"""Simple Test for PATO-Qwen2.5-VL Forward/Backward

This script tests:
1. Model initialization
2. Forward pass with dummy data
3. Backward pass with gradient computation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from pato_integration.pato_config import PATOQwen2_5_VLConfig, PATOConfig
from pato_integration.pato_model import PATOQwen2_5_VLModel
from pato_integration.loss import PATOLoss


def test_forward_backward():
    """Test forward and backward pass"""
    
    print("="*60)
    print("PATO-Qwen2.5-VL Forward/Backward Test")
    print("="*60)
    
    # ============================================================
    # Step 1: Create Configuration
    # ============================================================
    print("\n[1/5] Creating configuration...")
    
    # Create minimal config for testing
    config = PATOQwen2_5_VLConfig(
        # Use small dimensions for testing
        vision_config={
            'hidden_size': 256,
            'num_heads': 4,
            'depth': 2,  # Only 2 layers for speed
            'spatial_merge_size': 2,
        },
        text_config={
            'hidden_size': 512,
            'intermediate_size': 1024,
            'num_hidden_layers': 2,  # Only 2 layers for speed
            'num_attention_heads': 8,
            'vocab_size': 1000,
        },
    )
    
    # Set PATO config
    config.pato_config.g_raw.enable = True
    config.pato_config.g_raw.target_size = (224, 224)  # Smaller for testing
    config.pato_config.g_raw.vision_dim = 128
    config.pato_config.g_raw.text_dim = config.text_config.hidden_size
    
    config.pato_config.token_sort.enable = True
    config.pato_config.token_sort.budgets = [64]  # Small budget for testing
    config.pato_config.token_sort.scorer_hidden_dim = 128
    
    config.pato_config.projector.vision_dim = config.vision_config.hidden_size
    config.pato_config.projector.hidden_dim = config.text_config.hidden_size
    
    # Freeze settings for PATO-only training
    config.pato_config.freeze_vision_encoder = True
    config.pato_config.freeze_llm = True
    config.pato_config.freeze_embeddings = True
    
    print(f"  Config created:")
    print(f"    Vision hidden: {config.vision_config.hidden_size}")
    print(f"    Text hidden: {config.text_config.hidden_size}")
    print(f"    g_raw enabled: {config.pato_config.g_raw.enable}")
    print(f"    Token sort enabled: {config.pato_config.token_sort.enable}")
    print(f"    Token budget: {config.pato_config.token_sort.budgets[0]}")
    
    # ============================================================
    # Step 2: Initialize Model
    # ============================================================
    print("\n[2/5] Initializing model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    try:
        model = PATOQwen2_5_VLModel(config)
        model = model.to(device)
        model.train()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Model initialized successfully!")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        print(f"    Frozen percentage: {100 * (1 - trainable_params/total_params):.1f}%")
        
    except Exception as e:
        print(f"  ERROR: Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Step 3: Create Dummy Input
    # ============================================================
    print("\n[3/5] Creating dummy input...")
    
    batch_size = 2
    seq_len = 32
    image_size = 224
    
    # Dummy pixel values
    pixel_values = torch.randn(
        batch_size, 3, image_size, image_size,
        device=device, dtype=torch.float32
    )
    
    # Dummy input IDs
    input_ids = torch.randint(
        0, config.text_config.vocab_size,
        (batch_size, seq_len),
        device=device
    )
    
    # Insert image token (assuming token ID 0 is image token)
    config.image_token_id = 0
    input_ids[:, 10] = config.image_token_id
    
    # Attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Grid THW (temporal, height, width)
    # For testing, assume single image per batch
    image_grid_thw = torch.tensor(
        [[1, image_size // 14, image_size // 14]],  # 1 temporal, H/patch_size, W/patch_size
        device=device
    ).repeat(batch_size, 1)
    
    # Labels for language modeling
    labels = input_ids.clone()
    labels[:, :11] = -100  # Ignore first tokens including image
    
    print(f"  Input created:")
    print(f"    Batch size: {batch_size}")
    print(f"    Sequence length: {seq_len}")
    print(f"    Image size: {image_size}×{image_size}")
    print(f"    Pixel values shape: {pixel_values.shape}")
    print(f"    Input IDs shape: {input_ids.shape}")
    print(f"    Grid THW shape: {image_grid_thw.shape}")
    
    # ============================================================
    # Step 4: Forward Pass
    # ============================================================
    print("\n[4/5] Running forward pass...")
    
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        
        print(f"  Forward pass successful!")
        print(f"    Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
        
        # Check auxiliary outputs
        if hasattr(outputs, 'aux_outputs'):
            print(f"    Auxiliary outputs: {list(outputs.aux_outputs.keys())}")
        
    except Exception as e:
        print(f"  ERROR: Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Step 5: Compute Loss and Backward
    # ============================================================
    print("\n[5/5] Computing loss and backward pass...")
    
    try:
        # Create loss function
        pato_loss = PATOLoss(
            lambda_distill=config.pato_config.lambda_distill,
            lambda_sort_reg=config.pato_config.lambda_sort_reg,
            lambda_contrast=config.pato_config.lambda_contrast,
        )
        
        # Compute language modeling loss
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Shift for LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        print(f"  Language modeling loss: {lm_loss.item():.4f}")
        
        # Compute PATO losses
        aux_outputs = outputs.aux_outputs if hasattr(outputs, 'aux_outputs') else None
        
        losses = pato_loss(
            lm_loss=lm_loss,
            aux_outputs=aux_outputs,
            g_raw_module=model.g_raw if hasattr(model, 'g_raw') else None,
            token_sorter=model.visual.token_sorter if hasattr(model.visual, 'token_sorter') else None,
        )
        
        total_loss = losses['total_loss']
        
        print(f"  Loss components:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.item():.4f}")
        
        print(f"\n  Running backward pass...")
        total_loss.backward()
        
        # Check gradients
        has_grad = False
        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if len(grad_norms) <= 3:  # Print first 3
                    print(f"    {name}: grad_norm = {grad_norm:.6f}")
        
        if has_grad:
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            print(f"  Average gradient norm: {avg_grad_norm:.6f}")
            print(f"  ✓ Backward pass successful!")
        else:
            print(f"  WARNING: No gradients computed!")
            return False
        
    except Exception as e:
        print(f"  ERROR: Loss computation or backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Model initialized with {trainable_params:,} trainable parameters")
    print(f"  - Forward pass completed successfully")
    print(f"  - Loss computed: {total_loss.item():.4f}")
    print(f"  - Backward pass completed with gradients")
    print("\nNext steps:")
    print("  1. Test with real images and text")
    print("  2. Implement data loader for VQA dataset")
    print("  3. Create training script")
    print("  4. Run inference test")
    
    return True


if __name__ == "__main__":
    success = test_forward_backward()
    exit(0 if success else 1)
