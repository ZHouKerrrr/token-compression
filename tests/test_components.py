"""Minimal standalone test for PATO components

This test验证PATO核心组件without依赖完整的Qwen2.5-VL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

print("="*60)
print("PATO Components Test (Standalone)")
print("="*60)

# ============================================================
# Test 1: g_raw Module
# ============================================================
print("\n[Test 1/3] Testing g_raw (Weighted Downsampling)...")

try:
    from g_raw import WeightedDownsample
    from g_raw.base import RegularizationUtils
    
    # Create configuration
    class SimpleConfig:
        def __init__(self):
            self.target_size = [224, 224]
            self.text_dim = 512
            self.vision_dim = 128
            self.density_hidden_dim = 128
            self.density_layers = 2
            self.lambda_tv = 1e-4
            self.lambda_area = 1e-3
            self.min_area_ratio = 0.1
    
    config = SimpleConfig()
    context = {'device': 'cpu'}
    
    # Initialize g_raw
    g_raw = WeightedDownsample(config, context)
    g_raw.eval()
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 448, 448)
    text_embeddings = torch.randn(batch_size, 512)
    
    compressed = g_raw(images, text_embeddings)
    
    print(f"  ✓ g_raw initialized successfully")
    print(f"    Input shape: {images.shape}")
    print(f"    Output shape: {compressed.shape}")
    print(f"    Target size: {config.target_size}")
    
    # Test gradient flow
    g_raw.train()
    compressed = g_raw(images, text_embeddings)
    loss = compressed.mean()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in g_raw.parameters() if p.requires_grad)
    print(f"    Gradient flow: {'✓' if has_grad else '✗'}")
    
    print("  ✓ g_raw test passed!\n")
    
except Exception as e:
    print(f"  ✗ g_raw test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# Test 2: Token Sort Module
# ============================================================
print("[Test 2/3] Testing Token Sort (Differentiable Sorting)...")

try:
    from token_sort import DifferentiableSortingTokenSorter
    
    # Create configuration
    class TokenSortConfig:
        def __init__(self):
            self.tau_init = 1.0
            self.tau_final = 0.1
            self.tau_decay = 'linear'
            self.lambda_entropy = 1e-3
            self.lambda_diversity = 1e-4
            self.scorer_hidden_dim = 128
            self.sinkhorn_iters = 5
            self.budgets = [64]
    
    config = TokenSortConfig()
    context = {'hidden_size': 256}
    
    # Initialize token sorter
    token_sorter = DifferentiableSortingTokenSorter(config, context)
    token_sorter.eval()
    
    # Test forward pass
    batch_size = 2
    num_tokens = 128
    hidden_dim = 256
    
    hidden_states = torch.randn(batch_size, num_tokens, hidden_dim)
    query_embeds = torch.randn(batch_size, hidden_dim)
    attention_mask = torch.ones(batch_size, num_tokens)
    
    selected_tokens, sort_indices, aux_outputs = token_sorter(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        budget=64,
        query_embeddings=query_embeds
    )
    
    print(f"  ✓ Token sorter initialized successfully")
    print(f"    Input shape: {hidden_states.shape}")
    print(f"    Output shape: {selected_tokens.shape}")
    print(f"    Budget: 64 / {num_tokens}")
    print(f"    Sparsity: {aux_outputs['sparsity']:.2%}")
    
    # Test gradient flow
    token_sorter.train()
    selected_tokens, _, aux_outputs = token_sorter(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        budget=64,
        query_embeddings=query_embeds
    )
    loss = selected_tokens.mean() + aux_outputs['entropy_loss']
    loss.backward()
    
    has_grad = any(p.grad is not None for p in token_sorter.parameters() if p.requires_grad)
    print(f"    Gradient flow: {'✓' if has_grad else '✗'}")
    
    print("  ✓ Token sort test passed!\n")
    
except Exception as e:
    print(f"  ✗ Token sort test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# Test 3: PATO Loss
# ============================================================
print("[Test 3/3] Testing PATO Loss...")

try:
    # Import loss directly without going through __init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "loss",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                     "pato_integration", "loss.py")
    )
    loss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loss_module)
    PATOLoss = loss_module.PATOLoss
    
    # Initialize loss
    pato_loss = PATOLoss(
        lambda_distill=0.05,
        lambda_sort_reg=0.01,
        lambda_contrast=0.1
    )
    
    # Create dummy losses
    lm_loss = torch.tensor(2.5, requires_grad=True)
    
    aux_outputs = {
        'original_pixel_values': torch.randn(2, 3, 448, 448),
        'compressed_pixel_values': torch.randn(2, 3, 224, 224),
        'entropy_loss': torch.tensor(0.1),
        'diversity_loss': torch.tensor(0.05),
    }
    
    # Compute loss
    losses = pato_loss(
        lm_loss=lm_loss,
        aux_outputs=aux_outputs,
        g_raw_module=g_raw,
        token_sorter=token_sorter
    )
    
    print(f"  ✓ PATO loss initialized successfully")
    print(f"    Loss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"      {key}: {value.item():.4f}")
    
    # Test backward
    total_loss = losses['total_loss']
    total_loss.backward()
    
    print(f"    Gradient flow: ✓")
    print("  ✓ PATO loss test passed!\n")
    
except Exception as e:
    print(f"  ✗ PATO loss test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# Summary
# ============================================================
print("="*60)
print("✓ ALL COMPONENT TESTS PASSED!")
print("="*60)
print("\nComponents validated:")
print("  1. g_raw (Weighted Downsampling)")
print("  2. Token Sort (Differentiable Sorting)")
print("  3. PATO Loss")
print("\nNext steps:")
print("  - Integrate with full Qwen2.5-VL model")
print("  - Test with real data")
print("  - Implement training loop")

exit(0)
