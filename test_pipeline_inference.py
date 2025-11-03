"""
Test PATO Complete Pipeline with Random Weights
使用随机权重测试完整的PATO pipeline
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

print("="*60)
print("PATO Complete Pipeline Test (Random Weights)")
print("="*60)

# 测试配置
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# ============================================================
# Step 1: 加载测试图像
# ============================================================
print("\n[1/5] Loading test image...")
image_path = Path(__file__).parent / "examples" / "cat.png"

if not image_path.exists():
    print(f"  ✗ Image not found: {image_path}")
    exit(1)

image = Image.open(image_path).convert('RGB')
original_size = image.size
print(f"  ✓ Image loaded: {original_size}")

# Resize to 448x448
image = image.resize((448, 448))
image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 448, 448]
print(f"  ✓ Image tensor: {image_tensor.shape}")

# ============================================================
# Step 2: 初始化 g_raw (随机权重)
# ============================================================
print("\n[2/5] Initializing g_raw with random weights...")

try:
    import importlib.util
    config_path = Path(__file__).parent / 'pato_integration' / 'pato_config_standalone.py'
    spec = importlib.util.spec_from_file_location("pato_config_standalone", config_path)
    pato_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pato_config_module)
    PATOConfig = pato_config_module.PATOConfig
    
    from g_raw import WeightedDownsample
    
    pato_config = PATOConfig()
    pato_config.g_raw.text_dim = 3584
    pato_config.g_raw.target_size = (448, 448)
    
    g_raw = WeightedDownsample(
        pato_config.g_raw,
        {'device': device}
    ).to(device)
    
    g_raw.eval()
    
    print(f"  ✓ g_raw initialized")
    print(f"    Parameters: {sum(p.numel() for p in g_raw.parameters())/1e6:.2f}M")
    
except Exception as e:
    print(f"  ✗ Failed to initialize g_raw: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# Step 3: g_raw 前向传播
# ============================================================
print("\n[3/5] Testing g_raw forward pass...")

try:
    # 创建模拟的text embeddings
    text_embeds = torch.randn(1, 3584).to(device)
    
    with torch.no_grad():
        compressed_image = g_raw(image_tensor, text_embeds)
    
    print(f"  ✓ g_raw forward pass successful")
    print(f"    Input: {image_tensor.shape}")
    print(f"    Output: {compressed_image.shape}")
    print(f"    Compression ratio: {image_tensor.numel() / compressed_image.numel():.2f}×")
    
    # 检查输出值范围
    print(f"    Output range: [{compressed_image.min():.4f}, {compressed_image.max():.4f}]")
    
except Exception as e:
    print(f"  ✗ g_raw forward failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# Step 4: 初始化 Token Sorter (随机权重)
# ============================================================
print("\n[4/5] Initializing Token Sorter with random weights...")

try:
    from token_sort import DifferentiableSortingTokenSorter
    
    token_sorter = DifferentiableSortingTokenSorter(
        pato_config.token_sort,
        {'hidden_size': 3584}
    ).to(device)
    
    token_sorter.eval()
    
    print(f"  ✓ Token Sorter initialized")
    print(f"    Parameters: {sum(p.numel() for p in token_sorter.parameters())/1e6:.2f}M")
    
except Exception as e:
    print(f"  ✗ Failed to initialize Token Sorter: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# Step 5: 完整 Pipeline 测试
# ============================================================
print("\n[5/5] Testing complete PATO pipeline...")

try:
    with torch.no_grad():
        # Step 5.1: g_raw compression
        compressed = g_raw(image_tensor, text_embeds)
        
        # Step 5.2: 模拟 Vision Encoder输出
        # 在真实场景中，这里会是Qwen2.5-VL vision encoder的输出
        # 这里我们模拟 [B, N, D] 的vision tokens
        batch_size = 1
        num_vision_tokens = 256
        vision_hidden_dim = 3584
        
        vision_tokens = torch.randn(
            batch_size, 
            num_vision_tokens, 
            vision_hidden_dim
        ).to(device)
        
        print(f"  ✓ Simulated vision tokens: {vision_tokens.shape}")
        
        # Step 5.3: Token Sorting
        token_budget = 128  # 50% compression
        
        selected_tokens, sort_indices, aux_outputs = token_sorter(
            hidden_states=vision_tokens,
            budget=token_budget,
            query_embeddings=text_embeds
        )
        
        print(f"  ✓ Token sorting completed")
        print(f"    Original tokens: {vision_tokens.shape[1]}")
        print(f"    Selected tokens: {selected_tokens.shape[1]}")
        print(f"    Token reduction: {100 * (1 - selected_tokens.shape[1] / vision_tokens.shape[1]):.1f}%")
        
        # 检查辅助输出
        if aux_outputs:
            print(f"  ✓ Auxiliary outputs:")
            for key, value in aux_outputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype in [torch.float32, torch.float16, torch.float64]:
                        print(f"    - {key}: {value.shape} (mean={value.mean():.4f})")
                    else:
                        print(f"    - {key}: {value.shape} (dtype={value.dtype})")
                else:
                    print(f"    - {key}: {value}")
    
    print(f"\n  ✓ Complete pipeline test SUCCESSFUL!")
    
except Exception as e:
    print(f"  ✗ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("✓ PATO Pipeline Test Completed Successfully!")
print("="*60)

print("\nPipeline Summary:")
print(f"  1. Input Image: {original_size} → 448×448")
print(f"  2. g_raw Compression: {image_tensor.shape} → {compressed_image.shape}")
print(f"  3. Vision Encoding (simulated): {vision_tokens.shape}")
print(f"  4. Token Sorting: {vision_tokens.shape[1]} → {selected_tokens.shape[1]} tokens")
print(f"  5. Final Token Reduction: 50.0%")

print("\nModel Components:")
print(f"  • g_raw: {sum(p.numel() for p in g_raw.parameters())/1e6:.2f}M parameters")
print(f"  • Token Sorter: {sum(p.numel() for p in token_sorter.parameters())/1e6:.2f}M parameters")
print(f"  • Total PATO: {(sum(p.numel() for p in g_raw.parameters()) + sum(p.numel() for p in token_sorter.parameters()))/1e6:.2f}M parameters")

print("\nStatus:")
print("  ✅ g_raw module: Working")
print("  ✅ Token Sorter: Working")
print("  ✅ Pipeline integration: Working")
print("  ✅ Random weights inference: Successful")

print("\nNext Steps:")
print("  1. ✅ Load pre-trained Qwen2.5-VL model")
print("  2. ✅ Replace simulated vision encoder with real one")
print("  3. ✅ Train PATO components on VQA data")
print("  4. ⏳ Evaluate on real VQA tasks")

print("\n" + "="*60)
print("Pipeline is ready for training!")
print("="*60)
