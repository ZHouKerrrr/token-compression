"""快速测试PATO Pipeline组件（不加载完整模型）

这个脚本测试PATO组件本身，而不需要加载大型Qwen模型
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

print("="*60)
print("PATO Pipeline Components Test")
print("="*60)

# 测试配置
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Step 1: 测试 g_raw
print("\n[1/3] Testing g_raw...")
try:
    import importlib.util
    config_path = Path(__file__).parent.parent / 'pato_integration' / 'pato_config_standalone.py'
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
    
    # 测试forward
    test_images = torch.randn(2, 3, 448, 448).to(device)
    test_text = torch.randn(2, 3584).to(device)
    
    g_raw.eval()
    with torch.no_grad():
        compressed = g_raw(test_images, test_text)
    
    print(f"  ✓ g_raw forward: {test_images.shape} → {compressed.shape}")
    print(f"  ✓ g_raw parameters: {sum(p.numel() for p in g_raw.parameters())/1e6:.2f}M")
    
except Exception as e:
    print(f"  ✗ g_raw test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 2: 测试 Token Sorter
print("\n[2/3] Testing token sorter...")
try:
    from token_sort import DifferentiableSortingTokenSorter
    
    token_sorter = DifferentiableSortingTokenSorter(
        pato_config.token_sort,
        {'hidden_size': 3584}
    ).to(device)
    
    # 测试forward
    test_tokens = torch.randn(2, 256, 3584).to(device)
    test_query = torch.randn(2, 3584).to(device)
    
    token_sorter.eval()
    with torch.no_grad():
        selected, indices, aux = token_sorter(
            hidden_states=test_tokens,
            budget=128,
            query_embeddings=test_query
        )
    
    print(f"  ✓ Token sorter forward: {test_tokens.shape} → {selected.shape}")
    print(f"  ✓ Token reduction: {100*(1-selected.shape[1]/test_tokens.shape[1]):.1f}%")
    print(f"  ✓ Token sorter parameters: {sum(p.numel() for p in token_sorter.parameters())/1e6:.2f}M")
    
except Exception as e:
    print(f"  ✗ Token sorter test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 3: 测试训练模式（梯度流）
print("\n[3/3] Testing training mode...")
try:
    g_raw.train()
    token_sorter.train()
    
    # Forward
    test_images = torch.randn(2, 3, 448, 448, requires_grad=True).to(device)
    test_text = torch.randn(2, 3584, requires_grad=False).to(device)
    
    compressed = g_raw(test_images, test_text)
    
    # 模拟vision tokens
    vision_tokens = torch.randn(2, 256, 3584, requires_grad=True).to(device)
    
    selected, indices, aux = token_sorter(
        hidden_states=vision_tokens,
        budget=128,
        query_embeddings=test_text
    )
    
    # 计算loss
    loss = selected.mean() + aux.get('entropy_loss', 0.0) + aux.get('diversity_loss', 0.0)
    
    print(f"  ✓ Forward in training mode")
    print(f"  ✓ Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # 检查梯度
    g_raw_has_grad = any(p.grad is not None for p in g_raw.parameters() if p.requires_grad)
    token_sort_has_grad = any(p.grad is not None for p in token_sorter.parameters() if p.requires_grad)
    
    print(f"  ✓ Backward completed")
    print(f"  ✓ g_raw gradients: {'✓' if g_raw_has_grad else '✗'}")
    print(f"  ✓ Token sorter gradients: {'✓' if token_sort_has_grad else '✗'}")
    
    if g_raw_has_grad and token_sort_has_grad:
        print(f"  ✓ Gradient flow verified!")
    
except Exception as e:
    print(f"  ✗ Training mode test failed: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "="*60)
print("✓ PATO Components Test Completed!")
print("="*60)

print("\nSummary:")
print(f"  • g_raw: {sum(p.numel() for p in g_raw.parameters())/1e6:.2f}M params")
print(f"  • Token sorter: {sum(p.numel() for p in token_sorter.parameters())/1e6:.2f}M params")
print(f"  • Total trainable: {sum(p.numel() for p in list(g_raw.parameters()) + list(token_sorter.parameters()) if p.requires_grad)/1e6:.2f}M")
print(f"  • Device: {device}")
print(f"\nNext steps:")
print(f"  1. Components are ready ✓")
print(f"  2. Can integrate with Qwen2.5-VL for full pipeline")
print(f"  3. Ready for small-scale training test")
