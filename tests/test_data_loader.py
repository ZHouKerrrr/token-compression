"""
快速测试VQA数据加载器
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_large_scale import VQADataset
from torch.utils.data import DataLoader

print("="*60)
print("VQA Data Loader Test")
print("="*60)

# 测试TextVQA数据集
print("\n[1/3] Testing TextVQA dataset...")
textvqa_dataset = VQADataset(
    image_dir='/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data',
    annotation_file='/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data/textvqa_cot_train.jsonl',
    max_samples=10
)

print(f"  ✓ Loaded {len(textvqa_dataset)} samples")
print("\n  Sample 0:")
sample = textvqa_dataset[0]
print(f"    • Image shape: {sample['image'].shape}")
print(f"    • Question: {sample['question'][:80]}...")
print(f"    • Answer: {sample['answer']}")

# 测试GQA数据集
print("\n[2/3] Testing GQA dataset...")
gqa_dataset = VQADataset(
    image_dir='/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data',
    annotation_file='/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data/gqa_cot_train_brief_alpaca.jsonl',
    max_samples=10
)

print(f"  ✓ Loaded {len(gqa_dataset)} samples")
print("\n  Sample 0:")
sample = gqa_dataset[0]
print(f"    • Image shape: {sample['image'].shape}")
print(f"    • Question: {sample['question'][:80]}...")
print(f"    • Answer: {sample['answer']}")

# 测试DataLoader
print("\n[3/3] Testing DataLoader...")
dataloader = DataLoader(
    textvqa_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

for batch in dataloader:
    print(f"  ✓ Batch loaded")
    print(f"    • Images: {batch['image'].shape}")
    print(f"    • Questions: {len(batch['question'])} samples")
    print(f"    • Answers: {len(batch['answer'])} samples")
    break

print("\n" + "="*60)
print("✓ Data Loader Test PASSED!")
print("="*60)
print("\nAvailable datasets:")
print("  • TextVQA: 18,524 samples")
print("  • GQA: 98,149 samples")
print("  • DocVQA: 33,453 samples")
print("  • Flickr30k: 135,735 samples")
print("  • Total: 532,414 samples")
print("\nReady for large-scale training!")
