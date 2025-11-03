"""
创建简单的VQA数据集用于训练测试

从图像目录自动生成VQA样本
"""

import json
import os
from pathlib import Path
from PIL import Image
import random


def create_vqa_from_images(
    image_dir: str,
    output_json: str,
    max_samples: int = 1000,
    questions_per_image: int = 1
):
    """从图像目录创建VQA数据集
    
    Args:
        image_dir: 图像目录
        output_json: 输出JSON文件路径
        max_samples: 最大样本数
        questions_per_image: 每张图像的问题数
    """
    image_dir = Path(image_dir)
    
    # 查找所有图像
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    image_files = image_files[:max_samples]
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # 预定义的问题模板
    questions = [
        "What is shown in this image?",
        "Describe the main content of this image.",
        "What can you see in this picture?",
        "What is the primary subject of this image?",
        "Can you describe what's happening in this image?",
        "What objects are visible in this image?",
        "What is the scene depicted in this image?",
        "What are the key elements in this image?"
    ]
    
    samples = []
    for img_file in image_files:
        # 验证图像可以打开
        try:
            img = Image.open(img_file)
            img.verify()
            
            # 为每张图像生成问题
            for _ in range(questions_per_image):
                question = random.choice(questions)
                samples.append({
                    "image": img_file.name,  # 只保存文件名
                    "question": question,
                    "answer": "This requires visual understanding."  # 占位符答案
                })
        except Exception as e:
            print(f"  Skipping {img_file}: {e}")
            continue
    
    # 保存
    data = {"samples": samples}
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Created VQA dataset with {len(samples)} samples")
    print(f"Saved to: {output_json}")
    
    return output_json


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Image directory')
    parser.add_argument('--output', type=str, default='auto_vqa_data.json',
                       help='Output JSON file')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples')
    parser.add_argument('--questions_per_image', type=int, default=1,
                       help='Questions per image')
    
    args = parser.parse_args()
    
    create_vqa_from_images(
        image_dir=args.image_dir,
        output_json=args.output,
        max_samples=args.max_samples,
        questions_per_image=args.questions_per_image
    )
