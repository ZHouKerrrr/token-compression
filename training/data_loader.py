"""VQA Dataset Loader for PATO Training

支持 VQAv2 和 TextVQA 数据集的加载和预处理。
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class VQADataset(Dataset):
    """VQA Dataset for PATO Training
    
    支持的数据格式:
    - VQAv2: COCO-style annotations
    - TextVQA: TextVQA format
    - 自定义格式: {"image": path, "question": str, "answer": str}
    
    Args:
        data_path: 数据集路径 (JSON文件或目录)
        image_dir: 图像目录
        processor: Qwen2.5-VL processor
        max_samples: 最大样本数 (用于快速测试)
        transform: 图像变换 (可选)
        dataset_type: 数据集类型 ('vqav2', 'textvqa', 'custom')
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        processor,
        max_samples: Optional[int] = None,
        transform: Optional[T.Compose] = None,
        dataset_type: str = 'custom'
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_samples = max_samples
        self.transform = transform
        self.dataset_type = dataset_type
        
        # 加载数据
        self.samples = self._load_data()
        
        # 限制样本数量
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {self.data_path}")
    
    def _load_data(self) -> List[Dict]:
        """加载数据集"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # 读取JSON文件
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 根据数据集类型解析
        if self.dataset_type == 'vqav2':
            return self._parse_vqav2(data)
        elif self.dataset_type == 'textvqa':
            return self._parse_textvqa(data)
        else:  # custom
            return self._parse_custom(data)
    
    def _parse_vqav2(self, data: Union[Dict, List]) -> List[Dict]:
        """解析 VQAv2 格式"""
        samples = []
        
        # VQAv2 通常有两个文件: questions 和 annotations
        # 这里假设已经合并或者只提供questions
        if isinstance(data, dict) and 'annotations' in data:
            annotations = data['annotations']
        elif isinstance(data, list):
            annotations = data
        else:
            raise ValueError(f"Unsupported VQAv2 format: {type(data)}")
        
        for item in annotations:
            image_id = item.get('image_id')
            question = item.get('question', '')
            
            # 答案处理
            if 'multiple_choice_answer' in item:
                answer = item['multiple_choice_answer']
            elif 'answers' in item and len(item['answers']) > 0:
                answer = item['answers'][0].get('answer', '')
            else:
                answer = ''
            
            # 图像路径
            image_path = self.image_dir / f"COCO_train2014_{image_id:012d}.jpg"
            if not image_path.exists():
                image_path = self.image_dir / f"COCO_val2014_{image_id:012d}.jpg"
            
            if image_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'question': question,
                    'answer': answer,
                    'image_id': image_id
                })
        
        return samples
    
    def _parse_textvqa(self, data: Union[Dict, List]) -> List[Dict]:
        """解析 TextVQA 格式"""
        samples = []
        
        if isinstance(data, dict) and 'data' in data:
            data_list = data['data']
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError(f"Unsupported TextVQA format: {type(data)}")
        
        for item in data_list:
            image_path = self.image_dir / item.get('image_id', item.get('image', ''))
            question = item.get('question', '')
            
            # 答案处理
            if 'answers' in item:
                answers = item['answers']
                answer = answers[0] if answers else ''
            else:
                answer = item.get('answer', '')
            
            if image_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'question': question,
                    'answer': answer
                })
        
        return samples
    
    def _parse_custom(self, data: Union[Dict, List]) -> List[Dict]:
        """解析自定义格式
        
        期望格式: List[{"image": path, "question": str, "answer": str}]
        """
        samples = []
        
        if isinstance(data, dict) and 'samples' in data:
            data_list = data['samples']
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError(f"Unsupported custom format: {type(data)}")
        
        for item in data_list:
            # 支持多种键名
            image_key = 'image' if 'image' in item else 'image_path'
            image_path = Path(item[image_key])
            
            # 如果是相对路径，补全
            if not image_path.is_absolute():
                image_path = self.image_dir / image_path
            
            if image_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'question': item.get('question', ''),
                    'answer': item.get('answer', '')
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本
        
        Returns:
            Dict containing:
            - pixel_values: 图像tensor
            - input_ids: 文本token IDs
            - attention_mask: attention mask
            - labels: 答案的token IDs (for training)
            - image_grid_thw: 图像网格信息
        """
        sample = self.samples[idx]
        
        try:
            # 加载图像
            image = Image.open(sample['image_path']).convert('RGB')
            
            # 应用自定义变换
            if self.transform is not None:
                image = self.transform(image)
            
            # 构建输入文本 (question)
            question = sample['question']
            text = f"Question: {question}\nAnswer:"
            
            # 使用 processor 处理
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            # 处理答案 (用于训练)
            answer = sample['answer']
            answer_inputs = self.processor(
                text=[answer],
                return_tensors="pt",
                padding=True
            )
            
            # 构建返回字典
            result = {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': answer_inputs['input_ids'].squeeze(0),
                'image_path': sample['image_path'],
                'question': question,
                'answer': answer
            }
            
            # 添加图像网格信息（如果有）
            if 'image_grid_thw' in inputs:
                result['image_grid_thw'] = inputs['image_grid_thw'].squeeze(0)
            
            return result
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self))
    
    def get_raw_sample(self, idx: int) -> Tuple[Image.Image, str, str]:
        """获取原始样本（不经过processor）
        
        Returns:
            (image, question, answer)
        """
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        return image, sample['question'], sample['answer']


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """自定义 collate 函数，处理变长序列
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched tensors with proper padding
    """
    # 提取所有字段
    pixel_values = [item['pixel_values'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Stack pixel values (assuming same shape)
    # 注意: Qwen2.5-VL 的 pixel_values 可能是变长的
    # 这里我们简单处理，实际使用时可能需要更复杂的padding
    try:
        pixel_values_stacked = torch.stack(pixel_values, dim=0)
    except:
        # 如果形状不同，保留为list
        pixel_values_stacked = pixel_values
    
    result = {
        'pixel_values': pixel_values_stacked,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
    }
    
    # 添加额外信息
    if 'image_grid_thw' in batch[0]:
        result['image_grid_thw'] = [item['image_grid_thw'] for item in batch]
    
    result['questions'] = [item['question'] for item in batch]
    result['answers'] = [item['answer'] for item in batch]
    result['image_paths'] = [item['image_path'] for item in batch]
    
    return result


def create_vqa_dataloader(
    data_path: str,
    image_dir: str,
    processor,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    dataset_type: str = 'custom'
) -> DataLoader:
    """创建 VQA DataLoader
    
    Args:
        data_path: 数据集JSON文件路径
        image_dir: 图像目录
        processor: Qwen2.5-VL processor
        batch_size: batch大小
        max_samples: 最大样本数
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        dataset_type: 数据集类型
    
    Returns:
        DataLoader instance
    """
    dataset = VQADataset(
        data_path=data_path,
        image_dir=image_dir,
        processor=processor,
        max_samples=max_samples,
        dataset_type=dataset_type
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


# =============================================================================
# 创建演示/测试数据集
# =============================================================================

def create_demo_dataset(
    output_path: str = "./demo_vqa_data.json",
    num_samples: int = 100
):
    """创建演示用的VQA数据集
    
    生成简单的合成数据用于快速测试。
    
    Args:
        output_path: 输出JSON文件路径
        num_samples: 生成的样本数量
    """
    import random
    
    # 定义一些模板
    questions = [
        "What color is the {object}?",
        "How many {object}s are in the image?",
        "Where is the {object} located?",
        "What is the {object} doing?",
        "Is there a {object} in the image?",
    ]
    
    objects = ["cat", "dog", "car", "tree", "person", "house", "bird", "flower"]
    colors = ["red", "blue", "green", "yellow", "white", "black", "brown"]
    numbers = ["one", "two", "three", "four", "five"]
    locations = ["left", "right", "center", "top", "bottom"]
    actions = ["sitting", "running", "standing", "sleeping", "eating"]
    
    samples = []
    for i in range(num_samples):
        obj = random.choice(objects)
        q_template = random.choice(questions)
        question = q_template.format(object=obj)
        
        # 生成对应的答案
        if "color" in question:
            answer = random.choice(colors)
        elif "how many" in question.lower():
            answer = random.choice(numbers)
        elif "where" in question.lower():
            answer = random.choice(locations)
        elif "doing" in question:
            answer = random.choice(actions)
        elif "is there" in question.lower():
            answer = random.choice(["yes", "no"])
        else:
            answer = "unknown"
        
        samples.append({
            "image": f"demo_image_{i:04d}.jpg",  # 占位符
            "question": question,
            "answer": answer
        })
    
    # 保存
    data = {"samples": samples}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Created demo dataset with {num_samples} samples at {output_path}")
    return output_path


if __name__ == "__main__":
    # 测试代码
    print("VQA Dataset Loader Test")
    print("=" * 60)
    
    # 创建演示数据集
    demo_data_path = create_demo_dataset(num_samples=10)
    
    # 注意: 实际使用需要真实的processor和图像
    # 这里只是展示数据集的结构
    with open(demo_data_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nDemo data structure:")
    print(f"Total samples: {len(data['samples'])}")
    print(f"\nFirst sample:")
    print(json.dumps(data['samples'][0], indent=2))
    
    print("\n✓ Data loader module ready!")
    print("\nUsage:")
    print("""
    from transformers import AutoProcessor
    from training.data_loader import create_vqa_dataloader
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    dataloader = create_vqa_dataloader(
        data_path="path/to/vqa_data.json",
        image_dir="path/to/images",
        processor=processor,
        batch_size=4,
        max_samples=1000
    )
    
    for batch in dataloader:
        # Training loop
        pass
    """)
