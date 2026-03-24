"""Complete PATO Training Pipeline

这个脚本实现了完整的PATO训练pipeline，包括:
1. 完整的forward pass (g_raw → vision encoding → token sort)
2. 知识蒸馏loss计算
3. 处理Qwen2.5-VL特殊格式
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as T

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from g_raw import WeightedDownsample
from token_sort import DifferentiableSortingTokenSorter


class CompletePATOPipeline(nn.Module):
    """完整的PATO Pipeline
    
    实现从原始图像到压缩视觉token的完整流程
    """
    
    def __init__(
        self,
        base_model,
        processor,
        g_raw: WeightedDownsample,
        token_sorter: DifferentiableSortingTokenSorter,
        device: str = 'cuda'
    ):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.g_raw = g_raw
        self.token_sorter = token_sorter
        self.device = device
        
        # 图像预处理
        self.image_transform = T.Compose([
            T.ToTensor(),
        ])
    
    def forward(
        self,
        raw_images: list,  # List of PIL Images
        text_embeddings: torch.Tensor,  # [B, D]
        token_budget: int = 256,
        return_intermediates: bool = False
    ):
        """完整前向传播
        
        Args:
            raw_images: List of PIL Images
            text_embeddings: 文本嵌入 [B, D]
            token_budget: Token预算
            return_intermediates: 是否返回中间结果
        
        Returns:
            selected_tokens: 压缩后的视觉token [B, M, D]
            aux_outputs: 辅助输出
            intermediates: 中间结果(如果return_intermediates=True)
        """
        batch_size = len(raw_images)
        intermediates = {}
        
        # Step 1: 将PIL Images转为tensor
        image_tensors = []
        for img in raw_images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img_tensor = self.image_transform(img)
            image_tensors.append(img_tensor)
        
        image_tensors = torch.stack(image_tensors).to(self.device)  # [B, 3, H, W]
        intermediates['raw_images'] = image_tensors
        
        # Step 2: g_raw 图像压缩
        with torch.set_grad_enabled(self.training):
            compressed_images = self.g_raw(image_tensors, text_embeddings)
        intermediates['compressed_images'] = compressed_images
        
        # Step 3: 将压缩图像转回PIL格式并通过processor
        compressed_pil_images = []
        for i in range(batch_size):
            # [3, H, W] -> [H, W, 3]
            img_np = compressed_images[i].detach().cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            compressed_pil_images.append(Image.fromarray(img_np))
        
        # Step 4: 通过processor处理
        # 注意: processor需要text输入，我们使用dummy text
        dummy_texts = [""] * batch_size
        
        with torch.no_grad():
            processed_inputs = self.processor(
                text=dummy_texts,
                images=compressed_pil_images,
                return_tensors="pt",
                padding=True
            )
        
        # Step 5: Vision encoding (使用冻结的encoder)
        with torch.no_grad():
            pixel_values = processed_inputs['pixel_values'].to(self.device)
            image_grid_thw = processed_inputs.get('image_grid_thw', None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.device)
            
            # 调用visual encoder
            vision_outputs = self.base_model.visual(
                pixel_values,
                grid_thw=image_grid_thw
            )
            
            if isinstance(vision_outputs, tuple):
                vision_tokens = vision_outputs[0]
            else:
                vision_tokens = vision_outputs
            
            # Reshape to [B, N, D] if needed
            if vision_tokens.dim() == 2:
                # 估计batch size (假设每个样本token数相同)
                tokens_per_sample = vision_tokens.shape[0] // batch_size
                vision_tokens = vision_tokens.view(batch_size, tokens_per_sample, -1)
            
            # Convert to float32
            vision_tokens = vision_tokens.float()
        
        intermediates['vision_tokens'] = vision_tokens
        
        # Step 6: Token sorting
        with torch.set_grad_enabled(self.training):
            selected_tokens, sort_indices, aux_outputs = self.token_sorter(
                hidden_states=vision_tokens,
                budget=token_budget,
                query_embeddings=text_embeddings
            )
        
        #step 7: Language model decoding (optional, not implemented here)
        

        intermediates['selected_tokens'] = selected_tokens
        intermediates['sort_indices'] = sort_indices
        
        if return_intermediates:
            return selected_tokens, aux_outputs, intermediates
        else:
            return selected_tokens, aux_outputs


class PATOTrainerV2:
    """改进的PATO训练器，包含完整pipeline"""
    
    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str = 'cuda'
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        
        print("="*60)
        print("PATO Trainer V2 - Complete Pipeline")
        print("="*60)
        
        # 加载模型
        print(f"\n[1/6] Loading Qwen2.5-VL model...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # 冻结基础模型
        print(f"\n[2/6] Freezing base model...")
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 获取配置
        model_config = self.base_model.config
        self.vision_hidden_size = model_config.vision_config.hidden_size
        self.text_hidden_size = (model_config.text_config.hidden_size 
                                 if hasattr(model_config, 'text_config') 
                                 else model_config.hidden_size)
        
        print(f"  Vision dim: {self.vision_hidden_size}")
        print(f"  Text dim: {self.text_hidden_size}")
        
        # 初始化PATO组件
        print(f"\n[3/6] Initializing PATO components...")
        self._init_pato_components()
        
        # 创建完整pipeline
        print(f"\n[4/6] Building complete pipeline...")
        self.pipeline = CompletePATOPipeline(
            base_model=self.base_model,
            processor=self.processor,
            g_raw=self.g_raw,
            token_sorter=self.token_sorter,
            device=device
        )
        
        # 初始化优化器和loss
        print(f"\n[5/6] Initializing optimizer and losses...")
        self._init_optimizer()
        self._init_losses()
        
        # 统计
        print(f"\n[6/6] Initialization complete!")
        trainable_params = self._count_trainable_params()
        print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
        
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def _init_pato_components(self):
        """初始化PATO组件"""
        import importlib.util
        config_path = Path(__file__).parent.parent / 'pato_integration' / 'pato_config_standalone.py'
        spec = importlib.util.spec_from_file_location("pato_config_standalone", config_path)
        pato_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pato_config_module)
        PATOConfig = pato_config_module.PATOConfig
        
        pato_config = PATOConfig()
        pato_config.g_raw.text_dim = self.text_hidden_size
        pato_config.g_raw.target_size = tuple(self.config.get('target_size', (448, 448)))
        pato_config.token_sort.budgets = self.config.get('token_budgets', [256])
        
        # 初始化g_raw
        self.g_raw = WeightedDownsample(
            pato_config.g_raw,
            {'device': self.device}
        ).to(self.device)
        
        # 初始化token_sorter (使用text_hidden_size)
        self.token_sorter = DifferentiableSortingTokenSorter(
            pato_config.token_sort,
            {'hidden_size': self.text_hidden_size}
        ).to(self.device)
        
        print(f"  ✓ g_raw initialized")
        print(f"  ✓ token_sorter initialized")
    
    def _init_optimizer(self):
        """初始化优化器"""
        trainable_params = []
        trainable_params += list(self.g_raw.parameters())
        trainable_params += list(self.token_sorter.parameters())
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        lr = self.config.get('learning_rate', 1e-4)
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('max_epochs', 10),
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        print(f"  ✓ Optimizer: AdamW (lr={lr})")
    
    def _init_losses(self):
        """初始化损失函数"""
        self.distill_weight = self.config.get('distill_weight', 0.5)
        self.budget_weight = self.config.get('budget_weight', 0.3)
        self.diversity_weight = self.config.get('diversity_weight', 0.2)
        
        print(f"  ✓ Loss weights: distill={self.distill_weight}, "
              f"budget={self.budget_weight}, diversity={self.diversity_weight}")
    
    def _count_trainable_params(self):
        """统计可训练参数"""
        total = 0
        for module in [self.g_raw, self.token_sorter]:
            total += sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total
    
    def compute_distillation_loss(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        """计算知识蒸馏损失
        
        Args:
            student_tokens: PATO压缩后的tokens [B, M, D]
            teacher_tokens: 原始vision tokens [B, N, D]
        
        Returns:
            distillation_loss
        """
        # 简单的MSE loss (可以改进为更复杂的蒸馏策略)
        # 由于token数量不同，我们对student做平均池化后比较
        student_mean = student_tokens.mean(dim=1)  # [B, D]
        teacher_mean = teacher_tokens.mean(dim=1)  # [B, D]
        
        loss = nn.functional.mse_loss(student_mean, teacher_mean)
        return loss
    
    def train_step(
        self,
        batch: Dict
    ) -> Dict:
        """单步训练
        
        Args:
            batch: 数据batch
        
        Returns:
            loss_dict: 各项损失
        """
        # 准备数据
        images = batch['images']  # List of PIL Images
        questions = batch['questions']  # List of strings
        
        # 获取文本嵌入
        with torch.no_grad():
            # 简单使用问题的embedding
            text_inputs = self.processor(
                text=questions,
                return_tensors="pt",
                padding=True
            )
            input_ids = text_inputs['input_ids'].to(self.device)
            text_embeds = self.base_model.model.embed_tokens(input_ids)
            text_embeds = text_embeds.mean(dim=1).float()  # [B, D]
        
        # 获取原始vision tokens (teacher)
        with torch.no_grad():
            original_inputs = self.processor(
                text=questions,
                images=images,
                return_tensors="pt",
                padding=True
            )
            pixel_values = original_inputs['pixel_values'].to(self.device)
            image_grid_thw = original_inputs.get('image_grid_thw', None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.device)
            
            vision_outputs = self.base_model.visual(
                pixel_values,
                grid_thw=image_grid_thw
            )
            
            if isinstance(vision_outputs, tuple):
                teacher_tokens = vision_outputs[0]
            else:
                teacher_tokens = vision_outputs
            
            if teacher_tokens.dim() == 2:
                batch_size = len(images)
                tokens_per_sample = teacher_tokens.shape[0] // batch_size
                teacher_tokens = teacher_tokens.view(batch_size, tokens_per_sample, -1)
            
            teacher_tokens = teacher_tokens.float()
        
        # PATO forward pass
        token_budget = self.config.get('token_budgets', [256])[0]
        student_tokens, aux_outputs = self.pipeline(
            raw_images=images,
            text_embeddings=text_embeds,
            token_budget=token_budget
        )
        
        # 计算损失
        # 1. 知识蒸馏损失
        distill_loss = self.compute_distillation_loss(student_tokens, teacher_tokens)
        
        # 2. Budget正则化损失 (从aux_outputs获取)
        budget_loss = aux_outputs.get('entropy_loss', 0.0)
        
        # 3. Diversity损失
        diversity_loss = aux_outputs.get('diversity_loss', 0.0)
        
        # 总损失
        total_loss = (
            self.distill_weight * distill_loss +
            self.budget_weight * budget_loss +
            self.diversity_weight * diversity_loss
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.g_raw.parameters() if p.requires_grad] +
            [p for p in self.token_sorter.parameters() if p.requires_grad],
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'distill_loss': distill_loss.item(),
            'budget_loss': budget_loss.item() if isinstance(budget_loss, torch.Tensor) else budget_loss,
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
            'token_reduction': 100 * (1 - student_tokens.shape[1] / teacher_tokens.shape[1])
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict:
        """训练一个epoch"""
        self.g_raw.train()
        self.token_sorter.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'distill_loss': 0.0,
            'budget_loss': 0.0,
            'diversity_loss': 0.0,
            'token_reduction': 0.0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                loss_dict = self.train_step(batch)
                
                # 更新统计
                for key, value in loss_dict.items():
                    epoch_losses[key] += value
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'distill': f"{loss_dict['distill_loss']:.4f}",
                    'token_red': f"{loss_dict['token_reduction']:.1f}%",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                self.global_step += 1
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算平均
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        epoch_losses['epoch'] = epoch
        epoch_losses['lr'] = self.optimizer.param_groups[0]['lr']
        
        return epoch_losses
    
    def save_checkpoint(self, save_dir: str, epoch: int, is_best: bool = False):
        """保存检查点"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'g_raw_state_dict': self.g_raw.state_dict(),
            'token_sorter_state_dict': self.token_sorter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"  ✓ Checkpoint saved: {ckpt_path}")
        
        if is_best:
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved: {best_path}")


# 简化的数据加载器用于测试
class SimplePATODataset:
    """简化的PATO数据集，用于快速测试"""
    
    def __init__(self, image_dir: str, num_samples: int = 100):
        self.image_dir = Path(image_dir)
        self.samples = []
        
        # 查找图像文件
        image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        image_files = image_files[:num_samples]
        
        for img_path in image_files:
            self.samples.append({
                'image_path': str(img_path),
                'question': "What is in this image?",
                'answer': "unknown"
            })
        
        print(f"Loaded {len(self.samples)} images from {image_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        return {
            'image': image,
            'question': sample['question'],
            'answer': sample['answer']
        }


def collate_fn_simple(batch):
    """简单的collate函数"""
    return {
        'images': [item['image'] for item in batch],
        'questions': [item['question'] for item in batch],
        'answers': [item['answer'] for item in batch]
    }


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='/data2/youneng/models/Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_v2')
    parser.add_argument('--device', type=str, default='cuda:3')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'target_size': (448, 448),
        'token_budgets': [256],
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'max_epochs': args.max_epochs,
        'distill_weight': 0.5,
        'budget_weight': 0.3,
        'diversity_weight': 0.2
    }
    
    # 创建trainer
    trainer = PATOTrainerV2(
        model_path=args.model_path,
        config=config,
        device=args.device
    )
    
    # 创建数据集
    print("\nCreating dataset...")
    dataset = SimplePATODataset(
        image_dir=args.image_dir,
        num_samples=args.max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_simple
    )
    
    print(f"  ✓ DataLoader ready: {len(dataloader)} batches")
    
    # 训练循环
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        print("-" * 60)
        
        train_stats = trainer.train_epoch(dataloader, epoch + 1)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: {train_stats['total_loss']:.4f}")
        print(f"  Distill Loss: {train_stats['distill_loss']:.4f}")
        print(f"  Token Reduction: {train_stats['token_reduction']:.1f}%")
        print(f"  Learning Rate: {train_stats['lr']:.2e}")
        
        # 更新学习率
        trainer.scheduler.step()
        
        # 保存checkpoint
        is_best = train_stats['total_loss'] < trainer.best_loss
        if is_best:
            trainer.best_loss = train_stats['total_loss']
        
        trainer.save_checkpoint(
            save_dir=args.save_dir,
            epoch=epoch + 1,
            is_best=is_best
        )
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()
