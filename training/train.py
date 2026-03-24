"""PATO-Qwen2.5-VL Training Script

训练策略:
1. 冻结 Qwen2.5-VL 的 Vision Encoder 和 LLM
2. 只训练 PATO 组件 (g_raw + token_sort + projector) ~8.7M 参数
3. 使用知识蒸馏损失保持性能
4. 添加预算正则化确保压缩效率
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import json
# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir("/data1/home/jfzhou/PATO")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm
from pato_integration import PATOQwen2_5_VLModel, PATOQwen2_5_VLConfig
from pato_integration.pato_config_standalone import PATOConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig


from g_raw import WeightedDownsample
from token_sort import DifferentiableSortingTokenSorter
from pato_integration.loss import PATOLoss
from training.data_loader import create_vqa_dataloader


class PATOTrainer:
    """PATO Training Manager
    
    管理完整的训练流程，包括:
    - 模型初始化和冻结
    - 数据加载
    - 训练循环
    - 检查点保存
    - 日志记录
    """
    
    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.model_dtype = dtype
        
        print("="*60)
        print("PATO-Qwen2.5-VL Trainer Initialization")
        print("="*60)
        
        # 加载基础模型和processor
        print(f"\n[1/5] Initializing base model and processor")
        self._init_model()
        if self.model.model is not None and self.model.model.get_input_embeddings() is not None:
            print("Model initialized.")
        else:
            raise ValueError("Model initialization failed.")
        print(f"  ✓ Model loaded on {device}")

        # 初始化损失函数
        print(f"\n[3/5] Initializing loss functions")
        self.criterion = PATOLoss(
            lambda_distill=config.get('lambda_distill', 0.05),
            lambda_sort_reg=config.get('lambda_sort_reg', 0.01),
            lambda_contrast=config.get('lambda_contrast', 0.1),
            lambda_g_raw_reg=config.get('lambda_g_raw_reg', 0.001)
        )
        print(f"  ✓ Loss initialized")
        # 初始化损失函数
        print(f"\n[4/5] Initializing optimizer")
        self.criterion = PATOLoss()
            
        # 初始化优化器
        print(f"\n[5/5] Initializing optimizer")
        self._init_optimizer()
        
        # 训练统计
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        print(f"\n✓ Trainer initialized successfully!")
        print(f"  Trainable parameters: {self._count_trainable_params()/1e6:.2f}M")

    def _init_model(self):
        """ Initialize base model and processor """

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True # 建议加上
        )
        """Load pretrained model with PATO enhancements"""
        base_model_config = Qwen2_5_VLConfig.from_pretrained(self.model_path)
        pato_config = PATOConfig()
        pato_config.g_raw.text_dim = (base_model_config.text_config.hidden_size 
                                 if hasattr(base_model_config, 'text_config') 
                                 else base_model_config.hidden_size)
        print("Initializing PATO model...")
        config = PATOQwen2_5_VLConfig(
                pato_config=pato_config,
                device=self.device,
                **base_model_config.to_dict(),
            )
        self.model = PATOQwen2_5_VLModel(
            config=config,
        ).to(self.device)
        print("Done!")

        # print("Loading language model and vision transformer weights...")
        # #Load weights from base model
        # base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     pretrained_model_name_or_path=self.model_path,
        #     config=base_model_config,
        #     device_map="cpu", # 强制放在 CPU，避免挤爆显存
        #     torch_dtype=self.model_dtype,
        # )
        # print("Done!")
        # print("Loading base model weights into PATO model...")
        # missing_lm, unexpected_lm = self.model.model.load_state_dict(
        #     base_model.model.state_dict(), strict=False
        # )
        # missing_vis, unexpected_vis = self.model.visual.load_state_dict(
        #     base_model.visual.state_dict(), strict=False
        # )
        # TODO lm_head for losses
        # print("   LM missing:", missing_lm)
        # print("   LM unexpected:", unexpected_lm)
        # print("   VIS missing:", missing_vis)
        # print("   VIS unexpected:", unexpected_vis)
        # print("Done!")
        # del base_model
        # torch.cuda.empty_cache()
        print(f"\n[2/4] Freezing base model parameters")
        self.model.model.requires_grad_(False)
        self.model.visual.requires_grad_(False)

        # 只训练 PATO 模块
        self.model.g_raw.requires_grad_(True)
        self.model.visual.token_sorter.requires_grad_(True)
        self.model.visual.projector.requires_grad_(True)
        print(f"  ✓ Base model frozen")
    def _init_pato_components(self):
        """初始化PATO组件"""
        # 直接加载配置模块
        import importlib.util
        config_path = Path(__file__).parent.parent / 'pato_integration' / 'pato_config_standalone.py'
        spec = importlib.util.spec_from_file_location("pato_config_standalone", config_path)
        pato_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pato_config_module)
        PATOConfig = pato_config_module.PATOConfig
        
        # 创建配置
        pato_config = PATOConfig()
        
        # 从base_model获取维度信息
        model_config = self.base_model.config
        vision_hidden_size = model_config.vision_config.hidden_size
        text_hidden_size = model_config.text_config.hidden_size if hasattr(model_config, 'text_config') else model_config.hidden_size
        
        # 配置PATO参数
        pato_config.g_raw.text_dim = text_hidden_size
        pato_config.g_raw.target_size = tuple(self.config.get('target_size', (448, 448)))
        
        pato_config.token_sort.budgets = self.config.get('token_budgets', [256])
        pato_config.projector.vision_dim = vision_hidden_size
        pato_config.projector.hidden_dim = text_hidden_size
        
        # 初始化组件
        self.g_raw = WeightedDownsample(
            pato_config.g_raw,
            {'device': self.device}
        ).to(self.device)
        
        # 注意: Qwen2.5-VL的vision输出是text_hidden_size维度
        # 所以token_sorter需要使用text_hidden_size
        self.token_sorter = DifferentiableSortingTokenSorter(
            pato_config.token_sort,
            {'hidden_size': vision_hidden_size}  # 使用text维度
        ).to(self.device)
        

        # 简化的投影器（如果需要）
        self.projector = nn.Linear(
            text_hidden_size,
            text_hidden_size
        ).to(self.device)
        
        print(f"  ✓ g_raw initialized")
        print(f"  ✓ token_sorter initialized")
        print(f"  ✓ projector initialized")
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 收集所有可训练参数
        trainable_params = []
        trainable_params += list(self.model.g_raw.parameters())
        trainable_params += list(self.model.visual.token_sorter.parameters())
        trainable_params += list(self.model.visual.projector.parameters())
        
        # 过滤requires_grad=True的参数
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        lr = self.config.get('learning_rate', 1e-4)
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('max_epochs', 10),
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        print(f"  ✓ Optimizer: AdamW (lr={lr})")
        print(f"  ✓ Scheduler: CosineAnnealing")
    
    def _count_trainable_params(self) -> int:
        """统计可训练参数数量"""
        total = 0
        for module in [self.model.g_raw, self.model.visual.token_sorter, self.model.visual.projector]:
            total += sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total
    
    def forward_pato_pipeline(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,

    ):
        """PATO完整前向传播
        
        Args:
            images: 原始图像 [B, 3, H, W]
            text_embeddings: 文本嵌入 [B, D]
            budget: Token预算
        
        Returns:
            compressed_tokens: 压缩后的视觉token
            aux_outputs: 辅助输出（用于损失计算）
        """
        raise NotImplementedError("NotImplementedError: forward_pato_pipeline needs to be implemented.")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> dict:
        """训练一个epoch
        
        Args:
            dataloader: 训练数据加载器
            epoch: 当前epoch数
        
        Returns:
            训练统计信息
        """
        self.model.g_raw.train()
        self.model.visual.token_sorter.train()
        self.model.visual.projector.train()
        
        total_loss = 0.0
        total_distill_loss = 0.0
        total_budget_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for _, batch in enumerate(progress_bar):
            # 注意: 实际训练需要完整实现forward pipeline
            # 这里提供训练循环的框架
            
            # 提取batch数据
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            image_grid_thw = batch['image_grid_thw']
            # 更加鲁棒的转换逻辑
            if isinstance(image_grid_thw, list):
                # 如果列表里面是张量，先用 stack；如果是数字，直接转
                if torch.is_tensor(image_grid_thw[0]):
                    image_grid_thw = torch.stack(image_grid_thw).to(self.device)
                else:
                    image_grid_thw = torch.tensor(image_grid_thw).to(self.device)
            elif torch.is_tensor(image_grid_thw):
                image_grid_thw = image_grid_thw.to(self.device)
            # 检查pixel_values类型、

            # 问题在于dataloader这里
            if isinstance(pixel_values, list):
                # Qwen2.5-VL返回的可能是list
                # 需要特殊处理
                pixel_values = torch.cat(pixel_values, dim=0)
            print(f"pixel_values shape: {pixel_values.shape}")
            pixel_values = pixel_values.to(self.device)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # 获取文本嵌入
            # 模型中已经有文本嵌入处理
            # with torch.no_grad():
            #     text_embeds = self.model.embed_tokens(input_ids)
            #     text_embeds = text_embeds.mean(dim=1).float()  # [B, D]
            
            # 简化训练循环示例
            # 实际使用需要完整的pipeline
            self.optimizer.zero_grad()
            
            # 模拟loss计算
            # 在完整实现中，这里应该是:
            # 1. g_raw压缩图像
            # 2. vision encoding
            # 3. token sorting
            # 4. 计算distillation loss
            # 5. 计算budget regularization loss

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                return_dict=True,
            )
            # language model loss
            
            dummy_loss = self.criterion(
                lm_loss=output.loss,
                aux_outputs=output.aux_outputs,
                g_raw_module=self.model.g_raw,
                token_sorter=self.model.visual.token_sorter,   
            )
            
            # Backward
            if dummy_loss.requires_grad:
                dummy_loss.backward()
                self.optimizer.step()
            
            # 更新统计
            total_loss += dummy_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{dummy_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            self.global_step += 1
        
        # Epoch统计
        stats = {
            'epoch': epoch,
            'total_loss': total_loss / len(dataloader),
            'distill_loss': total_distill_loss / max(len(dataloader), 1),
            'budget_loss': total_budget_loss / max(len(dataloader), 1),
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return stats
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> dict:
        """评估模型
        
        Args:
            dataloader: 评估数据加载器
        
        Returns:
            评估统计信息
        """
        self.model.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 评估逻辑
                # 类似训练，但不更新参数
                pass
        
        return {
            'eval_loss': total_loss / max(len(dataloader), 1)
        }
    
    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        is_best: bool = False
    ):
        """保存检查点
        
        Args:
            save_dir: 保存目录
            epoch: 当前epoch
            is_best: 是否是最佳模型
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'g_raw_state_dict': self.model.g_raw.state_dict(),
            'token_sorter_state_dict': self.model.visual.token_sorter.state_dict(),
            'projector_state_dict': self.model.visual.projector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # 保存最新checkpoint
        ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"  ✓ Checkpoint saved to {ckpt_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved to {best_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str
    ):
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.g_raw.load_state_dict(checkpoint['g_raw_state_dict'])
        self.model.visual.token_sorter.load_state_dict(checkpoint['token_sorter_state_dict'])
        self.model.visual.projector.load_state_dict(checkpoint['projector_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"  ✓ Checkpoint loaded from {checkpoint_path}")
        print(f"    Resuming from epoch {self.epoch}, step {self.global_step}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PATO-Qwen2.5-VL Training")
    
    # 模型参数
    parser.add_argument(
        '--model_path',
        type=str,
        default='./Qwen/Qwen2.5-VL-3B-Instruct',
        help='Qwen2.5-VL model path'
    )
    
    # 数据参数
    parser.add_argument(
        '--data_path',
        type=str,
        #required=True, 
        default='./datas/metadata/textvqa_cot_train.jsonl',
        help='VQA dataset JSON file path'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        # required=True,
        default="./datas/cot/textvqa",
        help='Image directory'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default="textvqa",
        choices=['vqav2', 'textvqa', 'custom'],
        help='Dataset type'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=2,
        help='Maximum number of samples (for quick testing)'
    )
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    # PATO参数
    parser.add_argument('--target_size', type=int, nargs=2, default=[448, 448], help='Target image size')
    parser.add_argument('--token_budget', type=int, default=256, help='Token budget')
    parser.add_argument('--distill_weight', type=float, default=0.5, help='Distillation loss weight')
    parser.add_argument('--budget_weight', type=float, default=0.3, help='Budget regularization weight')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval (epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    
    # 构建配置字典
    config = {
        'target_size': tuple(args.target_size),
        'token_budgets': [args.token_budget],
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_epochs': args.max_epochs,
        'distill_weight': args.distill_weight,
        'budget_weight': args.budget_weight,
    }
    
    # 初始化trainer
    trainer = PATOTrainer(
        model_path=args.model_path,
        config=config,
        device=args.device
    )
    
    # 加载checkpoint（如果需要）
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    
    # 创建数据加载器
    print("\nCreating dataloaders...")
    train_loader = create_vqa_dataloader(
        data_path=args.data_path,
        image_dir=args.image_dir,
        processor=trainer.processor,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        shuffle=True,
        dataset_type=args.dataset_type
    )
    
    print(f"  ✓ Train loader: {len(train_loader)} batches")
    
    # 训练循环
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(trainer.epoch, args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        print("-" * 60)
        
        # 训练
        train_stats = trainer.train_epoch(train_loader, epoch + 1)
        
        # 打印统计信息
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: {train_stats['total_loss']:.4f}")
        print(f"  Learning Rate: {train_stats['lr']:.2e}")
        
        # 更新学习率
        trainer.scheduler.step()
        
        # 保存checkpoint
        if (epoch + 1) % args.save_interval == 0:
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
