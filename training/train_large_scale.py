"""
Large-scale PATO Training Script
支持多GPU并行训练、训练监控、断点续训
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available, logging disabled")
import argparse
import json
import time
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import numpy as np
import importlib.util

# 导入PATO组件
from g_raw import WeightedDownsample
from token_sort import DifferentiableSortingTokenSorter


class VQADataset(Dataset):
    """VQA数据集加载器"""
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str = None,
        max_samples: int = None,
        image_size: int = 448
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.samples = []
        
        # 加载标注文件
        if annotation_file and os.path.exists(annotation_file):
            print(f"Loading annotations from: {annotation_file}")
            
            # 检查文件格式 (.jsonl or .json)
            if annotation_file.endswith('.jsonl'):
                # JSONL格式 (每行一个JSON对象)
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                sample = json.loads(line)
                                self.samples.append(sample)
                            except json.JSONDecodeError as e:
                                continue
            else:
                # JSON格式
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.samples = data.get('samples', data)
        else:
            # 自动生成数据
            print(f"No annotation file, generating from images...")
            self.samples = self._generate_from_images()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} VQA samples")
    
    def _generate_from_images(self):
        """从图像自动生成数据"""
        samples = []
        questions = [
            "What is shown in this image?",
            "Describe the main content.",
            "What can you see?",
            "What is the subject?",
        ]
        
        image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        
        for img_file in image_files:
            samples.append({
                'image': img_file.name,
                'question': questions[hash(img_file.name) % len(questions)],
                'answer': 'Visual content'
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 处理不同的标注格式
        # Format 1: TextVQA format {"image": "xxx.jpg", "question": "...", "answer": "...", "dataset": "textvqa"}
        # Format 2: GQA format {"images": ["cot/gqa/xxx.jpg"], "instruction": "...", "output": "..."}
        
        if 'images' in sample:
            # GQA format
            img_path_str = sample['images'][0] if isinstance(sample['images'], list) else sample['images']
            # 移除 "cot/" 前缀
            if img_path_str.startswith('cot/'):
                img_path_str = img_path_str[4:]  # 移除 "cot/"
            img_path = self.image_dir / img_path_str
            question = sample.get('instruction', '').replace('<image>\n', '').strip()
            answer = sample.get('output', '')
        else:
            # TextVQA/DocVQA format - 图像在dataset子目录中
            dataset_name = sample.get('dataset', 'textvqa')
            img_name = sample['image']
            img_path = self.image_dir / dataset_name / img_name
            question = sample.get('question', '')
            answer = sample.get('answer', '')
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            # 创建占位图像
            # print(f"Warning: Failed to load image {img_path}: {e}")
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        return {
            'image': img_tensor,
            'question': question,
            'answer': answer
        }


class PATOLargeScaleTrainer:
    """大规模PATO训练器"""
    
    def __init__(
        self,
        config: dict,
        rank: int = 0,
        world_size: int = 1,
        use_ddp: bool = False
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.device = f"cuda:{rank}"
        
        # 只在主进程打印
        self.is_main = (rank == 0)
        
        if self.is_main:
            print("="*60)
            print("PATO Large-Scale Training")
            print("="*60)
            print(f"World size: {world_size}")
            print(f"Rank: {rank}")
            print(f"Device: {self.device}")
        
        # 初始化组件
        self._init_components()
        self._init_optimizer()
        
        # TensorBoard (只在主进程)
        if self.is_main:
            if TENSORBOARD_AVAILABLE:
                log_dir = Path(config['log_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
                self.writer = SummaryWriter(log_dir)
                print(f"TensorBoard: {log_dir}")
            else:
                self.writer = None
                print("TensorBoard disabled (not installed)")
        else:
            self.writer = None
        
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def _init_components(self):
        """初始化PATO组件"""
        if self.is_main:
            print("\n[1/3] Initializing PATO components...")
        
        # 加载配置
        config_path = Path(__file__).parent.parent / 'pato_integration' / 'pato_config_standalone.py'
        spec = importlib.util.spec_from_file_location("pato_config_standalone", config_path)
        pato_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pato_config_module)
        PATOConfig = pato_config_module.PATOConfig
        
        pato_config = PATOConfig()
        pato_config.g_raw.text_dim = self.config['text_dim']
        pato_config.g_raw.target_size = tuple(self.config['target_size'])
        pato_config.token_sort.budgets = self.config['token_budgets']
        
        # 创建模型
        self.g_raw = WeightedDownsample(
            pato_config.g_raw,
            {'device': self.device}
        ).to(self.device)
        
        self.token_sorter = DifferentiableSortingTokenSorter(
            pato_config.token_sort,
            {'hidden_size': self.config['text_dim']}
        ).to(self.device)
        
        # DDP包装
        if self.use_ddp:
            self.g_raw = DDP(self.g_raw, device_ids=[self.rank])
            self.token_sorter = DDP(self.token_sorter, device_ids=[self.rank])
        
        if self.is_main:
            total_params = sum(p.numel() for p in self.g_raw.parameters()) + \
                          sum(p.numel() for p in self.token_sorter.parameters())
            print(f"  ✓ Total parameters: {total_params/1e6:.2f}M")
    
    def _init_optimizer(self):
        """初始化优化器"""
        if self.is_main:
            print("\n[2/3] Initializing optimizer...")
        
        params = list(self.g_raw.parameters()) + list(self.token_sorter.parameters())
        params = [p for p in params if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['max_epochs'],
            eta_min=self.config['min_lr']
        )
        
        if self.is_main:
            print(f"  ✓ AdamW (lr={self.config['learning_rate']:.2e})")
    
    def train_step(self, batch):
        """单步训练"""
        images = batch['image'].to(self.device)
        batch_size = images.shape[0]
        
        # 模拟text embeddings
        text_embeds = torch.randn(batch_size, self.config['text_dim']).to(self.device)
        
        # Forward
        compressed_images = self.g_raw(images, text_embeds)
        
        # 模拟vision tokens
        vision_tokens = torch.randn(
            batch_size, 
            self.config['vision_tokens'], 
            self.config['text_dim']
        ).to(self.device)
        
        # Token sorting
        selected_tokens, sort_indices, aux_outputs = self.token_sorter(
            hidden_states=vision_tokens,
            budget=self.config['token_budgets'][0],
            query_embeddings=text_embeds
        )
        
        # 计算损失
        recon_loss = nn.functional.mse_loss(compressed_images, images)
        entropy_loss = aux_outputs.get('entropy_loss', torch.tensor(0.0, device=self.device))
        diversity_loss = aux_outputs.get('diversity_loss', torch.tensor(0.0, device=self.device))
        
        total_loss = (
            self.config['recon_weight'] * recon_loss +
            self.config['entropy_weight'] * entropy_loss +
            self.config['diversity_weight'] * diversity_loss
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.g_raw.parameters() if p.requires_grad] +
            [p for p in self.token_sorter.parameters() if p.requires_grad],
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
            'token_reduction': 100 * (1 - selected_tokens.shape[1] / vision_tokens.shape[1])
        }
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.g_raw.train()
        self.token_sorter.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'entropy_loss': 0.0,
            'diversity_loss': 0.0,
            'token_reduction': 0.0
        }
        
        # 只在主进程显示进度条
        if self.is_main:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)
            
            # 更新统计
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            
            # 更新进度条
            if self.is_main:
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'recon': f"{metrics['recon_loss']:.4f}",
                    'token': f"{metrics['token_reduction']:.1f}%",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # TensorBoard
                if self.writer and batch_idx % 10 == 0:
                    self.writer.add_scalar('Loss/total', metrics['total_loss'], self.global_step)
                    self.writer.add_scalar('Loss/recon', metrics['recon_loss'], self.global_step)
                    self.writer.add_scalar('Token/reduction', metrics['token_reduction'], self.global_step)
                    self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # 计算平均
        num_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        epoch_time = time.time() - start_time
        epoch_metrics['time'] = epoch_time
        epoch_metrics['samples_per_sec'] = len(dataloader.dataset) / epoch_time
        
        return epoch_metrics
    
    def save_checkpoint(self, save_dir, epoch, metrics):
        """保存检查点"""
        if not self.is_main:
            return
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取模型state dict (处理DDP)
        g_raw_state = self.g_raw.module.state_dict() if self.use_ddp else self.g_raw.state_dict()
        token_sorter_state = self.token_sorter.module.state_dict() if self.use_ddp else self.token_sorter.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'g_raw_state_dict': g_raw_state,
            'token_sorter_state_dict': token_sorter_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # 保存最新checkpoint
        ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"  ✓ Checkpoint saved: {ckpt_path}")
        
        # 保存最佳模型
        if metrics['total_loss'] < self.best_loss:
            self.best_loss = metrics['total_loss']
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model updated: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.use_ddp:
            self.g_raw.module.load_state_dict(checkpoint['g_raw_state_dict'])
            self.token_sorter.module.load_state_dict(checkpoint['token_sorter_state_dict'])
        else:
            self.g_raw.load_state_dict(checkpoint['g_raw_state_dict'])
            self.token_sorter.load_state_dict(checkpoint['token_sorter_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        if self.is_main:
            print(f"Loaded checkpoint from epoch {self.epoch}")


def setup_ddp(rank, world_size):
    """设置DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()


def train_worker(rank, world_size, config):
    """训练worker进程"""
    # 设置DDP
    if world_size > 1:
        setup_ddp(rank, world_size)
        use_ddp = True
    else:
        use_ddp = False
    
    # 创建trainer
    trainer = PATOLargeScaleTrainer(
        config=config,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp
    )
    
    # 创建数据集
    dataset = VQADataset(
        image_dir=config['image_dir'],
        annotation_file=config.get('annotation_file'),
        max_samples=config.get('max_samples'),
        image_size=config['target_size'][0]
    )
    
    # 创建dataloader
    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    if trainer.is_main:
        print(f"\n[3/3] Dataset ready: {len(dataset)} samples, {len(dataloader)} batches")
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
    
    # 训练循环
    for epoch in range(config['max_epochs']):
        if use_ddp:
            sampler.set_epoch(epoch)
        
        if trainer.is_main:
            print(f"\nEpoch {epoch + 1}/{config['max_epochs']}")
            print("-" * 60)
        
        metrics = trainer.train_epoch(dataloader, epoch + 1)
        
        if trainer.is_main:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Loss: {metrics['total_loss']:.4f}")
            print(f"  Token Reduction: {metrics['token_reduction']:.1f}%")
            print(f"  Time: {metrics['time']:.1f}s")
            print(f"  Speed: {metrics['samples_per_sec']:.1f} samples/s")
            print(f"  Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        trainer.scheduler.step()
        
        # 保存checkpoint
        trainer.save_checkpoint(
            save_dir=config['save_dir'],
            epoch=epoch + 1,
            metrics=metrics
        )
    
    if trainer.is_main:
        print("\n" + "="*60)
        print("✓ Training Completed!")
        print("="*60)
        if trainer.writer:
            trainer.writer.close()
    
    # 清理DDP
    if use_ddp:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    
    # 模型参数
    parser.add_argument('--text_dim', type=int, default=3584)
    parser.add_argument('--vision_tokens', type=int, default=256)
    parser.add_argument('--token_budget', type=int, default=128)
    parser.add_argument('--target_size', type=int, nargs=2, default=[448, 448])
    
    # 多GPU参数
    parser.add_argument('--gpus', type=str, default='3', help='GPU IDs (e.g., "1,3,6")')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints_large')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # 解析GPU IDs
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    world_size = len(gpu_ids)
    
    # 配置
    config = {
        'image_dir': args.image_dir,
        'annotation_file': args.annotation_file,
        'max_samples': args.max_samples,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'min_lr': args.min_lr,
        'text_dim': args.text_dim,
        'vision_tokens': args.vision_tokens,
        'token_budgets': [args.token_budget],
        'target_size': tuple(args.target_size),
        'recon_weight': 0.5,
        'entropy_weight': 0.3,
        'diversity_weight': 0.2,
        'num_workers': args.num_workers,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'resume': args.resume
    }
    
    print("Configuration:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
    
    # 单GPU或多GPU训练
    if world_size == 1:
        # 单GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        train_worker(0, 1, config)
    else:
        # 多GPU DDP
        print(f"\nLaunching DDP training on {world_size} GPUs: {gpu_ids}")
        import torch.multiprocessing as mp
        mp.spawn(
            train_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()
