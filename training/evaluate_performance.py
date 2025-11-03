"""
PATO性能评估脚本
评估准确率、推理速度、内存占用
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
import time
import psutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import importlib.util

from g_raw import WeightedDownsample
from token_sort import DifferentiableSortingTokenSorter


class PATOEvaluator:
    """PATO评估器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda:3',
        config: dict = None
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print("="*60)
        print("PATO Performance Evaluator")
        print("="*60)
        print(f"Device: {device}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # 加载checkpoint
        self._load_checkpoint(checkpoint_path, config)
        
        # 评估模式
        self.g_raw.eval()
        self.token_sorter.eval()
        
        print("✓ Models loaded and ready for evaluation")
    
    def _load_checkpoint(self, checkpoint_path, config):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取配置
        if config is None:
            config = checkpoint.get('config', {})
        
        self.config = config
        text_dim = config.get('text_dim', 3584)
        target_size = config.get('target_size', (448, 448))
        token_budgets = config.get('token_budgets', [128])
        
        # 初始化模型
        config_path = Path(__file__).parent.parent / 'pato_integration' / 'pato_config_standalone.py'
        spec = importlib.util.spec_from_file_location("pato_config_standalone", config_path)
        pato_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pato_config_module)
        PATOConfig = pato_config_module.PATOConfig
        
        pato_config = PATOConfig()
        pato_config.g_raw.text_dim = text_dim
        pato_config.g_raw.target_size = tuple(target_size)
        pato_config.token_sort.budgets = token_budgets
        
        self.g_raw = WeightedDownsample(
            pato_config.g_raw,
            {'device': self.device}
        ).to(self.device)
        
        self.token_sorter = DifferentiableSortingTokenSorter(
            pato_config.token_sort,
            {'hidden_size': text_dim}
        ).to(self.device)
        
        # 加载权重
        self.g_raw.load_state_dict(checkpoint['g_raw_state_dict'])
        self.token_sorter.load_state_dict(checkpoint['token_sorter_state_dict'])
        
        print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    def evaluate_reconstruction_quality(self, dataloader, num_samples=100):
        """评估重建质量"""
        print("\n" + "="*60)
        print("1. Reconstruction Quality Evaluation")
        print("="*60)
        
        total_mse = 0.0
        total_psnr = 0.0
        count = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating quality", total=min(num_samples, len(dataloader)))
            
            for batch in pbar:
                if count >= num_samples:
                    break
                
                images = batch['image'].to(self.device)
                batch_size = images.shape[0]
                
                # 模拟text embeddings
                text_embeds = torch.randn(batch_size, self.config['text_dim']).to(self.device)
                
                # Forward
                compressed = self.g_raw(images, text_embeds)
                
                # 计算MSE和PSNR
                mse = nn.functional.mse_loss(compressed, images).item()
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))
                
                total_mse += mse * batch_size
                total_psnr += psnr * batch_size
                count += batch_size
                
                pbar.set_postfix({'MSE': f'{mse:.6f}', 'PSNR': f'{psnr:.2f}dB'})
        
        avg_mse = total_mse / count
        avg_psnr = total_psnr / count
        
        results = {
            'mse': avg_mse,
            'psnr': avg_psnr,
            'num_samples': count
        }
        
        print(f"\nResults:")
        print(f"  • Average MSE: {avg_mse:.6f}")
        print(f"  • Average PSNR: {avg_psnr:.2f} dB")
        print(f"  • Samples evaluated: {count}")
        
        return results
    
    def evaluate_inference_speed(self, dataloader, num_samples=100):
        """评估推理速度"""
        print("\n" + "="*60)
        print("2. Inference Speed Evaluation")
        print("="*60)
        
        # 预热
        print("Warming up...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:
                    break
                images = batch['image'].to(self.device)
                batch_size = images.shape[0]
                text_embeds = torch.randn(batch_size, self.config['text_dim']).to(self.device)
                _ = self.g_raw(images, text_embeds)
        
        # 测速
        print("Measuring speed...")
        times_g_raw = []
        times_token_sort = []
        times_total = []
        count = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Speed test", total=min(num_samples, len(dataloader)))
            
            for batch in pbar:
                if count >= num_samples:
                    break
                
                images = batch['image'].to(self.device)
                batch_size = images.shape[0]
                text_embeds = torch.randn(batch_size, self.config['text_dim']).to(self.device)
                
                # 测量g_raw
                torch.cuda.synchronize()
                start = time.time()
                compressed = self.g_raw(images, text_embeds)
                torch.cuda.synchronize()
                time_g_raw = time.time() - start
                
                # 模拟vision tokens
                vision_tokens = torch.randn(
                    batch_size, 
                    self.config.get('vision_tokens', 256), 
                    self.config['text_dim']
                ).to(self.device)
                
                # 测量token sorting
                torch.cuda.synchronize()
                start = time.time()
                selected, _, _ = self.token_sorter(
                    hidden_states=vision_tokens,
                    budget=self.config['token_budgets'][0],
                    query_embeddings=text_embeds
                )
                torch.cuda.synchronize()
                time_token_sort = time.time() - start
                
                time_total = time_g_raw + time_token_sort
                
                times_g_raw.append(time_g_raw / batch_size)
                times_token_sort.append(time_token_sort / batch_size)
                times_total.append(time_total / batch_size)
                
                count += batch_size
                
                pbar.set_postfix({
                    'total': f'{time_total*1000:.2f}ms',
                    'g_raw': f'{time_g_raw*1000:.2f}ms',
                    'sort': f'{time_token_sort*1000:.2f}ms'
                })
        
        avg_g_raw = np.mean(times_g_raw) * 1000  # ms
        avg_token_sort = np.mean(times_token_sort) * 1000
        avg_total = np.mean(times_total) * 1000
        throughput = 1000.0 / avg_total  # samples/sec
        
        results = {
            'g_raw_ms': avg_g_raw,
            'token_sort_ms': avg_token_sort,
            'total_ms': avg_total,
            'throughput': throughput,
            'num_samples': count
        }
        
        print(f"\nResults (per sample):")
        print(f"  • g_raw: {avg_g_raw:.2f} ms")
        print(f"  • Token sort: {avg_token_sort:.2f} ms")
        print(f"  • Total: {avg_total:.2f} ms")
        print(f"  • Throughput: {throughput:.2f} samples/sec")
        
        return results
    
    def evaluate_memory_usage(self, batch_sizes=[1, 2, 4, 8]):
        """评估内存占用"""
        print("\n" + "="*60)
        print("3. Memory Usage Evaluation")
        print("="*60)
        
        results = {}
        
        for batch_size in batch_sizes:
            # 清空缓存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 创建输入
            images = torch.randn(batch_size, 3, 448, 448).to(self.device)
            text_embeds = torch.randn(batch_size, self.config['text_dim']).to(self.device)
            
            # Forward
            with torch.no_grad():
                compressed = self.g_raw(images, text_embeds)
                
                vision_tokens = torch.randn(
                    batch_size,
                    self.config.get('vision_tokens', 256),
                    self.config['text_dim']
                ).to(self.device)
                
                selected, _, _ = self.token_sorter(
                    hidden_states=vision_tokens,
                    budget=self.config['token_budgets'][0],
                    query_embeddings=text_embeds
                )
            
            # 获取内存使用
            mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            mem_peak = torch.cuda.max_memory_allocated(self.device) / 1024**2
            
            results[batch_size] = {
                'allocated_mb': mem_allocated,
                'reserved_mb': mem_reserved,
                'peak_mb': mem_peak
            }
            
            print(f"\nBatch size {batch_size}:")
            print(f"  • Allocated: {mem_allocated:.2f} MB")
            print(f"  • Reserved: {mem_reserved:.2f} MB")
            print(f"  • Peak: {mem_peak:.2f} MB")
        
        return results
    
    def evaluate_token_reduction(self, dataloader, num_samples=100):
        """评估token压缩率"""
        print("\n" + "="*60)
        print("4. Token Reduction Evaluation")
        print("="*60)
        
        total_original = 0
        total_selected = 0
        count = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Token reduction", total=min(num_samples, len(dataloader)))
            
            for batch in pbar:
                if count >= num_samples:
                    break
                
                images = batch['image'].to(self.device)
                batch_size = images.shape[0]
                text_embeds = torch.randn(batch_size, self.config['text_dim']).to(self.device)
                
                # 模拟vision tokens
                vision_tokens = torch.randn(
                    batch_size,
                    self.config.get('vision_tokens', 256),
                    self.config['text_dim']
                ).to(self.device)
                
                # Token sorting
                selected, indices, aux = self.token_sorter(
                    hidden_states=vision_tokens,
                    budget=self.config['token_budgets'][0],
                    query_embeddings=text_embeds
                )
                
                total_original += vision_tokens.shape[1] * batch_size
                total_selected += selected.shape[1] * batch_size
                count += batch_size
                
                reduction = 100 * (1 - selected.shape[1] / vision_tokens.shape[1])
                pbar.set_postfix({'reduction': f'{reduction:.1f}%'})
        
        avg_reduction = 100 * (1 - total_selected / total_original)
        compression_ratio = total_original / total_selected
        
        results = {
            'original_tokens': total_original / count,
            'selected_tokens': total_selected / count,
            'reduction_percent': avg_reduction,
            'compression_ratio': compression_ratio,
            'num_samples': count
        }
        
        print(f"\nResults:")
        print(f"  • Original tokens: {total_original / count:.1f}")
        print(f"  • Selected tokens: {total_selected / count:.1f}")
        print(f"  • Reduction: {avg_reduction:.1f}%")
        print(f"  • Compression ratio: {compression_ratio:.2f}×")
        
        return results
    
    def run_full_evaluation(self, dataloader, output_file='evaluation_results.json'):
        """运行完整评估"""
        print("\n" + "="*60)
        print("STARTING FULL EVALUATION")
        print("="*60)
        
        all_results = {}
        
        # 1. 重建质量
        all_results['reconstruction'] = self.evaluate_reconstruction_quality(dataloader)
        
        # 2. 推理速度
        all_results['inference_speed'] = self.evaluate_inference_speed(dataloader)
        
        # 3. 内存占用
        all_results['memory_usage'] = self.evaluate_memory_usage()
        
        # 4. Token压缩
        all_results['token_reduction'] = self.evaluate_token_reduction(dataloader)
        
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        print(f"\nResults saved to: {output_file}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\n📊 Reconstruction:")
        print(f"  • PSNR: {all_results['reconstruction']['psnr']:.2f} dB")
        
        print(f"\n⚡ Speed:")
        print(f"  • Throughput: {all_results['inference_speed']['throughput']:.2f} samples/sec")
        print(f"  • Latency: {all_results['inference_speed']['total_ms']:.2f} ms/sample")
        
        print(f"\n💾 Memory (batch=4):")
        mem_4 = all_results['memory_usage'].get('4', all_results['memory_usage'].get(4, {}))
        print(f"  • Peak: {mem_4.get('peak_mb', 0):.2f} MB")
        
        print(f"\n🗜️  Compression:")
        print(f"  • Token reduction: {all_results['token_reduction']['reduction_percent']:.1f}%")
        print(f"  • Compression ratio: {all_results['token_reduction']['compression_ratio']:.2f}×")
        
        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--image_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--output', type=str, default='evaluation_results.json')
    
    args = parser.parse_args()
    
    # 创建数据集
    from train_large_scale import VQADataset
    
    dataset = VQADataset(
        image_dir=args.image_dir,
        max_samples=args.num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 创建评估器
    evaluator = PATOEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 运行评估
    results = evaluator.run_full_evaluation(dataloader, args.output)


if __name__ == "__main__":
    main()
