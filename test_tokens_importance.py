import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import argparse
from pathlib import Path
from typing import Optional
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    Trainer, 
    TrainingArguments,
)
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from torch.utils.data import (
    DataLoader, 
    Dataset, 
    IterableDataset, 
    RandomSampler, 
    SequentialSampler,
)
from training.data import (
    PATODataset, 
    PATOCollator,
    RepeatRandomSampler
)
from tqdm import tqdm
from pato_integration import PATOQwen2_5_VLModel, PATOQwen2_5_VLConfig
from pato_integration.pato_config import PATOConfig, create_default_pato_config
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLConfig,
)
from g_raw import WeightedDownsample
from token_sort import DifferentiableSortingTokenSorter
from pato_integration.loss import PATOLoss
from training.data_loader import create_vqa_dataloader

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.models import unwrap_model_for_generation
from dataclasses import dataclass, field, asdict

from accelerate.utils import is_peft_model, set_seed


from training.utils import (
    norm_bboxes, 
    extract_one_bbox_from_str, 
    cal_paired_ious,
    print_rank0
)
"""
    PATOScriptArgurment:
        保存数据集、训练集、图片目录、随机种子数、最大粉笔那缕、最大输入序列、最大输入保留序列等参数
    PATOTrainingArgument:
        保存训练相关参数，如输出目录、训练轮次、batch大小、学习率、权重衰减、日志与保存间隔等
    PATOModelConfig:
        保存模型相关配置，如模型路径、目标图像大小、token预算、蒸馏损失权重、预算正则化权重等
"""
@dataclass
class PATOScriptArgurment:
    train_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training dataset config."},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation dataset config."},
    )
    img_dir: str = field(
        default="datas",
        metadata={"help": "Path to the image directory."},
    )    
    sampling_seed: int = field(
        default=42,
        metadata={"help": "Random seed for sampling."},
    )
    max_pixels: int = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image."},
    )
    max_input_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum sequence length for the input."},
    )
    max_input_remain_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum remaining sequence length for the input."},
    )
    resume: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to resume training from a checkpoint."},
    )
    min_image_size: Optional[int] = field(
        default=224,
        metadata={"help": "The min legal size of an image.It should be equal to g_raw.target_size."},
    )
@dataclass
class PATOTrainingArgument(TrainingArguments):
    """
        除了自定义的训练参数外，还继承了 transformers.TrainingArguments 的所有参数
        包括了：batch_size等基本训练参数
    """
    kd_weight: float = field(
        default=None,
        metadata={"help": "Weight for knowledge distillation loss."},
    )
@dataclass
class PATOModelConfig:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    projector_enable : bool = field(
        default=False,
        metadata={"help": "Whether to enable the visual projector."}
    )
    g_raw_enable : bool = field(
        default=False,
        metadata={"help": "Whether to enable the graph raw process."}
    )
    token_sort_enable : bool = field(
        default=True,
        metadata={"help": "Whether to enable the visual token sort."}
    )

# ---------- Main ----------

def patch_processor(processor):
    if not hasattr(processor.tokenizer, "eos_token_id"):
        eos_token = getattr(processor.tokenizer, "eos_token")
        eos_token_id = processor.tokenizer.convert_tokens_to_ids(eos_token)
        processor.tokenizer.eos_token_id = eos_token_id
    print_rank0("eos_token_id:", processor.tokenizer.eos_token_id)
    
    if not hasattr(processor.tokenizer, "pad_token_id"):
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print_rank0("pad_token_id:", processor.tokenizer.pad_token_id)
    processor.tokenizer.padding_side = "left"

def attention_scores(
    model,
    processor,
    data_loader,
    out_dir,
):
    q_list = []
    a_list = []
    for batch_idx, batch in enumerate(data_loader):
        print(f"正在处理第 {batch_idx} 个 Batch...")
        
        device = model.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        sample_idx = 0 # 验证 Batch 里的第 0 个样本
        input_ids_0 = inputs["input_ids"][sample_idx]
        
        # 定位 Image Tokens
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_token_indices = torch.where(input_ids_0 == image_pad_id)[0]
        if len(image_token_indices) == 0:
            print("该样本没有图像，跳过...")
            continue

        grid_t, grid_h, grid_w = inputs["image_grid_thw"][sample_idx].tolist()
        grid_h = grid_h // 2
        grid_w = grid_w // 2
        if len(image_token_indices) != grid_t * grid_h * grid_w:
            print(f"警告: Image Token 数量不匹配，已截断。")
            image_token_indices = image_token_indices[:grid_t * grid_h * grid_w]
        # ==========================================
        # 步骤 B: 直接从文件路径加载原图 (极简且高清)
        # ==========================================
        try:
            # 获取当前样本的图像路径
            image_path = inputs["img_path"]
            print(inputs["img_path"])
            # 兼容有些 collator 可能会把 path 包装在 list/tuple 里的情况
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0]
                
            # OpenCV 默认读取为 BGR 格式，正适合后续直接 imwrite 保存
            original_img = cv2.imread(image_path)
            
            if original_img is None:
                raise ValueError(f"图片路径无效或无法读取: {image_path}")
                
        except Exception as e:
            print(f"读取原始图像失败 ({e})，将使用纯白背景图...")
            # 兜底：如果没读到图，就生成一张白底图
            original_img = np.full((grid_h * 28, grid_w * 28, 3), 255, dtype=np.uint8)
        
        query = inputs["query"][sample_idx]
        answer = inputs["answer"][sample_idx]
        q_list.append(query)
        a_list.append(answer)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        all_attentions = outputs.attentions # tuple (tensor)
        I = torch.eye(all_attentions[0].shape)
        A_b = 0.5 * all_attentions + 0.5 * I
        attn_len = len(all_attentions)
        
        for attn_layer in range(attn_len):
            avg_attention  = all_attentions[attn_layer].mean(dim=1)
            print(f"avg_attention.shape: {avg_attention.shape}")
            # ==========================================
            # 步骤 C: 提取 Attention 并与高清原图叠加
            # ==========================================
            # 选择tokens与tokens之间的关系
            last_token_attention = avg_attention[sample_idx, image_token_indices, :] # [image_token_indices, :]
            last_token_attention = last_token_attention[:, image_token_indices]
            print(last_token_attention.shape)
            last_token_attention = last_token_attention.sum(dim=0).float().cpu().numpy()
            attention_2d = last_token_attention.reshape(grid_h, grid_w)
            attention_norm = (attention_2d - attention_2d.min()) / (attention_2d.max() - attention_2d.min() + 1e-8)
            # 获取原图的真实高宽
            img_h, img_w = original_img.shape[:2]
            
            # 将热力图缩放到原图的实际像素大小
            heatmap_resized = cv2.resize(attention_norm, (img_w, img_h))
            # 转换为彩色热力图
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

            # ✨ 图像融合: 0.5 原图 + 0.5 热力图
            alpha = 0.7
            overlay_img = cv2.addWeighted(heatmap_color, alpha, original_img, 1 - alpha, 0)

            # 7. 动态命名并保存图片
            output_path = f'{out_dir}batch_{batch_idx}'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            img_save_path = os.path.join(out_dir, f"batch_{batch_idx}/attention_heatmap_layer_{attn_layer}.jpg")
            cv2.imwrite(img_save_path, overlay_img)
        

        if batch_idx >= 9:
            print(f"\n🎉 验证完成，前 10 个 Batch 的图文数据均已保存在 {out_dir} 中，停止绘图。")
            break
        
    txt_save_path = os.path.join(out_dir, f"query_answer.txt")
    with open(txt_save_path, "w", encoding="utf-8") as f:
        for i in range(len(q_list)):
            f.write(f"idx: {i}  Query: {q_list[i]}  Answer: {a_list[i]}\n")
    print("红色区域代表模型认为分数最高、最关心的 Tokens，蓝色代表可被剪枝的低分 Tokens。")
    # exit(0)  # 如果你跑完验证就想直接停掉程序，可以取消这行注释

def attention_rollout(    
    model,
    processor,
    data_loader,
    out_dir,
):
    q_list = []
    a_list = []
    for batch_idx, batch in enumerate(data_loader):
        print(f"正在处理第 {batch_idx} 个 Batch...")
        
        device = model.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        sample_idx = 0 # 验证 Batch 里的第 0 个样本
        input_ids_0 = inputs["input_ids"][sample_idx]

        # 定位 Image Tokens
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_token_indices = torch.where(input_ids_0 == image_pad_id)[0]
        if len(image_token_indices) == 0:
            print("该样本没有图像，跳过...")
            continue

        grid_t, grid_h, grid_w = inputs["image_grid_thw"][sample_idx].tolist()
        grid_h = grid_h // 2
        grid_w = grid_w // 2
        if len(image_token_indices) != grid_t * grid_h * grid_w:
            print(f"警告: Image Token 数量不匹配，已截断。")
            image_token_indices = image_token_indices[:grid_t * grid_h * grid_w]
        # ==========================================
        # 步骤 B: 直接从文件路径加载原图 (极简且高清)
        # ==========================================
        try:
            # 获取当前样本的图像路径
            image_path = inputs["img_path"]
            print(inputs["img_path"])
            # 兼容有些 collator 可能会把 path 包装在 list/tuple 里的情况
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0]
                
            # OpenCV 默认读取为 BGR 格式，正适合后续直接 imwrite 保存
            original_img = cv2.imread(image_path)
            
            if original_img is None:
                raise ValueError(f"图片路径无效或无法读取: {image_path}")

        except Exception as e:
            print(f"读取原始图像失败 ({e})，将使用纯白背景图...")
            # 兜底：如果没读到图，就生成一张白底图
            original_img = np.full((grid_h * 28, grid_w * 28, 3), 255, dtype=np.uint8)
        
        query = inputs["query"][sample_idx]
        answer = inputs["answer"][sample_idx]
        q_list.append(query)
        a_list.append(answer)
        
        with torch.no_grad():
            outputs = model(
                **inputs, 
                use_cache=True,
                output_attentions=True
            )
        all_attentions = outputs.attentions # tuple (tensor)
        B, num_heads, seq_len, seq_len = all_attentions[0].shape
        I = torch.eye(seq_len, device=device).to(all_attentions[0].dtype)
        # 假设 outputs 是 Qwen2_5_VLCausalLMOutputWithPast
        pkv = outputs.past_key_values 
        all_attentions = outputs.attentions # [Layer, Batch, Head, Seq, Seq]

        prompt_length = input_ids_0.shape[0]
        # 找到 assistant 开始标记的 token id
        # 注意：具体 id 取决于你的 tokenizer
        assistant_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>assistant")

        # 在 input_ids 中找到这个 id 的位置
        # 假设我们处理的是 batch 中的第 0 个样本
        input_ids_list = input_ids_0.tolist()
        try:
            # assistant 后的第一个 token 就是答案的开始
            answer_start_idx = input_ids_list.index(assistant_start_token_id) + 2 # +1 是 im_start, +2 是换行符之后
        except ValueError:
            # 如果没找到，说明可能还没进入 assistant 阶段
            answer_start_idx = prompt_length 

        answer_indices = range(answer_start_idx, len(input_ids_list))

        num_layers = len(all_attentions)
        seq_len = all_attentions[0].shape[-1]
        device = all_attentions[0].device

        # 初始化 Rollout 矩阵 R
        R = torch.eye(seq_len, device=device).to(all_attentions[0].dtype)

        for i in range(num_layers):
            # --- A. 获取并处理注意力矩阵 ---
            # A_raw 形状: [Seq, Seq] (对多头取平均)
            A_raw = all_attentions[i][0].mean(dim=0)

            v_states = pkv[i][1] 
            v_states = v_states[0] 
            v_norm = torch.norm(v_states, dim=-1).mean(dim=0) 
            A_weighted = A_raw * v_norm.unsqueeze(0)
            A_weighted[:, 0] = 0 

            # --- E. 重新归一化与残差融合 ---
            # 重新归一化行和为 1
            A_bar = A_weighted / (A_weighted.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 融合残差连接
            # I = torch.eye(seq_len, device=device)
            A_final = 0.5 * A_bar + 0.5 * I

            # --- F. Rollout 累乘 ---
            R = torch.matmul(A_final, R)

        # attn_len = len(all_attentions)
        # for attn_layer in range(attn_len):
        #     A_i = 0.9 * all_attentions[attn_layer].mean(dim=1) + 0.1 * I
        #     if attn_layer == 0:
        #         R_i = I
        #         continue
        #     else:
        #         R_i = torch.matmul(A_i, R_i)
        print(f"R: {R}")
        target_token_row = R[answer_indices, :] 
        image_scores = target_token_row[:, image_token_indices] # 形状: [answer, image_token]
        importance_map = image_scores.mean(dim=0).reshape(grid_h, grid_w).float().cpu().numpy()
        img_h, img_w = original_img.shape[:2]
        v_min, v_max = importance_map.min(), importance_map.max()
        importance_map = (importance_map - v_min) / (v_max - v_min + 1e-8)
        heatmap_resized = cv2.resize(importance_map, (img_w, img_h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # ✨ 图像融合: 0.6 原图 + 0.4 热力图
        alpha = 0.6
        overlay_img = cv2.addWeighted(heatmap_color, alpha, original_img, 1 - alpha, 0)

        # 7. 动态命名并保存图片
        output_path = f'{out_dir}batch_{batch_idx}'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img_save_path = os.path.join(out_dir, f"batch_{batch_idx}/answer_heatmap.jpg")
        cv2.imwrite(img_save_path, overlay_img)
        

        if batch_idx >= 9:
            print(f"\n🎉 验证完成，前 10 个 Batch 的图文数据均已保存在 {out_dir} 中，停止绘图。")
            break
        
    txt_save_path = os.path.join(out_dir, f"query_answer.txt")
    with open(txt_save_path, "w", encoding="utf-8") as f:
        for i in range(len(q_list)):
            f.write(f"idx: {i}  Query: {q_list[i]}  Answer: {a_list[i]}\n")
    print("红色区域代表模型认为分数最高、最关心的 Tokens，蓝色代表可被剪枝的低分 Tokens。")
    # exit(0)  # 如果你跑完验证就想直接停掉程序，可以取消这行注释

def gradient(    
    model,
    processor,
    data_loader,
    out_dir,
):
    """
    计算并保存前10个样本中，Qwen vision encoder 输出的 image_embeds 对模型Loss的梯度重要性。
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 设置模型状态
    model.eval()
    device = next(model.parameters()).device
    
    # ==========================================
    # 2. 定位输出 image_embeds 的模块
    # ==========================================
    # 在 Qwen 源码中，model.visual 模块的直接输出就是最终的 image_embeds
    if hasattr(model, "visual"):
        visual_module = model.visual            # Qwen2-VL
    elif hasattr(model, "transformer") and hasattr(model.transformer, "visual"):
        visual_module = model.transformer.visual # Qwen-VL (第一代)
    else:
        raise ValueError("无法定位到视觉模块，请确认加载的是 Qwen-VL 系列模型。")

    # ==========================================
    # 3. 注册 Hook，专门拦截 image_embeds
    # ==========================================
    intercepted_data = {} # 用于将模型内部的 image_embeds 偷渡出来
    
    def image_embeds_hook(module, args, output):
        # 这里的 output 就是官方源码 `image_embeds = self.visual(...)` 的直接结果
        # 其形状通常为 (总图像Token数, hidden_size)
        image_embeds = output[0] if isinstance(output, tuple) else output
        
        # 核心逻辑：命令 PyTorch 在反向传播时不要丢弃这个中间变量的梯度
        if image_embeds.requires_grad:
            image_embeds.retain_grad() 
            
        intercepted_data['image_embeds'] = image_embeds
        return output

    # 挂载 Hook
    hook_handle = visual_module.register_forward_hook(image_embeds_hook)

    results =[]
    max_samples = 10
    sample_count = 0
    
    # ==========================================
    # 4. 开始遍历数据
    # ==========================================
    for batch_idx, batch in enumerate(data_loader):
        if sample_count >= max_samples:
            break
            
        # 将数据移至 GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if "pixel_values" not in batch:
            print(f"Batch {batch_idx} 没有 pixel_values，跳过...")
            continue
            
        # 【极其关键】由于模型在 eval() 模式下，默认输入不计算梯度。
        # 必须显式开启 pixel_values 的梯度，反向传播（backward）的链路才能一直贯穿到 image_embeds
        batch["pixel_values"].requires_grad_(True)
        
        # 清理状态
        model.zero_grad()
        intercepted_data.clear() 
        
        # 5. 前向传播
        # 这步执行时，底层会自动调用 visual_module，从而触发我们的 hook 并保存 image_embeds
        outputs = model(
            **batch,

        )

        loss = outputs.loss
        if loss is None:
            print(f"Batch {batch_idx} 未返回 loss，请确保传入了 labels 用于计算梯度！")
            continue
            
        # 6. 反向传播
        loss.backward()
        
        # 7. 提取 image_embeds 及其梯度
        image_embeds = intercepted_data.get('image_embeds')
        if image_embeds is None or image_embeds.grad is None:
            print(f"⚠️ 截获 image_embeds 梯度失败！请检查输入张量的 requires_grad 属性。")
            continue
            
        image_embeds_grad = image_embeds.grad
        
        # 8. 评估重要性 (使用经典的 激活值 * 梯度 绝对值)
        # 维度对齐处理：(num_image_tokens, hidden_size) -> (num_image_tokens,)
        # x * grad(x) = importance
        _, grid_h, grid_w = batch["image_grid_thw"][0].tolist()
        grid_h = grid_h // 2
        grid_w = grid_w // 2
        #importance_scores = image_embeds_grad.abs().reshape(grid_h, grid_w).detach().float().cpu().numpy()
        importance_scores = (image_embeds * image_embeds_grad)
        importance_scores = torch.sum(image_embeds * image_embeds_grad, dim=-1).reshape(grid_h, grid_w).detach().float().cpu().numpy()
        v_min, v_max = importance_scores.min(), importance_scores.max()
        importance_scores = (importance_scores - v_min) / (v_max - v_min + 1e-8)
        # scores_l2 = image_embeds.grad.detach().norm(p=2, dim=-1).reshape(grid_h, grid_w).detach().float().cpu().numpy()
        # v_min, v_max = scores_l2.min(), scores_l2.max()
        # importance_scores_l2 = (scores_l2 - v_min) / (v_max - v_min + 1e-8)
        try:
            # 获取当前样本的图像路径
            image_path = batch["img_path"]
            print(batch["img_path"])
            # 兼容有些 collator 可能会把 path 包装在 list/tuple 里的情况
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0]
                
            # OpenCV 默认读取为 BGR 格式，正适合后续直接 imwrite 保存
            original_img = cv2.imread(image_path)
            
            if original_img is None:
                raise ValueError(f"图片路径无效或无法读取: {image_path}")
        
        except Exception as e:
            print(f"读取原始图像失败 ({e})，将使用纯白背景图...")
            # 兜底：如果没读到图，就生成一张白底图
            original_img = np.full((grid_h * 28, grid_w * 28, 3), 255, dtype=np.uint8)
        img_h, img_w = original_img.shape[:2]
        
        heatmap_resized = cv2.resize(importance_scores, (img_w, img_h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # ✨ 图像融合: 0.4 原图 + 0.6 热力图
        alpha = 0.6
        overlay_img = cv2.addWeighted(heatmap_color, alpha, original_img, 1 - alpha, 0)

        # 7. 动态命名并保存图片
        # output_path = f'{out_dir}batch_{batch_idx}'
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        img_save_path = os.path.join(out_dir, f"{batch_idx}_gradient_heatmap.jpg")
        cv2.imwrite(img_save_path, overlay_img)
        
        if batch_idx >= 9:
            print(f"\n🎉 验证完成，前 10 个 Batch 的图文数据均已保存在 {out_dir} 中，停止绘图。")
            break
        
    # ==========================================
    # 10. 收尾清理
    # ==========================================
    hook_handle.remove() # 务必移除 Hook
    
    pt_path = os.path.join(out_dir, "image_embeds_gradients.pt")
    torch.save(results, pt_path)
    
    json_path = os.path.join(out_dir, "image_embeds_gradients.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
def main():
    """主训练函数"""
    parser = TrlParser((PATOScriptArgurment, PATOTrainingArgument, PATOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    set_seed(training_args.seed)
    
    # ------------ init model ------------ #
    # ⚠️ 修改点 1：验证 Attention 必须使用 eager 模式，不能使用 flash_attention_2！
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        device_map="cuda",
        attn_implementation="eager",  # <--- 必须修改为 eager
        torch_dtype=model_args.torch_dtype
    )
    # dump_param_freeze_status(model, training_args.output_dir)

    # ------------ loading training dataset ------------ #
    processor = AutoProcessor.from_pretrained(
        model_args.teacher_model_name_or_path,
        max_pixels=script_args.max_pixels,
    )
    # patch_processor(processor) # 假设这是你的自定义函数
    
    train_dataset = PATODataset(
        script_args.train_dataset,
        processor,
        script_args,
    ) if script_args.train_dataset else None

    collator = PATOCollator(
        processor=processor,
        is_sft=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer
    )
    data_loader = trainer.get_train_dataloader()

    # ==========================================
    # TODO 添加: 提取并保存 Attention 热力图验证
    # ==========================================
    print("开始进行 Vision Tokens 重要性(Attention)验证...")
    model.eval() # 切换到评估模式

   # ==========================================
    # TODO 添加: 提取并保存 Attention 热力图验证
    # ==========================================
    print("开始进行 Vision Tokens 重要性(Attention)验证...")
    model.eval() # 切换到评估模式
    
    # 提前准备好输出目录
    out_dir = "result/Qwen2.5-VL-3B-Instruct/gradient/"
    os.makedirs(out_dir, exist_ok=True)

    # attention_scores(
    #     model=model,
    #     processor=processor,
    #     data_loader=data_loader,
    #     out_dir=out_dir,
    # )
    gradient(
        model=model,
        processor=processor,
        data_loader=data_loader,
        out_dir=out_dir,
    )
    
if __name__ == "__main__":
    main()