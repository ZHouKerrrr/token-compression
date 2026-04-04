import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from typing import List
import re
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from transformers.trainer import _is_peft_model
from transformers import TrainerCallback, TrainerState, TrainerControl
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


def norm_bboxes(bboxes, height, width, bbox_type="xyxy"):
    assert bbox_type in ["xyxy", "xywh", "xyxy_norm1000"]
    normed_bboxes = []
    for bbox in bboxes:
        if bbox_type == "xyxy":
            x1, y1, x2, y2 = bbox
            normed_bboxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
        elif bbox_type == "xyxy_norm1000":
            x1, y1, x2, y2 = bbox
            normed_bboxes.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
        else:
            x1, y1, w, h = bbox
            normed_bboxes.append([x1 / width, y1 / height, (x1 + w) / width, (y1 + h) / height])
    return normed_bboxes


def extract_one_bbox_from_str(bbox_str: str) -> List[float]:
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(bbox_pattern, bbox_str)
    if match:
        try:
            coords_str = match.groups()
            bbox_coords = [float(coord) for coord in coords_str]
            return bbox_coords
        except ValueError:
            return [0, 0, 0, 0] # Or raise an error
    else:
        return [0, 0, 0, 0]
    

def cal_paired_ious(bboxes_1: np.ndarray, bboxes_2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between a pair of bounding boxes.
    Args:
        bboxes_1 (np.ndarray): Array of shape (N, 4) for first set of boxes.
        bboxes_2 (np.ndarray): Array of shape (N, 4) for second set of boxes.
    Returns:
        np.ndarray: IoU of shape (N, ) where N is the number of boxes.
    """
    assert bboxes_1.shape == bboxes_2.shape, "Bounding boxes must have the same shape"
    
    x1 = np.maximum(bboxes_1[:, 0], bboxes_2[:, 0])
    y1 = np.maximum(bboxes_1[:, 1], bboxes_2[:, 1])
    x2 = np.minimum(bboxes_1[:, 2], bboxes_2[:, 2])
    y2 = np.minimum(bboxes_1[:, 3], bboxes_2[:, 3])
    
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area_1 = (bboxes_1[:, 2] - bboxes_1[:, 0]) * (bboxes_1[:, 3] - bboxes_1[:, 1])
    area_2 = (bboxes_2[:, 2] - bboxes_2[:, 0]) * (bboxes_2[:, 3] - bboxes_2[:, 1])
    
    union_area = area_1 + area_2 - intersection_area
    
    iou = intersection_area / (union_area + 1e-6) # Add small value to avoid division by zero
    return iou


def print_rank0(*args, **kwargs):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(*args, **kwargs)


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


def dump_param_freeze_status(model, filepath):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        with open(filepath, "w") as f:
            total = 0
            trainable = 0

            for name, p in model.named_parameters():
                numel = p.numel()
                total += numel
                if p.requires_grad:
                    trainable += numel

                f.write(
                    f"{name}\t"
                    f"requires_grad={p.requires_grad}\t"
                    f"shape={tuple(p.shape)}\t"
                    f"numel={numel}\n"
                )

            f.write("\n")
            f.write(f"TOTAL_NUMEL={total}\n")
            f.write(f"TRAINABLE_NUMEL={trainable}\n")
            f.write(f"TRAINABLE_RATIO={trainable / total:.6f}\n")


def test_model(
    model,
    trainer,
    train_dataset,
    collator,
    scorer_path="visual.token_sorter.token_scorer",  # 如果你的路径是 visual.token_sorter.token_scorer 可改
    use_amp=False,  # 如果你训练是 AMP，这里仍建议先关掉做 debug
):
    """
    核心测试：
      - 取一个 batch
      - 前向拿 distortion_loss / rate_loss
      - 用 torch.autograd.grad 计算它们对 token_scorer 参数的梯度
      - 输出每个参数来自各项 loss 的梯度强度与相似度
    """
    def _move_batch_to_device(batch, device):
        """
        支持 dict / list / tuple 的 batch，把 tensor 移到 device。
        """
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            typ = type(batch)
            return typ(_move_batch_to_device(v, device) for v in batch)
        return batch


    def _extract_losses_from_outputs(outputs):
        """
        从 model(**batch) 的输出里取 distortion_loss / rate_loss。
        你可以按你项目实际输出结构微调这里。

        支持几种常见形式：
        1) outputs 是 dict，含 'distortion_loss' / 'rate_loss'
        2) outputs 是 dict，含 'aux_outputs'，其中含 'rate_loss'
        3) outputs 是 tuple/list，最后一个元素可能是 dict aux_outputs
        """
        distortion_loss = None
        rate_loss = None

        # dict
        if isinstance(outputs, dict):
            if "aux_outputs" in outputs and isinstance(outputs["aux_outputs"], dict):
                aux = outputs["aux_outputs"]
                if "rate_loss" in aux:
                    rate_loss = aux["rate_loss"]
                elif "keep_ratio" in aux:
                    keep_ratio = aux["keep_ratio"]
                    rate_loss = (keep_ratio.mean(dim=0) - 0.25) ** 2
                if "distortion_loss" in aux:
                    distortion_loss = aux["distortion_loss"]
                else:
                    distortion_loss = outputs["loss"]
        
        return distortion_loss, rate_loss


    def _safe_zero_grad(model, optimizer=None):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            for p in model.parameters():
                p.grad = None


    @torch.no_grad()
    def _print_param_grad_summary(grads_by_name, title):
        """
        grads_by_name: {name: grad_tensor_or_None}
        """
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        for n, g in grads_by_name.items():
            if g is None:
                print(f"{n:60s}  NO_GRAD")
            else:
                mean_abs = float(g.abs().mean().cpu())
                norm = float(g.norm().cpu())
                mx = float(g.abs().max().cpu())
                print(f"{n:60s}  norm={norm:.3e}  mean_abs={mean_abs:.3e}  max_abs={mx:.3e}")


    def _cosine(a, b):
        if a is None or b is None:
            return 0.0
        a = a.flatten()
        b = b.flatten()
        denom = (a.norm() * b.norm()).clamp_min(1e-12)
        return float((a @ b) / denom)
    
    device = next(model.parameters()).device
    model.train()

    # 1) 取 batch
    if hasattr(trainer, "get_train_dataloader"):
        data_loader = trainer.get_train_dataloader()
        batch = next(iter(data_loader))
    else:
        data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator)
        batch = next(iter(data_loader))

    # 2) 拿 lambda 系数
    lam_d = 1.0
    lam_kd = 1.0
    lam_r = 0.1
    

    # 3) 定位 token_scorer 参数
    # 支持 scorer_path 的两种情况：
    # - 直接 model.token_sorter.token_scorer
    # - model.visual.token_sorter.token_scorer 等
    def _get_submodule(root, path: str):
        cur = root
        for part in path.split("."):
            cur = getattr(cur, part)
        return cur

    try:
        token_scorer = _get_submodule(model, scorer_path)
    except Exception as e:
        raise RuntimeError(
            f"找不到 scorer_path='{scorer_path}' 对应的模块。"
            f"请改成你的真实路径，比如 'visual.token_sorter.token_scorer'。原错误: {e}"
        )

    scorer_named_params = list(token_scorer.named_parameters())
    scorer_params = [p for _, p in scorer_named_params]
    assert scorer_params is not None, "scorer_params is none"
    # 4) 前向得到 outputs
    # 兼容 trainer.compute_loss
    distortion_loss = None
    rate_loss = None

    if hasattr(trainer, "compute_loss"):
        # 许多 trainer.compute_loss(model, inputs, return_outputs=True)
        try:
            # 尽量拿 return_outputs=True
            out = trainer.compute_loss(model, batch, return_outputs=True)
            # out 可能是 (loss, outputs)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                _, outputs = out
            else:
                outputs = out
            distortion_loss, rate_loss = _extract_losses_from_outputs(outputs)
        except TypeError:
            # compute_loss 签名不一样，fallback 用 model(**batch)
            outputs = model(**batch) if isinstance(batch, dict) else model(batch)
            distortion_loss, rate_loss = _extract_losses_from_outputs(outputs)
    else:
        outputs = model(**batch) if isinstance(batch, dict) else model(batch)
        distortion_loss, rate_loss = _extract_losses_from_outputs(outputs)

    if distortion_loss is None or rate_loss is None:
        raise RuntimeError(
            "无法从一次 forward 输出中同时提取 distortion_loss 和 rate_loss。\n"
            "请在 _extract_losses_from_outputs() 里按你的项目输出结构补齐字段路径，"
            "或让 model/outputs 显式返回这两个 loss。"
        )

    # 5) 用 autograd.grad 拆分梯度来源
    # 注意：不用 backward，不污染 .grad，更适合 debug
    # retain_graph=True 是为了同一个 forward 图上算两次 grad
    if use_amp and torch.cuda.is_available():
        # debug 时建议关 AMP；如果必须，用 autocast 包住前向，但这里我们已做过前向
        pass
    def safe_autograd_grad(loss, params, name, retain_graph):
        if not torch.is_tensor(loss) or not loss.requires_grad:
            print(f"[SKIP] {name} is not differentiable: "
                f"type={type(loss)}, requires_grad={getattr(loss,'requires_grad',None)}, grad_fn={getattr(loss,'grad_fn',None)}")
            return [None for _ in params]
        return torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    g_dist = safe_autograd_grad(
        lam_d * distortion_loss,
        scorer_params,
        "distortion_loss",
        retain_graph=True,
    )
    g_rate = safe_autograd_grad(
        lam_r * rate_loss,
        scorer_params,
        "rate_loss",
        retain_graph=True,
    )
    g_total = safe_autograd_grad(
        lam_d * distortion_loss + lam_r * rate_loss,
        scorer_params,
        "loss",
        retain_graph=False,
    )

    # 6) 组织输出
    dist_by_name = {}
    rate_by_name = {}
    total_by_name = {}
    for (n, _), gd, gr, gt in zip(scorer_named_params, g_dist, g_rate, g_total):
        dist_by_name[n] = None if gd is None else gd.detach()
        rate_by_name[n] = None if gr is None else gr.detach()
        total_by_name[n] = None if gt is None else gt.detach()

    _print_param_grad_summary(dist_by_name, f"[token_scorer] grad from lambda_distortion * distortion_loss (lambda={lam_d})")
    _print_param_grad_summary(rate_by_name, f"[token_scorer] grad from lambda_rate * rate_loss (lambda={lam_r})")
    _print_param_grad_summary(total_by_name, "[token_scorer] grad from total loss")

    # 7) 打印每个参数：谁贡献更大 + cosine 相似度（是否冲突/抵消）
    print("\n" + "-" * 80)
    print("Per-parameter attribution (norms + cosine similarity)")
    print("-" * 80)
    for (n, _), gd, gr, gt in zip(scorer_named_params, g_dist, g_rate, g_total):
        dn = 0.0 if gd is None else float(gd.norm().detach().cpu())
        rn = 0.0 if gr is None else float(gr.norm().detach().cpu())
        tn = 0.0 if gt is None else float(gt.norm().detach().cpu())
        c_td = _cosine(gt.detach() if gt is not None else None, gd.detach() if gd is not None else None)
        c_tr = _cosine(gt.detach() if gt is not None else None, gr.detach() if gr is not None else None)
        c_dr = _cosine(gd.detach() if gd is not None else None, gr.detach() if gr is not None else None)

        dominant = "rate" if rn > dn else "distortion"
        print(
            f"{n:30s}  ||dist||={dn:.3e}  ||rate||={rn:.3e}  ||total||={tn:.3e}  "
            f"dom={dominant:10s}  cos(total,dist)={c_td:+.3f}  cos(total,rate)={c_tr:+.3f}  cos(dist,rate)={c_dr:+.3f}"
        )

    print("\nDone.")
    
    
def evaluate_batch(model, train_dataloader, processor):
    for batch in train_dataloader:
        model.eval()
        
        # ==========================================
        # 1. 计算 Loss (前向传播)
        # ==========================================
        with torch.no_grad():
            outputs = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )
        loss = outputs.loss.item() if outputs.loss is not None else "N/A"
        print("loss:", loss)

        # 假设 batch size = 1
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]

        # ==========================================
        # 2. 找到 Prompt 和 Answer 的分界线
        # labels 中非 -100 的第一个位置，就是模型回答的起点
        # ==========================================
        valid_indices = (labels != -100).nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            print("Warning: No valid labels found in this batch.")
            continue
            
        # 找到答案的起始位置
        prompt_end_idx = valid_indices[0].item()

        # 提取真正的 Query 和 Target 文本
        query_ids = input_ids[:prompt_end_idx]
        query_text = processor.tokenizer.decode(query_ids, skip_special_tokens=True)
        
        # 只提取非 -100 的标签部分
        target_ids = labels[labels != -100]
        target_text = processor.tokenizer.decode(target_ids, skip_special_tokens=True)

        # ==========================================
        # 3. 让模型完整自由生成 (Auto-regressive Generation)
        # ==========================================
        # 准备生成所需的输入：把答案截断，只给模型喂 Prompt 部分
        gen_inputs = {}
        for k, v in batch.items():
            if k in ["input_ids", "attention_mask"]:
                gen_inputs[k] = v[:, :prompt_end_idx]
            elif k != "labels":
                # 保留其它多模态参数（如 VLM 需要的 pixel_values 等）
                gen_inputs[k] = v

        pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id

        with torch.no_grad():
            generated_ids = model.generate(
                **gen_inputs,
                max_new_tokens=50,           # 允许生成的最大长度（根据需要调整）
                pad_token_id=pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,             # 贪心解码，保证测试的确定性
                use_cache=True
            )

        # ==========================================
        # 4. 提取模型新生成的部分并解码
        # generated_ids 包含了 [Prompt + 新预测的文本]
        # ==========================================
        new_token_ids = generated_ids[0][prompt_end_idx:]
        pred_text = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # 输出结果
        print("-" * 50)
        print(f"Query (Input): \n{query_text.strip()}\n")
        print(f"Target Label (Ground Truth): \n{target_text.strip()}\n")
        print(f"Model Prediction (Full Generation): \n{pred_text.strip()}\n")
        print("-" * 50)
        
        # 如果只想打印第一个 batch 进行观测，可以使用 break
        # break 
        
    return


def init_pato_config(
    base_model_config,
    model_args,
):
    pato_config : PATOConfig = create_default_pato_config(**asdict(model_args))

    pato_config.g_raw.text_hidden_size = base_model_config.hidden_size
    pato_config.projector.vision_hidden_size = base_model_config.vision_config.hidden_size
    pato_config.projector.hidden_size = base_model_config.hidden_size

    return pato_config