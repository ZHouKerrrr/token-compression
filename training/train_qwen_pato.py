import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import argparse
from pathlib import Path
from typing import Optional
import json
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
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
    lambda_kd: float = field(
        default=1.0,
        metadata={"help": "Weight for knowledge distillation loss."},
    )
    lambda_distortion: float = field(
        default=0.5,
        metadata={"help": "Weight for distortion loss."},
    )
    lambda_rate: float = field(
        default=1.0,
        metadata={"help": "Weight for rate loss."},
    )
    target_rate: float = field(
        default=0.25,
        metadata={"help": "Weight for rate loss."},
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

class TauAnnealingCallback(TrainerCallback):
    """
    用于向底层 Token 剪枝模块同步训练进度的回调函数
    """
    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, model, **kwargs):
        # 1. 计算当前进度 progress (0.0 到 1.0)
        # state.max_steps 是总训练步数
        if state.max_steps > 0:
            progress = state.global_step / state.max_steps
        else:
            progress = 0.0
            
        # 2. 遍历模型，找到 HardTokenSorter 并注入 progress
        # 注意：使用 model.modules() 可以穿透 DDP / DeepSpeed 等各种包装器，直达最底层的子模块
        for module in model.modules():
            # 判断是不是你的剪枝模块。可以通过类名，或者检查是否含有 specific 属性
            if type(module).__name__ == "HardTokenSorter" or hasattr(module, 'current_progress'):
                module.current_progress = progress

class PATOTrainer(Trainer):
    """PATO Training 
        TODO:   1. 需要自主进行loss设计     def compute_loss(self, model, inputs, return_outputs=False) 
                2. 需要实现模型保存与加载    
    """
    def __init__(
            self, 
            teacher_model: nn.Module = None,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.lambda_kd = self.args.lambda_kd
        self.lambda_distortion = self.args.lambda_distortion
        self.lambda_rate = self.args.lambda_rate
        self.target_rate = self.args.target_rate


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        ### ========================= ###
        ### output and teacher output ###
        ### ========================= ###
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        if self.teacher is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    **inputs, 
                    use_cache=False,
                    return_dict=True,
                )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if teacher_outputs is not None:
                T = 1.0
            
                aux_outputs = outputs.aux_outputs
                logits_mask = aux_outputs["logits_mask"]
                student_logits = outputs.logits
                teacher_logits = teacher_outputs.logits[logits_mask]
                student_labels = inputs.get("labels", labels)
                student_labels = student_labels[logits_mask]
                valid_token_mask = (student_labels != -100).view(-1)
                student_logits = student_logits[0][valid_token_mask]
                teacher_logits = teacher_logits[valid_token_mask]
                # 【新增修复代码】：动态对齐词表维度
                min_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
                student_logits = student_logits[..., :min_vocab_size]
                teacher_logits = teacher_logits[..., :min_vocab_size]
                student_log_probs = F.log_softmax(student_logits / T, dim=-1)   # [B, T, V]
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)   # [B, T, V]
                
                kl = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none"
                ).sum(dim=-1)  # [B, T]

                kd_loss = kl.mean()
                kd_loss = kd_loss * (T ** 2)
                if self.state.max_steps > 0:
                    progress = self.state.global_step / self.state.max_steps
                else:
                    progress = 0.0

                self.target_rate_pre = self.target_rate + 0.5 * (1.0 - self.target_rate) * (
                    1.0 + math.cos(math.pi * progress)
                )
                keep_ratio = aux_outputs["keep_ratio"]
                rate_loss = (keep_ratio.mean(dim=0) - self.target_rate_pre) ** 2
                distortion_loss = aux_outputs["distortion_loss"]
                loss = self.lambda_distortion * distortion_loss + self.lambda_kd * kd_loss + self.lambda_rate * rate_loss
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            unwrapped_model = self.accelerator.unwrap_model(model)
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        # print_rank0(f"target ratio: {self.target_rate_pre}")
        # print_rank0(f"Distortion Loss: {distortion_loss},  KD Loss: {kd_loss},  Rate Loss: {rate_loss},  Loss: {loss}")
        return (loss, outputs) if return_outputs else loss
    

    def _save(self, output_dir: str = None, state_dict=None):
        """保存PATO组件"""
        if not self.accelerator.is_main_process:
            return
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        pato_save_path = os.path.join(output_dir, "pato_components.pt")

        with unwrap_model_for_generation(self.model, self.accelerator) as model:
            base_model = self.accelerator.unwrap_model(model)
            pato_state_dict = base_model.get_pato_components()
            torch.save(pato_state_dict, pato_save_path)
            print_rank0(f"✓ PATO components saved to {pato_save_path}")


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

def _init_pato_config(
    base_model_config : Qwen2_5_VLConfig = None,
    model_args: PATOModelConfig = None,
):
    vision_hidden_size = base_model_config.vision_config.hidden_size
    text_hidden_size = base_model_config.hidden_size
    # 配置PATO参数，对齐模型维度
    pato_config : PATOConfig = create_default_pato_config(**asdict(model_args))
    pato_config.g_raw.text_hidden_size = text_hidden_size
    pato_config.projector.vision_hidden_size = vision_hidden_size
    pato_config.projector.hidden_size = text_hidden_size
    
    pato_config.g_raw.enable = model_args.g_raw_enable
    pato_config.token_sort.enable = model_args.token_sort_enable
    pato_config.projector.enable = model_args.projector_enable
    return pato_config

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

def test_dataset(
    dataset,
    collator,
):
    print_rank0("============== Testing dataset ==============")
    sampler = RepeatRandomSampler(
        data_source=dataset,
        mini_repeat_count=1,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collator,
    )
    for _, batch in enumerate(loader):
        if batch["image_grid_thw"].shape[0] > 1:
            print(batch)
            return
        input_ids = batch["input_ids"]
        for input_id in input_ids:
            e_idx = torch.where(input_id == 151653)
            print("e_idx", e_idx)
            if e_idx[0].numel() > 1:
                print(batch)
                return
    print("done")
    return

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

            # 常见：aux_outputs 里带 rate_loss
            if "aux_outputs" in outputs and isinstance(outputs["aux_outputs"], dict):
                aux = outputs["aux_outputs"]
                if "rate_loss" in aux:
                    rate_loss = aux["rate_loss"]
                if "distortion_loss" in aux:
                    distortion_loss = aux["distortion_loss"]

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
    lam_r = 1.0
    

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

def main():
    """主训练函数"""
    parser = TrlParser((PATOScriptArgurment, PATOTrainingArgument, PATOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    set_seed(training_args.seed)
    
    # ------------ init model ------------ #
    print_rank0("\n" + "="*60)
    print_rank0("Initializing PATO Model")
    print_rank0("="*60)

    model_path = model_args.model_name_or_path
    base_model_config = Qwen2_5_VLConfig.from_pretrained(model_path)
    pato_config = _init_pato_config(base_model_config, model_args)
    config = PATOQwen2_5_VLConfig(
        pato_config,
        **base_model_config.to_dict()
    )
    # print(config)
    model = PATOQwen2_5_VLModel.from_pretrained(
        model_path,
        config=config,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        torch_dtype=model_args.torch_dtype
    )
    print_rank0("\n" + "="*60)
    print_rank0("Initializing Teacher Model")
    print_rank0("="*60)
    teacher_model_path = model_args.teacher_model_name_or_path
    teacher_model_config = Qwen2_5_VLConfig.from_pretrained(teacher_model_path)
    teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        teacher_model_path,
        config=teacher_model_config,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        torch_dtype=model_args.torch_dtype
    )
    pato_state_dict = None
    # pato_state_dict_path = "output/qwen2_5_3b_pato/pato_components.pt"
    if script_args.resume:
        raise NotImplementedError("Resume from checkpoint is not implemented yet.")
    model.load_pato_components(pato_state_dict=pato_state_dict,)
    # dump_param_freeze_status(model, training_args.output_dir)

    # ------------ loading training dataset ------------ #
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        max_pixels=script_args.max_pixels,
    )
    patch_processor(processor)
    
    train_dataset = PATODataset(
        script_args.train_dataset,
        processor,
        script_args,
    ) if script_args.train_dataset else None

    collator = PATOCollator(
        processor=processor,
        is_sft=True,
    )


    trainer = PATOTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer,
        callbacks=[TauAnnealingCallback()],
    )
    # train_dataloader = trainer.get_train_dataloader()
    # evaluate_batch(model, train_dataloader, processor)
    # return
    # test model
    # test_model(model=model, trainer=trainer, train_dataset=train_dataset, collator=collator)
    # return
    # ------------ Training ------------ #
    # 训练循环
    print_rank0("\n" + "="*60)
    print_rank0("Starting Training")
    print_rank0("="*60)

    # 开始训练
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    
    print_rank0("\n" + "="*60)
    print_rank0("✓ Training completed!")
    print_rank0("="*60)


if __name__ == "__main__":
    main()
