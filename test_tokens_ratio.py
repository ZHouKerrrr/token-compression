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
        default=1,
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

def ratio(
    model,
    processor,
    data_loader,
    out_dir,
):
    keep_ratio = 0.0
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing", ncols=100), start=1):
        with torch.no_grad():
            keep_ratio_i = model(**batch)
        keep_ratio = keep_ratio + keep_ratio_i
        if batch_idx % 10 == 0:
            print(f"\nTotal samples: {batch_idx}")
            print(f"Average of keep_ratio: {keep_ratio / batch_idx}")
    keep_ratio = keep_ratio / batch_idx
    
    print(f"Total samples: {batch_idx}")
    print(f"Average of keep_ratio: {keep_ratio}")

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
    
    pato_config.g_raw.enable = False
    pato_config.token_sort.enable = True
    pato_config.projector.enable = False
    pato_config.evaluate = True
    return pato_config   

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


def main():
    """主训练函数"""
    parser = TrlParser((PATOScriptArgurment, PATOTrainingArgument, PATOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    set_seed(training_args.seed)
    
    # ------------ init model ------------ #
    # ⚠️ 修改点 1：验证 Attention 必须使用 eager 模式，不能使用 flash_attention_2！
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
    # dump_param_freeze_status(model, training_args.output_dir)

    # ------------ loading training dataset ------------ #
    processor = AutoProcessor.from_pretrained(
        model_args.teacher_model_name_or_path,
        max_pixels=script_args.max_pixels,
    )
    patch_processor(processor) # 假设这是你的自定义函数
    pato_state_dict_path = "output/qwen2_5_3b_pato/pato_components.pt"
    model.load_pato_components(pato_state_dict_path=pato_state_dict_path,)
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
    model.eval() # 切换到评估模式
    
    # 提前准备好输出目录
    out_dir = "result/pato/ratio/"
    os.makedirs(out_dir, exist_ok=True)

    ratio(
        model=model,
        processor=processor,
        data_loader=data_loader,
        out_dir=out_dir,
    )
    
if __name__ == "__main__":
    main()