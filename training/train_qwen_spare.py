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
from pato_integration import (
    SPAREQwen2_5_VLForConditionalGeneration,
    SPAREQwen2_5_VLConfig,
)
from pato_integration.spare_config import (
    SPAREConfig, 
    SPARELossConfig,
    create_default_spare_config,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLConfig,
)
from g_raw import WeightedDownsample
from token_sort import PruneMergeTokenSorter
from pato_integration.spare_loss import create_spare_loss

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.models import unwrap_model_for_generation
from dataclasses import dataclass, field, asdict

from accelerate.utils import is_peft_model, set_seed


from training.utils import *
"""
    SPAREScriptArgurment:
        保存数据集、训练集、图片目录、随机种子数、最大粉笔那缕、最大输入序列、最大输入保留序列等参数
    SPARETrainingArgument:
        保存训练相关参数，如输出目录、训练轮次、batch大小、学习率、权重衰减、日志与保存间隔等
    SPAREModelConfig:
        保存模型相关配置，如模型路径、目标图像大小、token预算、蒸馏损失权重、预算正则化权重等
"""
@dataclass
class SPAREScriptArgurment:
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
    state_dict_path: Optional[str] = field(
        default=None,
        metadata={"help": "Resume from checkpoint's path"},
    )
@dataclass
class SPARETrainingArgument(TrainingArguments):
    """
        除了自定义的训练参数外，还继承了 transformers.TrainingArguments 的所有参数
        包括了：batch_size等基本训练参数
    """
    lambda_loss : Optional[dict] = field(
        default=None,
        metadata={"help": "Loss weights for different loss components."}
    )
@dataclass
class SPAREModelConfig:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    teacher_config: Optional[dict] = field(
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
    resume_from_prev: Optional[bool] = field(
        default=False,
        metadata={"help": "Determine whether to resume from previous checkpoint."}
    )
    model_args : Optional[dict] = field(
        default=None,
        metadata={"help": "arguments of spare model."}
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
            if isinstance(module, PruneMergeTokenSorter):
                module.current_progress = progress

class SPARETrainer(Trainer):
    """SPARE Training 
        TODO:   1. 需要自主进行loss设计     def compute_loss(self, model, inputs, return_outputs=False) 
                2. 需要实现模型保存与加载    
    """
    def __init__(
            self, 
            teacher_model: Optional[torch.nn.Module] = None,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        self.args = kwargs["args"]
        self.lambda_loss = kwargs["args"].lambda_loss
        self.spare_loss_fct = create_spare_loss()
        self.print_loss = 0
        

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
        
        # --------- student route forward --------- #

        if self.teacher is not None:
            output_hidden_states = True
        else:
            output_hidden_states = False

        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )

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
            
            # -------------------------------------- #
            #           LOSS 设计核心代码             #
            # -------------------------------------- #

            if self.teacher is not None:
                aux_outputs = outputs.aux_outputs
                labels = inputs["labels"]
                teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}

                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        **teacher_inputs,
                        use_cache=False,
                        return_dict=True,
                        output_hidden_states=output_hidden_states,
                    )
                # -----  rate loss 反向退火  ----- #
                if "lambda_rate" in self.lambda_loss:
                    progress = self.state.global_step / self.state.max_steps # 0.00 -> 1.00
                    progress = torch.clamp(torch.tensor(progress, dtype=torch.float32), 0.0, 1.0)
                    lambda_loss = self.lambda_loss.copy()
                    lambda_loss['lambda_rate'] = lambda_loss['lambda_rate_start'] + (lambda_loss['lambda_rate'] -  lambda_loss['lambda_rate_start']) * 0.5 * (1 - torch.cos(torch.pi * progress)) # 0.10 -> 1.00
                losses = self.spare_loss_fct(
                    inputs=inputs,
                    labels=labels,
                    lambda_loss=lambda_loss,
                    students_outputs=outputs,
                    teacher_outputs=teacher_outputs,
                )

                loss = sum(losses.values())
                if self.state.global_step % 50 == 0:
                    if self.print_loss % 2 == 0: 
                        min_key = min([k for k in aux_outputs.keys() if isinstance(k, int)]) # 不考虑不存在的情况
                        s_aux = aux_outputs[min_key]
                        print_rank0(f"keep ratio: first: {s_aux['keep_ratio'].item()}, tot: {aux_outputs['keep_ratio'].item()}")
                    if self.print_loss % 8 == 0: 
                        print_rank0(f"progress, lambda", progress, lambda_loss['lambda_rate'],)
                        for key in losses.keys():
                            print_rank0(f"{key}: {losses[key]}")
                    self.print_loss += 1
                    
            else:
                keep_ratio = aux_outputs["keep_ratio"]
                rate_loss = keep_ratio.mean(dim=0)
                distortion_loss = outputs["loss"]
                loss = self.lambda_loss["lambda_distortion"] * distortion_loss + self.lambda_loss["lambda_rate"] * rate_loss
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        
        return (loss, outputs) if return_outputs else loss
    

    def _save(self, output_dir: str = None, state_dict=None):
        """保存SPARE组件"""
        if not self.accelerator.is_main_process:
            return
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        spare_save_path = os.path.join(output_dir, "spare_components.pt")

        with unwrap_model_for_generation(self.model, self.accelerator) as model:
            base_model = self.accelerator.unwrap_model(model)
            spare_state_dict = base_model.get_spare_components()
            torch.save(spare_state_dict, spare_save_path)
            print_rank0(f"✓ SPARE components saved to {spare_save_path}")




def main():
    """主训练函数"""
    parser = TrlParser((SPAREScriptArgurment, SPARETrainingArgument, SPAREModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    set_seed(training_args.seed)
    
    # ------------ init model ------------ #
    print_rank0("\n" + "="*60)
    print_rank0("Initializing SPARE Model")
    print_rank0("="*60)

    model_path = model_args.model_name_or_path
    base_model_config = Qwen2_5_VLConfig.from_pretrained(model_path)
    spare_config: SPAREConfig = create_default_spare_config(**model_args.model_args)
    config = SPAREQwen2_5_VLConfig(
        spare_config,
        **base_model_config.to_dict()
    )

    # print(config)
    model = SPAREQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        device_map="cuda",
        attn_implementation="eager",
        torch_dtype=model_args.torch_dtype
    )

    print_rank0("\n" + "="*60)
    print_rank0("Initializing Teacher Model")
    print_rank0("="*60)
    teacher_config = model_args.teacher_config
    if teacher_config["enable"]:
        print_rank0("Distillation enabled. Loading teacher model...")
        teacher_model_path = teacher_config["teacher_model_name_or_path"]
        teacher_model_config = Qwen2_5_VLConfig.from_pretrained(teacher_model_path)
        teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            teacher_model_path,
            config=teacher_model_config,
            device_map="cuda",
            attn_implementation='eager',
            torch_dtype=model_args.torch_dtype
        )
    else:
        teacher_model = None
    
    if script_args.resume:
        spare_state_dict_path = script_args.state_dict_path
        model.init_spare_components(spare_state_dict_path=spare_state_dict_path, freeze=False)
    elif model_args.resume_from_prev:
        spare_state_dict_path = script_args.state_dict_path
        model.init_spare_components(spare_state_dict_path=spare_state_dict_path, freeze=True)
    else:
        model.init_spare_components()
            
        
    
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
    trainer = SPARETrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer,
        callbacks=[TauAnnealingCallback()],
    )

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
