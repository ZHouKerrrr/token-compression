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
    PATOQwen2_5_VLForConditionalGeneration,
    PATOQwen2_5_VLConfig,
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


from training.utils import *
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
        default=1.0,
        metadata={"help": "Weight for distortion loss."},
    )
    lambda_rate: float = field(
        default=1e-3,
        metadata={"help": "Weight for rate loss."},
    )
    target_rate: float = field(
        default=0.25,
        metadata={"help": "Weight for rate loss."},
    )
    kd_temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for knowledge distillation loss."},
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
    distill: Optional[bool] = field(
        default=False,
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
    token_sort_mode: str = field(
        default="dynamic_token_sorter",
        metadata={"help": "The method of token sorting. Options: 'dynamic_token_sorter','dynamic_token_sorter_v2', 'hard_token_sorter'."}
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
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        self.lambda_kd = self.args.lambda_kd
        self.lambda_distortion = self.args.lambda_distortion
        self.lambda_rate = self.args.lambda_rate
        self.target_rate = self.args.target_rate
        self.kd_temperature = self.args.kd_temperature if hasattr(self.args, "kd_temperature") else 2.0

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
        query = inputs.pop("query", None)
        if query is not None:
            query_id = self.processing_class(
                query,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device=inputs["input_ids"].device)
            inputs.update({
                "query_input_ids": query_id.input_ids,
                "query_attention_mask": query_id.attention_mask,
            })
        else:
            raise NotImplementedError("Currently only supports models with 'query' input. Please modify the code to fit your model's input format.")
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        aux_outputs = outputs.aux_outputs
        
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
            
            # -------------------------------------- #
            #           LOSS 设计核心代码             #
            # -------------------------------------- #
            if self.teacher is not None:
                labels = inputs["labels"]

                # teacher 不需要 labels，省掉一份无用 CE
                teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                with torch.no_grad():
                    teacher_logits = self.teacher(
                        **teacher_inputs,
                        use_cache=False,
                        return_dict=True,
                    ).logits
                # 1. distortion loss
                distortion_loss = outputs["loss"]
                
                # 2. knowledge distillation loss (logits kd loss and hidden_states kd loss)
                T = self.kd_temperature  # 先试 2.0 或 4.0
                student_logits = outputs.logits.float()
                teacher_logits = teacher_logits.float()
                # 和 CE 完全对齐
                shift_student = student_logits[:, :-1, :].contiguous()
                shift_teacher = teacher_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                valid = shift_labels.ne(-100)   # [B, T-1]

                shift_student = shift_student[valid]   # [N_valid, V]
                shift_teacher = shift_teacher[valid]   # [N_valid, V]

                student_log_probs = F.log_softmax(shift_student / T, dim=-1)
                teacher_log_probs = F.log_softmax(shift_teacher / T, dim=-1)
                kd_loss = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="batchmean",
                    log_target=True,
                ) * (T * T)
                
                # 弃用：温度缩放后 loss 过小，难以调节 lambda_kd 权重
                # if self.state.max_steps > 0:
                #     progress = self.state.global_step / self.state.max_steps
                # else:
                #     progress = 0.0
                # self.target_rate_pre = self.target_rate + 0.5 * (1.0 - self.target_rate) * (
                #     1.0 + math.cos(math.pi * progress)
                # ) # 1 -> 0.25
                
                # 3. rate loss
                keep_ratio = aux_outputs["keep_ratio"]
                rate_loss = keep_ratio.mean(dim=0)
                
                loss = self.lambda_distortion * distortion_loss + self.lambda_kd * kd_loss + self.lambda_rate * rate_loss 
                if self.state.global_step % 50 == 0:
                    print_rank0(f"keep ratio: {keep_ratio}")
                    print_rank0(f"Distortion Loss: {self.lambda_distortion * distortion_loss}, KD Loss: {self.lambda_kd * kd_loss}, Rate Loss: {self.lambda_rate * rate_loss[0]},  Loss: {loss[0]}")
            else:
                keep_ratio = aux_outputs["keep_ratio"]
                rate_loss = keep_ratio.mean(dim=0)
                distortion_loss = outputs["loss"]
                loss = self.lambda_distortion * distortion_loss + self.lambda_rate * rate_loss
            unwrapped_model = self.accelerator.unwrap_model(model)
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

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
    pato_config = init_pato_config(base_model_config, model_args)
    config = PATOQwen2_5_VLConfig(
        pato_config,
        **base_model_config.to_dict()
    )
    # print(config)
    model = PATOQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        device_map="cuda",
        attn_implementation={
            "vision_config": "flash_attention_2",
            "text_config": "pato",
        },
        torch_dtype=model_args.torch_dtype
    )
    print_rank0("\n" + "="*60)
    print_rank0("Initializing Teacher Model")
    print_rank0("="*60)
    teacher_model_config = model_args.teacher_model
    if teacher_model_config.enable:
        print_rank0("Distillation enabled. Loading teacher model...")
        teacher_model_path = teacher_model_config.teacher_model_name_or_path
        teacher_model_config = Qwen2_5_VLConfig.from_pretrained(teacher_model_path)
        teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            teacher_model_path,
            config=teacher_model_config,
            device_map="cuda",
            attn_implementation='flash_attention_2',
            torch_dtype=model_args.torch_dtype
        )
    else:
        teacher_model = None
    
    # pato_state_dict_path = "output/qwen2_5_3b_pato/pato_components.pt"
    if script_args.resume:
        raise NotImplementedError("Resume from checkpoint is not implemented yet.")
    else: 
        model.load_pato_components()
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
