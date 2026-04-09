import os
import re
import base64
import numpy as np
from io import BytesIO
from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm

import torch
from accelerate import Accelerator, DistributedType
import decord

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLConfig,
)
from qwen_vl_utils import process_vision_info

# lmms-eval 核心组件
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance
from lmms_eval import utils
from lmms_eval.utils import eval_logger

from pato_integration import PATOQwen2_5_VLForConditionalGeneration, PATOQwen2_5_VLConfig
from pato_integration.pato_config import PATOConfig, create_default_pato_config



@register_model("pato_qwen2_5_vl")  # 注册模型名称，运行脚本时传入此名字
class PATO_Qwen2_5_VL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 2048 * 28 * 28,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        layer_list: Optional[str] = "0",               # 设定默认值防止报错
        image_token_ratio_list: Optional[str] = "1.0", # 设定默认值防止报错
        image_token_ratio: Optional[float] = 1.0,
        pato_state_dict_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # 移除内部框架传入的特定评测参数，防止 assert 报错
        if "max_length" in kwargs:
            self._max_length = kwargs.pop("max_length")
        else:
            self._max_length = 2048

        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert not interleave_visuals, "Interleave visuals is not supported yet."

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
            
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
        }
        
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # 1. 正常加载预训练权重
        pato_state_dict = torch.load(
            pato_state_dict_path,
            map_location=device,
            weights_only=False,
        )
        self._config = pato_state_dict["config"]
        
        self._model = PATOQwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained, 
            config=self._config,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).eval()
        self._model.set_route("pato")
        self._model.load_pato_components(pato_state_dict=pato_state_dict)
        self._model.model.layer_list = list(map(int, str(layer_list).split('-')))
        self._model.model.image_token_ratio_list = list(map(float, str(image_token_ratio_list).split('-')))
        self._model.image_token_ratio = image_token_ratio
        # print("layer_list: ", self._model.model.layer_list)
        # print("image_token_ratio_list", self._model.model.image_token_ratio_list)
        # print("image_token_ratio", XCUself._model.image_token_ratio)
        processor_kwargs = {"padding_side": "left"}
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        self.max_num_frames = max_num_frames
        
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
            
        self.processor = Qwen2_5_VLProcessor.from_pretrained(pretrained, **processor_kwargs)
        self._tokenizer = self.processor.tokenizer
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        
        # 3. 处理分布式 (FSDP / DDP)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in[
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res =[]

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            
            # 兼容 LMMS-Eval 的视觉信息读取
            visual_list =[doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list]")

            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)
            query = contexts.copy()
            batched_messages =[]
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")): 
                        # 处理视频
                        vr = decord.VideoReader(visual)
                        processed_visuals.append({"type": "video", "video": visual})
                    elif isinstance(visual, Image.Image):  
                        # 处理单图/多图
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})

                if self.interleave_visuals is False:
                    message.append({
                        "role": "user",
                        "content": processed_visuals +[{"type": "text", "text": context}],
                    })
                else:
                    raise NotImplementedError("Interleaved logic needs custom mapping.")

                batched_messages.append(message)

            texts =[self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
                
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            # 推理生成
            if query is not None:
                query_id = self._tokenizer(
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
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=False,
            )

            # 截取新生成的文本
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # 后处理：依据 until 词汇截断
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
                
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")