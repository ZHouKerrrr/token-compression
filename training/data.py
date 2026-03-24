import os
import re
import ast
import math
import yaml
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized, Dict, Tuple, List, Literal, Type

import numpy as np
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import datasets

from PIL import Image

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.models import unwrap_model_for_generation

from transformers import (
    TrainingArguments, 
    Trainer,
    GenerationConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_safetensors_available, 
    is_peft_available
)
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLProcessor,
)
if is_safetensors_available():
    import safetensors.torch
from peft import PeftConfig, get_peft_model, PeftModel
from accelerate.utils import is_peft_model, set_seed

from qwen_vl_utils import process_vision_info

from .utils import (
    norm_bboxes, 
    extract_one_bbox_from_str, 
    cal_paired_ious,
    print_rank0
)

from transformers.trainer import (
    logger,
    TRAINING_ARGS_NAME,
    CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    FSDP_MODEL_NAME,
)



# ---------- Datasets ----------


QUERY_KEY = "query"
IMG_PATH_KEY = "img_path"
ANSWER_KEY = "answer"
NORMED_BBOXES_KEY = "normed_bboxes"
SCORE_FUNCS_KEY = "score_funcs"


REMAIN_KEYS = [
    QUERY_KEY,
    IMG_PATH_KEY,
    NORMED_BBOXES_KEY,
    ANSWER_KEY,
    SCORE_FUNCS_KEY,
]


MAPPER_REGISTRY = {}

FILTER_REGISTRY = {}


def register_mappers():
    def wrapper(func):
        name = func.__name__.replace("_dataset_mapper", "")
        MAPPER_REGISTRY[name] = func
        return func
    return wrapper


def register_filters():
    def wrapper(func):
        name = func.__name__.replace("_dataset_filter", "")
        FILTER_REGISTRY[name] = func
        return func
    return wrapper



@register_mappers()
def cot_train_dataset_mapper(one_data, **kwargs):
    query = one_data['question']
    if 'prompt' in kwargs:
        query = kwargs['prompt'].format(query)
    answer = one_data['answer']
    image = one_data['image']
    dataset = one_data['dataset']
    img_path = os.path.join(kwargs['img_dir'], "cot", dataset, image)
    bboxes = one_data['bboxs']
    
    return {
        QUERY_KEY: query,
        ANSWER_KEY: answer,
        IMG_PATH_KEY: img_path,
        NORMED_BBOXES_KEY: bboxes,
    }    
    

@register_mappers()
def cot_train_fullmask_dataset_mapper(one_data, **kwargs):
    query = one_data['question']
    if 'prompt' in kwargs:
        query = kwargs['prompt'].format(query)
    answer = one_data['answer']
    image = one_data['image']
    dataset = one_data['dataset']
    img_path = os.path.join(kwargs['img_dir'], "cot", dataset, image)
    normed_bboxes = [[0.0, 0.0, 1.0, 1.0]]
    
    return {
        QUERY_KEY: query,
        ANSWER_KEY: answer,
        IMG_PATH_KEY: img_path,
        NORMED_BBOXES_KEY: normed_bboxes,
    }    
    
    
@register_mappers()
def norm_bboxes_dataset_mapper(one_data, **kwargs):
    bboxes = one_data.pop(NORMED_BBOXES_KEY)
    if 'width' in one_data:
        width = one_data['width']
        height = one_data['height']
    else:
        img_path = one_data[IMG_PATH_KEY]
        img_pil = Image.open(img_path)
        width, height = img_pil.size
        img_pil.close()
    normed_bboxes = norm_bboxes(bboxes, height, width, bbox_type=kwargs['bbox_type'])
    one_data[NORMED_BBOXES_KEY] = normed_bboxes
    return one_data

    
@register_filters()
def min_image_filter(one_data, **kwargs):
    img_path = one_data[IMG_PATH_KEY]
    min_image_size = kwargs.get('min_image_size', None)
    try:
        img = Image.open(img_path)
        w, h = img.size
        img.close()  # Close the image to free resources
        if w * h >= min_image_size * min_image_size:
            return True  # Image exists and is valid
        else:
            return False
    except (FileNotFoundError, OSError) as e:
        print_rank0(f"Image not found or invalid: {img_path}. Error: {e}")
        return False
    except Exception as e:
        print_rank0(f"Unexpected error while checking image: {img_path}. Error: {e}")
        return False
    
@register_filters()
def inputs_seq_length_dataset_filter(one_data, **kwargs):
    processor = kwargs['processor']
    max_input_seq_length = kwargs.get('max_input_seq_length', None)
    max_input_remain_seq_length = kwargs.get('max_input_remain_seq_length', None)
    min_image_size = kwargs.get('min_image_size', None)
    if max_input_seq_length is None and max_input_remain_seq_length is None:
        return True
    img_path = one_data[IMG_PATH_KEY]
    query = one_data[QUERY_KEY]
    # normed_bboxes = [one_data[NORMED_BBOXES_KEY]] if max_input_remain_seq_length is not None else None
    messages = [[{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}]]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            # normed_bboxes=normed_bboxes,
            padding=True,
            return_tensors="pt",
        )
    seq_length = inputs.input_ids.shape[1]
    if max_input_seq_length is not None and seq_length > max_input_seq_length:
        # print_rank0(f"Input sequence length {seq_length} exceeds max limit {max_input_seq_length}. Filtering out.")
        return False
    
    if max_input_remain_seq_length is not None:
        ref_token_masks = inputs.ref_token_masks[0]
        reduced_num = ref_token_masks.numel() - ref_token_masks.sum().item()
        remain_seq_length = seq_length - reduced_num
        if remain_seq_length > max_input_remain_seq_length:
            # print_rank0(f"Remaining sequence length {remain_seq_length} exceeds max limit {max_input_remain_seq_length}. Filtering out.")
            return False
    # print_rank0(f"inputs.image_grid_thw{inputs.image_grid_thw.shape}")
    grid_t, grid_h, grid_w = inputs.image_grid_thw[0]

    print_rank0("min_image_size")
    if min_image_size is not None and grid_t * grid_h * grid_w > min_image_size:
        print_rank0("work")
        return False
    return True

"""
    Collator for datasets.
"""

# ---------- Dataset & Collator & Sampler ----------

class PATODataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that loads and combines multiple datasets
    based on a YAML configuration file. It handles sampling
    and applies specified mapping functions.
    """
    @classmethod
    def _load_config(cls, config_path: str) -> Dict[str, Any]:
        """Loads configuration from a YAML file."""
        print_rank0(f"Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if config is None or 'datasets' not in config:
                 raise ValueError("YAML config is empty or missing 'datasets' key.")
            print_rank0("Configuration loaded successfully.")
            return config
        except FileNotFoundError:
            print_rank0(f"Error: Configuration file not found at {config_path}")
            raise
        except yaml.YAMLError as e:
            print_rank0(f"Error: Could not parse YAML configuration: {e}")
            raise
        except Exception as e:
            print_rank0(f"An unexpected error occurred during config loading: {e}")
            raise

    @classmethod
    def _apply_sampling(cls, dataset: datasets.Dataset, strategy: Optional[str], seed: Optional[int] = None) -> datasets.Dataset:
        """Applies sampling strategy to a dataset."""
        if not strategy:
            print_rank0("No sampling strategy specified, using full dataset.")
            return dataset

        try:
            parts = strategy.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid sampling strategy format: '{strategy}'. Expected 'type:value'.")

            strat_type, strat_value = parts[0].lower(), parts[1]
            num_samples = int(strat_value)
            total_size = len(dataset)

            if num_samples <= 0:
                 raise ValueError(f"Sampling value must be positive, got: {num_samples} [{strategy}]")
            # Ensure sample size isn't larger than dataset, prevents errors in select/slice
            num_samples = min(num_samples, total_size)


            print_rank0(f"Applying sampling: {strategy} ({num_samples} samples) to dataset of size {total_size}")

            if strat_type == "first":
                return dataset.select(range(num_samples))
            elif strat_type == "end":
                 # Ensure we don't request more than available from the end
                start_index = max(0, total_size - num_samples)
                return dataset.select(range(start_index, total_size))
            elif strat_type == "random":
                if seed is None:
                    print_rank0("Warning: Random sampling without a fixed seed. Results may not be reproducible.")
                shuffled_dataset = dataset.shuffle(seed=seed)
                return shuffled_dataset.select(range(num_samples))
            else:
                print_rank0(f"Warning: Unknown sampling strategy type: '{strat_type}'. Using full dataset.")
                return dataset
        except ValueError as e:
            print_rank0(f"Error parsing sampling strategy '{strategy}': {e}. Using full dataset.")
            return dataset
        except Exception as e:
            print_rank0(f"An unexpected error occurred during sampling: {e}. Using full dataset.")
            return dataset
        
    @classmethod
    def _all_processed_datasets(cls, config, processor, args):
        all_processed_datasets: Dict[str, datasets.Dataset] = {}
        for i, dataset_config in enumerate(config['datasets']):
            print_rank0(f"\nProcessing dataset entry {i+1}/{len(config['datasets'])}...")
            json_path = dataset_config.get('json_path')
            base_name = '.'.join(os.path.basename(json_path).split('.')[:-1])
            dataset_name = dataset_config.get('dataset_name', base_name)
            if not json_path:
                print_rank0(f"Warning: Skipping dataset entry {i+1} due to missing 'json_path'.")
                continue

            sampling_strategy = dataset_config.get('sampling_strategy', None)
            mapper_name = dataset_config.get('mapper')
            bbox_type = dataset_config.get('bbox_type')
            img_dir = dataset_config.get('img_dir', args.img_dir)
            additional_mappers = dataset_config.get('additional_mappers', [])
            prompt = dataset_config.get('prompt', None)
            max_input_seq_length = dataset_config.get('max_input_seq_length', args.max_input_seq_length)
            max_input_remain_seq_length = dataset_config.get('max_input_remain_seq_length', args.max_input_remain_seq_length)
            # 过滤小样本
            min_image_size = dataset_config.get('min_image_size', args.min_image_size)
            
            try:
                print_rank0(f"Loading raw data from: {json_path}")
                # Assuming JSON Lines format, common with `datasets`
                raw_dataset = datasets.load_dataset('json', data_files=json_path, split='train')
                print_rank0(f"Loaded {len(raw_dataset)} examples raw.")

                # Apply sampling
                sampled_dataset = cls._apply_sampling(raw_dataset, sampling_strategy, args.sampling_seed)
                if len(sampled_dataset) == 0:
                    print_rank0("Dataset is empty after sampling, skipping.")
                    continue
                print_rank0(f"Dataset size after sampling: {len(sampled_dataset)}")

                # Apply mapping
                mapper_func = MAPPER_REGISTRY[mapper_name]
                print_rank0(f"Applying mapper: '{mapper_name}'")
                # Prepare arguments for the mapper function
                mapper_kwargs = {
                    'img_dir': img_dir,
                }
                if prompt is not None:
                    mapper_kwargs['prompt'] = prompt
                print_rank0(f"Mapper arguments: {mapper_kwargs}")
                processed_dataset = sampled_dataset.map(
                    mapper_func,
                    num_proc=8,
                    fn_kwargs=mapper_kwargs,
                )

                processed_dataset = processed_dataset.remove_columns(
                    [col for col in processed_dataset.column_names if col not in REMAIN_KEYS]
                )
                    
                # Filtering
                print_rank0("Applying dataset filter: 'min_image_filter'")
                processed_dataset = processed_dataset.filter(
                    min_image_filter,
                    num_proc=8,
                    fn_kwargs={
                        'min_image_size': min_image_size,
                    }
                )
                print_rank0(f"Processed dataset size after min_image_filter: {len(processed_dataset)}")
                
                # Additional filtering
                if max_input_seq_length is not None or max_input_remain_seq_length is not None:
                    processed_dataset = processed_dataset.filter(
                        inputs_seq_length_dataset_filter,
                        num_proc=8,
                        fn_kwargs={
                            'processor': processor,
                            'max_input_seq_length': max_input_seq_length,
                            'max_input_remain_seq_length': max_input_remain_seq_length,
                            'min_image_size' : min_image_size,
                        }
                    )
                    print_rank0(f"Processed dataset size after inputs_seq_length_dataset_filter(with min size of images): {len(processed_dataset)}")

                # Additional mapping
                for additional_mapper in additional_mappers:
                    mapper_func = MAPPER_REGISTRY[additional_mapper]
                    print_rank0(f"Applying additional mapper: '{additional_mapper}'")
                    processed_dataset = processed_dataset.map(
                        mapper_func,
                        num_proc=8,
                        fn_kwargs={
                            'bbox_type': bbox_type,
                        }
                    )
                print_rank0(f"Processed dataset size: {len(processed_dataset)}")
                if len(processed_dataset) == 0:
                    print_rank0(f"Warning: Processed dataset {dataset_name} is empty after mapping. Skipping.")
                    continue
                # Store the processed dataset
                if dataset_name in all_processed_datasets:
                    dataset_name_with_uuid = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    print_rank0(f"Warning: Dataset name '{dataset_name}' already exists. Renaming to '{dataset_name_with_uuid}'")
                    all_processed_datasets[dataset_name_with_uuid] = processed_dataset
                else:
                    all_processed_datasets[dataset_name] = processed_dataset                

            except FileNotFoundError:
                print_rank0(f"Error: Data file not found for dataset entry {i+1}: {json_path}. Skipping.")
            except Exception as e:
                print_rank0(f"Error processing dataset entry {i+1} ({json_path}): {e}. Skipping.")
                
        return all_processed_datasets
        

    def __init__(self, config_path: str, processor: Qwen2_5_VLProcessor, script_args: Optional[Any] = None):
        """
        Initializes the GPDataset.

        Args:
            config_path (str): Path to the YAML configuration file.
            processor (Qwen2_5_VLProcessor): Processor for handling text and vision data.
            script_args (Any, optional): Additional arguments passed from the script
                                         (e.g., training args, could contain seed). Defaults to None.
        """
        super().__init__()
        self.args = script_args
        self.config = self._load_config(config_path)
        self.processor = processor
        all_processed_datasets = self._all_processed_datasets(self.config, self.processor, self.args)
        # Combine all processed datasets
        if all_processed_datasets:
            print_rank0(f"\nConcatenating {len(all_processed_datasets)} processed dataset(s)...")
            # Note: Concatenation works best if all datasets have the exact same features/columns.
            # The `map` function should ensure consistent output structure.
            # Consider using `features=...` argument in `concatenate_datasets` if schemas might differ slightly
            # and you know how to resolve them.
            self.final_dataset = datasets.concatenate_datasets(list(all_processed_datasets.values()))
            if len(self.final_dataset) == 0:
                raise ValueError("Final dataset is empty after concatenation.")
            print_rank0(f"Final combined dataset size: {len(self.final_dataset)}")
            # Optionally print final features/columns
            print_rank0(f"Final dataset features: {self.final_dataset.features}")
        else:
            # print_rank0("No datasets were successfully processed.")
            raise ValueError("No datasets were successfully processed. Please check your configuration.")
            self.final_dataset = None

    def __len__(self) -> int:
        """Returns the total number of samples in the combined dataset."""
        return len(self.final_dataset) if self.final_dataset else 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieves a single sample from the combined dataset."""
        if self.final_dataset is None:
            raise IndexError("Dataset is not initialized or is empty.")
        if not 0 <= index < len(self.final_dataset):
             raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.final_dataset)}")
        # `datasets` objects behave like lists/dicts for access
        return self.final_dataset[index]
    
    
    @classmethod
    def get_processed_dataset_dict(cls, config_path: str, processor: Qwen2_5_VLProcessor, script_args: Optional[Any] = None) -> Dict[str, datasets.Dataset]:
        """
        Class method to get processed datasets based on the YAML configuration.

        Args:
            config_path (str): Path to the YAML configuration file.
            script_args (Any, optional): Additional arguments passed from the script
                                         (e.g., training args). Defaults to None.

        Returns:
            Dict[str, datasets.Dataset]: Dictionary of processed datasets.
        """
        config = cls._load_config(config_path)
        all_processed_datasets = cls._all_processed_datasets(config, processor, script_args)
        return all_processed_datasets



class PATOCollator:
    def __init__(self, processor, is_sft):
        self.processor = processor
        self.is_sft = is_sft
        self.im_start_id = self.processor.tokenizer.encode("<|im_start|>")[0]
        
    def _prepare_labels_from_input_ids(self, input_ids):
        """
        Message sample:
            '<|im_start|>system\n
            You are a helpful assistant.<|im_end|>\n
            <|im_start|>user\n
            <|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>
            <|im_start|>assistant\n
            This is a stop sign in Australia.<|im_end|>\n'
        TODO:
            search the last  token, 
            and mask all tokens before last  + 3 (including 1, role token, and first vision token)
        """

        B, L = input_ids.shape
        labels = input_ids.clone()
        mask = input_ids == self.im_start_id
        flipped_mask = mask.flip(dims=(1,))  # Reverse the mask to find the last <|im_start|> token
        first_idx_in_flipped = torch.argmax(flipped_mask.int(), dim=1)
        last_pos = (L - 1) - first_idx_in_flipped
        mask_until_idx = last_pos + 3
        mask_until_idx = torch.clamp(mask_until_idx, max=L)
        
        arange_l = torch.arange(L, device=input_ids.device).expand(B, -1)
        modification_mask = arange_l < mask_until_idx.unsqueeze(1)
        
        labels[modification_mask] = -100   # ignore index of CrossEntropyLoss
        return labels
        
    
    def __call__(self, features):
        messages = []
        # normed_bboxes = []
        answers = []
        querys = []
        for feature in features:
            query = feature[QUERY_KEY]
            answer = feature[ANSWER_KEY]
            img_path = feature[IMG_PATH_KEY]
            if self.is_sft:
                messages.append([{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}, {"role": "assistant", "content": [{"type": "text", "text": answer}]}])
            else:
                messages.append([{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}])
            # normed_bboxes.append(feature[NORMED_BBOXES_KEY])
            querys.append(query)
            answers.append(answer)
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(not self.is_sft)
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            # normed_bboxes=normed_bboxes,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        if self.is_sft:
            labels = self._prepare_labels_from_input_ids(inputs.input_ids)
            inputs["labels"] = labels
        
        inputs[QUERY_KEY] = querys
        inputs[ANSWER_KEY] = answers
        inputs[IMG_PATH_KEY] = img_path
        return inputs
            

class RepeatRandomSampler(torch.utils.data.Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

#def main():
 