import sys
import os
from typing import Optional, Tuple, List, Union, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss

def reorganize_tensor(
    tensor,
    tensor_size,
    pad,
    batch_valid_counts,
    mask,
):
    B = batch_valid_counts.shape[0]
    
    new_tensor = torch.full(
        tensor_size,
        pad,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    for i in range(B):
        valid_len = batch_valid_counts[i]
        mask_i = mask[i]   # [old_seq_len]
        if len(tensor_size) == 2:
            new_tensor[i, :valid_len] = tensor[i, mask_i]
        else:
            for d in range(3):
                # 获取当前维度、当前样本的有效 mask
                pos_mask_d_i = mask[d, i] # [old_seq_len]
                valid_pos = tensor[d, i][pos_mask_d_i]
                new_tensor[d, i, :valid_len] = valid_pos
    return new_tensor