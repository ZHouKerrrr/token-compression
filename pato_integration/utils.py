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
        if len(tensor_size) == 2:
            mask_i = mask[i]   # [old_seq_len]
            new_tensor[i, :valid_len] = tensor[i, mask_i]
        else:
            for d in range(3):
                # 获取当前维度、当前样本的有效 mask
                pos_mask_d_i = mask[d, i] # [old_seq_len]
                new_tensor[d, i, :valid_len] = tensor[d, i][pos_mask_d_i]
    return new_tensor

def expand_vis_transform_to_full(T_vis: torch.Tensor, seq_len: int, s: int, e: int):
    """
    T_vis:  [N_new, N]
    seq_len: 原始整条序列长度 S
    s, e:   视觉 token 段 [s:e]

    return:
        T_full: [S_new, S]
    """
    device = T_vis.device
    dtype = T_vis.dtype

    N_new, N = T_vis.shape
    assert e - s == N, f"视觉段长度不匹配: e-s={e-s}, but T_vis.shape[1]={N}"

    tail_len = seq_len - e
    new_seq_len = s + N_new + tail_len

    T_full = torch.zeros(new_seq_len, seq_len, device=device, dtype=dtype)

    if s > 0:
        T_full[:s, :s] = torch.eye(s, device=device, dtype=dtype)

    T_full[s:s + N_new, s:e] = T_vis

    if tail_len > 0:
        T_full[s + N_new:, e:] = torch.eye(tail_len, device=device, dtype=dtype)

    return T_full



def print_rank0(*args, **kwargs):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(*args, **kwargs)