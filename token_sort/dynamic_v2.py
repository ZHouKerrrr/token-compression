from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTokenSorter, register_token_sort
from typing import List, Sequence, Tuple

import torch


def apply_transform_to_vision_tokens(
    vision_tensor: torch.Tensor,   # [N_old, D]
    transform_matrix: torch.Tensor # [N_new, N_old]
) -> torch.Tensor:
    """
    对 vision hidden states 复用变换矩阵 T
    """
    return transform_matrix @ vision_tensor

def apply_transform_to_vision_position_ids(
    vision_position_ids: torch.Tensor,   # [C, N_old]，C=3(常见) 或 1
    transform_matrix: torch.Tensor,      # [N_new, N_old]
) -> torch.Tensor:
    """
    对 vision position_ids 复用同一个 T
    new_pos = old_pos @ T^T

    说明：
      - 如果是 3D multimodal rope position_ids，shape 通常是 [3, N_old]
      - 输出仍是 [3, N_new]
      - 因为 position_ids 一般是整数，这里最后 round + cast 回原 dtype
    """
    orig_dtype = vision_position_ids.dtype
    new_pos = vision_position_ids.to(torch.float32) @ transform_matrix.transpose(0, 1).to(torch.float32)
    if torch.is_floating_point(torch.empty((), dtype=orig_dtype)):
        return new_pos.to(orig_dtype)
    return new_pos.round().to(orig_dtype)

def apply_transform_to_full_llm_position_ids(
    position_ids: torch.Tensor,
    vision_spans: Sequence[Tuple[int, int]],
    transform_matrices: Sequence[torch.Tensor],
    shift_text: bool = False,
    pad_value: int = 0,
):
    """
    只改 full LLM position_ids 里的 vision 段，text 段保持不变。

    支持两种常见布局：
      1) [3, B, L]
      2) [B, 3, L]

    Args:
        position_ids:
            full LLM position_ids
        vision_spans:
            长度为 B 的列表，每个元素是 (start, end)，表示该 sample 中 vision 段在 full seq 中的位置
        transform_matrices:
            长度为 B 的列表，每个元素 T_i 形状 [N'_i, N_i]
        shift_text:
            False: text 段 position_ids 不变
            True : 将 vision 段后面的 text position_ids 整体减去 delta_len
        pad_value:
            重新 pad batch 后的填充值

    Returns:
        new_position_ids:
            和输入 layout 一致，但 seq_len 可能变为新的 max_len
        new_seq_lens:
            [B]
    """
    # 统一转成 [B, C, L]
    input_layout = "BCL"
    if position_ids.shape[0] == 3:
        # [3, B, L] -> [B, 3, L]
        pos = position_ids.permute(1, 0, 2).contiguous()
        input_layout = "CBL"
    else:
        pos = position_ids

    B, C, L = pos.shape
    if len(vision_spans) != B or len(transform_matrices) != B:
        raise ValueError("vision_spans / transform_matrices 的长度必须等于 batch size。")

    new_samples: List[torch.Tensor] = []
    new_seq_lens: List[int] = []

    for b in range(B):
        start, end = vision_spans[b]
        T = transform_matrices[b]          # [N_new, N_old]

        if not (0 <= start <= end <= L):
            raise ValueError(f"Invalid vision span for sample {b}: {(start, end)} vs seq_len={L}")

        old_vlen = end - start
        if T.size(1) != old_vlen:
            raise ValueError(
                f"Sample {b}: transform_matrix expects old vision len={T.size(1)}, "
                f"but vision span len={old_vlen}."
            )

        sample_pos = pos[b]                        # [C, L]
        before = sample_pos[:, :start]             # [C, start]
        vision_seg = sample_pos[:, start:end]      # [C, old_vlen]
        after = sample_pos[:, end:]                # [C, L-end]

        new_vision_seg = apply_transform_to_vision_position_ids(
            vision_position_ids=vision_seg,
            transform_matrix=T,
        )   # [C, new_vlen]

        if shift_text:
            delta = old_vlen - T.size(0)
            after = after - delta

        new_sample = torch.cat([before, new_vision_seg, after], dim=-1)   # [C, L_new]
        new_samples.append(new_sample)
        new_seq_lens.append(int(new_sample.size(-1)))

    max_len = max(new_seq_lens)
    new_pos = pos.new_full((B, C, max_len), pad_value)

    for b, sample in enumerate(new_samples):
        cur_len = sample.size(-1)
        new_pos[b, :, :cur_len] = sample

    new_seq_lens = torch.as_tensor(new_seq_lens, device=pos.device, dtype=torch.long)

    if input_layout == "CBL":
        new_pos = new_pos.permute(1, 0, 2).contiguous()   # [3, B, L_new]

    return new_pos, new_seq_lens

class TokenScorer(nn.Module):
    def __init__(
        self,
        query_dim,
        token_dim,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(query_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.Dropout(0.1)
        )
        self.in_conv = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
        )
        self.out_conv = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim // 4),
            nn.GELU(),
            nn.Linear(token_dim // 4, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, query_embeds):
        """
        x : vision tokens
        query_embeds : text tokens
        """
        x = self.in_conv(x)
        B, N, C = x.shape
        local_x = x[:, :, :C // 2]
        global_x = x[:, :, C // 2:].sum(dim=1, keepdim=True)
        query_embeds = self.projector(query_embeds)
        x = torch.cat([local_x, global_x.expand(B, N, C//2), query_embeds], dim=-1) # B, N, 2C
        x = self.out_conv(x)
        return x


@register_token_sort("dynamic_token_sorter_v2")
class DynamicTokenSorterV2(BaseTokenSorter):
    """
    推理阶段：
      1) 将 keep 概率分为 drop / merge / keep 三档
      2) keep tokens 原顺序保留
      3) merge tokens 聚成 K_merge 个 merged tokens，并统一追加到队尾
      4) 用一个变换矩阵 T 完成：
            new_hidden = T @ old_hidden
         后续可对 vision position_ids 复用同一个 T
    """

    def _config_value(self, name: str, default: Any) -> Any:
        if hasattr(self.config, name):
            return getattr(self.config, name)
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return default

    def _setup_module(self) -> None:
        self.token_dim = self.context.get("out_hidden_size", 2048)
        self.query_dim = self.context.get("out_hidden_size", 2048)
        self.scorer_hidden_dim = self.context.get("scorer_hidden_dim", 256)

        # 三档阈值
        self.prune_low = float(self._config_value("prune_low", 0.2))
        self.keep_high = float(self._config_value("keep_high", 0.5))

        # merge 数量策略：min(ceil(M / merge_divisor), ceil(N * max_merge_ratio))
        self.merge_divisor = int(self._config_value("merge_divisor", 10))
        self.max_merge_ratio = float(self._config_value("max_merge_ratio", 0.05))
        self.min_merge_tokens = int(self._config_value("min_merge_tokens", 8))   # merge 候选太少时可不 merge
        self.eps = float(self._config_value("eps", 1e-6))

        self.token_scorer = TokenScorer(
            query_dim=self.query_dim,
            token_dim=self.token_dim,
        )

    def _ensure_scorer_device(self, device: torch.device) -> None:
        first_param = next(self.token_scorer.parameters())
        if first_param.device != device:
            self.token_scorer.to(device)

    def _build_token_coords(
        self,
        valid_len: int,
        grid_thw: torch.Tensor,   # [3] = (T, H, W)
        device: torch.device,
    ) -> torch.Tensor:
        """
        为前 valid_len 个 vision tokens 构造 (t, h, w) 坐标。
        默认假设 hidden_states 的前 valid_len 个视觉 token 是按 T-H-W 展开的。
        """
        T, H, W = [int(v) for v in grid_thw.tolist()]
        total = T * H * W
        if total < valid_len:
            raise ValueError(
                f"image_grid_thw={grid_thw.tolist()} gives total grid size={total}, "
                f"but valid_len={valid_len}. 请确认 hidden_states 中只有 vision patch tokens，"
                f"或者先把 special tokens 剥离。"
            )

        tt, hh, ww = torch.meshgrid(
            torch.arange(T, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        coords = torch.stack([tt, hh, ww], dim=-1).reshape(-1, 3)
        return coords[:valid_len]   # [valid_len, 3]

    def _choose_grid_bins(
        self,
        grid_thw: torch.Tensor,    # [3]
        target_k: int,
    ) -> Tuple[int, int, int]:
        """
        选择 (n_t, n_h, n_w)，使得：
          n_t * n_h * n_w <= target_k
        且尽量大，保证 coarse grid 的簇数不超过 target_k。
        """
        T, H, W = [int(v) for v in grid_thw.tolist()]
        target_k = max(1, int(target_k))

        best_prod = 1
        best_bins = (1, 1, 1)

        for nt in range(1, T + 1):
            for nh in range(1, H + 1):
                max_nw = min(W, target_k // (nt * nh))
                if max_nw < 1:
                    continue
                nw = max_nw
                prod = nt * nh * nw
                if prod > best_prod:
                    best_prod = prod
                    best_bins = (nt, nh, nw)

        return best_bins

    def _build_merge_transform(
        self,
        valid_len: int,
        merge_idx: torch.Tensor,         # [M]
        merge_weight: torch.Tensor,      # [M]
        grid_thw: torch.Tensor,          # [3]
        target_k: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        构造 merge 部分的变换矩阵 T_merge: [K_merge_eff, valid_len]
        满足：
            merged_hidden = T_merge @ hidden
        """
        if merge_idx.numel() == 0 or target_k <= 0:
            empty = torch.zeros((0, valid_len), device=device, dtype=dtype)
            return empty, {
                "merge_cluster_count": 0,
                "merge_cluster_bins": (0, 0, 0),
                "merge_assign_matrix": torch.zeros((0, 0), device=device, dtype=dtype),
            }

        target_k = min(int(target_k), int(merge_idx.numel()))
        coords = self._build_token_coords(valid_len, grid_thw, device=device)   # [N, 3]
        coords_merge = coords[merge_idx]                                         # [M, 3]

        # 选 coarse bins
        n_t, n_h, n_w = self._choose_grid_bins(grid_thw, target_k)
        Tdim, Hdim, Wdim = [int(v) for v in grid_thw.tolist()]

        # 将 merge tokens 映射到 coarse grid cell
        bin_t = torch.clamp((coords_merge[:, 0] * n_t) // Tdim, max=n_t - 1)
        bin_h = torch.clamp((coords_merge[:, 1] * n_h) // Hdim, max=n_h - 1)
        bin_w = torch.clamp((coords_merge[:, 2] * n_w) // Wdim, max=n_w - 1)
        coarse_id = bin_t * (n_h * n_w) + bin_h * n_w + bin_w                    # [M]

        k_raw = n_t * n_h * n_w
        A = F.one_hot(coarse_id, num_classes=k_raw).to(dtype=dtype)              # [M, k_raw]

        # 去掉空簇
        non_empty = A.sum(dim=0) > 0
        A = A[:, non_empty]                                                       # [M, K_eff]

        # 用 merge_weight 做簇内归一化加权
        weighted = A * merge_weight.unsqueeze(1)                                  # [M, K_eff]
        denom = weighted.sum(dim=0, keepdim=True).clamp_min(self.eps)             # [1, K_eff]
        T_merge_local = (weighted / denom).transpose(0, 1)                        # [K_eff, M]

        # 扩展回对原始 valid_len 的变换矩阵
        T_merge = torch.zeros(
            (T_merge_local.size(0), valid_len),
            device=device,
            dtype=dtype,
        )
        T_merge[:, merge_idx] = T_merge_local

        return T_merge, {
            "merge_cluster_count": int(T_merge_local.size(0)),
            "merge_cluster_bins": (int(n_t), int(n_h), int(n_w)),
            "merge_assign_matrix": A,  # [M, K_eff]
        }

    def _build_keep_transform(
        self,
        valid_len: int,
        keep_idx: torch.Tensor,         # [K_keep]
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        keep 部分：one-hot 选择，保持原顺序。
        """
        k_keep = int(keep_idx.numel())
        T_keep = torch.zeros((k_keep, valid_len), device=device, dtype=dtype)
        if k_keep > 0:
            T_keep[torch.arange(k_keep, device=device), keep_idx] = 1.0
        return T_keep

    def _compute_target_merge_count(
        self,
        total_tokens: int,
        merge_tokens: int,
    ) -> int:
        """
            计算聚类数量
        默认策略：
            K_merge = min(ceil(M / merge_divisor), ceil(N * max_merge_ratio))
        """
        if merge_tokens <= 0:
            return 0
        if merge_tokens < self.min_merge_tokens:
            return 1

        cap_by_ratio = max(1, math.ceil(total_tokens * self.max_merge_ratio))
        cap_by_merge = max(1, math.ceil(merge_tokens / self.merge_divisor))
        return min(cap_by_ratio, cap_by_merge, merge_tokens)

    def forward(
        self,
        hidden_states: torch.Tensor,                 # [B, N, D] 只包含 vision tokens
        lengths: torch.Tensor,                       # [B]
        query_embeddings: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,   # [B, 3]
        training: Optional[bool] = False,
    ):
        """
        Returns:
            training=True:
                (None, aux_outputs)

            training=False:
                filtered_hidden: [sum_i N'_i, D]
                aux_outputs:
                    - filtered_lengths: [B]
                    - transform_matrices: List[Tensor], each [N'_i, N_i]
                    - keep_mask / merge_mask / drop_mask
                    - keep_prob
        """
        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        if query_embeddings is None:
            query_embeddings = torch.zeros(
                B,
                self.query_dim,
                device=device,
                dtype=dtype,
            )
        elif query_embeddings.dtype != dtype:
            query_embeddings = query_embeddings.to(dtype=dtype)

        self._ensure_scorer_device(device)

        idx = torch.arange(N, device=device).unsqueeze(0)     # [1, N]
        valid_mask = idx < lengths.unsqueeze(-1)              # [B, N]

        query_expanded = query_embeddings.unsqueeze(1).expand(B, N, D)
        log_probs = self.token_scorer(hidden_states, query_expanded)   # [B, N, 2]

        aux_outputs: Dict[str, Any] = {}

        if training:
            hard_mask = F.gumbel_softmax(log_probs, hard=True)[:, :, 0:1]   # [B, N, 1]
            hard_mask = hard_mask * valid_mask.unsqueeze(-1)
            keep_ratio = hard_mask.sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1)

            aux_outputs = {
                "soft_prune_mask": hard_mask,
                "keep_prob": keep_prob,
                "keep_ratio": keep_ratio.squeeze(-1),
            }
            return None, aux_outputs

        # ---------------------------
        # inference: 三档离散化 + 统一变换矩阵
        # ---------------------------
        if image_grid_thw is None:
            raise ValueError("training=False 时需要提供 image_grid_thw，供 merge 聚类使用。")
        keep_prob = F.softmax(log_probs, dim=-1)[..., 0]                            # [B, N]，恢复成 0~1 概率
        drop_mask = (keep_prob < self.prune_low) & valid_mask
        merge_mask = (keep_prob >= self.prune_low) & (keep_prob < self.keep_high) & valid_mask 
        keep_mask = (keep_prob >= self.keep_high) & valid_mask

        # 将 [prune_low, keep_high) 线性映射到 [0, 1]
        # merge_weight_full = ((keep_prob - self.prune_low) / (self.keep_high - self.prune_low)).clamp(0.0, 1.0)
        merge_weight_full = keep_prob * merge_mask.to(dtype)

        filtered_hidden_list: List[torch.Tensor] = []
        filtered_lengths: List[int] = []
        transform_matrices: List[torch.Tensor] = []
        merge_cluster_infos: List[Dict[str, Any]] = []

        raw_keep_ratio = keep_mask.sum(dim=1) / lengths.clamp(min=1)

        for i in range(B):
            valid_len = int(lengths[i].item())
            hidden_i = hidden_states[i, :valid_len]                 # [N_i, D]
            keep_prob_i = keep_prob[i, :valid_len]                  # [N_i]
            keep_idx = torch.nonzero(keep_mask[i, :valid_len], as_tuple=False).squeeze(-1)   # [K_keep]
            merge_idx = torch.nonzero(merge_mask[i, :valid_len], as_tuple=False).squeeze(-1) # [M]
            merge_weight_i = merge_weight_full[i, :valid_len][merge_idx]                      # [M]

            target_merge_count = self._compute_target_merge_count(
                total_tokens=valid_len,
                merge_tokens=int(merge_idx.numel()),
            )
            T_keep = self._build_keep_transform(
                valid_len=valid_len,
                keep_idx=keep_idx,
                dtype=dtype,
                device=device,
            )   # [K_keep, N_i]
            T_merge, merge_info = self._build_merge_transform(
                valid_len=valid_len,
                merge_idx=merge_idx,
                merge_weight=merge_weight_i,
                grid_thw=image_grid_thw[i],
                target_k=target_merge_count,
                dtype=dtype,
                device=device,
            )   # [K_merge_eff, N_i]
            col_has_positive = (T_merge > 0).any(dim=0)   # shape: [N]
            valid = torch.all(col_has_positive == merge_mask[i, :valid_len])
            assert valid, f"Sample {i}: merge_mask 和 merge_matrix 不匹配，可能存在 merge_mask=True 但 merge_matrix 全零的情况。请检查模型输出的 merge_mask 和 merge_matrix 是否正确。"
            # 输出顺序：keep 在前，merged 在后（按你的要求，全部追加到 vision 段尾部）
            T_i = torch.cat([T_keep, T_merge], dim=0)   # [N'_i, N_i]

            if T_i.size(0) == 0:
                top1 = int(torch.argmax(keep_prob_i).item())
                T_i = torch.zeros((1, valid_len), device=device, dtype=dtype)
                T_i[0, top1] = 1.0

            hidden_out_i = T_i @ hidden_i   # [N'_i, D]

            filtered_hidden_list.append(hidden_out_i)
            filtered_lengths.append(int(hidden_out_i.size(0)))
            transform_matrices.append(T_i)
            merge_cluster_infos.append(merge_info)

        filtered_lengths_tensor = torch.as_tensor(filtered_lengths, device=device, dtype=torch.long)
        effective_keep_ratio = filtered_lengths_tensor.float() / lengths.clamp(min=1)

        filtered_hidden = torch.cat(filtered_hidden_list, dim=0)

        aux_outputs.update({
            "filtered_lengths": filtered_lengths_tensor,     # [B]
            "transform_matrices": transform_matrices,        # List[[N'_i, N_i]]
            "keep_prob": keep_prob,                          # [B, N]
            "drop_mask": drop_mask,                          # [B, N]
            "merge_mask": merge_mask,                        # [B, N]
            "keep_mask": keep_mask,                          # [B, N]
            "merge_weight": merge_weight_full,               # [B, N]
            "raw_keep_ratio": raw_keep_ratio,                # [B]
            "effective_keep_ratio": effective_keep_ratio,    # [B]
            "merge_cluster_infos": merge_cluster_infos,      # List[dict]
        })
        return filtered_hidden, aux_outputs


__all__ = ["DynamicTokenSorterV2"]