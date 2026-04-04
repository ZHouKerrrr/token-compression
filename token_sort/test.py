import math
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_printoptions(sci_mode=False, precision=4)


def apply_transform_to_vision_tokens(vision_tensor, transform_matrix):
    return transform_matrix @ vision_tensor


def apply_transform_to_vision_position_ids(vision_position_ids, transform_matrix):
    orig_dtype = vision_position_ids.dtype
    new_pos = vision_position_ids.to(torch.float32) @ transform_matrix.transpose(0, 1).to(torch.float32)
    if torch.is_floating_point(torch.empty((), dtype=orig_dtype)):
        return new_pos.to(orig_dtype)
    return new_pos.round().to(orig_dtype)


def apply_transform_to_full_llm_position_ids(position_ids, vision_spans, transform_matrices, shift_text=False, pad_value=0):
    input_layout = "BCL"
    if position_ids.shape[0] == 3:
        pos = position_ids.permute(1, 0, 2).contiguous()
        input_layout = "CBL"
    else:
        pos = position_ids

    B, C, _ = pos.shape
    new_samples = []
    new_seq_lens = []

    for b in range(B):
        start, end = vision_spans[b]
        T = transform_matrices[b]

        before = pos[b, :, :start]
        vision_seg = pos[b, :, start:end]
        after = pos[b, :, end:]

        new_vision_seg = apply_transform_to_vision_position_ids(vision_seg, T)

        if shift_text:
            delta = (end - start) - T.size(0)
            after = after - delta

        new_sample = torch.cat([before, new_vision_seg, after], dim=-1)
        new_samples.append(new_sample)
        new_seq_lens.append(int(new_sample.size(-1)))

    max_len = max(new_seq_lens)
    new_pos = pos.new_full((B, C, max_len), pad_value)
    for b, sample in enumerate(new_samples):
        new_pos[b, :, : sample.size(-1)] = sample

    new_seq_lens = torch.as_tensor(new_seq_lens, dtype=torch.long)

    if input_layout == "CBL":
        new_pos = new_pos.permute(1, 0, 2).contiguous()

    return new_pos, new_seq_lens


class FakeTokenScorer(nn.Module):
    def __init__(self, keep_prob):
        super().__init__()
        self.register_buffer("keep_prob_template", torch.tensor(keep_prob, dtype=torch.float32))

    def forward(self, x, query_embeds):
        B, N, _ = x.shape
        p = self.keep_prob_template[:N].unsqueeze(0).expand(B, N).clamp(1e-6, 1 - 1e-6)
        return torch.stack([p.log(), (1 - p).log()], dim=-1)


class DemoDynamicTokenSorterV2(nn.Module):
    def __init__(self, config: Dict[str, Any], token_dim: int, keep_prob_for_demo: Sequence[float]):
        super().__init__()
        self.config = config
        self.context = {"out_hidden_size": token_dim, "scorer_hidden_dim": 256}
        self._setup_module()
        self.token_scorer = FakeTokenScorer(keep_prob_for_demo)

    def _config_value(self, name: str, default: Any) -> Any:
        return self.config.get(name, default)

    def _setup_module(self) -> None:
        self.token_dim = self.context["out_hidden_size"]
        self.query_dim = self.context["out_hidden_size"]
        self.prune_low = float(self._config_value("prune_low", 0.2))
        self.keep_high = float(self._config_value("keep_high", 0.6))
        self.merge_divisor = int(self._config_value("merge_divisor", 10))
        self.max_merge_ratio = float(self._config_value("max_merge_ratio", 0.05))
        self.min_merge_tokens = int(self._config_value("min_merge_tokens", 8))
        self.eps = float(self._config_value("eps", 1e-6))

    def _build_token_coords(self, valid_len: int, grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
        T, H, W = [int(v) for v in grid_thw.tolist()]
        tt, hh, ww = torch.meshgrid(
            torch.arange(T, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        coords = torch.stack([tt, hh, ww], dim=-1).reshape(-1, 3)
        return coords[:valid_len]

    def _choose_grid_bins(self, grid_thw: torch.Tensor, target_k: int) -> Tuple[int, int, int]:
        T, H, W = [int(v) for v in grid_thw.tolist()]
        best_prod, best_bins = 1, (1, 1, 1)
        for nt in range(1, T + 1):
            for nh in range(1, H + 1):
                max_nw = min(W, target_k // (nt * nh))
                if max_nw < 1:
                    continue
                prod = nt * nh * max_nw
                if prod > best_prod:
                    best_prod = prod
                    best_bins = (nt, nh, max_nw)
        return best_bins

    def _build_keep_transform(self, valid_len: int, keep_idx: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        T_keep = torch.zeros((keep_idx.numel(), valid_len), dtype=dtype, device=device)
        if keep_idx.numel() > 0:
            T_keep[torch.arange(keep_idx.numel(), device=device), keep_idx] = 1.0
        return T_keep

    def _build_merge_transform(self, valid_len: int, merge_idx: torch.Tensor, merge_weight: torch.Tensor, grid_thw: torch.Tensor, target_k: int, dtype: torch.dtype, device: torch.device):
        if merge_idx.numel() == 0 or target_k <= 0:
            return torch.zeros((0, valid_len), dtype=dtype, device=device), {"merge_cluster_count": 0, "merge_cluster_bins": (0, 0, 0)}

        target_k = min(int(target_k), int(merge_idx.numel()))
        coords = self._build_token_coords(valid_len, grid_thw, device)
        coords_merge = coords[merge_idx]

        n_t, n_h, n_w = self._choose_grid_bins(grid_thw, target_k)
        Tdim, Hdim, Wdim = [int(v) for v in grid_thw.tolist()]

        bin_t = torch.clamp((coords_merge[:, 0] * n_t) // Tdim, max=n_t - 1)
        bin_h = torch.clamp((coords_merge[:, 1] * n_h) // Hdim, max=n_h - 1)
        bin_w = torch.clamp((coords_merge[:, 2] * n_w) // Wdim, max=n_w - 1)
        coarse_id = bin_t * (n_h * n_w) + bin_h * n_w + bin_w

        A = F.one_hot(coarse_id, num_classes=n_t * n_h * n_w).to(dtype=dtype)
        A = A[:, A.sum(dim=0) > 0]
        weighted = A * merge_weight.unsqueeze(1)
        denom = weighted.sum(dim=0, keepdim=True).clamp_min(self.eps)
        T_merge_local = (weighted / denom).transpose(0, 1)

        T_merge = torch.zeros((T_merge_local.size(0), valid_len), dtype=dtype, device=device)
        T_merge[:, merge_idx] = T_merge_local
        return T_merge, {"merge_cluster_count": int(T_merge_local.size(0)), "merge_cluster_bins": (n_t, n_h, n_w)}

    def _compute_target_merge_count(self, total_tokens: int, merge_tokens: int) -> int:
        if merge_tokens <= 0:
            return 0
        if merge_tokens < self.min_merge_tokens:
            return 0
        cap_by_ratio = max(1, math.ceil(total_tokens * self.max_merge_ratio))
        cap_by_merge = max(1, math.ceil(merge_tokens / self.merge_divisor))
        return min(cap_by_ratio, cap_by_merge, merge_tokens)

    def forward(self, hidden_states, lengths, query_embeddings=None, image_grid_thw=None):
        B, N, D = hidden_states.shape
        dtype, device = hidden_states.dtype, hidden_states.device

        if query_embeddings is None:
            query_embeddings = torch.zeros(B, D, dtype=dtype, device=device)

        idx = torch.arange(N, device=device).unsqueeze(0)
        valid_mask = idx < lengths.unsqueeze(-1)

        log_probs = self.token_scorer(hidden_states, query_embeddings.unsqueeze(1).expand(B, N, D))
        keep_prob = F.softmax(log_probs, dim=-1)[..., 0]

        drop_mask = (keep_prob < self.prune_low) & valid_mask
        merge_mask = (keep_prob >= self.prune_low) & (keep_prob < self.keep_high) & valid_mask
        keep_mask = (keep_prob >= self.keep_high) & valid_mask

        merge_weight_full = ((keep_prob - self.prune_low) / (self.keep_high - self.prune_low)).clamp(0.0, 1.0)
        merge_weight_full = merge_weight_full * merge_mask.to(dtype)

        filtered_hidden_list = []
        transform_matrices = []
        filtered_lengths = []
        merge_infos = []

        for i in range(B):
            valid_len = int(lengths[i].item())
            hidden_i = hidden_states[i, :valid_len]
            keep_idx = torch.nonzero(keep_mask[i, :valid_len], as_tuple=False).squeeze(-1)
            merge_idx = torch.nonzero(merge_mask[i, :valid_len], as_tuple=False).squeeze(-1)
            merge_weight_i = merge_weight_full[i, :valid_len][merge_idx]

            target_merge_count = self._compute_target_merge_count(valid_len, int(merge_idx.numel()))
            T_keep = self._build_keep_transform(valid_len, keep_idx, dtype, device)
            T_merge, merge_info = self._build_merge_transform(valid_len, merge_idx, merge_weight_i, image_grid_thw[i], target_merge_count, dtype, device)
            T_i = torch.cat([T_keep, T_merge], dim=0)

            filtered_hidden_list.append(T_i @ hidden_i)
            transform_matrices.append(T_i)
            filtered_lengths.append(T_i.size(0))
            merge_infos.append(merge_info)

        return torch.cat(filtered_hidden_list, dim=0), {
            "filtered_lengths": torch.tensor(filtered_lengths),
            "transform_matrices": transform_matrices,
            "keep_prob": keep_prob,
            "keep_mask": keep_mask,
            "merge_mask": merge_mask,
            "drop_mask": drop_mask,
            "merge_weight": merge_weight_full,
            "merge_cluster_infos": merge_infos,
        }


def main():
    hidden_states = torch.arange(1, 1 + 8 * 4, dtype=torch.float32).reshape(1, 8, 4)
    lengths = torch.tensor([8], dtype=torch.long)
    query_embeddings = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    image_grid_thw = torch.tensor([[1, 2, 4]], dtype=torch.long)

    keep_prob_for_demo = [0.90, 0.85, 0.70, 0.55, 0.50, 0.45, 0.10, 0.05]

    config = {
        "prune_low": 0.20,
        "keep_high": 0.60,
        "merge_divisor": 2,
        "max_merge_ratio": 0.50,
        "min_merge_tokens": 2,
        "eps": 1e-6,
    }

    sorter = DemoDynamicTokenSorterV2(config=config, token_dim=4, keep_prob_for_demo=keep_prob_for_demo)
    filtered_hidden, aux_outputs = sorter(
        hidden_states=hidden_states,
        lengths=lengths,
        query_embeddings=query_embeddings,
        image_grid_thw=image_grid_thw,
    )

    T = aux_outputs["transform_matrices"][0]

    vision_position_ids = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 2, 3, 0, 1, 2, 3],
    ], dtype=torch.long)

    new_vision_tokens = apply_transform_to_vision_tokens(hidden_states[0], T)
    new_vision_position_ids = apply_transform_to_vision_position_ids(vision_position_ids, T)

    full_position_ids = torch.tensor([[
        [100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 200, 201],
        [100, 101, 0, 0, 0, 0, 1, 1, 1, 1, 200, 201],
        [100, 101, 0, 1, 2, 3, 0, 1, 2, 3, 200, 201],
    ]], dtype=torch.long)

    new_full_position_ids, new_seq_lens = apply_transform_to_full_llm_position_ids(
        position_ids=full_position_ids,
        vision_spans=[(2, 10)],
        transform_matrices=[T],
        shift_text=True,
        pad_value=-1,
    )

    print("hidden_states.shape =", tuple(hidden_states.shape))
    print("lengths =", lengths)
    print("query_embeddings.shape =", tuple(query_embeddings.shape))
    print("image_grid_thw =", image_grid_thw)
    print()

    print("keep_prob =")
    print(aux_outputs["keep_prob"])
    print()

    print("keep_mask =")
    print(aux_outputs["keep_mask"].int())
    print()

    print("merge_mask =")
    print(aux_outputs["merge_mask"].int())
    print()

    print("drop_mask =")
    print(aux_outputs["drop_mask"].int())
    print()

    print("merge_weight =")
    print(aux_outputs["merge_weight"])
    print()

    print("transform_matrix T =")
    print(T)
    print()

    print("filtered_lengths =")
    print(aux_outputs["filtered_lengths"])
    print()

    print("filtered_hidden =")
    print(filtered_hidden)
    print()

    print("new_vision_tokens =")
    print(new_vision_tokens)
    print()

    print("new_vision_position_ids =")
    print(new_vision_position_ids)
    print()

    print("new_full_position_ids =")
    print(new_full_position_ids)
    print()

    print("new_seq_lens =")
    print(new_seq_lens)


if __name__ == "__main__":
    main()
