"""PATO-Qwen2.5-VL Model Implementation

This module extends Qwen2.5-VL with PATO components:
- g_raw: Pixel-level conditional precompression
- Token Sort: Query-conditional token selection
- Simplified Projector: Linear projection from vision to LLM space
"""

import sys
import os
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss

from accelerate import init_empty_weights

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig, 
    Qwen2_5_VLConfig,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
    ModelOutput,
)
from pathlib import Path
import importlib.util
from .pato_config import PATOQwen2_5_VLConfig
from g_raw import get_graw_class
from token_sort import get_token_sort_class

def print_rank0(*args, **kwargs):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(*args, **kwargs)

@dataclass
class PATOQwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    keep_ratio: Optional[torch.FloatTensor] = None
    distortion_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    aux_outputs: Optional[Dict[str, Any]] = None

class PATOSimplifiedProjector(nn.Module):
    """Simplified Projector for PATO (Method A)
    
    Linear projection from vision encoder output to LLM hidden dimension.
    This is the simplest approach that preserves all selected tokens.
    """
    
    def __init__(self, vision_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        
        # Simple linear projection
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """Project vision tokens to LLM space
        
        Args:
            vision_tokens: [B, N, vision_dim]
            
        Returns:
            projected_tokens: [B, N, hidden_dim]
        """
        return self.projection(vision_tokens)


class PATOQwen2_5_VisionTransformer(Qwen2_5_VisionTransformerPretrainedModel):
    """Extended Vision Transformer with Token Sort
    
    This class extends the original Qwen2.5 vision transformer to insert
    Token Sort module after vision encoding.
    """
    
    def __init__(self, 
                 config : PATOQwen2_5_VLConfig = None,
                 ):
        super().__init__(config.vision_config)
        
        self.pato_config = config.pato_config
        self.config = config.vision_config
        # Create Token Sort module if enabled
        if self.pato_config.token_sort.enable:
            token_sort_class = get_token_sort_class(self.pato_config.token_sort.mode)
            context = {
                'hidden_size': config.vision_config.hidden_size,  # Vision encoder hidden size
                'out_hidden_size': config.vision_config.out_hidden_size,   
                # Vision encoder hidden size is not the final output size
            }
            self.token_sorter = token_sort_class(
                self.pato_config.token_sort, 
                context)
        else:
            self.token_sorter = None

        # Replace merger with simplified projector
        if self.pato_config.projector.enable and \
         self.pato_config.projector.mode == 'A':
            # Simplified linear projection
            self.projector = PATOSimplifiedProjector(
                vision_dim=config.vision_config.hidden_size,
                hidden_dim=self.pato_config.projector.hidden_dim,
                dropout=self.pato_config.projector.dropout
            )
        else:
            # Keep original merger for now (safely handle if merger doesn't exist)
            self.projector = getattr(self, 'merger', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        query_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass with Token Sort
        
        Args:
            hidden_states: [B*num_patches, C, pH, pW] patch embeddings
            grid_thw: [num_images, 3] temporal/height/width grid info
            query_embeddings: [B, D] query embeddings from text
            token_budget: Target number of tokens to select
            
        Returns:
            image_embeds: [B, M, hidden_dim] projected visual tokens
            aux_outputs: Auxiliary outputs (if training)
        """
        # ============================================================
        # Step 1: Patch Embedding (original Qwen2.5-VL)
        # ============================================================
        hidden_states = self.patch_embed(hidden_states)
        
        # Get rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        
        # ============================================================
        # Step 2: Vision Transformer Blocks (original Qwen2.5-VL)
        # ============================================================
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        # Run through transformer blocks
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
        
        # Output shape: [total_patches, hidden_size]
        # hidden_states = self.merger(hidden_states)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        # ============================================================
        # Step 3: Token Sort (NEW - PATO)
        # ============================================================
    
        aux_outputs = {}
        if self.token_sorter is not None and self.pato_config.token_sort.enable:
            # Calculate tokens per image after merger (Downsampling usually /4)
            B = grid_thw.shape[0]
            sequence = []
            seq_idx = 0
            for i in range(B):
                tokens_num = (grid_thw[i, 0] * grid_thw[i, 1] * grid_thw[i, 2]) // (self.spatial_merge_size ** 2)
                sequence.append(hidden_states[seq_idx : seq_idx+tokens_num])
                seq_idx = seq_idx + tokens_num
            device = hidden_states.device
            hidden_states = pad_sequence(
                sequences=sequence,
                batch_first=True,
            ).to(device) # [B, max_len, dim]
            lengths = torch.as_tensor([v.size(0) for v in hidden_states], device=device).to(hidden_states.device)
            assert B == 1, "The batch size in prune work must be 1."
            # hidden_states = hidden_states.unsqueeze(0)
            
            # Sort and Select
            sorted_tokens, sort_aux = self.token_sorter(
                hidden_states=hidden_states,
                lengths=lengths,
                query_embeddings=query_embeddings,
                training=self.training
            ) # [seq, dim]
            aux_outputs.update(sort_aux)

        else:
            # If sort is disabled, ensure variable name consistency
            return hidden_states
        del hidden_states
        
        
        # ============================================================
        # Step 4: Simplified Projector (NEW - PATO)
        # ============================================================

        if self.pato_config.projector.enable:
            # Use simplified projector
            sorted_tokens = self.projector(sorted_tokens)  # [B, M, hidden_dim]

        return sorted_tokens, aux_outputs



class PATOQwen2_5_VLModel(Qwen2_5_VLForConditionalGeneration):
    """PATO-enhanced Qwen2.5-VL Model
    
    This model extends Qwen2.5-VL with:
    1. g_raw: Conditional pixel-level precompression (applied before visual encoder)
    2. Token Sort: Query-conditional token selection (applied after visual encoder)
    3. Simplified Projector: Linear projection (replaces patch merger)
    """
    
    config_class = PATOQwen2_5_VLConfig
    base_model_prefix = "model"

    def __init__(self, 
                 config: PATOQwen2_5_VLConfig,
                *model_args,
                **model_kwargs,):
        # Initialize base model
        
        Qwen2_5_VLPreTrainedModel.__init__(self, config._base_config(), *model_args,**model_kwargs)
        self.config = config
        self.pato_config = config.pato_config
        self.base_model_config = config._base_config()
        self.vision_config = self.base_model_config.vision_config
        self.lambda_rate = self.pato_config.lambda_rate # default 0.5
        self.lambda_distortion = self.pato_config.lambda_distortion # default 0.5
        # Create g_raw module if enabled
        if self.pato_config.g_raw.enable:
            g_raw_class = get_graw_class(self.pato_config.g_raw.mode)
            context = {
            }
            self.g_raw = g_raw_class(self.pato_config.g_raw, context)
        else:
            self.g_raw = None
        
        # Create extended visual encoder
        self.visual = PATOQwen2_5_VisionTransformer(self.config)
        # Create language model (unchanged)
        self.model = Qwen2_5_VLModel(self.base_model_config)
        self.rope_deltas = None
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # Apply freezing if configured
        self._apply_freezing()
        
        # Initialize weights
        self.post_init()
    
    def _apply_freezing(self):
        """Freeze specified components for training only PATO modules"""
        if self.config.pato_config.freeze_vision_encoder:
            # Freeze original vision encoder components
            for param in self.visual.patch_embed.parameters():
                param.requires_grad = False
            for param in self.visual.blocks.parameters():
                param.requires_grad = False
            if hasattr(self.visual, 'merger'):
                for param in self.visual.merger.parameters():
                    param.requires_grad = False
        
        if self.config.pato_config.freeze_llm:
            # Freeze language model
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
        
        if self.config.pato_config.freeze_embeddings:
            # Freeze embeddings
            if hasattr(self.model, 'embed_tokens'):
                for param in self.model.embed_tokens.parameters():
                    param.requires_grad = False

    def get_text_embeddings(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract text embeddings for g_raw and token sort
        
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            
        Returns:
            text_embeds: [B, hidden_dim(model_config != vision_config)]
        """
        # Get token embeddings
        embeds = self.model.embed_tokens(input_ids)  # [B, seq_len, hidden_dim]
        
        # Apply attention-weighted pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            embeds_masked = embeds * mask_expanded
            text_embeds = embeds_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            text_embeds = embeds.mean(dim=1)
        
        return text_embeds  # [B, hidden_dim]
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """Forward pass with PATO enhancements"""

        # Store auxiliary outputs for loss computation
        aux_outputs = {}

        # Config extraction
        B = input_ids.shape[0]
        patch_size = self.vision_config.patch_size                      # 14
        temporal_patch_size = self.vision_config.temporal_patch_size    # 2
        spatial_merge_size = self.vision_config.spatial_merge_size      # 2
        in_chans = self.vision_config.in_chans                          # 3
        target_size = self.pato_config.g_raw.target_size[0]             # (224 * 224)
        vision_tokens_num = (target_size * target_size) // (patch_size * patch_size * spatial_merge_size * spatial_merge_size) # graw后的视觉token数量 64
        vision_seq_len = (target_size * target_size) // (patch_size * patch_size) # 256
        vision_token_id = self.config.vision_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id
        pad_token_id = 151643
        label_pad = -100

        # ============================================================
        # Step 1: Extract text embeddings for conditioning
        # ============================================================
        if input_ids is not None:
            query_embeds = self.get_text_embeddings(input_ids, attention_mask)
        else:
            query_embeds = None


        # ============================================================
        # Step 2: g_raw Compression (显存优化版)
        # ============================================================
        # g_raw (224, 224)
        if self.g_raw is not None and pixel_values is not None and query_embeds is not None:
            # ✅ 不要 inplace 改原来的 image_grid_thw
            # TODO：
            # 暂时不考虑g_raw
            pass

        # ============================================================
        # Step 3: Call parent forward with potentially modified inputs
        # ============================================================
        # We need to handle the visual encoding ourselves to pass query_embeds
        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            rope_deltas = None
        
        if pixel_values is not None:
            # Encode vision
            grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
            vision_aux = None

            if self.visual.token_sorter is not None:
                image_embeds, vision_aux = self.visual(
                    pixel_values,
                    grid_thw=grid_thw,
                    query_embeddings=query_embeds,
                )
                aux_outputs.update(vision_aux)
            else:
                image_embeds = self.visual(
                    pixel_values,
                    grid_thw=grid_thw,
                    query_embeddings=query_embeds,
                )

            if vision_aux is not None:
                # TODO 可以设计一个函数，减少代码复用
                sorter_mask = vision_aux.get("sorter_mask", None)
                if sorter_mask is None:
                    raise ValueError("Mask could not be none")
                # mask of position_ids and input_ids and attn_mask
                pos_mask = torch.ones(
                    position_ids.shape,
                    device=position_ids.device,
                    dtype=torch.bool,
                ) # [3, B, vt]
                input_attn_mask = torch.ones(
                    input_ids.shape,
                    device=input_ids.device,
                    dtype=torch.bool,
                ) # [B, vt]

                for i in range(B):
                    s_idx = torch.where(input_ids[i] == self.config.vision_start_token_id)[0].item() + 1
                    e_idx = torch.where(input_ids[i] == self.config.vision_end_token_id)[0].item()

                    vision_token_num = e_idx - s_idx
                    vision_seq_len = (image_grid_thw[i, 0] * image_grid_thw[i, 1] * image_grid_thw[i, 2]) // (self.vision_config.spatial_merge_size ** 2)
                    assert vision_seq_len == vision_token_num, "Vision tokens in input_ids have equal to image_grids"
                    
                    pos_mask[:, i, s_idx:e_idx] = sorter_mask[i, : vision_token_num].unsqueeze(0) # [1, 1, len]
                    input_attn_mask[i, s_idx:e_idx] = sorter_mask[i, : vision_token_num]

                aux_outputs["logits_mask"] = input_attn_mask

                # ================= 核心修改部分 Start =================
                
                # 1. 计算每个样本裁剪后剩余的有效长度
                # input_attn_mask shape: [B, old_seq_len]
                batch_valid_counts = input_attn_mask.sum(dim=1) # [B]
                max_seq_len = batch_valid_counts.max().item()

                # 2. 初始化新的 Tensor (使用 padding value 填充)
                # Input IDs 填充 pad_token_id
                def reorganize_tensor(
                    tensor,
                    tensor_size,
                    pad,
                    
                ):
                    new_tensor = torch.full(
                        tensor_size,
                        pad,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    for i in range(B):
                        valid_len = batch_valid_counts[i]
                        mask_i = input_attn_mask[i]   # [old_seq_len]
                        if len(tensor_size) == 2:
                            new_tensor[i, :valid_len] = tensor[i, mask_i]
                        else:
                            for d in range(3):
                                # 获取当前维度、当前样本的有效 mask
                                pos_mask_d_i = pos_mask[d, i] # [old_seq_len]
                                valid_pos = tensor[d, i][pos_mask_d_i]
                                new_tensor[d, i, :valid_len] = valid_pos
                    return new_tensor
                    
                # new_input_ids = torch.full(
                #     (B, max_seq_len), 
                #     pad_token_id, 
                #     dtype=input_ids.dtype, 
                #     device=input_ids.device
                # )
                # new_labels = torch.full(
                #     (B, max_seq_len), 
                #     label_pad, 
                #     dtype=labels.dtype, 
                #     device=labels.device
                # )
                # new_attention_mask = torch.zeros(
                #     (B, max_seq_len), 
                #     dtype=attention_mask.dtype, 
                #     device=attention_mask.device
                # )
                # new_position_ids = torch.zeros(
                #     (3, B, max_seq_len), 
                #     dtype=position_ids.dtype, 
                #     device=position_ids.device
                # )

                # # 3. 逐个样本(Batch)将有效数据填入新 Tensor
                # for i in range(B):
                #     valid_len = batch_valid_counts[i]
                #     mask_i = input_attn_mask[i]   # [old_seq_len]
                #     new_input_ids[i, :valid_len] = input_ids[i, mask_i]
                #     new_labels[i, :valid_len] = labels[i, mask_i]
                #     new_attention_mask[i, :valid_len] = attention_mask[i, mask_i]

                #     for d in range(3):
                #         # 获取当前维度、当前样本的有效 mask
                #         pos_mask_d_i = pos_mask[d, i] # [old_seq_len]
                #         valid_pos = position_ids[d, i][pos_mask_d_i]
                #         new_position_ids[d, i, :valid_len] = valid_pos

                # # 4. 替换原有变量
                # input_ids = new_input_ids
                # labels = new_labels
                # attention_mask = new_attention_mask
                # position_ids = new_position_ids
                input_ids = reorganize_tensor(input_ids, (B, max_seq_len), pad_token_id)
                attention_mask = reorganize_tensor(attention_mask, (B, max_seq_len), 0)
                position_ids = reorganize_tensor(position_ids, (3, B, max_seq_len), 0)
                if labels is not None:
                    labels = reorganize_tensor(labels, (B, max_seq_len), label_pad)
                    
                    # # ================= 核心修改部分 End =================
            # TODO 经过Token_sort后，token数量不等同于input_ids中的image_token数量
            
            # TODO : 根据vision_position_ids调整position_ids
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds must be provided")
                else:
                    inputs_embeds = self.model.embed_tokens(input_ids)
            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            image_embeds = None
        if self.pato_config.evaluate:
            return aux_outputs['keep_ratio']
        # Call language model
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        distortion_loss = None
        # # TODO 对于batch_size等于1的情况，labels可以直接进行简单处理，而复杂情况需要另外考虑
        if self.training:
            
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                distortion_loss = loss_fct(shift_logits, shift_labels)
                aux_outputs["distortion_loss"] = distortion_loss
                del shift_logits, shift_labels
            del hidden_states
        #
        if return_dict:
            return PATOQwen2_5_VLCausalLMOutputWithPast(
                loss=distortion_loss,
                logits=logits,
                rope_deltas=self.rope_deltas,
                aux_outputs=aux_outputs,
            )
        else:
            return outputs, aux_outputs, loss

    """
        init model backbone -> load pato components -> forward
    """
    def load_pato_components(
            self, 
            pato_state_dict: Dict[str, Any] = None,
            pato_state_dict_path: Dict[str, Path] = None):
        """Load PATO-specific components from a state dict"""
        device = self.model.device
        if pato_state_dict_path is not None:
            pato_state_dict = torch.load(
                pato_state_dict_path,
                map_location=device,
                weights_only=False,
            )
            print(type(pato_state_dict))
            print(pato_state_dict.keys())
            self.config = pato_state_dict["config"]
            
        if self.pato_config.g_raw.enable:
            g_raw_class = get_graw_class(self.config.pato_config.g_raw.mode)
            context = {}
            self.g_raw = g_raw_class(self.config.pato_config.g_raw, context).to(device)
        else:
            self.g_raw = None

        if self.pato_config.token_sort.enable:
            token_sort_class = get_token_sort_class(self.pato_config.token_sort.mode)
            context = {
                'hidden_size': self.config.vision_config.hidden_size,  # Vision encoder hidden size
                'out_hidden_size': self.config.vision_config.out_hidden_size,   
                # Vision encoder hidden size is not the final output size
            }
            self.visual.token_sorter = token_sort_class(
                self.pato_config.token_sort, 
                context
            ).to(device)
        else:
            self.visual.token_sorter = None
        
        # Simplified linear projection
        if self.pato_config.projector.enable:
            self.visual.projector = PATOSimplifiedProjector(
                vision_dim=self.config.vision_config.hidden_size,
                hidden_dim=self.config.pato_config.projector.hidden_size,
                dropout=self.config.pato_config.projector.dropout
            ).to(device)
        else:
            self.visual.projector = None
        
        if pato_state_dict is not None:
            if  pato_state_dict.get("g_raw"):
                self.g_raw.load_state_dict(pato_state_dict["g_raw"], strict=False)
            if pato_state_dict is not None and pato_state_dict["token_sorter"] is not None:
                self.visual.token_sorter.load_state_dict(pato_state_dict["token_sorter"], strict=False)
            if pato_state_dict.get("projector") is not None:
                self.visual.projector.load_state_dict(pato_state_dict["projector"], strict=False)

    def get_pato_components(self):
        pato_components = {}
        pato_components['config'] = self.config
        if self.pato_config.g_raw.enable and self.g_raw is not None:
            pato_components['g_raw'] = self.g_raw.state_dict()
        if self.pato_config.token_sort.enable and self.visual.token_sorter is not None:
            pato_components['token_sorter'] = self.visual.token_sorter.state_dict()
        if self.pato_config.projector.enable and self.visual.projector is not None:
            pato_components['projector'] = self.visual.token_sorter.state_dict()
        return pato_components
    
# Export
__all__ = [
    'PATOSimplifiedProjector',
    'PATOQwen2_5_VisionTransformer',
    'PATOQwen2_5_VLModel',
]
