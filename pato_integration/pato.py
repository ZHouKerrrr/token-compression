"""PATO-Qwen2.5-VL Model Implementation

This module extends Qwen2.5-VL with PATO components:
- g_raw: Pixel-level conditional precompression
- Token Sort: Query-conditional token selection
- Simplified Projector: Linear projection from vision to LLM space
"""
"""
TODO:
    !!!!!!!!!! CLEAR all endpoint for testing
"""
import sys
import os
from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)
from typing import Optional, Tuple, List, Union, Dict, Any
from pathlib import Path
import math
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
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLConfig,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLAttention,
    Qwen2_5_VLVisionFlashAttention2,
    ModelOutput,
    BaseModelOutputWithPast,
    QWEN2_5_VL_ATTENTION_CLASSES,
    QWEN2_5_VL_VISION_ATTENTION_CLASSES,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
from transformers import AttentionInterface
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

from .pato_config import PATOQwen2_5_VLConfig
from .utils import *
from g_raw import get_graw_class
from token_sort import get_token_sort_class
logger = logging.get_logger(__name__)




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
                context
            )
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
        
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        query_embeddings: Optional[torch.Tensor] = None,
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
            sorted_tokens, sort_aux = self.token_sorter(
                hidden_states=hidden_states,
                lengths=lengths,
                query_embeddings=query_embeddings,
                image_grid_thw=grid_thw,
                training=self.training
            ) # [seq, dim]
            aux_outputs.update(sort_aux)

        else:
            # If sort is disabled, ensure variable name consistency
            return hidden_states
        
        
        # ============================================================
        # Step 4: Simplified Projector (NEW - PATO)
        # ============================================================

        if self.pato_config.projector.enable:
            # Use simplified projector
            sorted_tokens = self.projector(sorted_tokens)  # [B, M, hidden_dim]
        # if token sorter has reorginize the tokens
        if sorted_tokens is not None:
            return sorted_tokens, aux_outputs
        else:
            return hidden_states, aux_outputs

class PATOQwen2_5_VLRotaryEmbedding(Qwen2_5_VLRotaryEmbedding):
    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__(config, device)

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class PATOAttention(Qwen2_5_VLAttention):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
    def softmax_with_mask(self, attn, prune_mask, eps=1e-6):
        B, N = prune_mask.size()
        B, H, N, N = attn.size()
        attn_prune_mask = prune_mask.reshape(B, 1, 1, N)  # * prune_mask.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_prune_mask.dtype, device=attn_prune_mask.device).view(1, 1, N, N)
        attn_prune_mask = attn_prune_mask + (1.0 - attn_prune_mask) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        # e^z * mask / (e^z) (mask == 1)
        attn = attn.to(torch.float32).exp_() * attn_prune_mask.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)
    
    def forward(
        self, 
        hidden_states, 
        attention_mask = None, 
        position_ids = None, 
        past_key_value = None, 
        output_attentions = False, 
        use_cache = False, 
        cache_position = None, 
        position_embeddings = None,
        prune_mask = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        if prune_mask is None:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        else:
            attn_weights = self.softmax_with_mask(attn=attn_weights, prune_mask=prune_mask)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class PATOQwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        # self.qwen_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation]
        self.self_attn = PATOAttention(config, layer_idx)
        # self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        prune_mask = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            prune_mask = prune_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PATOQwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PATOQwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        aux_outputs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # TODO prune position embeding here
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            prune_mask = aux_outputs.get("llm_prune_mask", None)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    prune_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    prune_mask=prune_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2_5_VLConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

class PATOQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
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
        
        Qwen2_5_VLPreTrainedModel.__init__(self, config, *model_args,**model_kwargs)
        self.config = config
        self.pato_config = config.pato_config
        self.vision_config = self.config.vision_config
        
        # Create g_raw module if enabled
        if self.pato_config.g_raw.enable:
            g_raw_class = get_graw_class(self.pato_config.g_raw.mode)
            context = {}
            self.g_raw = g_raw_class(self.pato_config.g_raw, context)
        else:
            self.g_raw = None
        
        # Create extended visual encoder
        self.visual = PATOQwen2_5_VisionTransformer(self.config)
        # Create language model (unchanged)
        self.model = PATOQwen2_5_VLModel(self.config)
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
            mask_expanded = attention_mask.unsqueeze(-1)  # [B, seq_len, 1]
            embeds_masked = embeds * mask_expanded
        else:
            embeds_masked = embeds
        return embeds_masked  # [B, text_len, hidden_dim]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        query_input_ids = None,
        query_attention_mask = None,
        **kwargs
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """Forward pass with PATO enhancements"""
        # Store auxiliary outputs for loss computation
        aux_outputs = {}
        # Config extraction
        patch_size = self.vision_config.patch_size                      # 14
        temporal_patch_size = self.vision_config.temporal_patch_size    # 2
        spatial_merge_size = self.vision_config.spatial_merge_size      # 2
        in_chans = self.vision_config.in_chans                          # 3

        vision_token_id = self.config.vision_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id
        pad_token_id = 151643
        label_pad = -100

        # ============================================================
        # Step 1: Extract text embeddings for conditioning
        # ============================================================
        if query_input_ids is not None and query_attention_mask is not None:
            query_embeds = self.get_text_embeddings(query_input_ids, query_attention_mask)
        else:
            query_embeds = None


        # ============================================================
        # Step 2: g_raw Compression (显存优化版)
        # ============================================================
        if self.g_raw is not None and pixel_values is not None and query_embeds is not None:
            # ✅ 不要 inplace 改原来的 image_grid_thw
            # TODO：
            # 暂时不考虑g_raw
            pass

        # ============================================================
        # Step 3: Call parent forward with potentially modified inputs
        # ============================================================
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
            llm_prune_mask = None
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
                hard_prune_mask = vision_aux.get("hard_prune_mask", None)
                soft_prune_mask = vision_aux.get("soft_prune_mask", None)
                transform_matrices = vision_aux.get("transform_matrices", None)
             # endpoint for testing
            if self.pato_config.evaluate:
                return vision_aux
            
            s_idx = (input_ids == vision_start_token_id).long().argmax(dim=1) + 1
            e_idx = (input_ids == vision_end_token_id).long().argmax(dim=1)
            vision_token_nums = e_idx - s_idx
            vision_seq_lens = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // (self.vision_config.spatial_merge_size ** 2)
            assert torch.equal(vision_seq_lens, vision_token_nums), "Vision tokens in input_ids have equal to image_grids"
            B, seq_len = input_ids.shape
            device = input_ids.device

            if hard_prune_mask is not None:
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
                seq_arange = torch.arange(seq_len, device=device).unsqueeze(0)
                vision_mask = (seq_arange >= s_idx.unsqueeze(1)) & (seq_arange < e_idx.unsqueeze(1))
                max_vision_len = hard_prune_mask.shape[1]
                sorter_arange = torch.arange(max_vision_len, device=device).unsqueeze(0)    # [1, max_vision_len]
                valid_hard_prune_mask = sorter_arange < vision_token_nums.unsqueeze(1)      # [B, vision_token_seq]
                valid_sorter_values = hard_prune_mask[valid_hard_prune_mask]                # [total_vision_seq_len, 1]
                input_attn_mask[vision_mask] = valid_sorter_values
                pos_mask[:, vision_mask] = valid_sorter_values
                batch_valid_counts = input_attn_mask.sum(dim=1) # [B]
                max_seq_len = batch_valid_counts.max().item()
                input_ids = reorganize_tensor(input_ids, (B, max_seq_len), pad_token_id, batch_valid_counts, input_attn_mask)
                attention_mask = reorganize_tensor(attention_mask, (B, max_seq_len), 0, batch_valid_counts, input_attn_mask)
                position_ids = reorganize_tensor(position_ids, (3, B, max_seq_len), 0, batch_valid_counts, pos_mask)

            elif soft_prune_mask is not None:
                seq_arange = torch.arange(seq_len, device=device).unsqueeze(0)
                vision_mask = (seq_arange >= s_idx.unsqueeze(1)) & (seq_arange < e_idx.unsqueeze(1))

                max_vision_len = soft_prune_mask.shape[1]
                sorter_arange = torch.arange(max_vision_len, device=device).unsqueeze(0)  
                valid_hard_prune_mask = sorter_arange < vision_token_nums.unsqueeze(1)  # (B, vision_token_seq)
                valid_sorter_values = soft_prune_mask[valid_hard_prune_mask].squeeze(-1) # (total_vision_seq_len, 1)
                llm_prune_mask = torch.ones(attention_mask.shape, dtype=soft_prune_mask.dtype, device=attention_mask.device)

                llm_prune_mask[vision_mask] = valid_sorter_values
                aux_outputs.update({"llm_prune_mask": llm_prune_mask})
            
            elif transform_matrices is not None:
                new_pos_ids_list = []
                seq_len = position_ids.shape[-1]
                for i, trans_matrix in enumerate(transform_matrices):
                    full_trans_matrix = expand_vis_transform_to_full(trans_matrix, seq_len, s_idx[i], e_idx[i]) # [S_new, S]
                    new_position_ids = torch.matmul(full_trans_matrix, position_ids[:, i, :].to(full_trans_matrix.dtype).transpose(0, 1)).transpose(0, 1) # [3, S_new]
                    new_pos_ids_list.append(new_position_ids)
                position_ids = pad_sequence(new_pos_ids_list, batch_first=True).transpose(0, 1) # [3, B, S_new]
                
                filtered_lengths = aux_outputs["filtered_lengths"]  # [B]
                max_vision_len = max(vision_token_nums).item()

                input_attn_mask = torch.ones(
                    input_ids.shape,
                    device=input_ids.device,
                    dtype=torch.bool,
                ) # [B, S]
                prune_mask = (
                    torch.arange(max_vision_len, device=filtered_lengths.device)
                    .unsqueeze(0)                                  # [1, max_vision_len]
                    < filtered_lengths.unsqueeze(1)                # [B, 1]
                ).to(device)                                   # 或者 .to(torch.bool)

                seq_arange = torch.arange(seq_len, device=device).unsqueeze(0)
                vision_mask = (seq_arange >= s_idx.unsqueeze(1)) & (seq_arange < e_idx.unsqueeze(1))
                
                sorter_arange = torch.arange(max_vision_len, device=device).unsqueeze(0)    # [1, max_vision_len]
                valid_prune_mask = sorter_arange < vision_token_nums.unsqueeze(1)           # [B, max_vision_len]
                valid_sorter_values = prune_mask[valid_prune_mask]                          # [total_vision_seq_len, 1]
                input_attn_mask[vision_mask] = valid_sorter_values
                batch_valid_counts = input_attn_mask.sum(dim=1) # [B]
                max_seq_len = batch_valid_counts.max().item()
                input_ids = reorganize_tensor(input_ids, (B, max_seq_len), pad_token_id, batch_valid_counts, input_attn_mask)
                attention_mask = reorganize_tensor(attention_mask, (B, max_seq_len), 0, batch_valid_counts, input_attn_mask)

                assert input_ids.shape[-1] == position_ids.shape[-1], "After transformation, position_ids should align with input_ids"

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
        
        # Call language model
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            aux_outputs=aux_outputs,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
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
            loss = loss_fct(shift_logits, shift_labels)
            del shift_logits, shift_labels
            
        if return_dict:
            return PATOQwen2_5_VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
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
        dtype = self.model.dtype
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
            g_raw_class = get_graw_class(self.pato_config.g_raw.mode)
            context = {}
            self.g_raw = g_raw_class(self.pato_config.g_raw, context).to(device=device, dtype=dtype)
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
            ).to(device=device, dtype=dtype)
        else:
            self.visual.token_sorter = None
        
        # Simplified linear projection
        if self.pato_config.projector.enable:
            self.visual.projector = PATOSimplifiedProjector(
                vision_dim=self.config.vision_config.hidden_size,
                hidden_dim=self.config.pato_config.projector.hidden_size,
                dropout=self.config.pato_config.projector.dropout
            ).to(device=device, dtype=dtype)
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
    'PATOQwen2_5_VLForConditionalGeneration',
]