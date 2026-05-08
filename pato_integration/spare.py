
"""
4. SPARE
Scoring-based Pruning for Adaptive Representation Efficiency
强调“评分驱动的稀疏化/效率优化”。
如果你想要一个更偏系统优化味道的名字，这个不错。
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

from .spare_config import SPAREQwen2_5_VLConfig
from .utils import *
from token_sort import get_token_sort_class
logger = logging.get_logger(__name__)


@dataclass 
class SPAREBaseModelOutputWithPast(BaseModelOutputWithPast):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    aux_outputs: Optional[Dict] = None


@dataclass
class SPAREQwen2_5_VLCausalLMOutputWithPast(ModelOutput):
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

class SPAREAttention(Qwen2_5_VLAttention):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
    
    @staticmethod
    def softmax_with_mask(attn, prune_mask, eps=1e-6):
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
            attn_weights = SPAREAttention.softmax_with_mask(attn=attn_weights, prune_mask=prune_mask)
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

class SPAREQwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        # self.qwen_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation]
        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.train_attn = SPAREAttention # QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation]
        # self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def train_forward(
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
        hidden_states, self_attn_weights, present_key_value = self.train_attn.forward(
            self.self_attn,
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
    
    def eval_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
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
    ):
        if self.training:
            return self.train_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,  # necessary, but kept here for BC
                prune_mask,
            )
        else:
            return self.eval_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )

class SPAREQwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    _model_class = SPAREQwen2_5_VLConfig

    def __init__(self, config: SPAREQwen2_5_VLConfig):
        super().__init__(config)
        self.config = config
        self.vision_config = config.vision_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SPAREQwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        
        self.post_init()

    def _get_prune_layers(self, compressor_config):
        if compressor_config.enable:
            if getattr(compressor_config, "prune_depth"):
                prune_layers = compressor_config.prune_depth
            elif getattr(compressor_config, "prune_layers"):
                prune_layers = compressor_config.prune_layers
            elif getattr(compressor_config, "prune_depth_ratio"):
                layers = self.config.num_hidden_layers
                prune_layers = [int(ratio * layers) for ratio in compressor_config.prune_depth_ratio if ratio < 1.0]
            else:
                prune_layers = []
        return prune_layers
                
    def _set_spare_componets(self):
        compressor_config = self.config.spare_config.compressor
        if compressor_config.enable:
            self.prune_layers = self._get_prune_layers(compressor_config)
            token_sort_mode = get_token_sort_class(compressor_config.mode)
            print_rank0(f"SPARE will prune layers: {self.prune_layers}")
            print_rank0(f"SPARE compressor mode: {token_sort_mode}")
            context = {
                'hidden_size': self.config.hidden_size,
            }
            self.compressor_layers = nn.ModuleDict({
                str(layer_idx) : token_sort_mode(context={**context, 'layer_idx': layer_idx}, config=compressor_config) for layer_idx in self.prune_layers
            })
        else:
            self.compressor_layers = {}

    
    @staticmethod
    def _sequence_to_batch(
        tensor: torch.Tensor,   # [Seq, ...]
        lengths: torch.Tensor,  # [B]
    ):
        assert lengths.sum(dim=0) == tensor.size(0)
        device = tensor.device
        sequence = torch.split(tensor, lengths.tolist(), dim=0)
        batch_tensor = pad_sequence(
            sequences=sequence,
            batch_first=True,
        ).to(device)
        
        N = lengths.max().item()
        idx = torch.arange(N, device=device).unsqueeze(0)   # (1, N)
        mask = idx < lengths.unsqueeze(-1)  
        
        return batch_tensor, mask
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SPAREBaseModelOutputWithPast]:
        ## --- setting  --- ##
        aux_outputs = {}
        # Config extraction
        patch_size = self.vision_config.patch_size                      # 14
        temporal_patch_size = self.vision_config.temporal_patch_size    # 2
        spatial_merge_size = self.vision_config.spatial_merge_size      # 2
        in_chans = self.vision_config.in_chans                          # 3

        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id
        pad_token_id = 151643
        label_pad = -100


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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

        # # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        dtype = hidden_states.dtype

        # prev_mask 指导评分器如何 mask 
        assert len(input_ids.shape) == 2, "input_ids should be [B, N]"
        bs_idxs, s_idxs = torch.where(input_ids == vision_start_token_id) # Tuple (tensor1,  tensor2)
        be_idxs, e_idxs = torch.where(input_ids == vision_end_token_id) # Tuple (tensor1,  tensor2)
        assert torch.equal(bs_idxs, be_idxs), "Vision tokens should be at the same batch idxs"

        image_lengths = (e_idxs - (s_idxs + 1))
        
        origin_lengths = image_lengths.clone()
        max_image_token = image_lengths.max().item()

        prev_img_mask = torch.arange(max_image_token, device=input_ids.device).unsqueeze(0) < image_lengths.unsqueeze(1)
        prev_img_mask = prev_img_mask.to(dtype=dtype)
        prev_mask = attention_mask.clone().to(dtype=dtype)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if layer_idx in self.prune_layers:
                # Get the image_embeds and reorginize to B, N
                image_token_mask = input_ids == image_token_id  # [B, N]
                
                image_embeds = hidden_states[image_token_mask]  # [Seq, C]
                image_embeds_list = torch.split(image_embeds, image_lengths.tolist(), dim=0)
                image_hidden_states = pad_sequence(
                    sequences=image_embeds_list,
                    batch_first=True,
                ).to(input_ids.device)  # [I_num, max_len, C]
                image_lengths = image_lengths.to(image_hidden_states.device)

                aux_output = self.compressor_layers[str(layer_idx)](
                    hidden_states=image_hidden_states,
                    lengths=image_lengths,
                )

                aux_outputs.update({layer_idx :aux_output})
                
                mask = aux_output["mask"]
                prev_img_mask = prev_img_mask * mask     # Long -> BFloat 16
                valid_img_mask = torch.arange(max_image_token, device=input_ids.device).unsqueeze(0) < image_lengths.unsqueeze(1)
                valid_prev_img_mask = prev_img_mask[valid_img_mask]
                prev_mask = prev_mask.masked_scatter(image_token_mask, valid_prev_img_mask)
                
                if not self.training:
                    """When evaluate
                        prune - input_ids, attention_mask, position_ids,
                                then prev_img_mask, prev_llm_mask
                    """
                    prev_mask_bool = prev_mask.bool()
                    input_ids = input_ids[prev_mask_bool]
                    hidden_states = hidden_states[prev_mask_bool]  # [seq_len, C]
                    position_ids = position_ids[:, prev_mask_bool] # [3, seq_len]
                    
                    input_lengths = prev_mask.long().sum(dim=-1)  # [B]

                    input_ids, attention_mask = self._sequence_to_batch(
                        tensor=input_ids,
                        lengths=input_lengths,
                    )
                    hidden_states, prev_mask = self._sequence_to_batch(
                        tensor=hidden_states,
                        lengths=input_lengths,
                    )  # prev_mask is equal to attention_mask
                    prev_mask = prev_mask.to(dtype=dtype)
                    
                    position_ids = position_ids.transpose(0, 1)   # [seq_len, 3]
                    position_ids, _ = self._sequence_to_batch(    # [B, N, 3]
                        tensor=position_ids,
                        lengths=input_lengths,
                    )
                    position_ids = position_ids.permute(2, 0, 1)  # [3, B, N]
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

                    image_lengths = prev_img_mask.long().sum(dim=-1)
                    max_image_token = image_lengths.max().item()
                    prev_img_mask = torch.arange(max_image_token, device=input_ids.device).unsqueeze(0) < image_lengths.unsqueeze(1)
                    prev_img_mask = prev_img_mask.to(dtype)


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
                    prev_mask,
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
                    prune_mask=prev_mask,
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

        if self.training:
            keep_ratio = prev_img_mask.sum(dim=-1) / image_lengths.clamp(min=1.0)
            aux_outputs.update({'keep_ratio': keep_ratio})
        else:
            keep_ratio = 1.0
            for layer_idx in self.prune_layers:
                aux_output = aux_outputs.get(layer_idx)
                keep_ratio = keep_ratio * aux_output['keep_ratio']
            aux_outputs.update({'keep_ratio': keep_ratio})
            _keep_ratio = image_lengths / origin_lengths
            aux_outputs.update({'_keep_ratio': _keep_ratio})

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return SPAREBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            aux_outputs=aux_outputs,
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

class SPAREQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """SPARE-enhanced Qwen2.5-VL Model
    
    This model extends Qwen2.5-VL with:
    1. g_raw: Conditional pixel-level precompression (applied before visual encoder)
    2. Token Sort: Query-conditional token selection (applied after visual encoder)
    3. Simplified Projector: Linear projection (replaces patch merger)
    """
    
    config_class = SPAREQwen2_5_VLConfig
    base_model_prefix = "model"
    # _keys_to_ignore_on_load_missing = [
    #     r"model\.compressor_layers\..*",
    # ]
    
    def __init__(self, 
                 config: SPAREQwen2_5_VLConfig,
                *model_args,
                **model_kwargs,):
        # Initialize base model
        
        Qwen2_5_VLPreTrainedModel.__init__(self, config, *model_args,**model_kwargs)
        self.config = config
        self.spare_config = config.spare_config
        self.vision_config = self.config.vision_config
        
        # Create extended visual encoder
        self.visual = Qwen2_5_VisionTransformerPretrainedModel(self.config.vision_config)
        # Create language model (unchanged)
        self.model = SPAREQwen2_5_VLModel(self.config)
        self.rope_deltas = None
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # Apply freezing if configured
        self._apply_freezing()
        
        # Initialize weights
        self.post_init()
    
    def _apply_freezing(self):
        """Freeze specified components for training only SPARE modules"""
        if self.config.spare_config.freeze_vision_encoder:
            # Freeze original vision encoder components
            for param in self.visual.patch_embed.parameters():
                param.requires_grad = False
            for param in self.visual.blocks.parameters():
                param.requires_grad = False
            if hasattr(self.visual, 'merger'):
                for param in self.visual.merger.parameters():
                    param.requires_grad = False
        
        if self.config.spare_config.freeze_llm:
            # Freeze language model
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
        
        if self.config.spare_config.freeze_embeddings:
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
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """Forward pass with SPARE enhancements"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            return SPAREQwen2_5_VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                rope_deltas=self.rope_deltas,
                aux_outputs=outputs.aux_outputs,
            )
        else:
            return outputs, outputs.aux_outputs, loss
    
    """
        init model backbone -> load spare components -> forward
        
        support loading unprefect match between model and state dict.
    """     
    
    def init_spare_components(
            self, 
            spare_state_dict: Dict[str, Any] = None,
            spare_state_dict_path: Dict[str, Path] = None,
            freeze: bool = True,
        ):
        """Load SPARE-specific components from a state dict"""

        device = self.model.device
        dtype = self.model.dtype
        
        self.model._set_spare_componets()
        compressor_layers = self.model.compressor_layers
        
        if spare_state_dict is None:
            if spare_state_dict_path is not None:    
                spare_state_dict = torch.load(
                    spare_state_dict_path,
                    map_location=device,
                    weights_only=False,
                )
            else:
                spare_state_dict = {}

        if spare_state_dict.get("config") is not None:
            ckpt_config = spare_state_dict["config"].spare_config.compressor
            ckpt_prune_layers = self.model._get_prune_layers(ckpt_config)
            print_rank0(f"SPARE ckpt prune layers:{ckpt_prune_layers}")
        if spare_state_dict.get("compressor") is not None:
            missing_keys, unexpected_keys = compressor_layers.load_state_dict(
                spare_state_dict['compressor'],
                strict=False,    
            )
            print_rank0("missing_keys:", missing_keys)
            print_rank0("unexpected_keys:", unexpected_keys)
            for layer_idx in ckpt_prune_layers:
                layer_idx_str = str(layer_idx)
                if layer_idx_str not in compressor_layers:
                    raise ValueError(f"Compressor layer {layer_idx} is missing in the provided state dict")

                if freeze:
                    compressor_layers[layer_idx_str]._apply_freezing()
                print_rank0(f'successfully load layer {layer_idx} and freeze it')

        compressor_layers.to(device=device, dtype=dtype)


    def get_spare_components(self):
        spare_components = {}
        spare_components['config'] = self.config

        if self.spare_config.compressor.enable and self.model.compressor_layers is not None:
            spare_components['compressor'] = self.model.compressor_layers.state_dict()

        return spare_components


# Export
__all__ = [
    'SPARESimplifiedProjector',
    'SPAREQwen2_5_VisionTransformer',
    'SPAREQwen2_5_VLForConditionalGeneration',
]