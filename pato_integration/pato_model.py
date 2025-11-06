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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast,
)
from .pato_config import PATOQwen2_5_VLConfig
from g_raw import get_graw_class
from token_sort import get_token_sort_class


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
    
    def __init__(self, config, pato_config):
        super().__init__(config)
        
        self.pato_config = pato_config
        
        # Create Token Sort module if enabled
        if pato_config.token_sort.enable:
            token_sort_class = get_token_sort_class(pato_config.token_sort.mode)
            context = {
                'hidden_size': config.hidden_size,  # Vision encoder hidden size
                'device': None,  # Will be inferred dynamically in forward
            }
            self.token_sorter = token_sort_class(pato_config.token_sort, context)
        else:
            self.token_sorter = None
        
        # Replace merger with simplified projector
        if pato_config.projector.mode == 'A':
            # Simplified linear projection
            self.projector = PATOSimplifiedProjector(
                vision_dim=config.hidden_size,
                hidden_dim=pato_config.projector.hidden_dim,
                dropout=pato_config.projector.dropout
            )
        else:
            # Keep original merger for now (safely handle if merger doesn't exist)
            self.projector = getattr(self, 'merger', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        query_embeddings: Optional[torch.Tensor] = None,
        token_budget: Optional[int] = None,
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
        
        # Run through transformer blocks
        for blk_idx, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    blk,
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    use_sliding_window=blk_idx not in self.fullatt_block_indexes,
                    window_index=window_index,
                    cu_window_seqlens=cu_window_seqlens,
                    use_reentrant=False,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    use_sliding_window=blk_idx not in self.fullatt_block_indexes,
                    window_index=window_index,
                    cu_window_seqlens=cu_window_seqlens,
                )
        
        # Output shape: [total_patches, hidden_size]
        vision_tokens = hidden_states
        
        # ============================================================
        # Step 3: Token Sort (NEW - PATO)
        # ============================================================
        aux_outputs = {}
        
        if self.token_sorter is not None and self.pato_config.token_sort.enable:
            # Reshape to [B, N, D] for token sorting
            # Need to split by images based on grid_thw
            batch_size = grid_thw.shape[0]
            
            # Calculate number of patches per image
            patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
            
            # Split tokens by image
            vision_tokens_list = torch.split(vision_tokens, patches_per_image, dim=0)
            
            # Pad to max length for batching
            max_patches = max(patches_per_image)
            padded_tokens = []
            attention_masks = []
            
            for tokens in vision_tokens_list:
                num_patches = tokens.shape[0]
                if num_patches < max_patches:
                    # Pad with zeros
                    padding = torch.zeros(
                        max_patches - num_patches,
                        tokens.shape[1],
                        device=tokens.device,
                        dtype=tokens.dtype
                    )
                    padded_tokens.append(torch.cat([tokens, padding], dim=0))
                    
                    # Create attention mask
                    mask = torch.cat([
                        torch.ones(num_patches, device=tokens.device, dtype=torch.long),
                        torch.zeros(max_patches - num_patches, device=tokens.device, dtype=torch.long)
                    ])
                    attention_masks.append(mask)
                else:
                    padded_tokens.append(tokens)
                    attention_masks.append(torch.ones(num_patches, device=tokens.device, dtype=torch.long))
            
            # Stack to batch
            vision_tokens_batch = torch.stack(padded_tokens, dim=0)  # [B, max_N, D]
            attention_mask = torch.stack(attention_masks, dim=0)  # [B, max_N]
            
            # Apply token sorting
            if query_embeddings is None:
                # Use zero query embeddings if not provided
                query_embeddings = torch.zeros(
                    batch_size,
                    self.pato_config.projector.hidden_dim,
                    device=vision_tokens.device,
                    dtype=vision_tokens.dtype
                )
            
            # Random budget sampling during training
            if self.training and self.pato_config.token_sort.random_budget_training:
                if token_budget is None:
                    token_budget = torch.randint(
                        self.pato_config.token_sort.budget_min,
                        self.pato_config.token_sort.budget_max + 1,
                        (1,)
                    ).item()
            elif token_budget is None:
                # Use default budget
                budgets = self.pato_config.token_sort.budgets
                if isinstance(budgets, list):
                    token_budget = budgets[0]
                else:
                    token_budget = int(budgets)
            
            # Sort and select tokens
            sorted_tokens, sort_indices, sort_aux = self.token_sorter(
                hidden_states=vision_tokens_batch,
                attention_mask=attention_mask,
                budget=token_budget,
                query_embeddings=query_embeddings
            )
            
            # Update aux outputs
            aux_outputs.update(sort_aux)
            
            # Use sorted tokens
            vision_tokens = sorted_tokens  # [B, M, vision_dim]
        else:
            # No token sorting, reshape to [B, N, D]
            batch_size = grid_thw.shape[0]
            patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
            vision_tokens_list = torch.split(vision_tokens, patches_per_image, dim=0)
            
            # Stack (assuming same number of patches per image for now)
            # In practice, you may need padding
            vision_tokens = torch.stack([t for t in vision_tokens_list], dim=0)
        
        # ============================================================
        # Step 4: Simplified Projector (NEW - PATO)
        # ============================================================
        merger = getattr(self, 'merger', None)
        if hasattr(self, 'projector') and self.projector is not merger:
            # Use simplified projector
            image_embeds = self.projector(vision_tokens)  # [B, M, hidden_dim]
        else:
            # Use original merger
            if merger is not None:
                image_embeds = merger(vision_tokens)
            else:
                raise RuntimeError("Neither projector nor merger is available")
        
        if self.training and aux_outputs:
            return image_embeds, aux_outputs
        else:
            return image_embeds


class PATOQwen2_5_VLModel(Qwen2_5_VLModel):
    """PATO-enhanced Qwen2.5-VL Model
    
    This model extends Qwen2.5-VL with:
    1. g_raw: Conditional pixel-level precompression (applied before visual encoder)
    2. Token Sort: Query-conditional token selection (applied after visual encoder)
    3. Simplified Projector: Linear projection (replaces patch merger)
    """
    
    config_class = PATOQwen2_5_VLConfig
    
    def __init__(self, config: PATOQwen2_5_VLConfig):
        # Initialize base model
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        
        # Create g_raw module if enabled
        if config.pato_config.g_raw.enable:
            g_raw_class = get_graw_class(config.pato_config.g_raw.mode)
            context = {
                'device': None,  # Will be inferred dynamically in forward
            }
            self.g_raw = g_raw_class(config.pato_config.g_raw, context)
        else:
            self.g_raw = None
        
        # Create extended visual encoder
        self.visual = PATOQwen2_5_VisionTransformer(
            config.vision_config,
            config.pato_config
        )
        
        # Create language model (unchanged)
        from modeling_qwen2_5_vl import Qwen2_5_VLTextModel
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        
        self.rope_deltas = None
        
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
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        if self.config.pato_config.freeze_embeddings:
            # Freeze embeddings
            if hasattr(self.language_model, 'embed_tokens'):
                for param in self.language_model.embed_tokens.parameters():
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
            text_embeds: [B, hidden_dim]
        """
        # Get token embeddings
        embeds = self.language_model.embed_tokens(input_ids)  # [B, seq_len, hidden_dim]
        
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, Qwen2_5_VLModelOutputWithPast]:
        """Forward pass with PATO enhancements"""
        
        # Store auxiliary outputs for loss computation
        aux_outputs = {}
        
        # ============================================================
        # Step 1: Extract text embeddings for conditioning
        # ============================================================
        if input_ids is not None:
            query_embeds = self.get_text_embeddings(input_ids, attention_mask)
        else:
            query_embeds = None
        
        # ============================================================
        # Step 2: Apply g_raw if enabled and pixel_values provided
        # ============================================================
        if self.g_raw is not None and pixel_values is not None and query_embeds is not None:
            # Apply conditional precompression
            original_pixel_values = pixel_values.clone()
            pixel_values = self.g_raw(
                images=pixel_values,
                text_embeddings=query_embeds
            )
            
            # Store for loss computation
            if self.training:
                aux_outputs['original_pixel_values'] = original_pixel_values
                aux_outputs['compressed_pixel_values'] = pixel_values
        
        # ============================================================
        # Step 3: Call parent forward with potentially modified inputs
        # ============================================================
        # We need to handle the visual encoding ourselves to pass query_embeds
        
        if pixel_values is not None:
            # Encode vision
            grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
            
            if self.training:
                image_embeds, vision_aux = self.visual(
                    pixel_values,
                    grid_thw=grid_thw,
                    query_embeddings=query_embeds
                )
                aux_outputs.update(vision_aux)
            else:
                image_embeds = self.visual(
                    pixel_values,
                    grid_thw=grid_thw,
                    query_embeddings=query_embeds
                )
        else:
            image_embeds = None
        
        # Now we need to merge with language model
        # Get position IDs
        if position_ids is None:
            position_ids, mrope_position_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
        else:
            mrope_position_deltas = None
        
        # Merge vision and text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        
        if image_embeds is not None:
            # Insert image embeddings
            # This follows Qwen2.5-VL's logic for embedding fusion
            # For simplicity, we assume single image case
            # TODO: Handle multiple images and complex fusion
            vision_start_token_id = self.config.vision_start_token_id
            image_token_id = self.config.image_token_id
            
            # Find image token positions
            image_positions = (input_ids == image_token_id).nonzero(as_tuple=False)
            
            if image_positions.numel() > 0:
                # Simple replacement for first image
                # In practice, need more sophisticated handling
                batch_idx = image_positions[0, 0]
                seq_idx = image_positions[0, 1]
                
                # Insert image embeddings
                num_image_tokens = image_embeds.shape[1]
                
                # This is a simplified version - real implementation needs proper fusion
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[batch_idx, seq_idx:seq_idx+num_image_tokens] = image_embeds[batch_idx]
        
        # Call language model
        outputs = self.language_model(
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
        
        # Add auxiliary outputs if training
        if self.training and aux_outputs:
            if return_dict:
                outputs.aux_outputs = aux_outputs
            else:
                outputs = outputs + (aux_outputs,)
        
        return outputs


# Export
__all__ = [
    'PATOSimplifiedProjector',
    'PATOQwen2_5_VisionTransformer',
    'PATOQwen2_5_VLModel',
]
