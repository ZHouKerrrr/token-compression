这段代码是 Qwen2-VL 的 Vision Transformer 前向传播逻辑，结合了 PATO 的 Token Sort 机制。

代码的核心性能瓶颈在于 **Step 3 (Token Sort)** 的数据准备阶段。目前的实现使用了 Python `for` 循环手动进行 Padding 和 Mask 生成，这在 `Batch Size > 1` 且图片分辨率差异较大（导致 `max_patches` 很大）时，会导致极大的显存开销和 CPU-GPU 同步延迟。

以下是优化后的代码及详细修改说明：

### 优化后的代码

```python
from torch.nn.utils.rnn import pad_sequence

def forward(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    query_embeddings: Optional[torch.Tensor] = None,
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    """Forward pass with Optimized Token Sort Preparation
    
    Args:
        hidden_states: [total_patches, C] flattened patch embeddings
        grid_thw: [B, 3] temporal/height/width grid info
        query_embeddings: [B, D] query embeddings from text
        
    Returns:
        sorted_tokens: [B * M, hidden_dim] or [B, M, hidden_dim] depending on usage
        aux_outputs: Auxiliary outputs (if training)
    """
    # ============================================================
    # Step 1: Patch Embedding & Rotary Pos Emb
    # ============================================================
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    # ============================================================
    # Step 2: Prepare Sequence Lengths (Calculate ONCE)
    # ============================================================
    # 优化点1：cu_seqlens 计算逻辑去重，只计算一次
    batch_size = grid_thw.shape[0]
    # Calculate number of patches per image (T * H * W)
    patches_per_image = (grid_thw[:, 1] * grid_thw[:, 2])  # [B]
    # Handle Temporal dimension if it affects patch count (usually multiply by grid_thw[:, 0])
    patches_per_image = patches_per_image * grid_thw[:, 0]
    
    cu_seqlens = torch.repeat_interleave(patches_per_image, 1).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # Window logic for Qwen2-VL
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    # Apply Windowing
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # ============================================================
    # Step 3: Transformer Blocks
    # ============================================================
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

    # ============================================================
    # Step 4: Merge & Restore Order
    # ============================================================
    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    vision_tokens = hidden_states[reverse_indices, :]
    
    # ============================================================
    # Step 5: Token Sort (PATO) - VECTORIZED
    # ============================================================
    aux_outputs = {}
    
    # Check if sorting is enabled
    if self.token_sorter is not None and self.pato_config.token_sort.enable:
        # Calculate tokens per image after merger (Downsampling usually /4)
        downsample_factor = self.config.spatial_merge_size * self.config.spatial_merge_size
        tokens_per_image_merged = (patches_per_image // downsample_factor).cpu().tolist()
        
        # 优化点2：使用 pad_sequence 代替手动循环 Padding
        # Split: fast view operation
        vision_tokens_split = list(torch.split(vision_tokens, tokens_per_image_merged, dim=0))
        
        # Pad: Highly optimized C++ implementation in PyTorch
        vision_tokens_batch = pad_sequence(
            vision_tokens_split, 
            batch_first=True, 
            padding_value=0.0
        ) # [B, Max_N, D]

        # 优化点3：向量化生成 Attention Mask
        max_len = vision_tokens_batch.size(1)
        # Create mask [B, Max_N] where valid positions are 1, padding is 0
        seq_lens_tensor = torch.tensor(tokens_per_image_merged, device=vision_tokens.device).unsqueeze(1) # [B, 1]
        range_tensor = torch.arange(max_len, device=vision_tokens.device).unsqueeze(0) # [1, Max_N]
        attention_mask = (range_tensor < seq_lens_tensor).long()
        
        # Sort and Select
        sorted_tokens, sort_aux = self.token_sorter(
            hidden_states=vision_tokens_batch,
            attention_mask=attention_mask,
            budget=self.token_budget,
            query_embeddings=query_embeddings
        )
        aux_outputs.update(sort_aux)
        
        # Flatten back: [B, Budget, D] -> [B * Budget, D]
        # 注意：这里假设 sorted_tokens 已经是筛选后的结果
        sorted_tokens = sorted_tokens.reshape(-1, self.config.out_hidden_size)
    else:
        # If sort is disabled, ensure variable name consistency
        sorted_tokens = vision_tokens

    # ============================================================
    # Step 6: Projector
    # ============================================================
    if self.pato_config.projector.enable:
        sorted_tokens = self.projector(sorted_tokens)

    if self.training and aux_outputs:
        return sorted_tokens, aux_outputs
    else:
        return sorted_tokens
```

### 主要修改与优化说明

#### 1. 移除冗余的 `cu_seqlens` 计算 (性能优化)
*   **原代码**：在 Step 1 前算了一次 `cu_seqlens`，在 Step 2 Transformer 前又根据 `grid_thw` 重新算了一遍。
*   **优化后**：统一在开头计算一次，并在后续复用。这减少了 CPU-GPU 同步和不必要的计算。

#### 2. 向量化 Padding 和 Mask 生成 (核心显存/速度优化)
*   **原代码问题**：
    ```python
    for tokens in vision_tokens_list:  # Python 循环，慢！
        if num_patches < max_patches:
             torch.cat(...)            # 多次小内存分配和拷贝，极慢且产生显存碎片！
    torch.stack(...)                   # 再次全量内存拷贝
    ```
    这种写法在 Batch Size 较大或序列较长时是显存杀手。
*   **优化后**：
    ```python
    vision_tokens_batch = pad_sequence(vision_tokens_split, batch_first=True)
    ```
    使用 `torch.nn.utils.rnn.pad_sequence`，这是 PyTorch 底层 C++ 实现的高效操作，一步完成内存分配和填充。
*   **Mask 生成优化**：使用广播机制 `(range < length)` 一次性生成所有 Mask，避免了在循环中反复创建 Tensor。

#### 3. 修正潜在的 Device 问题
*   在生成 `attention_mask` 和处理 `grid_thw` 时，显式保证了新创建的 Tensor (如 `range_tensor`) 位于 `vision_tokens.device` 上，防止多卡训练时出现 "Tensor on CPU vs Tensor on CUDA:0" 的报错。

#### 4. 关于 TODO 的解答
> `TODO: 考虑merger，应该在token_sorter之前完成，有必要考虑进入LLM时，添加projector进行投影？`

*   **Merger 位置**：是的，Merger 应该在 Sorter 之前。
    *   **原因**：Merger (Patch Merging) 会将 Token 数量减少 4 倍（通常 2x2 pooling）。如果在 Merger 之前做 Sort，计算量是 Merger 之后的 4 倍。既然 PATO 的目的是筛选重要 Token，在 Merger 之后的特征更稠密、更具语义，筛选更有效率。
*   **Projector 位置**：Projector 应该放在最后。
    *   **原因**：Projector (通常是 MLP) 用来对齐 Vision 维度和 LLM 维度。如果 PATO 筛选掉了 50% 的 Token，那么先把 Token 扔掉，再对剩下的 Token 做 Projection，可以节省 50% 的 Projection 计算量。

这个优化版本应该能显著缓解你之前遇到的 OOM 问题（尤其是 Step 3 产生的中间变量内存），并且在 Batch Size > 1 时训练速度会有提升。


如果是**训练阶段**出现 79GB 的显存占用，且模型仅为 3B，这说明主要瓶颈不在模型参数（Weights），而在**激活值（Activations）**。

在训练时，PyTorch 必须保存所有中间层的输出（Feature Maps）以便在反向传播（Backward）时计算梯度。对于高分辨率图像（Qwen-VL 处理图片的分辨率通常很高）或视频数据，视觉编码器的激活值显存占用是巨大的。

以下是针对**训练阶段**的优化方案，核心在于**用计算换显存（Gradient Checkpointing）**和**及时的内存清理**。



## ------------------------------------------------------------------------------------------------------------- ##



### 核心修改点

1.  **引入梯度检查点 (Gradient Checkpointing)**：这是解决训练显存爆炸的最有效手段。通过不保存中间激活值，而在反向传播时重新计算，可以将显存占用降低 50%-70%。
2.  **及时释放显存**：在算出 `image_embeds` 后，立即删除巨大的 `pixel_values`。
3.  **减少中间变量复制**：你的代码在 `for` 循环里进行了大量的 `permute` + `contiguous`，这会产生显存副本。

### 修改后的代码

```python
import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Union, Tuple

# 假设相关的类定义已存在
# ... 

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
        """Forward pass with PATO enhancements (Training Optimized)"""
        
        aux_outputs = {}
        use_cache = False if self.training else use_cache # 训练时通常不需要 cache
        
        # ============================================================
        # Step 1: Text Embeddings
        # ============================================================
        if input_ids is not None:
            query_embeds = self.get_text_embeddings(input_ids, attention_mask)
        else:
            query_embeds = None

        # ============================================================
        # Step 2: g_raw Compression (显存优化版)
        # ============================================================
        if self.g_raw is not None and pixel_values is not None and query_embeds is not None:
            B = image_grid_thw.shape[0]
            
            # Config extraction
            patch_size = self.vision_config.patch_size
            temporal_patch_size = self.vision_config.temporal_patch_size 
            spatial_merge_size = self.vision_config.spatial_merge_size
            in_chans = self.vision_config.in_chans
            
            # 预计算所有切片位置，避免在循环中做复杂逻辑导致的显存驻留
            seq = 0
            compress_values = []
            
            for i in range(B):
                grid_t, grid_h, grid_w = image_grid_thw[i]
                
                # 校验逻辑保持不变...
                ts = self.pato_config.g_raw.target_size[0]
                # assertions...

                grid_size = grid_t * grid_h * grid_w
                seq_len = grid_size * patch_size * patch_size * temporal_patch_size * in_chans
                
                # 切片，注意：切片操作本身不复制内存，是 View
                pixel_values_i = pixel_values[seq : seq+seq_len, :]
                
                # [优化关键点] 将复杂的 reshape/permute 封装到一个函数中
                # 并在可能的情况下使用 checkpoint 来节省显存
                def process_single_image(p_val, q_emb, g_t, g_h, g_w):
                    # 1. Complex Reshape & Permute Input
                    p_val = p_val.reshape(
                        g_t, g_h // spatial_merge_size, g_w // spatial_merge_size, 
                        spatial_merge_size, spatial_merge_size, in_chans, 
                        temporal_patch_size, patch_size, patch_size                                        
                    ).permute(0, 6, 1, 3, 7, 2, 4, 8, 5).contiguous()
                    
                    p_val = p_val.reshape(
                        g_t * temporal_patch_size, in_chans,
                        g_h * patch_size, g_w * patch_size
                    )
                    
                    # 2. Expand Query
                    q_expand = q_emb.unsqueeze(0).expand(
                        g_t * temporal_patch_size, self.config.hidden_size
                    ).reshape(-1, self.config.hidden_size)

                    # 3. Apply g_raw
                    # 如果 g_raw 层很重，这里应该再次 checkpoint，但外层可能已经做
                    c_val = self.g_raw(images=p_val, text_embeddings=q_expand)
                    
                    # 4. Post-process Reshape
                    _, _, h_out, w_out = c_val.shape
                    g_h_i = h_out // patch_size
                    g_w_i = w_out // patch_size
                    
                    c_val = c_val.reshape(
                        g_t, temporal_patch_size, in_chans,
                        g_h_i // spatial_merge_size, spatial_merge_size, patch_size, 
                        g_w_i // spatial_merge_size, spatial_merge_size, patch_size,
                    ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
                    
                    c_val = c_val.reshape(g_t * g_h_i * g_w_i, -1)
                    return c_val, g_h_i, g_w_i

                # [核心优化]: 使用 Gradient Checkpointing 运行图像处理
                # 这会牺牲大约 20-30% 的速度，但极大降低显存（不保存中间 Conv/Attn 激活）
                if self.training and self.gradient_checkpointing:
                    # 注意：checkpoint 要求输入必须有 requires_grad=True 才能传递梯度
                    # 如果 pixel_values 是输入数据(False)，我们需要临时设为 True 或包装一下
                    # 这里假设 query_embeds 有梯度，可以传递
                    compress_values_i, grid_h_i, grid_w_i = torch.utils.checkpoint.checkpoint(
                        process_single_image, 
                        pixel_values_i, 
                        query_embeds[i], 
                        grid_t, grid_h, grid_w,
                        use_reentrant=False # 推荐设置为 False
                    )
                else:
                    compress_values_i, grid_h_i, grid_w_i = process_single_image(
                        pixel_values_i, query_embeds[i], grid_t, grid_h, grid_w
                    )

                # Update metadata (same as your code)
                # ... seq_len check ...
                image_grid_thw[i, 1] = grid_h_i
                image_grid_thw[i, 2] = grid_w_i
                
                compress_values.append(compress_values_i)
                seq += seq_len
                
                # [内存清理] 删除循环内的临时引用
                del pixel_values_i
            
            # 合并结果
            new_pixel_values = torch.cat(compress_values, dim=0)
            
            # [重要] 彻底删除旧的大变量
            del pixel_values 
            pixel_values = new_pixel_values
            del compress_values # 删除列表引用

        # ============================================================
        # Step 3: Vision Encoder
        # ============================================================
        
        # Position ID logic (Pre-computation)
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
        
        image_embeds = None
        
        if pixel_values is not None:
            grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
            
            # [核心优化]: 对 Vision Encoder 整体应用 Checkpointing
            # 视觉编码器通常是显存杀手 (ViT 内部有大量 Attention)
            if self.training and self.gradient_checkpointing:
                # 定义一个 wrapper 来适配 checkpoint 的参数传递
                def vision_forward_wrapper(pv, g_thw, q_emb):
                    return self.visual(pv, grid_thw=g_thw, query_embeddings=q_emb)
                
                image_embeds, vision_aux = torch.utils.checkpoint.checkpoint(
                    vision_forward_wrapper,
                    pixel_values,
                    grid_thw,
                    query_embeds,
                    use_reentrant=False
                )
            else:
                image_embeds, vision_aux = self.visual(
                    pixel_values,
                    grid_thw=grid_thw,
                    query_embeddings=query_embeds,
                )
            
            if self.training:
                aux_outputs.update(vision_aux)

            # [内存清理] 视觉特征提取完毕，立刻删除 pixel_values
            del pixel_values 

            # ============================================================
            # Token Sorting Logic (保持你的逻辑，但增加注释)
            # ============================================================
            if vision_aux is not None:
                sort_indices = vision_aux.get('sort_indices', None)
                if sort_indices is None:
                    raise ValueError("sort_indices cant be none")
                budget = self.visual.token_budget
                B = input_ids.shape[0]
                sorted_position_ids = []
                sorted_input_ids = []
                sorted_attention_mask = []
                
                # 这里的循环难以完全避免，但只处理轻量的 LongTensor，显存影响不大
                for i in range(B):
                    s_idx = int(torch.where(input_ids[i] == self.config.vision_start_token_id)[0]) + 1
                    e_idx = int(torch.where(input_ids[i] == self.config.vision_end_token_id)[0]) 
                    
                    middle_part = position_ids[:, i, s_idx:e_idx] 
                    topk_indices = sort_indices[i, :budget]
                    
                    # Gather
                    index = topk_indices.unsqueeze(0).expand(3, -1) 
                    sorted_position_ids_i = torch.gather(middle_part, dim=1, index=index)
                    
                    # Concatenate
                    position_ids_i = torch.cat([position_ids[:, i, :s_idx], sorted_position_ids_i, position_ids[:, i, e_idx:]], dim=1)
                    input_ids_i = torch.cat([input_ids[i, :s_idx], input_ids[i, s_idx:s_idx+budget], input_ids[i, e_idx:]], dim=0)
                    attention_mask_i = torch.cat([attention_mask[i, :s_idx], attention_mask[i, s_idx:s_idx+budget], attention_mask[i, e_idx:]], dim=0)
                    
                    sorted_position_ids.append(position_ids_i)
                    sorted_input_ids.append(input_ids_i)
                    sorted_attention_mask.append(attention_mask_i)
                
                # Stack
                position_ids = torch.stack(sorted_position_ids, dim=1) # Note dim=1 based on your logic
                input_ids = torch.stack(sorted_input_ids, dim=0)
                attention_mask = torch.stack(sorted_attention_mask, dim=0)

            # Embedding Fusion
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            
            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            # expand_as 不会复制内存，只是改变 stride
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            
            # cast & scatter
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            # [内存清理] 融合完成后，删除不再需要的 mask 和 image_embeds
            del image_mask, mask_expanded, image_embeds, mask_unsqueezed

        # ============================================================
        # Step 4: LLM Forward
        # ============================================================
        # 确保 LLM 本体也开启了 Gradient Checkpointing (通常在 config 里设置)
        if self.gradient_checkpointing and self.training:
             self.model.gradient_checkpointing_enable()

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
        
        # Labels processing
        loss = None
        if labels is not None:
            # Shift Logic
            # 注意：由于 input_ids 长度变了，这里需要确保 labels 已经被外部逻辑截断或对齐
            # 如果 labels 也是全量的，需要像 input_ids 一样经过 sort_indices 的筛选
            # 这里的代码假设 labels 是对应 logits 的尾部
            labels = labels[:, (labels.shape[1] - logits.shape[1]):]
            
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        if return_dict:
            return PATOQwen2_5_VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
                aux_outputs=aux_outputs,
            )
        else:
            return outputs, aux_outputs, loss
```

### 关键操作指南（必须执行）

为了让上面的代码生效，你需要在初始化模型或开始训练前，显式开启 gradient checkpointing：

```python
# 在训练脚本中
model.gradient_checkpointing_enable() 
model.train()

# 确保你的 config 中也有相关设置
# model.config.use_cache = False  # 训练时必须关闭 cache，否则 checkpointing 会报错
```

### 为什么这些修改能解决 79GB 问题？

1.  **Checkpointing 原理**：
    *   **原代码**：`g_raw` 和 `visual` 运行完后，生成的 Feature Maps (假设大小为 [Batch, Seq_Len, Hidden]) 全部驻留在显存中，直到 Loss 反向传播结束。对于 3B 模型处理视频/大图，这个中间量可能高达 40-60GB。
    *   **新代码**：运行 `checkpoint(process_single_image)` 时，只保存输入 tensor，中间所有的卷积/Attention 结果计算完就丢弃。等到反传时，再重新跑一遍 forward。**显存占用从 $O(N_{layers})$ 降到了 $O(1)$**。

2.  **`del pixel_values`**：
    *   `pixel_values` 原始 tensor 可能很大。一旦转换成 embedding 融合进 `inputs_embeds`，原始数据就是垃圾显存了。Python 的 GC 机制有时比较滞后，显式 `del` 能保证在进入最耗显存的 LLM 层之前腾出空间。

3.  **速度问题**：
    *   训练速度慢主要是因为显存换入换出（Swapping）或者 Python 循环开销。
    *   虽然 Checkpointing 会多计算一次 Forward，但因为它避免了显存溢出导致的 GPU Page Faults（严重拖慢速度），通常在显存受限场景下反而能让训练更稳定流畅。
    *   如果依然很慢，检查 `g_raw` 内部是否用了 Flash Attention。如果没有，普通的 Attention 是 $O(N^2)$ 复杂度，在长序列下是龟速且显存杀手。