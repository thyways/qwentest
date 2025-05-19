from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math
from transformers.models.qwen2_vl.modeling_qwen2_vl import(
    PreTrainedModel,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    ACT2FN,
    Qwen2VLRotaryEmbedding
    
)
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from flash_attn import flash_attn_varlen_func
from transformers.modeling_outputs import CausalLMOutputWithPast,ModelOutput

from torch.nn import LayerNorm

from transformers.modeling_flash_attention_utils import _flash_attention_forward

from model.config_qwen2_vl import Qwen2VLConfig,Qwen2VLVisionConfig
from model.cache import Cache, RetrievalCache , FlashSimpleCache

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, device: torch.device = None) -> None:
        super().__init__()
        
        device = device if device is not None else torch.device("cpu")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        target_dtype = self.proj.weight.dtype
        device = self.proj.weight.device
        hidden_states = hidden_states.to(device)
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.attn = VisionAttention(config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1,keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2VLAttention(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config                              
        self.layer_idx = layer_idx                        
        self.hidden_size = config.hidden_size             
        self.num_heads = config.num_attention_heads       
        self.head_dim = self.hidden_size // self.num_heads  
        self.num_key_value_heads = config.num_key_value_heads  
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  
        self.max_position_embeddings = config.max_position_embeddings 
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size} and num_heads={self.num_heads})."
            )
        
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        
        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        spec=False,
        use_retrieval = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if isinstance(graph_cache, RetrievalCache):
            key_states, value_states = graph_cache.init_graph_cache(key_states, value_states, self.layer_idx,)

        elif isinstance(kv_cache, FlashSimpleCache):
            key_states, value_states = kv_cache.init_kv_cache(key_states, value_states, self.layer_idx,)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled(): 
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2).to(query_states)       
        value_states = value_states.transpose(1, 2).to(query_states) 

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).to(torch.float16)
        
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2VLAttention(config=config, layer_idx=layer_idx)       
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        gamma_offset: int = 1,
        spec: bool = False,
        use_retrieval: bool = False,
    ):
        device = next(self.parameters()).device
        hidden_states = hidden_states.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out= self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            graph_cache=graph_cache,
            spec=spec,
            use_retrieval = use_retrieval,
        )       
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)

class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)   

        self.layer_list = [8, 16, 24,]
        self.vision_token_ratio_list = [1, 1, 1, 1]

        self.vision_token_posi = None
        self.prompt_len = None
        self.prompt_len_user = None
        self.vision_token_num = None

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        gamma_offset: int = 1,
        spec: bool = False,
        use_retrieval: bool = False,
    ):
        
        device = next(self.parameters()).device
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]
        kv_cache_length = kv_cache.seq_len if kv_cache is not None else 0

        if position_ids is None:
            
            position_ids = torch.arange(
                kv_cache_length, seq_length + kv_cache_length, dtype=torch.long, device=device
            ).view(1,1,-1).expand(3,batch_size,-1)
        else:
            position_ids = position_ids.to(device)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                graph_cache=graph_cache,
                storage_ids=storage_ids,
                gamma_offset=gamma_offset,
                spec=spec,
                use_retrieval = use_retrieval
            )
            hidden_states=outputs
            # if use_retrieval:
            #     rank_layer = layer_idx+1
            #     if rank_layer in self.layer_list:
            #         if hidden_states.shape[1]!=1:  
            #             stage = self.layer_list.index(rank_layer)
            #             position_ids, attention_mask, hidden_states, position_embeddings = self.rank_drop(
            #                                                                         cur_num = stage,
            #                                                                         rank_layer = rank_layer,
            #                                                                         hidden_states = hidden_states,
            #                                                                         position_ids = position_ids,
            #                                                                         attention_mask = attention_mask,
            #                                                                         graph_cache =  graph_cache,
            #                                                                         position_embeddings = position_embeddings
            #             )
            #             if self.config._attn_implementation == "flash_attention_2":
            #                 attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
    def rank_drop(
        self, cur_num, rank_layer, hidden_states ,
        position_ids, attention_mask, graph_cache, position_embeddings
    ):
        
        _position_ids = position_ids
        _attention_mask = attention_mask

        batch_size = hidden_states.shape[0]
        cur_vision_token_num = self.vision_token_num
        vision_tokens_num = int(cur_vision_token_num * self.vision_token_ratio_list[cur_num])
        next_stage_vision_tokens_num = int(cur_vision_token_num * self.vision_token_ratio_list[cur_num + 1])

        if attention_mask is None:
                attention_mask = torch.ones((batch_size,hidden_states.shape[1]), dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.bool()
        
        rank_drop_hidden_states = hidden_states.clone().detach()
        self_attn = self.layers[rank_layer].self_attn
        hidden_states = self.layers[rank_layer].input_layernorm(rank_drop_hidden_states)

        num_heads = self_attn.num_heads
        num_key_value_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim

        bsz, q_len, _ = hidden_states.size()

        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        cos, sin = self_attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, self_attn.rope_scaling["mrope_section"])

        
        for i in range(batch_size):
            vision_index = self.prompt_len_user

            cur_key_states = key_states
            cur_query_states = query_states

            prompt_total_len = self.prompt_len_user + vision_tokens_num + self.prompt_len

            text_query_states = cur_query_states[:,:,prompt_total_len-1,:].unsqueeze(2)

            cur_key_states = repeat_kv(key_states, self_attn.num_key_value_groups)

            attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(2, 3)) / math.sqrt(head_dim) 
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 

            attention_avg_head = torch.mean(attn_weights, dim=1) 
            attention_avg_head = attention_avg_head[:,:,vision_index:vision_index+vision_tokens_num] 
            attention_avg_text = torch.mean(attention_avg_head, dim=1)[0]

            top_rank_index = attention_avg_text.topk(next_stage_vision_tokens_num).indices
            top_rank_index = top_rank_index + vision_index  
            top_rank_index= top_rank_index.sort().values

            start_index = vision_index + vision_tokens_num
            rank_drop_hidden_states = torch.cat([rank_drop_hidden_states[ :, :vision_index, :] ,rank_drop_hidden_states[ :, top_rank_index, :], rank_drop_hidden_states[:, start_index:, :]], dim=1)
            attention_mask = torch.cat([attention_mask[:, :vision_index], attention_mask[:, top_rank_index], attention_mask[:, start_index:]], dim=1)
            position_ids = torch.cat([position_ids[:,:, :vision_index], position_ids[:,:,top_rank_index], position_ids[:,:,start_index:]], dim=2)
            graph_cache.key_cache[rank_layer-1] = torch.cat([key_states[:,:, :vision_index, :], key_states[:,:,top_rank_index, :], key_states[:,:,start_index:, :]], dim=2)
            graph_cache.value_cache[rank_layer-1] = torch.cat([value_states[:,:, :vision_index, :], value_states[:,:,top_rank_index, :], value_states[:,:,start_index:, :]], dim=2)
            position_embeddings = list(position_embeddings)
            for j in range(2):
                position_embeddings[j] = torch.cat([position_embeddings[j][:,:, :vision_index, :], position_embeddings[j][:,:,top_rank_index, :], position_embeddings[j][:,:,start_index:, :]], dim=2)
            position_embeddings = tuple(position_embeddings)

        return position_ids, attention_mask, rank_drop_hidden_states, position_embeddings


class Qwen2VLForConditionalGeneration_target(Qwen2VLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  
        self.rope_deltas = None  
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                self.model.prompt_len_user = ed

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                self.model.prompt_len = text_len

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            self.model.vision_token_posi = position_ids

            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


    def _update_model_kwargs_for_generation(self,
                                              outputs: ModelOutput,
                                              model_kwargs: Dict[str, Any],
                                              is_encoder_decoder: bool = False,
                                              num_new_tokens: int = 1) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas
        return model_kwargs

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Cache = None,
                graph_cache: Optional[Cache] = None,
                storage_ids: Optional[torch.LongTensor] = None,
                gamma_offset: int = 0,
                spec=False,
                use_retrieval = False,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None) -> Union[Tuple, CausalLMOutputWithPast]:

        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()

                self.model.vision_token_num = n_image_tokens
                
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()

                self.model.vision_token_num = n_video_tokens

                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2) and use_retrieval == False:
            if kv_cache is not None:
                if (kv_cache.seq_len == 0) or (self.rope_deltas is None):
                    position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
                    self.rope_deltas = rope_deltas
                else:
                    if not spec:
                        batch_size, seq_length, _ = inputs_embeds.shape
                        delta = torch.tensor(kv_cache.seq_len) + self.rope_deltas-1 if kv_cache is not None else 0
                        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                        if kv_cache is not None:  
                            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                            delta = delta.to(position_ids.device)
                        position_ids = position_ids.add(delta)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                    if spec:
                        batch_size, seq_length, _ = inputs_embeds.shape
                        
                        delta = torch.tensor(kv_cache.seq_len) + self.rope_deltas-1 if graph_cache is not None else 0
                        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                        if graph_cache is not None:  
                            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                            delta = delta.to(position_ids.device)
                        position_ids = position_ids.add(delta)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            if graph_cache is not None:
                if (graph_cache.seq_len == 0) or (self.rope_deltas is None):
                    position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
                    self.rope_deltas = rope_deltas
                else:
                    if not spec:
                        batch_size, seq_length, _ = inputs_embeds.shape
                        delta = torch.tensor(graph_cache.seq_len) + self.rope_deltas-1 if graph_cache is not None else 0
                        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                        if graph_cache is not None:  
                            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                            delta = delta.to(position_ids.device)
                        position_ids = position_ids.add(delta)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                    if spec:
                        batch_size, seq_length, _ = inputs_embeds.shape
                        
                        delta = torch.tensor(graph_cache.seq_len) + self.rope_deltas-1 if graph_cache is not None else 0
                        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                        if graph_cache is not None:  
                            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                            delta = delta.to(position_ids.device)
                        position_ids = position_ids.add(delta)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            graph_cache=graph_cache,
            storage_ids=storage_ids,
            gamma_offset=gamma_offset,
            spec=spec,
            use_retrieval = use_retrieval
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states).float()
        return CausalLMOutputWithPast(logits=logits)
        