from typing import Any, Dict, List, Optional, Tuple, Union
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
    
)

from transformers.modeling_outputs import CausalLMOutputWithPast,ModelOutput
#from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from torch.nn import LayerNorm

from flash_attn import flash_attn_with_kvcache

from model.config_yarn_new import Qwen2VLConfig,Qwen2VLVisionConfig
from model.cache import Cache, RetrievalCache

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, device: torch.device = None) -> None:
        super().__init__()
        # 如果提供 device 则直接创建在该 device 上
        device = device if device is not None else torch.device("cpu")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        # 保证 seq 在 inv_freq.device 上
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
        # 保证输入和卷积权重在同一 device 上
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
    # def __init__(self, dim: int, num_heads: int = 16) -> None:
    #     super().__init__()
    #     self.num_heads = num_heads
    #     self.head_dim = dim // num_heads
    #     self.qkv = nn.Linear(dim, dim * 3, bias=True)
    #     self.proj = nn.Linear(dim, dim)

    # def forward(
    #     self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    # ) -> torch.Tensor:
    #     seq_length = hidden_states.shape[0]
    #     qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)
    #     q, k, v = qkv.unbind(0)
    #     q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    #     k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    #     # 构造 attention mask（块数通常较少，此处循环不会成为性能瓶颈）
    #     attention_mask = torch.full(
    #         [1, seq_length, seq_length],
    #         torch.finfo(q.dtype).min,
    #         device=q.device,
    #         dtype=q.dtype,
    #     )
    #     for i in range(1, len(cu_seqlens)):
    #         start = cu_seqlens[i - 1]
    #         end = cu_seqlens[i]
    #         attention_mask[..., start:end, start:end] = 0
    #     q = q.transpose(0, 1)
    #     k = k.transpose(0, 1)
    #     v = v.transpose(0, 1)
    #     attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
    #     attn_weights = attn_weights + attention_mask
    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    #     attn_output = torch.matmul(attn_weights, v)
    #     attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
    #     attn_output = self.proj(attn_output)
    #     return attn_output
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

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
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


class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=128,
        base=10000,
        max_position_embeddings=2048,
        device: torch.device = None,
        config: Optional[Qwen2VLConfig] = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.config = config
        device = device if device is not None else torch.device("cpu")
        # 初始化频率，直接创建在指定 device 上
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # 扩展 inv_freq 到 (3, 1, dim//2, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        # 调整 position_ids 维度到 (3, bs, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

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
        self.config = config                              # 配置类
        self.layer_idx = layer_idx                        # 模型层 id
        self.hidden_size = config.hidden_size             # 隐藏层维度
        self.num_heads = config.num_attention_heads       # q 注意力头数
        self.head_dim = self.hidden_size // self.num_heads  # 每个头维度
        self.num_key_value_heads = config.num_key_value_heads  # k, v 注意力头数
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # GQA 分组数
        self.max_position_embeddings = config.max_position_embeddings 
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size} and num_heads={self.num_heads})."
            )

        # 定义 q, k, v 投影及输出投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        # 此处简化，只使用默认的 RoPE 实现
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
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        gamma_offset: int = -1,
        
    ):
        
        # 获取统一设备
        device = hidden_states.device
        bsz, q_len, _ = hidden_states.size()
        
        # 投影得到 Q, K, V，确保运算在 device 上
        query_states = self.q_proj(hidden_states).to(device=device)
        key_states = self.k_proj(hidden_states).to(device=device)
        value_states = self.v_proj(hidden_states).to(device=device)
        
        # reshape 并 transpose 为 (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # 计算 RoPE 所需的 cos, sin，注意这里 position_ids 已经预先计算好并在 device 上
        cos, sin = self.rotary_emb(value_states, position_ids)
        # 应用多模态 Rotary 位置嵌入；此函数内部需保证所有张量在 device 上
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # transpose 恢复 shape 为 (bsz, seq_len, num_heads, head_dim)
        if gamma_offset >= 0: # graph spec
            key_states, value_states = graph_cache.spec_update(new_k_cache=key_states, new_v_cache=value_states, layer_idx=self.layer_idx, gamma_offset=gamma_offset)
            
            kv_seq_len = gamma_offset + graph_cache.start_size + graph_cache.recent_size + 1

            # query_states = query_states.transpose(1, 2)
            # key_states = key_states.transpose(1, 2)
        #dropout_rate = 0.0 if not self.training else self.attention_dropout
        # 更新 kv cache 或 graph cache（保证内部所有张量在 device 上）
        # if spec:  # spec decoding
        #     new_k_states = key_states
        #     new_k_states = new_k_states.transpose(1,2)
        #     new_v_states = value_states
        #     new_v_states = new_v_states.transpose(1,2)
        #     new_k_states = repeat_kv(new_k_states,self.num_key_value_groups)
        #     new_v_states = repeat_kv(new_v_states,self.num_key_value_groups)
        #     key_states, value_states = graph_cache.update(new_k_cache=new_k_states, new_v_cache=new_v_states, layer_idx=self.layer_idx)
        else:
            kv_seq_len = key_states.shape[-3]
            kv_seq_len += kv_cache.seq_len
            key_states, value_states = kv_cache.update(key_states, value_states, layer_idx=self.layer_idx)
            # new_cache = kv_cache
            # k_states = new_cache.key_cache[self.layer_idx][:,:4096]
            # k_states = k_states.transpose(1,2)
            # v_states = new_cache.value_cache[self.layer_idx][:,:4096]
            # v_states = v_states.transpose(1,2)
            # k_states = repeat_kv(k_states,self.num_key_value_groups)
            # v_states = repeat_kv(v_states,self.num_key_value_groups)
            # if query_states.shape[1] == 1 and isinstance(graph_cache, RetrievalCache):
            #     if not graph_cache.init_graph:
                    
            #         graph_cache.init_graph_cache(k_states,v_states, query_states, self.layer_idx)
            #     else:
            #         graph_cache.update_graph_cache_retrieval(new_cache,k_states,v_states, query_states, self.layer_idx)
        # head_dim_tensor = torch.tensor(self.head_dim, device=device, dtype=torch.float16)
        # with torch.cuda.stream(torch.cuda.Stream()):
        #     softmax_scale = 1 / torch.sqrt(head_dim_tensor)
        # 调用 FlashAttention 接口，统一转换到 float16 进行加速计算
        #softmax_scale = 1 / torch.sqrt(torch.tensor(self.head_dim, device=device, dtype=torch.float16))
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # torch.save(
        #     {
        #         "hidden":hidden_states,
        #        "q":query_states,
        #        "k":key_states,
        #        "v":value_states,
        #     },"/data1/bks/liurunze/qwentest/utils/qkv_save_new.pt"
        # )
        attn_output = flash_attn_with_kvcache(
            q=query_states.to(torch.float16),
            k_cache=key_states,
            v_cache=value_states,
            softmax_scale=1/torch.sqrt(torch.tensor(self.head_dim)),
            causal=True,
        )
        # torch.save(
        #     {
        #         "attn":attn_output,
        #     },"/data1/bks/liurunze/qwentest/utils/qkv_save_new.pt"
        # )
        # 处理输出形状并转换回目标精度
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).to(torch.float16)
        #attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1).to(torch.bfloat16)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self Attention 层
        self.self_attn = Qwen2VLAttention(config=config, layer_idx=layer_idx)
        # MLP 层
        self.mlp = Qwen2MLP(config)
        # LayerNorm 层
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
        gamma_offset: int = -1,
        
    ):
        device = next(self.parameters()).device
        hidden_states = hidden_states.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)

        # 残差连接与 layer norm
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        # Self-Attention 部分
        attn_out= self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            graph_cache=graph_cache,
            storage_ids=storage_ids,
            gamma_offset=gamma_offset,
            
        )
        
        hidden_states = residual + attn_out
        # torch.save(
        #     {
        #         "hidden":present_kv,
        #     },"/data1/bks/liurunze/qwentest/utils/qkv_save_new.pt"
        # )
        # MLP 部分：先 layer norm，再 MLP，再残差连接
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        #outputs = (hidden_states,)
        # if kv_cache is not None:
        #     outputs+=(present_kv,)
        
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

    def _compute_single_sample_pos_ids(self, t: int, h: int, w: int) -> torch.Tensor:
        """
        计算单个样本的二维位置索引，并重复 t 次得到最终位置索引矩阵。
        利用 reshape 和 permute 实现对 h 和 w 方向的分块处理。
        """
        # 使用模型所在设备
        device = self.get_device()
        # 构造 h, w 索引矩阵
        h_range = torch.arange(h, device=device).unsqueeze(1).expand(h, w)
        w_range = torch.arange(w, device=device).unsqueeze(0).expand(h, w)
        # 将 h, w 重塑为 (h_merge, merge, w_merge, merge) 格式
        h_merge = h // self.spatial_merge_size
        w_merge = w // self.spatial_merge_size
        h_indices = h_range.reshape(h_merge, self.spatial_merge_size, w_merge, self.spatial_merge_size)
        w_indices = w_range.reshape(h_merge, self.spatial_merge_size, w_merge, self.spatial_merge_size)
        # 调整维度顺序，并展平每个 patch grid
        h_indices = h_indices.permute(0, 2, 1, 3).flatten()
        w_indices = w_indices.permute(0, 2, 1, 3).flatten()
        # 拼接 h 和 w 索引，得到形状 (num_tokens, 2)
        pos_ids = torch.stack([h_indices, w_indices], dim=-1)
        # 重复 t 次，代表 t 帧或 t 个视觉样本
        return pos_ids.repeat(t, 1)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        grid_thw: Tensor，每行格式为 (t, h, w)
        计算所有样本的二维位置索引，并利用 VisionRotaryEmbedding 得到最终 rotary positional embeddings。
        """
        device = grid_thw.device
        pos_ids_list = []
        # 利用 unbind 遍历每个样本（避免使用 tolist() 转换成 python list）
        for row in grid_thw.unbind(dim=0):
            # row 为形状 (3,) 的 tensor，转换为 Python 标量（仅 3 个数字，开销可忽略）
            t = int(row[0].item())
            h = int(row[1].item())
            w = int(row[2].item())
            pos_ids_sample = self._compute_single_sample_pos_ids(t, h, w)
            pos_ids_list.append(pos_ids_sample.to(device))
        # 拼接所有样本的二维位置索引，形状为 (total_tokens, 2)
        pos_ids = torch.cat(pos_ids_list, dim=0)
        # 取 grid_thw 中 h, w 的最大值作为索引范围
        max_grid_size = int(grid_thw[:, 1:].max().item())
        # 计算全量的 rotary embedding，形状例如 (max_grid_size, embedding_dim)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        # 使用 pos_ids 对 rotary_pos_emb_full 进行索引，并展平最后一维
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: 原始图像（或视频帧）输入
        grid_thw: 每个样本的 (t, h, w) 信息，决定视觉特征图的尺寸（应位于同一设备上）
        """
        device = self.get_device()
        # 确保输入都在模型所在设备上
        hidden_states = hidden_states.to(device)
        grid_thw = grid_thw.to(device)

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # 计算每个样本的累计序列长度（cu_seqlens）
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        # 依次通过所有视觉 transformer block
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
        gamma_offset: int = -1,
        
    ):
        # 统一获取模型所在设备
        device = next(self.parameters()).device
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        #input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]
        kv_cache_length = kv_cache.seq_len if kv_cache is not None else 0

        if position_ids is None:
            # 在目标设备上生成 position_ids
            position_ids = torch.arange(
                kv_cache_length, seq_length + kv_cache_length, dtype=torch.long, device=device
            ).view(1,1,-1).expand(3,batch_size,-1)
        else:
            position_ids = position_ids.to(device)

        # 计算输入 embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # 按层执行 decoder 层计算
        for decoder_layer in self.layers:
            outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                graph_cache=graph_cache,
                storage_ids=storage_ids,
                gamma_offset=gamma_offset,
                
            )
            hidden_states=outputs
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class Qwen2VLForConditionalGeneration_draft(Qwen2VLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  # 默认左侧填充
        self.rope_deltas = None  # 初始化 rope_deltas
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

    # def _compute_llm_pos_ids(self, text_length, grid_shape, spatial_merge_size, start_index):
    #     """
    #     计算单个多模态段（文本部分＋视觉部分）的 RoPE 位置
    #     :param text_length: 文本token数量
    #     :param grid_shape: (t, h, w) 视觉网格形状
    #     :param spatial_merge_size: 空间合并因子
    #     :param start_index: 本段起始索引
    #     :return: 文本位置张量和视觉网格位置张量，均为形状 (3, num_tokens)
    #     """
    #     # 文本部分：简单递增
    #     pos_text = torch.arange(text_length).view(1, -1) + start_index
    #     # 视觉部分：先根据 grid_shape 计算实际 grid 大小
    #     t, h, w = grid_shape
    #     llm_grid_h = h // spatial_merge_size
    #     llm_grid_w = w // spatial_merge_size
    #     num_grid_tokens = t * llm_grid_h * llm_grid_w
    #     # 构造三个维度的位置索引（这里仅为示例，可根据实际需求调整）
    #     t_index = torch.arange(t,device=t.device).repeat_interleave(llm_grid_h * llm_grid_w)
    #     h_index = torch.arange(llm_grid_h,device=llm_grid_h.device).repeat(t * llm_grid_w)
    #     w_index = torch.arange(llm_grid_w,device=llm_grid_w.device).repeat(t * llm_grid_h)
    #     pos_grid = torch.stack([t_index, h_index, w_index], dim=0) + (start_index + text_length)
    #     return pos_text.expand(3, -1), pos_grid

    # def _compute_llm_pos_ids(self, text_length, grid_shape, spatial_merge_size, start_index, device):
    #     """
    #     计算单个多模态段（文本部分＋视觉部分）的 RoPE 位置，不转换为 tuple，
    #     而是直接利用 tensor 的 unbind 操作提取 t, h, w（转换为 int 用于 torch.arange）。
    #     :param text_length: 文本 token 数量
    #     :param grid_shape: 形如 tensor([t, h, w])，不转换为 tuple
    #     :param spatial_merge_size: 空间合并因子
    #     :param start_index: 本段起始索引
    #     :param device: 使用的设备（例如 input_ids.device）
    #     :return: 文本位置张量和视觉网格位置张量，均为形状 (3, num_tokens)
    #     """
    #     pos_text = torch.arange(text_length, device=device).view(1, -1) + start_index
    #     # 直接利用 tensor unbind 提取各维度（0-dim tensor），再转换为 int
    #     t, h, w = grid_shape.unbind()  # 得到三个标量 tensor
    #     t = int(t.item())
    #     h = int(h.item())
    #     w = int(w.item())
    #     llm_grid_h = h // spatial_merge_size
    #     llm_grid_w = w // spatial_merge_size
    #     t_index = torch.arange(t, device=device).repeat_interleave(llm_grid_h * llm_grid_w)
    #     h_index = torch.arange(llm_grid_h, device=device).repeat(t * llm_grid_w)
    #     w_index = torch.arange(llm_grid_w, device=device).repeat(t * llm_grid_h)
    #     pos_grid = torch.stack([t_index, h_index, w_index], dim=0) + (start_index + text_length)
    #     return pos_text.expand(3, -1), pos_grid

    # def get_rope_index(self,
    #                    input_ids: torch.LongTensor,
    #                    image_grid_thw: Optional[torch.LongTensor] = None,
    #                    video_grid_thw: Optional[torch.LongTensor] = None,
    #                    attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    #     spatial_merge_size = self.config.vision_config.spatial_merge_size
    #     image_token_id = self.config.image_token_id
    #     video_token_id = self.config.video_token_id
    #     vision_start_token_id = self.config.vision_start_token_id
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
    #     mrope_position_deltas = []
    #     batch_size, seq_len = input_ids.shape
    #     position_ids = torch.ones(3, batch_size, seq_len, dtype=input_ids.dtype, device=input_ids.device)

    #     if image_grid_thw is not None or video_grid_thw is not None:
    #         for i in range(batch_size):
    #             tokens = input_ids[i]
    #             if attention_mask is not None:
    #                 tokens = tokens[attention_mask[i] == 1]
    #             token_list = tokens.tolist()
    #             llm_pos_ids_list = []
    #             st = 0
    #             image_count = token_list.count(image_token_id)
    #             video_count = token_list.count(video_token_id)
    #             remain_images, remain_videos = image_count, video_count

    #             while remain_images > 0 or remain_videos > 0:
    #                 try:
    #                     next_image = token_list.index(image_token_id, st) if remain_images > 0 else seq_len + 1
    #                 except ValueError:
    #                     next_image = seq_len + 1
    #                 try:
    #                     next_video = token_list.index(video_token_id, st) if remain_videos > 0 else seq_len + 1
    #                 except ValueError:
    #                     next_video = seq_len + 1

    #                 if next_image < next_video:
    #                     if image_grid_thw is None:
    #                         break
    #                     else:
    #                         # 直接取出 tensor，不转换为 tuple
    #                         grid_shape = image_grid_thw[image_count - remain_images]
    #                         remain_images -= 1
    #                         ed = next_image
    #                 else:
    #                     if video_grid_thw is None:
    #                         break
    #                     else:
    #                         grid_shape = video_grid_thw[video_count - remain_videos]
    #                         remain_videos -= 1
    #                         ed = next_video

    #                 text_len = ed - st
    #                 st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
    #                 # 传入 input_ids.device 保证所有张量在同一设备上
    #                 pos_text, pos_grid = self._compute_llm_pos_ids(text_len, grid_shape, spatial_merge_size, st_idx, input_ids.device)
    #                 llm_pos_ids_list.append(pos_text)
    #                 llm_pos_ids_list.append(pos_grid)
    #                 st = ed + grid_shape[0].item() * ((grid_shape[1].item() // spatial_merge_size)) * ((grid_shape[2].item() // spatial_merge_size))
    #             if st < len(token_list):
    #                 st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
    #                 remaining = len(token_list) - st
    #                 llm_pos_ids_list.append(torch.arange(remaining, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)
    #             # 在拼接前确保所有张量都在 input_ids.device 上
    #             llm_positions = torch.cat([t.to(input_ids.device) for t in llm_pos_ids_list], dim=1)
    #             position_ids[:, i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
    #             mrope_position_deltas.append(llm_positions.max().item() + 1 - len(token_list))
    #         mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    #         return position_ids, mrope_position_deltas
    #     else:
    #         # 仅文本情况
    #         if attention_mask is not None:
    #             position_ids = attention_mask.long().cumsum(-1) - 1
    #             position_ids.masked_fill_(attention_mask == 0, 1)
    #             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
    #             max_ids = position_ids.max(dim=0)[0].max(dim=1, keepdim=True)[0]
    #             mrope_position_deltas = max_ids + 1 - seq_len
    #         else:
    #             position_ids = torch.arange(seq_len, device=input_ids.device).view(1, 1, -1).expand(3, batch_size, -1)
    #             mrope_position_deltas = torch.zeros(batch_size, 1, device=input_ids.device, dtype=input_ids.dtype)
    #         return position_ids, mrope_position_deltas
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
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

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
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
                gamma_offset: int = -1,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None) -> Union[Tuple, CausalLMOutputWithPast]:

        # 若未传入 inputs_embeds，则先从 token embedding 获取
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # 封装一个辅助函数处理视觉输入替换
            def replace_tokens(token_id, pixel_vals, grid_thw):
                if pixel_vals is not None:
                    pixel_vals = pixel_vals.type(self.visual.get_dtype()).to(inputs_embeds.device)
                    visual_embeds = self.visual(pixel_vals, grid_thw=grid_thw)
                    n_tokens = (input_ids == token_id).sum().item()
                    if n_tokens != visual_embeds.shape[0]:
                        raise ValueError(f"Image/video tokens and features mismatch: tokens {n_tokens}, features {visual_embeds.shape[0]}")
                    mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    return inputs_embeds.masked_scatter(mask, visual_embeds)
                return inputs_embeds
            
            inputs_embeds = replace_tokens(self.config.image_token_id, pixel_values, image_grid_thw)
            inputs_embeds = replace_tokens(self.config.video_token_id, pixel_values_videos, video_grid_thw)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype()).to(inputs_embeds.device)
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype()).to(inputs_embeds.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 处理 position_ids 的计算
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            if kv_cache is not None:
                if (kv_cache.seq_len == 0) or (self.rope_deltas is None):
                    position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
                    self.rope_deltas = rope_deltas
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = torch.tensor(kv_cache.seq_len) + self.rope_deltas if kv_cache is not None else 0
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if kv_cache is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                        delta = delta.to(position_ids.device)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            # if draft_cache is not None:
            #     if (draft_cache.seq_len == 0) or (self.rope_deltas is None):
            #         position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
            #         self.rope_deltas = rope_deltas
            #     else:
            #         batch_size, seq_length, _ = inputs_embeds.shape
            #         delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            #         pos_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1) + delta
            #         position_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)
        #torch.save({"P_ids":position_ids},"/data1/bks/liurunze/qwentest/utils/qkv_save_new.pt")
        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            graph_cache=graph_cache,
            storage_ids=storage_ids,
            gamma_offset=gamma_offset,
        )

        hidden_states = outputs
        #torch.save({"hidden":hidden_states},"/data1/bks/liurunze/qwentest/utils/qkv_save_new.pt")
        logits = self.lm_head(hidden_states).float()
        #torch.save({"logits":logits},"/data1/bks/liurunze/qwentest/utils/qkv_save_new.pt")
        return CausalLMOutputWithPast(logits=logits)
        