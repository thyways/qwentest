
from typing import Any, Dict, List, Optional, Tuple
from numpy import dtype
import torch
import math
import torch.nn.functional as F
from einops import rearrange, einsum
from transformers.models.qwen2_vl.modeling_qwen2_vl import repeat_kv
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")


############## Single GPU Cache ###############
##单纯把k v cache到cpu上，然后从cpu上copy到gpu上
class FlashSimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads 
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)

        self.scores = []

    def print_status(self):
        print("[Full Cache] Cached:", self.seq_len, "| Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states

        key = self.key_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]

        return key, value

class OffloadingFlashSimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype
        self.device = model.device

        self.key_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu').pin_memory()
        self.value_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu').pin_memory()

        # init layer cache buffer on chip
        self.key_cache_buffer = torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=self.device)
        self.value_cache_buffer = torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=self.device)

        self.load_stream = torch.cuda.Stream(device=self.device)

    def print_status(self):
        print("[Offloading Flash Simple Cache] Cached Size:", self.seq_len, "| Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # copy incoming k v cache to cpu
        # preloaded_key_states = key_states.clone().to(self.key_cache.device)
        # self.key_cache[layer_idx][:, self.seq_len : self.seq_len + preloaded_key_states.shape[-3]] = preloaded_key_states
        print(f"cache:{self.key_cache.shape[-1]}")
        print(f"value:{self.seq_len + key_states.shape[-3]}")
        print(key_states.shape)
        key_states = key_states.transpose(1,2)
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states[:,:,:4,:].cpu()
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states[:,:,:4,:].cpu()

        # copy k v cache to buffer
        self.key_cache_buffer.copy_(self.key_cache[layer_idx], non_blocking=True)
        self.value_cache_buffer.copy_(self.value_cache[layer_idx], non_blocking=True)
        
        key = self.key_cache_buffer[:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache_buffer[:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]

        return key, value

class RetrievalCache(Cache):
    def __init__(self, model, max_budget=1024, prefill=1024, chunk_size=8, gamma=6) -> None:
        
        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.gamma = gamma
        self.max_budget = max_budget
        # assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        # assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"

        self.real_budget = max_budget + gamma + 1
        #self.seq_len=0
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.num_attention_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers
        assert self.num_attention_heads%self.num_heads==0, "num_query_heads must be divisible by num_kv_heads"
        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)

        self.init_graph = False

    def print_status(self):
        print("[Retrieval Cache] Budget:", self.max_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)
    
    

    # def init_graph_cache(self, k_states, v_states, query_states, layer_idx):
    #     # query_states: (bsz, 1, num_heads, head_dim)
    #     # k_states: (bsz, seq_len, num_kv_heads, head_dim)
    #     # v_states: (bsz, seq_len, num_kv_heads, head_dim)
    #     assert query_states.shape[1] == 1, "query_states should be 1 for init"

    #     # 获取参数
    #     bsz, seq_len_q, num_heads, head_dim = query_states.shape
    #     _, seq_len_k, num_kv_heads, _ = k_states.shape

    #     # 计算每组的大小
    #     group_size = num_heads // num_kv_heads

    #     # 将 Key 和 Value 分块
    #     # 假设每个块的大小为 chunk_size
    #     chunk_size = self.chunk_size
    #     chunks = seq_len_k // chunk_size  # 计算块的数量
    #     k_chunks = rearrange(k_states, "b (c s) h d -> b c s h d", s=chunk_size)  # (bsz, chunks, chunk_size, num_kv_heads, head_dim)
    #     v_chunks = rearrange(v_states, "b (c s) h d -> b c s h d", s=chunk_size)  # (bsz, chunks, chunk_size, num_kv_heads, head_dim)
    #     torch.save({"k_chunks":k_chunks},"/data1/bks/liurunze/qwen_final/utils/qkv_save.pt")
    #     # 将 Query 头分组
    #     query = rearrange(query_states, "b n (h g) d -> b g h n d", g=group_size)  # (bsz, group_size, num_kv_heads, seq_len_q, head_dim)

    #     # 将 Key 和 Value 复制到每个组
    #     key = rearrange(k_chunks, "b c s h d -> b h c s d")  # (bsz, num_kv_heads, chunks, chunk_size, head_dim)
    #     value = rearrange(v_chunks, "b c s h d -> b h c s d")  # (bsz, num_kv_heads, chunks, chunk_size, head_dim)

    #     # 计算注意力分数
    #     scores = einsum(query, key, "b g h n d, b h c s d -> b h n c s")  # (bsz, num_kv_heads, seq_len_q, chunks, chunk_size)
       
    #     # 计算每个块的注意力分数（对 chunk_size 维度取均值）
    #     chunk_attn = scores.mean(dim=-1)  # (bsz, num_kv_heads, seq_len_q, chunks)
    #     chunk_attn = chunk_attn.mean(dim=2)  # (bsz, seq_len_q, chunks)

    #     # 确保 select_sets 不超过 chunks 的数量
    #     select_sets = min(self.select_sets, chunks)
    #     _, topk_idx = torch.topk(chunk_attn, k=select_sets, dim=-1)  # (bsz, seq_len_q, select_sets)

    #     # 扩展索引以匹配块形状
    #     expanded_index_tensor = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
    #         -1, -1, -1, chunk_size, head_dim
    #     ).permute(0, 2, 3, 1, 4)  # 调整维度顺序以匹配 k_chunks 的形状

    #     # 使用索引选择块
    #     selected_key = torch.gather(k_chunks, 1, expanded_index_tensor)  # 从 k_chunks 中选 Top-K 块
    #     selected_value = torch.gather(v_chunks, 1, expanded_index_tensor)  # 从 v_chunks 中选 Top-K 块
    #     torch.save({"select":select_sets},"/data1/bks/liurunze/qwen_final/utils/qkv_save_new.pt")
    #     # 更新缓存
    #     self.key_cache[layer_idx][:, :select_sets*chunk_size] = selected_key.reshape(
    #         1, select_sets*chunk_size, self.num_heads, self.head_dim
    #     ).clone()
    #     self.value_cache[layer_idx][:, :select_sets*chunk_size] = selected_value.reshape(
    #         1, select_sets*chunk_size, self.num_heads, self.head_dim
    #     ).clone()

    #     if layer_idx == self.layers - 1:
    #         self.init_graph = True
    def init_graph_cache(self, k_states, v_states, query_states, layer_idx):
        # query_states: (bsz, 1, num_heads, head_dim)
        # k_states: (bsz, seq_len, num_kv_heads, head_dim)
        # v_states: (bsz, seq_len, num_kv_heads, head_dim)
        assert query_states.shape[1] == 1, "query_states should be 1 for init"
        if query_states.shape[2]>k_states.shape[2]:
            key = k_states.repeat_interleave(query_states.size(-2)//k_states.size(-2), -2)
            value = v_states.repeat_interleave(query_states.size(-2)//v_states.size(-2), -2)
        #attn_weight = F.scaled_dot_product_attention(query=query_states,key=k_states,value=v_states,enable_gqa=True)
        query_states = query_states.permute(0,2,1,3)
        key = key.permute(0,2,3,1)
        scores = torch.matmul(query_states,key).squeeze(2)
        scores = scores.mean(dim=1)
        _,topk = torch.topk(scores,k=self.max_budget,dim=-1)
        expanded_index_tensor = topk.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1,self.num_heads, self.head_dim
        )  # 调整维度顺序以匹配 key_chunks 的形状
        # 计算 Top-K 块的索引
        # 使用索引选择块
        selected_key = torch.gather(k_states, 1, expanded_index_tensor)  # 从 k_states 中选 Top-K 块
        selected_value = torch.gather(v_states, 1, expanded_index_tensor)  # 从 v_states 中选 Top-K 块

        # 更新缓存
        self.key_cache[layer_idx][:, :self.max_budget] = selected_key
        
        self.value_cache[layer_idx][:, :self.max_budget] = selected_value

        if layer_idx == self.layers - 1:
            self.init_graph = True
    #def init_graph_cache(self, k_states,v_states, query_states, layer_idx):
        
        # query_states: (bsz, 1, 12, head_dim) --> (bsz, 12, 1, head_dim)
        # key_cache: (bsz, seq_len, 12, head_dim) --> (bsz, 12, head_dim, seq_len)
        # print(query_states.shape, self.chunk_k[layer_idx].shape)
        #assert 1 == query_states.shape[1], "query_states should be 1 for init"
        #group_size = self.num_attention_heads // self.num_heads
        #new_k_states = kv_cache.key_cache[layer_idx][:,:self.prefill].unsqueeze(2).expand(-1,-1,group_size,-1,-1)
        # new_k_states = k_states[:,:self.prefill].clone()
        # chunk_k = new_k_states.cuda().view(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        # key_chunks = k_states.reshape(1,self.chunks,self.chunk_size,self.num_heads,self.head_dim)
        # value_chunks = v_states[:,:self.prefill].reshape(1,self.chunks,self.chunk_size,self.num_heads,self.head_dim)
        # query = query_states.permute(0,2,1,3)
        # key = key_chunks.permute(0,1,3,4,2)
        # attn_scores = torch.matmul(query,key)
        # attn_scores = attn_scores.squeeze(3)
        # chunk_attn = attn_scores.mean(dim=-1)
        # chunk_attn = chunk_attn.permute(0,2,1)
        # _,topk_idx = torch.topk(chunk_attn,k=self.select_sets,dim=-1)
        # topk_idx = topk_idx.permute(0,2,1)
        # expanded_index_tensor = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,self.chunk_size,self.head_dim).permute(0,1,3,2,4)
        # selected_key = torch.gather(key_chunks,1,expanded_index_tensor)
        # selected_value = torch.gather(value_chunks, 1, expanded_index_tensor)
        # self.key_cache[layer_idx][:,:self.select_sets*self.chunk_size]=selected_key.reshape(1,self.select_sets*self.chunk_size,self.num_heads,self.head_dim).clone()
        # self.value_cache[layer_idx][:,:self.select_sets*self.chunk_size]=selected_value.reshape(1,self.select_sets*self.chunk_size,self.num_heads,self.head_dim).clone()
        # if layer_idx == self.layers - 1:
        #     self.init_graph = True
        # if chunk_k.device != query_states.device:
        #     chunk_k = chunk_k.to(query_states.device)
        # # (bsz, 32, chunks)
        # chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2)
        # # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        # _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1)
        # topk_idx_rest += 1
        # topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        # topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)
        # expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        # key_ = k_states[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        # key_ = key_.permute(0, 1, 3, 2, 4)
        # #torch.save({"chunk_k":key_},"/data1/bks/liurunze/qwen_final/utils/qkv_save.pt")
        # if key_.device != expanded_index_tensor.device:
        #     key_ = key_.to(expanded_index_tensor.device)
        # result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # #torch.save({"topk":result_tensor},"/data1/bks/liurunze/qwen_final/utils/qkv_save_new.pt")
        # # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        # self.key_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()
        
        # value_ = v_states[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        # value_ = value_.permute(0, 1, 3, 2, 4)
        # if value_.device != expanded_index_tensor.device:
        #     value_ = value_.to(expanded_index_tensor.device)
        # result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        # self.value_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        # if layer_idx == self.layers-1:
        #     self.init_graph = True

    def update_graph_cache(self, kv_cache=None):
        self.value_cache[:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.value_cache[:,:, self.prefill:kv_cache.seq_len].clone()
        self.key_cache[:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.key_cache[:,:, self.prefill:kv_cache.seq_len].clone()

    def update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int):
        
        # new_k_cache = new_k_cache.transpose(1,2)
        # new_v_cache = new_v_cache.transpose(1,2)
        # self.key_cache[layer_idx][:, self.real_budget-self.gamma-1:self.real_budget-self.gamma] = new_k_cache.clone()
        # self.value_cache[layer_idx][:, self.real_budget-self.gamma-1:self.real_budget-self.gamma] = new_v_cache.clone()
        self.key_cache[layer_idx][:, self.real_budget-self.gamma-1:] = new_k_cache.clone()
        self.value_cache[layer_idx][:, self.real_budget-self.gamma-1:] = new_v_cache.clone()
        # if layer_idx==self.layers-1:
        #     self.gamma-=1
        # if(self.gamma==0):
        #     self.gamma=6
        return self.key_cache[layer_idx][:,:self.real_budget], self.value_cache[layer_idx][:,:self.real_budget]

    def update_graph_cache_retrieval(self, kv_cache,k_states,v_states,query_states, layer_idx):
        self.init_graph_cache(k_states,v_states, query_states, layer_idx)
        self.value_cache[layer_idx,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.value_cache[layer_idx,:, self.prefill:kv_cache.seq_len].clone()
        self.key_cache[layer_idx,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.key_cache[layer_idx,:, self.prefill:kv_cache.seq_len].clone()

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
class StreamingLLMEvictionCache(Cache):

    def __init__(self, model, gamma=6, start_size=16, recent_size=496) -> None:

        self.gamma = gamma
        self.start_size = start_size
        self.recent_size = recent_size
        self.real_budget = self.start_size + self.recent_size + self.gamma + 1 + 1 + 1

        self.seq_len = 0 # just for prefill usage

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        self.key_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
    
    def print_status(self):
        print("[StreamingLLM Cache] Start Size:", self.start_size, "| Recent Size:", self.recent_size, "| Gamma:", self.gamma, "| Real Budget:", self.real_budget, "| Cached:", self.seq_len)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        
        incoming = key_states.shape[-3]

        assert self.seq_len + incoming <= self.start_size + self.recent_size
        self.key_cache[layer_idx][:, self.seq_len:self.seq_len + incoming] = key_states.clone()
        self.value_cache[layer_idx][:, self.seq_len:self.seq_len + incoming] = value_states.clone()

        key = self.key_cache[layer_idx][:, :self.seq_len + incoming]
        value = self.value_cache[layer_idx][:, :self.seq_len + incoming]

        if layer_idx == self.layers-1:
            self.seq_len += incoming
        return key, value

    def spec_update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int, gamma_offset=0):

        start = self.real_budget-self.gamma-3
        end = self.real_budget-self.gamma-3+new_k_cache.shape[-3]
        new_k_cache_resized = new_k_cache.clone()
        self.key_cache[layer_idx][:, start:end] = new_k_cache_resized
        #.key_cache[layer_idx][:, start:end] = new_k_cache.clone()
        new_v_cache_resized = new_v_cache.clone()
        self.value_cache[layer_idx][:, start:end] = new_v_cache_resized
        #self.value_cache[layer_idx][:, start:end] = new_v_cache.clone()

        return self.key_cache[layer_idx][:,:end], self.value_cache[layer_idx][:,:end]

    def reset(self):
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def evict_prefill(self, incoming):
        # evict
        if self.seq_len + incoming <= self.start_size + self.recent_size:
            return
        for layer_idx in range(self.layers):
            size_keep = self.recent_size - incoming
            self.key_cache[layer_idx][:, self.start_size:self.start_size+size_keep] = self.key_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()
            self.value_cache[layer_idx][:, self.start_size:self.start_size+size_keep] = self.value_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()

        self.seq_len = self.start_size + self.recent_size - incoming

    def evict_for_spec(self, current_seq_len):
        self.key_cache[:,:,self.start_size:self.start_size+self.recent_size] = self.key_cache[:,:, current_seq_len-self.recent_size:current_seq_len].clone()
        self.value_cache[:,:, self.start_size:self.start_size+self.recent_size] = self.value_cache[:,:, current_seq_len-self.recent_size:current_seq_len].clone()
