from typing import Any, Dict, List, Optional, Tuple
from numpy import dtype
import torch
import math
import torch.nn.functional as F
from einops import rearrange, einsum
from transformers.models.qwen2_vl.modeling_qwen2_vl import repeat_kv

class Cache:
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")
############## Single GPU Cache ###############
class FlashSimpleCache(Cache):
    def __init__(self, model,) -> None:
        self.seq_len = 0
        
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads 
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.scores = []
    
    def reset(self):
        self.seq_len=0
        with torch.inference_mode():
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
    
    def init_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if layer_idx == 0:
            self.seq_len += key_states.shape[-2]
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0: 
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def reset_kv_cache(self):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :self.seq_len, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :self.seq_len, :]

    

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

        self.key_cache: List[torch.Tensor] = [torch.zeros(1, self.max_budget+10,  self.num_heads, self.head_dim, dtype=torch.float16, device='cpu').pin_memory() for _ in range(self.layers)]
        self.value_cache: List[torch.Tensor] = [torch.zeros(1, self.max_budget+10,  self.num_heads, self.head_dim, dtype=torch.float16, device='cpu').pin_memory() for _ in range(self.layers)]

        # init layer cache buffer on chip
        self.key_cache_buffer: List[torch.Tensor] = [torch.zeros(1, self.max_budget+10,  self.num_heads, self.head_dim, dtype=torch.float16, device=self.device) for _ in range(self.layers)]
        self.value_cache_buffer: List[torch.Tensor] = [torch.zeros(1, self.max_budget+10,  self.num_heads, self.head_dim, dtype=torch.float16, device=self.device) for _ in range(self.layers)]

        self.load_stream = torch.cuda.Stream(device=self.device)

    def print_status(self):
        print("[Offloading Flash Simple Cache] Cached Size:", self.seq_len, "| Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len=0
        with torch.inference_mode():
            for layer_idx in range(len(self.key_cache)):
                # In-place ops prevent breaking the static address
                self.key_cache[layer_idx].zero_()
                self.value_cache[layer_idx].zero_()
    
    def init_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states.cpu()
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states.cpu()

        # copy k v cache to buffer
        self.key_cache_buffer.copy_(self.key_cache[layer_idx], non_blocking=True)
        self.value_cache_buffer.copy_(self.value_cache[layer_idx], non_blocking=True)

        key = self.key_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]
        
        return key, value
    
    def update_kv_cache(self,kv_cache) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx][:, self.seq_len :].zero_()
            self.value_cache[layer_idx][:, self.seq_len :].zero_()

    def reset_kv_cache(self):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :self.seq_len, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :self.seq_len, :]
 
class RetrievalCache(Cache):
    def __init__(self, model) -> None:
        self.seq_len=0
        self.gamma=6
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.num_attention_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers
        assert self.num_attention_heads%self.num_heads==0, "num_query_heads must be divisible by num_kv_heads"
        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.init_graph = False

    def reset(self):
        self.seq_len=0
        with torch.inference_mode():
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []

    def init_graph_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if layer_idx == 0:
            self.seq_len += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset_graph_cache(self):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :self.seq_len, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :self.seq_len, :]
    
    def update(self, kv_cache, count):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx],kv_cache.key_cache[layer_idx][..., kv_cache.seq_len-self.gamma-1+count:, :]], dim=-2)
            self.value_cache[layer_idx]= torch.cat([self.value_cache[layer_idx],kv_cache.value_cache[layer_idx][..., kv_cache.seq_len-self.gamma-1+count:, :]], dim=-2)
        self.seq_len = self.key_cache[layer_idx].shape[2]
       

class StreamingLLMEvictionCache(Cache):

    def __init__(self, model, gamma=6, start_size=16, recent_size=496) -> None:

        self.gamma = gamma
        self.start_size = start_size
        self.recent_size = recent_size
        self.real_budget = self.start_size + self.recent_size + self.gamma + 1 + 1 + 1

        self.seq_len = 0 

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
