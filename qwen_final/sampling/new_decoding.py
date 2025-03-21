import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample,max_fn
from utils.misc import log_csv, spec_stream
import numpy as np
import time
from model.cache import RetrievalCache
from transformers import AutoProcessor
def TriForce(processor,target,cache,graph_cache, inputs, gamma=6, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=True):
    cache.reset()
    graph_cache.reset()
    logits = target(input_ids=inputs.input_ids[:,:-1],pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw,kv_cache=cache,graph_cache=None).logits
    logits = target(input_ids=inputs.input_ids[:,-1:],kv_cache=cache,graph_cache=graph_cache).logits
    torch.save({"key":cache.key_cache,"graph":graph_cache.key_cache},"/data1/bks/liurunze/qwen_final/utils/qkv_save.pt")
    if verbose:
        cache.print_status()
        graph_cache.print_status()
    resample_count = 0
    accepted_count = 0
    draft_count = 0
    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    if verbose:
        spec_stream(next_token[0], processor, 'cyan')
    n = 0
    time1 = time.time()
    
    pass_token = []
    pass_token.append(next_token.item())
    while n < max_len:
        generated_tokens = []
        
        specualtion_probs = []
        verify_probs = []
        count = 0
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        #position_ids = torch.arange(cache.seq_len, cache.seq_len+gamma+1, device=target.device).unsqueeze(0).unsqueeze(0)
        # speculative decoding for draft (68m) and retrieval 7b model
        pred_token_idx = next_token
        verify_tokens = torch.full((1, gamma + 1), 100, device=target.device)
        verify_tokens[:,0]=pred_token_idx
        while count<gamma:
            logits = target(input_ids=pred_token_idx,spec=True,graph_cache=graph_cache,kv_cache=cache,gamma_offset=count).logits
            #logits = target(input_ids=pred_token_idx,kv_cache=cache).logits
            #torch.save({"graph":logits_graph,"key":logits},"/data1/bks/liurunze/qwen_final/utils/qkv_save_new.pt")
            next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
            if next_token.shape == torch.Size([1]):
                next_token = next_token.unsqueeze(0)
            verify_tokens[:,count+1:count+2]=next_token
            speculation_prob = norm_logits(logits[0],temperature=temperature ,top_k=top_k, top_p=top_p)
            generated_tokens.append(next_token.item())
            specualtion_probs.append(speculation_prob[0])
            count+=1
            draft_count+=1
        target_logits =target(input_ids=verify_tokens[:,0:],kv_cache = cache,graph_cache=graph_cache).logits
        probs = norm_logits(target_logits[0],temperature=temperature ,top_k=top_k, top_p=top_p)
        for i in range(gamma+1):
            verify_probs.append(probs[i])
        # verify_probs = verify_probs.to(target.device)
        # specualtion_probs = specualtion_probs.to(target.device)
        total=0
        for i,speculation_prob,verify_prob in zip(generated_tokens,specualtion_probs,verify_probs):
            verify_prob[i] = verify_prob[i].to(target.device)
            speculation_prob[i] = speculation_prob[i].to(target.device)
              # 检查 r 的设备
            r = torch.rand(1)
            r = r.to(verify_prob[i].device)
            m = torch.tensor([1])
            m = m.to(verify_prob[i].device)
            if r < torch.min(m, (verify_prob[i] / speculation_prob[i])):
                accepted_count+=1
                n+=1
                pass_token.append(i)
                next_token = torch.tensor([[i]]).to(target.device)
                total+=1
                if processor.tokenizer.eos_token_id==i:
                    draft_count-=len(generated_tokens)-total
                    break
            else:
                next_token = sample(max_fn(verify_prob-speculation_prob))
                resample_count+=1
                n+=1
                if next_token.shape == torch.Size([1]):
                    next_token = next_token.unsqueeze(0)
                pass_token.append(next_token.item())
                break
            if processor.tokenizer.eos_token_id == next_token.item():
                    #draft_count-=len(generated_tokens)-total
                    break
        cache.seq_len-=(len(generated_tokens)-total)
        graph_cache.update_graph_cache(cache)
        if(total==len(generated_tokens)):
            n+=1
            pred_token = sample(verify_probs[-1])
            pass_token.append(pred_token.item())
            next_token=pred_token
    time2 = time.time()
    acceptance_rate = accepted_count/draft_count
    
    generated_text = processor.tokenizer.decode(pass_token, skip_special_tokens=True)
    return acceptance_rate,n/(time2-time1),generated_text