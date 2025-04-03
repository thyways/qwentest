import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample,max_fn
from utils.misc import log_csv, spec_stream
import numpy as np
import time
import math
from model.cache import RetrievalCache
from transformers import AutoProcessor
def draft_run(draft,draft_cache,input_ids: torch.LongTensor, gamma_offset: int=0, probs=False, temperature=0.6, top_p=0.9):
        if input_ids.shape[-1] > 64: # prefill
            iter_prefill = math.ceil(input_ids.shape[1] / 64)
            for i in range(iter_prefill):
                draft_cache.evict_prefill(64)
                logits = draft(
                    input_ids=input_ids[:, i*64:(i+1)*64],
                    kv_cache=draft_cache,
                    graph_cache=None,
                ).logits
        else: # decoding
            logits = draft(input_ids=input_ids, kv_cache=draft_cache, graph_cache=draft_cache, gamma_offset=gamma_offset).logits

        if probs: # without top_p
            return norm_logits(logits[0], temperature=temperature, top_k=-1, top_p=top_p)[-1]
        return logits
def TriForce(processor,target,cache,graph_cache,draft,draft_cache, inputs, gamma=6, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=True):
    cache.reset()
    graph_cache.reset()
    draft_cache.reset()
    with torch.no_grad():
        logits = target(input_ids=inputs.input_ids[:,:-1],pixel_values_videos=inputs.pixel_values_videos,video_grid_thw=inputs.video_grid_thw,kv_cache=cache,graph_cache=None).logits
        logits = target(input_ids=inputs.input_ids[:,-1:],kv_cache=cache,graph_cache=graph_cache).logits
        _ = draft_run(draft=draft,draft_cache=draft_cache,input_ids=inputs.input_ids)
    if verbose:
        cache.print_status()
        graph_cache.print_status()
        draft_cache.print_status()
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0
    generated = []
    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    generated.append(next_token.item())
    if verbose:
        spec_stream(next_token[0], processor, 'cyan')

    acc_rate_middle_list = []
    n = 0
    time1 = time.time()
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        pred_token_idx = next_token
        verify_tokens, speculation_probs, acc_rate_middle = Middle_Spec(pred_token_idx, target,cache,graph_cache,draft,draft_cache, gamma, False, processor)
        acc_rate_middle_list.append(acc_rate_middle)
        generated_ids = verify_tokens[1:]
        draft_count += len(speculation_probs)

        gamma2 = len(generated_ids)
        
        # speculative decoding retrieval 7b model and target model
        verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(target.device)], dim=1)
        logits = target(input_ids=verify_tokens,kv_cache=cache).logits

        count = 0
        verify_probs = []
    
        probs = norm_logits(logits[0], temperature=temperature ,top_k=top_k, top_p=top_p)
        for i in range(gamma2 + 1):
            verify_probs.append(probs[i])

        pass_tokens = torch.full((1, gamma2 + 2), 100, device=target.device)
        pass_tokens[:, 0] = next_token
        
        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
            r = torch.rand(1, device = target.device)
            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(target.device)
                pass_tokens[:, count] = pred_token_idx
                if verbose:
                    spec_stream(i, processor, 'green')
                # if eos
                if processor.tokenizer.eos_token_id == i:
                    draft_count -= gamma2 - count
                    break
                generated.append(pred_token_idx.item())
            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                pass_tokens[:, count+1] = pred_token_idx
                generated.append(pred_token_idx.item())
                if verbose:
                    spec_stream(pred_token_idx, processor, 'red')
                break

            if processor.tokenizer.eos_token_id == pred_token_idx:
                break

        # update 7b cache
        cache.seq_len -= (len(generated_ids) - count)
        graph_cache.update_graph_cache(cache)
        
        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            pass_tokens[:, count+1] = pred_token_idx
            if verbose:
                spec_stream(pred_token_idx, processor, 'blue')
            count += 1

        # update cache for 68m
        draft_run(draft=draft,draft_cache=draft_cache,input_ids=pass_tokens, gamma_offset = gamma2 + 1)
        current_seq_len = draft_cache.start_size + draft_cache.recent_size + count
        draft_cache.evict_for_spec(current_seq_len)
        next_token = pred_token_idx
        
    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    text = processor.decode(generated,skip_special_tokens=True)
    return acceptance_rate, time2 - time1,text
def Middle_Spec(next_token,target,cache,graph_cache,draft,draft_cache,gamma, verbose, processor):
    n=0
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0
    pred_token_idx = next_token
    return_generated_ids = []
    return_speculation_probs = []
    return_generated_ids.append(next_token.item())

    verify_tokens = torch.full((1, gamma + 1), 100, device=target.device)
    verify_tokens[:, 0] = next_token

    #position_ids = torch.arange(graph_engine.engine.kv_cache.seq_len, graph_engine.engine.kv_cache.seq_len+gamma+1, device=graph_engine.engine.model.device).unsqueeze(0)

    while n < gamma:
        speculation_prob = draft_run(draft=draft,draft_cache=draft_cache,input_ids=verify_tokens[:,:n+1], gamma_offset = n,probs=True)
        
        pred_token_idx = sample(speculation_prob)
        token_idx = pred_token_idx.item()
        draft_count += 1

        verify_tokens[:, n+1:n+2] = pred_token_idx
        verify_logits = target(input_ids=verify_tokens,spec=True,kv_cache=cache,graph_cache=graph_cache).logits
        verify_prob = norm_logits(verify_logits[0],temperature=0.01,top_k=-1,top_p=0.9)
        r = torch.rand(1, device = target.device)
        if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[n, token_idx] / speculation_prob[token_idx])):
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(token_idx)
            if verbose:
                spec_stream(pred_token_idx, processor, 'green')
            accepted_count += 1
            n += 1
        
            pred_token_idx = sample(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, processor, 'blue')
            target_sample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx
        
        else:
            pred_token_idx = sample(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, processor, 'red')
            resample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx
    
    acceptance_rate = accepted_count / draft_count
    return return_generated_ids, return_speculation_probs, acceptance_rate
