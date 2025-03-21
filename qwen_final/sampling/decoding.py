import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample,max_fn
from utils.misc import log_csv, spec_stream
import numpy as np
import time
from model.cache import RetrievalCache
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained('/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct')
@torch.inference_mode()
def autoregressive_sampling(processor, graph_engine, inputs, max_len=10, top_k=-1, top_p=0.9, temperature=0.01, verbose=False):
    # reset all cache
    graph_engine.engine.kv_cache.reset()
    

    logits = graph_engine.new_token_inference(input_ids=inputs.input_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw)
    
    #logits = graph_engine.new_token_inference(input_ids=inputs.input_ids,pixel_values=None,image_grid_thw=None)
    if verbose:
        graph_engine.engine.kv_cache.print_status()

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    if verbose:
        spec_stream(next_token[0], processor, 'cyan')

    n = 0
    next_token_flat = next_token.view(-1)
    generated_ids = torch.cat((inputs.input_ids, next_token_flat.view(1, -1)), dim=1)

    time1 = time.time()
    while n < max_len:
        logits = graph_engine.engine.model(input_ids=next_token, kv_cache=graph_engine.engine.kv_cache, graph_cache=None).logits
        #next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
        next_token = torch.argmax(logits[:,-1,:],dim=-1).unsqueeze(-1)
        # if processor.tokenizer.eos_token_id==next_token:
        #     break
        n += 1
        if verbose:
            spec_stream(next_token[0], processor, 'cyan')
        next_token_flat = next_token.view(-1)
        generated_ids = torch.cat((generated_ids, next_token_flat.view(1, -1)), dim=1)
    time2 = time.time()
    return n / (time2 - time1),generated_ids
# def TriForce(processor,target,cache,draft,draft_cache, inputs, gamma=6, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=True, file_path=None, spec_args=None):
#     print(inputs.input_ids[:,:-1].shape)
#     cache.reset()
#     draft_cache.reset()
#     #logits = target(input_ids=inputs.input_ids,kv_cache=cache,draft_cache=draft_cache).logits
#     #logits = draft(input_ids=inputs.input_ids[:,:-1],kv_cache=cache,draft_cache=draft_cache).logits
#     logits = target(input_ids=inputs.input_ids[:,:-1],pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw,kv_cache=cache,draft_cache=draft_cache).logits
#     logits = draft(input_ids=inputs.input_ids[:,-1:],kv_cache=cache,draft_cache=draft_cache).logits
#     #logits = graph_engine.inference(input_ids=input_ids[:,-1:])
#     torch.save({"key_cache_org":cache.key_cache},"/data1/bks/liurunze/qwen_final/utils/qkv_save.pt")

#     if verbose:
#         cache.print_status()
#         draft_cache.print_status()

#     resample_count = 0
#     accepted_count = 0
#     draft_count = 0

#     next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

#     if verbose:
#         spec_stream(next_token[0], processor, 'cyan')

#     n = 0
#     time1 = time.time()
#     generated_tokens = []
#     generated_tokens.append(next_token.item())
#     pass_token = []
#     pass_token.append(next_token.item())
#     while n < max_len:
#         specualtion_probs = []
#         verify_probs = []
#         count = 0
#         if next_token.shape == torch.Size([1]):
#             next_token = next_token.unsqueeze(0)
#         #position_ids = torch.arange(cache.seq_len, cache.seq_len+gamma+1, device=target.device).unsqueeze(0).unsqueeze(0)
#         # speculative decoding for draft (68m) and retrieval 7b model
#         pred_token_idx = next_token
        
#         verify_tokens = torch.full((1, gamma + 1), 100, device=target.device)
#         verify_tokens[:,0]=pred_token_idx
#         while count<gamma:
#             logits = draft(input_ids=pred_token_idx,spec=True,draft_cache=draft_cache).logits
#             next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
#             if next_token.shape == torch.Size([1]):
#                 next_token = next_token.unsqueeze(0)
#             verify_tokens[:,count+1:count+2]=next_token
#             speculation_prob = norm_logits(logits[0],temperature=temperature ,top_k=top_k, top_p=top_p)
#             generated_tokens.append(next_token.item())
#             specualtion_probs.append(speculation_prob[0])
#             count+=1
#             draft_count+=1
#             if processor.tokenizer.eos_token_id==next_token.item():
#                 break
#         target_logits =draft(input_ids=verify_tokens[:,1:],kv_cache = cache).logits
#         probs = norm_logits(target_logits[0],temperature=temperature ,top_k=top_k, top_p=top_p)
#         for i in range(gamma):
#             verify_probs.append(probs[i])
#         # verify_probs = verify_probs.to(target.device)
#         # specualtion_probs = specualtion_probs.to(target.device)
#         m=0
#         for i,speculation_prob,verify_prob in zip(generated_tokens,specualtion_probs,verify_probs):
#             verify_prob[i] = verify_prob[i].to(target.device)
#             speculation_prob[i] = speculation_prob[i].to(target.device)
#               # 检查 r 的设备
#             r = torch.rand(1)
#             r = r.to(verify_prob[i].device)
#             m = torch.tensor([1])
#             m = m.to(verify_prob[i].device)
#             if r < torch.min(m, (verify_prob[i] / speculation_prob[i])):
#                 accepted_count+=1
#                 n+=1
#                 pass_token.append(i)
#             else:
#                 next_token = sample(max_fn(verify_prob-speculation_prob))
#                 print(next_token)
#                 resample_count+=1
#                 n+=1
#                 if next_token.dim() == 0:
#                     next_token = next_token.unsqueeze(0) 
                
#                 pass_token.append(next_token.item())
                
            
#             if processor.tokenizer.eos_token_id == next_token:
#                     break
#         #cache.seq_len-=(gamma-accepted_count)
#         #draft_cache.update_graph_cache(cache)
#     time2 = time.time()
#     acceptance_rate = accepted_count/draft_count
    
#     generated_text = processor.tokenizer.decode(pass_token, skip_special_tokens=True)
#     return acceptance_rate,n/(time2-time1),generated_text
def TriForce(processor, graph_engine, inputs, gamma=4, max_len=40, top_k=-1, top_p=0.9, temperature=0.6, verbose=False):

    # reset all cache
    graph_engine.engine.kv_cache.reset()
    graph_engine.engine.graph_cache.reset()
    graph_engine.engine.draft_cache.reset()

    logits = graph_engine.new_token_inference(input_ids=inputs.input_ids[:,:-1],pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw)
    logits = graph_engine.new_token_inference(input_ids=inputs.input_ids[:,-1:])
    _ = graph_engine.graph_draft_prefill(input_ids=inputs.input_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw)

    if verbose:
        graph_engine.engine.kv_cache.print_status()
        graph_engine.engine.graph_cache.print_status()
        graph_engine.engine.draft_cache.print_status()

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
        
        # speculative decoding for draft (68m) and retrieval 7b model
        pred_token_idx = next_token
        verify_tokens, speculation_probs, acc_rate_middle = Middle_Spec(pred_token_idx, graph_engine, gamma, False, processor)
        acc_rate_middle_list.append(acc_rate_middle)
        generated_ids = verify_tokens[1:]
        draft_count += len(speculation_probs)

        gamma2 = len(generated_ids)
        
        # speculative decoding retrieval 7b model and target model
        verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(graph_engine.engine.model.device)], dim=1)
        logits = graph_engine.inference(input_ids=verify_tokens)

        count = 0
        verify_probs = []
    
        probs = norm_logits(logits[0], temperature=temperature ,top_k=top_k, top_p=top_p)
        for i in range(gamma2 + 1):
            verify_probs.append(probs[i])

        pass_tokens = torch.full((1, gamma2 + 2), 100, device=graph_engine.engine.model.device)
        pass_tokens[:, 0] = next_token
        
        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
            r = torch.rand(1, device = graph_engine.engine.model.device)
            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(graph_engine.engine.model.device)
                pass_tokens[:, count] = pred_token_idx
                if verbose:
                    spec_stream(i, processor, 'green')
                # if eos
                if processor.tokenizer.eos_token_id == i:
                    draft_count -= gamma2 - count
                    break
            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                pass_tokens[:, count+1] = pred_token_idx
                if verbose:
                    spec_stream(pred_token_idx, processor, 'red')
                break

            if processor.tokenizer.eos_token_id == pred_token_idx:
                break

        # update 7b cache
        graph_engine.engine.kv_cache.seq_len -= (len(generated_ids) - count)
        graph_engine.update_graph_cache()
        
        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            pass_tokens[:, count+1] = pred_token_idx
            if verbose:
                spec_stream(pred_token_idx, processor, 'blue')
            count += 1

        # update cache for 68m
        graph_engine.graph_draft_inference(input_ids=pass_tokens, gamma_offset = gamma2 + 1)
        current_seq_len = graph_engine.engine.draft_cache.start_size + graph_engine.engine.draft_cache.recent_size + count
        graph_engine.engine.draft_cache.evict_for_spec(current_seq_len)

        next_token = pred_token_idx
        generated.append(next_token.item())
    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    text = processor.decode(generated,skip_special_tokens=True)
    if verbose:
        print(f"Use {time2 - time1} sec to generate {n} tokens (now {graph_engine.engine.kv_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True)
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    # if file_path is not None:
    #     header = "target,acceptance_rate,token/s,avg_tokens,prefill,gen_len,dataset,acc_rate_middle,latency\n"
    #     entry = f"{graph_engine.engine.model.config._name_or_path},{acceptance_rate},{n / (time2 - time1)},{avg_tokens},{inputs.input_ids.shape[1]},{n},{dataset},{np.array(acc_rate_middle_list).mean()},{(time2 - time1)/n}\n"

    #     if spec_args is not None:
    #         for k, v in spec_args.items():
    #             header=header.replace("\n", f",{k}\n")
    #             entry=entry.replace("\n", f",{v}\n")
    #     log_csv(file_path, header, entry)

    return acceptance_rate, n / (time2 - time1),text

@torch.inference_mode()
def Middle_Spec(next_token, graph_engine, gamma, verbose, processor):

    n = 0
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    pred_token_idx = next_token

    return_generated_ids = []
    return_speculation_probs = []
    return_generated_ids.append(next_token.item())

    verify_tokens = torch.full((1, gamma + 1), 100, device=graph_engine.engine.model.device)
    verify_tokens[:, 0] = next_token

    #position_ids = torch.arange(graph_engine.engine.kv_cache.seq_len, graph_engine.engine.kv_cache.seq_len+gamma+1, device=graph_engine.engine.model.device).unsqueeze(0)

    while n < gamma:
        speculation_prob = graph_engine.graph_draft_inference(input_ids=verify_tokens[:,:n+1], gamma_offset = n)
        
        pred_token_idx = sample(speculation_prob)
        token_idx = pred_token_idx.item()
        draft_count += 1

        verify_tokens[:, n+1:n+2] = pred_token_idx
        verify_prob = graph_engine.graph_verify(input_ids=verify_tokens)

        r = torch.rand(1, device = graph_engine.engine.model.device)
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
# def TriForce(processor, target, cache, draft, draft_cache, inputs, gamma=6, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=True, file_path=None, spec_args=None):

#     cache.reset()
#     draft_cache.reset()

#     # 初始化目标模型（修复双调用问题）
#     logits = target(
#         input_ids=inputs.input_ids[:,:-1],
#         pixel_values=inputs.pixel_values,
#         image_grid_thw=inputs.image_grid_thw,
#         kv_cache=cache,
#         draft_cache=draft_cache
#     ).logits
#     _ = target(input_ids=inputs.input_ids[:,-1:],kv_cache=cache,draft_cache=draft_cache)
#     if verbose:
#         cache.print_status()
#         draft_cache.print_status()

#     resample_count = 0
#     accepted_count = 0
#     draft_count = 0

#     # 初始token生成
#     next_token = sample(norm_logits(logits[:, -1, :], temperature, top_k, top_p))
#     generated_tokens = [next_token.item()]
#     pass_token = [next_token.item()]

#     if verbose:
#         spec_stream(next_token[0], processor, 'cyan')

#     n = 1  # 已生成1个token
#     time1 = time.time()
    
#     while n < max_len:
#         if next_token.dim() == 1:
#             next_token = next_token.unsqueeze(0)

#         # ==== 草案阶段 ====
#         draft_tokens = [next_token]
#         spec_probs = []
#         for _ in range(gamma):
#             logits = draft(
#                 input_ids=draft_tokens[-1],
#                 kv_cache=cache,
#                 spec=True,
#                 draft_cache=draft_cache
#             ).logits
            
#             # 正确获取最后一个位置的logits
#             draft_prob = norm_logits(logits[:, -1, :], temperature, top_k, top_p)
#             next_draft = sample(draft_prob)
            
#             spec_probs.append(draft_prob[0])  # 保存概率分布
#             draft_tokens.append(next_draft)
#             draft_count += 1
            
#             if next_draft.item() == processor.tokenizer.eos_token_id:
#                 break

#         # ==== 验证阶段 ====
#         verify_tokens = torch.cat(draft_tokens, dim=1)
#         target_logits = target(input_ids=verify_tokens, kv_cache=cache).logits
        
#         # 正确提取验证概率
#         verify_probs = [
#             norm_logits(target_logits[:, i, :], temperature, top_k, top_p)[0]
#             for i in range(verify_tokens.shape[1]-1)
#         ]

#         # ==== 接受判断 ====
#         accepted = 0
#         for t in range(len(verify_probs)):
#             draft_token = draft_tokens[t+1].item()  # 第t个草案token
            
#             # 正确获取对应位置的概率值
#             q = spec_probs[t][draft_token] 
#             p = verify_probs[t][draft_token]
#             r = torch.rand(1,device=target.device)
#             if r < (p / q).clamp(max=1.0):
#                 accepted += 1
#                 pass_token.append(draft_token)
#             else:
#                 # 从调整后的分布重新采样
#                 adjusted_probs = (verify_probs[t] - spec_probs[t]).clamp(min=0)
#                 resampled = sample(adjusted_probs)
#                 pass_token.append(resampled.item())
#                 break
#         else:  # 全接受
#             accepted += 1
        
#         accepted_count += accepted
#         n += accepted
        
#         # 更新next_token
#         next_token = torch.tensor([[pass_token[-1]]], device=target.device)

#         if pass_token[-1] == processor.tokenizer.eos_token_id:
#             break

#     # ==== 统计结果 ====
#     time2 = time.time()
#     acceptance_rate = accepted_count / draft_count if draft_count > 0 else 0.0
#     generated_text = processor.tokenizer.decode(pass_token, skip_special_tokens=True)
    
#     return acceptance_rate, n/(time2-time1), generated_text
#         # speculative decoding retrieval 7b model and target model
#         verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(target.device)], dim=1)
#         logits = target(input_ids=verify_tokens,pixel_values=None,image_grid_thw=None)
        
#         count = 0
#         verify_probs = []
    
#         probs = norm_logits(logits[0], temperature=temperature ,top_k=top_k, top_p=top_p)
#         for i in range(gamma2 + 1):
#             verify_probs.append(probs[i])

#         pass_tokens = torch.full((1, gamma2 + 2), 100, device=target.device)
#         pass_tokens[:, 0] = next_token
        
#         for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
#             r = torch.rand(1, device = target.device)
#             if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
#                 count += 1
#                 accepted_count += 1
#                 n += 1
#                 pred_token_idx = torch.tensor([[i]]).to(target.device)
#                 pass_tokens[:, count] = pred_token_idx
#                 generated_tokens.append(i) 
#                 if verbose:
#                     spec_stream(i, processor, 'green')
#                 # if eos
#                 if processor.tokenizer.eos_token_id == i:
#                     draft_count -= gamma2 - count
#                     break
#             else:
#                 resample_count += 1
#                 n += 1
#                 pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
#                 pass_tokens[:, count+1] = pred_token_idx
#                 generated_tokens.append(pred_token_idx.item())
#                 if verbose:
#                     spec_stream(pred_token_idx, processor, 'red')
#                 break

#             if processor.tokenizer.eos_token_id == pred_token_idx:
#                 break

#         # update 7b cache
#         cache.seq_len -= (len(generated_ids) - count)
#         #graph_engine.update_graph_cache()
        
#         if count == len(generated_ids):
#             target_sample_count += 1
#             n += 1
#             pred_token_idx = sample(verify_probs[-1])
#             pass_tokens[:, count+1] = pred_token_idx
#             if verbose:
#                 spec_stream(pred_token_idx, processor, 'blue')
#             count += 1

#         # update cache for 68m
#         draft_cache.update_graph_cache(kv_cache=cache)
        
#         next_token = pred_token_idx

#     time2 = time.time()
#     acceptance_rate = accepted_count / draft_count
#     avg_tokens = accepted_count / draft_count * gamma
#     if verbose:
#         print(f"Use {time2 - time1} sec to generate {n} tokens (now {cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True)
#         print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

#     if file_path is not None:
#         header = "target,acceptance_rate,token/s,avg_tokens,prefill,gen_len,dataset,acc_rate_middle,latency\n"
#         entry = f"{target.model.config._name_or_path},{acceptance_rate},{n / (time2 - time1)},{avg_tokens},{input.input_ids.shape[1]},{n},{np.array(acc_rate_middle_list).mean()},{(time2 - time1)/n}\n"

#         if spec_args is not None:
#             for k, v in spec_args.items():
#                 header=header.replace("\n", f",{k}\n")
#                 entry=entry.replace("\n", f",{v}\n")
#         log_csv(file_path, header, entry)
#     generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
#     return acceptance_rate, n / (time2 - time1),generated_text

# @torch.inference_mode()
# def Middle_Spec(next_token, target,cache,draft,draft_cache:RetrievalCache, gamma, verbose, processor):

#     n = 0
#     resample_count = 0
#     accepted_count = 0
#     target_sample_count = 0
#     draft_count = 0

#     pred_token_idx = next_token

#     return_generated_ids = []
#     return_speculation_probs = []
    
#     return_generated_ids.append(next_token.item())

#     verify_tokens = torch.full((1, gamma + 1), 100, device=target.device)
#     verify_tokens[:, 0] = next_token

#     position_ids = torch.arange(cache.seq_len, cache.seq_len+gamma+1, device=target.device).unsqueeze(0)

#     while n < gamma:
#         speculation_prob = draft(input_ids=verify_tokens[:,:n+1], gamma_offset = n)
#         pred_token_idx = sample(speculation_prob)
#         token_idx = pred_token_idx.item()
#         draft_count += 1
#         # if pred_token_idx.shape==torch.Size([1]):
#         #     pred_token_idx = pred_token_idx.unsqueeze(0)
#         verify_tokens[:, n+1:n+2] = pred_token_idx
#         verify_prob = target(input_ids=verify_tokens, position_ids=position_ids)
        
#         r = torch.rand(1, device = target.device)
#         if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[n, token_idx] / speculation_prob[token_idx])):
#             return_speculation_probs.append(verify_prob[n])
#             return_generated_ids.append(token_idx)
#             if verbose:
#                 spec_stream(pred_token_idx, processor, 'green')
#             accepted_count += 1
#             n += 1
        
#             pred_token_idx = sample(verify_prob[n])
#             return_speculation_probs.append(verify_prob[n])
#             return_generated_ids.append(pred_token_idx.item())
#             if verbose:
#                 spec_stream(pred_token_idx, processor, 'blue')
#             target_sample_count += 1
#             n += 1

#             verify_tokens[:, n:n+1] = pred_token_idx
        
#         else:
#             pred_token_idx = sample(verify_prob[n])
#             # if pred_token_idx.shape==torch.Size([1]):
#             #     pred_token_idx = pred_token_idx.unsqueeze(0)
#             return_speculation_probs.append(verify_prob[n])
#             return_generated_ids.append(pred_token_idx.item())
#             if verbose:
#                 spec_stream(pred_token_idx, processor, 'red')
#             resample_count += 1
#             n += 1

#             verify_tokens[:, n:n+1] = pred_token_idx
    
#     acceptance_rate = accepted_count / draft_count
#     return return_generated_ids, return_speculation_probs, acceptance_rate

