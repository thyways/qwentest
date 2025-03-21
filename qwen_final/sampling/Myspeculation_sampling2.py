
import torch
from tqdm import tqdm
import torch
import time
from termcolor import colored

from sampling.utils import norm_logits, sample, max_fn


def speculative_sampling(inputs, draft_model,target_model,max_len,gamma,temperature,top_k,top_p):

    input_ids=inputs.input_ids

    next_logits = draft_model(input_ids=input_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw).logits
    next_token = sample(norm_logits(next_logits[0,-1:], temperature=temperature ,top_k=top_k, top_p=top_p))

    next_token_flat = next_token.view(-1)
    generation = torch.cat((input_ids, next_token_flat.view(1, -1)), dim=1)

    resample_count = 0
    accepted_count = 0
    draft_count = 0
    n=0
    
    time1 = time.time()
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
    #draft model sampling
        next_token_flat = next_token.view(-1)
        verify_tokens = torch.cat((generation, next_token_flat.view(1, -1)), dim=1)

        speculation_probs = []
        generated_ids = []
        for i in range(gamma):
            logits = draft_model(input_ids=verify_tokens,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw).logits

            speculation_prob =  norm_logits(logits[:,-1,:], temperature, top_k, top_p)
            spec_token = sample(speculation_prob)
            speculation_probs.append(speculation_prob.unsqueeze(0))
            verify_tokens_flat = spec_token.view(-1)
            verify_tokens = torch.cat((verify_tokens, verify_tokens_flat.view(1, -1)), dim=1)

            generated_ids.append(spec_token)

        draft_count += gamma
        gamma2 = len(generated_ids)

    #target model sampling
        target_logits = target_model(input_ids=verify_tokens,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw).logits

        count = 0
        verify_probs = []
        for j in range(gamma):
            probs = norm_logits(target_logits[:,-gamma-1+j,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            verify_probs.append(probs.unsqueeze(0))

        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
            r = torch.rand(1, device = target_model.device)
            
            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[0,0,i] / speculation_prob[0,0,i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(target_model.device)
                next_token_flat = pred_token_idx.view(-1)
                generation = torch.cat((generation, next_token_flat.view(1, -1)), dim=1)
                
                
            else:
                resample_count += 1
                n += 1
                x=verify_prob[0,:151936].unsqueeze(0)
                pred_token_idx = sample(max_fn(x-speculation_prob[0]))
                next_token_flat = pred_token_idx.view(-1)
                generation = torch.cat((generation, next_token_flat.view(1, -1)), dim=1)
                break

        next_token = pred_token_idx

    time2 = time.time()
    acceptance_rate = accepted_count / draft_count

    return acceptance_rate, n / (time2 - time1), generation


    