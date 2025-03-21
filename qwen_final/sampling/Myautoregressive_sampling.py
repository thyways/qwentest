import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample
from transformers import Qwen2VLForConditionalGeneration

import time


@torch.no_grad()
def autoregressive_sampling(inputs : torch.Tensor, model : torch.nn.Module, max_len : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    model_outputs=model(input_ids=inputs.input_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw)
    logits = model_outputs.logits
    past_key_values = model_outputs.past_key_values

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    n = 0
    next_token_flat = next_token.view(-1)
    generated_ids = torch.cat((inputs.input_ids, next_token_flat.view(1, -1)), dim=1)

    time1 = time.time()
    while n < max_len:
        #model_outputs1=model(input_ids=next_token,past_key_values=past_key_values,use_cache=True)
        #model_outputs2=model(input_ids=generated_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw)
        model_outputs=model(input_ids=inputs.input_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw)
        logits1 = model_outputs.logits[:,-1,:]
        #logits2 = model_outputs2.logits[:,-1,:]
        #assert torch.allclose(logits1,logits2)
        #print(logits1.eq(logits2))
        next_token = sample(norm_logits(logits1, temperature=temperature ,top_k=top_k, top_p=top_p))
        n += 1

        next_token_flat = next_token.view(-1)
        generated_ids = torch.cat((generated_ids, next_token_flat.view(1, -1)), dim=1)

    time2 = time.time()
    return n / (time2 - time1),generated_ids