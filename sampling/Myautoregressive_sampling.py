import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample

import time


@torch.no_grad()
def autoregressive_sampling(inputs : torch.Tensor, model : torch.nn.Module, max_len : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    input_ids=inputs.input_ids
    logits=model(input_ids=input_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw).logits
    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    n = 0
    next_token_flat = next_token.view(-1)
    generated_ids = torch.cat((input_ids, next_token_flat.view(1, -1)), dim=1)

    time1 = time.time()
    while n < max_len:
        logits=model(input_ids=generated_ids,pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw).logits
        next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
        n += 1

        next_token_flat = next_token.view(-1)
        generated_ids = torch.cat((generated_ids, next_token_flat.view(1, -1)), dim=1)

    time2 = time.time()
    return n / (time2 - time1),generated_ids