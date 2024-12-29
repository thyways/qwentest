# %%
import argparse
import torch 

# %%
from PIL import Image
import os
import requests
from io import BytesIO
from torchvision import io
from transformers import TextStreamer
from tqdm import tqdm
from termcolor import colored


# %%
from transformers import logging
from sampling.Myspeculation_sampling2 import speculative_sampling
from sampling.Myautoregressive_sampling import autoregressive_sampling
from sampling.qwen2_sampling import qwen2_sampling

# %%
logging.set_verbosity_error()

# %%
from qwen2_vl import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info 

# %%
def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


# %%
def main(args):
    disable_torch_init()

    random_seed=args.seed

    print('Loading  draft  model...')
  
    draft_model = Qwen2VLForConditionalGeneration.from_pretrained(args.draft_model_path,torch_dtype = torch.bfloat16,device_map="cuda:0")
    draft_model = draft_model.eval()

    print('Loading  target  model...')

    target_model = Qwen2VLForConditionalGeneration.from_pretrained(args.target_model_path,torch_dtype = torch.bfloat16,device_map="cuda:0")
    target_model = target_model.eval()

    processor = AutoProcessor.from_pretrained(args.target_model_path)
        ######## sampling parameters ########

    top_k = 1
    top_p = args.top_p
    temperature = args.temp

    prefill = args.prefill
    gen_len = args.gen_len
    gamma = args.gamma
    verbose = args.verbose
    chunk_size = args.chunk_size
    max_budget = args.budget
    max_len = args.max_new_tokens

    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": "/data1/bks/wumengke/code/qwentest/image/5Z6kFPE1v9.jpg",
                                },
                                {"type": "text", "text": "Describe this image."},
                            ],
                        }
                    ]

# Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    input_tokens = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
    input_tokens = input_tokens.to(target_model.device)

####### qwen2生成 ########
    qwen2_sampling(processor=processor, model=target_model , input_tokens=input_tokens, max_new_tokens=max_len)
    
###### Warm up for baseline ########
    torch.manual_seed(seed=random_seed)
    # n_warmups = 1
    # for i in tqdm(range(n_warmups), desc="Autoregressive Warmup"): 
    #     autoregressive_sampling( input_tokens, target_model , max_len=max_len, top_k=top_k, top_p=top_p, temperature=temperature)

    all_speed = []
    for i in tqdm(range(1), desc="Autoregressive Test"):
        speed, generated_ids = autoregressive_sampling(input_tokens, target_model , max_len=max_len, top_k=top_k, top_p=top_p, temperature=temperature)
        all_speed.append(speed)

    baseline_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"[Autoregressive] average latency: {baseline_latency} ms", "red"))

    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_tokens.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output_text)
######## Warm up for our method ########

    torch.manual_seed(seed=random_seed)
    n_warmups = 1
    for i in tqdm(range(n_warmups), desc="Speculative Warmup"):
        speculative_sampling(input_tokens, draft_model, target_model,  max_len=gen_len, gamma=gamma, top_k=top_k, top_p=top_p, temperature=temperature)

    all_acceptance_rate = []
    all_speed = []
    for i in tqdm(range(5), desc="Speculative Test"):
        input_tokens = input_tokens.to(target_model.device)

        acceptance_rate, speed ,generated_ids = speculative_sampling(input_tokens, draft_model, target_model,  max_len=gen_len, gamma=gamma, top_k=top_k, top_p=top_p, temperature=temperature)
        all_acceptance_rate.append(acceptance_rate)
        all_speed.append(speed)

    method_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"average acceptance rate (NOT per token): {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
    print(colored(f"[TriForce] average latency: {method_latency} ms", "red"))
    print(colored(f"[E2E Speedup]: {baseline_latency / method_latency}", "red"))

    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_tokens.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output_text)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--draft_model_path', type=str, default="/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct")
    parser.add_argument('--target_model_path', type=str, default="/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct")
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--prefill', type=int, default=4096, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=30, help='generation length')
    parser.add_argument('--gamma', type=int, default=8, help='gamma')
    parser.add_argument("--max_new_tokens", type=int, default=30)

    parser.add_argument('--dataset', type=str, default='gs', help='dataset')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument('--draft_cache_budget', type=int, default=256, help='draft cache budget')
    parser.add_argument('--chunk_size', type=int, default=8, help='chunk size')
    parser.add_argument('--seed', '-s', type=int, default=66, help='set a random seed, which can makes the result reproducible')
    args = parser.parse_args()
    main(args)


