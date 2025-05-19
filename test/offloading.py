# CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --prefill 124928 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6 --dataset 128k

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import json
from pathlib import Path

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from termcolor import colored
from tqdm import tqdm
from model.modeling_qwen2_vl_sparsity import Qwen2VLForConditionalGeneration_target
from model.modeling_qwen2_2b import Qwen2VLForConditionalGeneration_draft
from model.modeling_qwen import Qwen2ForCausalLM
from transformers import Qwen2VLForConditionalGeneration
from model.cache import OffloadingFlashSimpleCache, RetrievalCache,StreamingLLMEvictionCache
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info 
from utils.graph_infer import GraphInferenceEngine
from transformers import logging

from utils.new_decoding import TriForce,Autoregressive,original_sampling
from utils.misc import print_config
from dataloader.SSv2 import SSv2Loader

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--draft_model_path', type=str, default="/home/share/pyz/model_weight/Qwen2.5-0.5B-Instruct/")
    parser.add_argument('--target_model_path', type=str, default="/home/share/pyz/model_weight/Qwen2-VL-7B-Instruct/")
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--layer_list', type=list, default=[8, 16, 24,], help='layer_list')
    parser.add_argument('--vision_token_ratio_list', type=list, default=[1, 0.5, 0.25, 0.125], help='vision_token_ratio_list')
    parser.add_argument('--gen_len', type=int, default=64, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--prefill', type=int, default=4096, help='prefill length')

    #parser.add_argument('--dataset', type=str, default='gs', help='dataset')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--draft_cache_budget', type=int, default=512, help='draft cache budget')
    parser.add_argument('--chunk_size', type=int, default=64, help='chunk size')

    parser.add_argument('--dataset_path', type=str, default="/home/lrz/spatical_PCA/dataset/20bn-something-something-v2/")
    parser.add_argument('--other_path', type=str, default="/home/wmk/data")
    parser.add_argument('--split', type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument('--batch_size', type=int, default=1)  

    args = parser.parse_args()
    
    return args

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

if __name__ == "__main__":

    args = parse_arguments()
    disable_torch_init()

    ######## model initialization ########
    print('Loading  draft  model...')
    draft_model = Qwen2ForCausalLM.from_pretrained(args.draft_model_path,torch_dtype = torch.float16,device_map="cuda:3")
    draft = draft_model.eval()
    print('Loading  target  model...')
    target_model = Qwen2VLForConditionalGeneration_target.from_pretrained(args.target_model_path,torch_dtype = torch.float16,device_map="cuda:3")
    target = target_model.eval()
    # print('Loading  original  model...')
    # original_model = Qwen2VLForConditionalGeneration.from_pretrained(args.target_model_path,torch_dtype = torch.float16,device_map="cuda:3")
    # original = original_model.eval()

    processor = AutoProcessor.from_pretrained(args.target_model_path)

    # dataset = SSv2Loader(args.dataset_path, args.other_path, args.split)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type":"video","video":"/home/wmk/code/Qwen2VL_Triforce/vision/xzg_515709.mp4"},
                                    {"type": "text", "text": "Describe the video"},
                                ],
                            }
                        ]


    # for batch in tqdm(dataloader, desc="Processing SSv2"):
    #     messages = []
    #     for video_path, template in zip(batch["video_path"], batch["template"]):
    #         messages.append({
    #             "role": "user",
    #             "content": [
    #                 {"type": "video", "video": video_path},
    #                 {"type": "text", "text": template}
    #             ]
    #         })

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
    inputs = inputs.to(target.device)

    vision_token_nums = inputs.video_grid_thw[0,0]*inputs.video_grid_thw[0,1]*inputs.video_grid_thw[0,2]/4
    ######## sampling parameters ########
    top_k = -1
    top_p = args.top_p
    temperature = args.temp

    prefill = args.prefill
    gen_len = args.gen_len

    layer_list = args.layer_list
    vision_token_ratio_list = args.vision_token_ratio_list

    gamma = args.gamma
    verbose = args.verbose
    chunk_size = args.chunk_size
    max_budget = args.budget

    print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=None, method="TriForce")
    ####### cache init #######

    draft_cache_budget = args.draft_cache_budget
    recent_size = draft_cache_budget - 16 - gamma

    cache =OffloadingFlashSimpleCache(target,max_budget=inputs.input_ids.shape[1]+gen_len+1)
    graph_cache = RetrievalCache(target, 
                                prefill=inputs.input_ids.shape[1],
                                gamma=gamma,
                                max_len = gen_len,
                                vision_token_nums=vision_token_nums,
                                layer_list=layer_list,
                                vision_token_ratio_list=vision_token_ratio_list)
    draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)
    
    graph_engine = GraphInferenceEngine(target, cache, graph_cache, draft, draft_cache)
    graph_engine.initialize_cuda_graph(gamma, probs=True, temperature=temperature, top_p=top_p)

    cache.print_status()
    graph_cache.print_status()
    draft_cache.print_status()
    print(colored(f"tokenized_prompts length: {inputs.input_ids.shape[1]}", "green"))

    # all_speed = []
    # for _ in tqdm(range(3), desc="Autoregressive Test"):
    #     speed = Autoregressive(processor, graph_engine, inputs=inputs, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=False)
    #     all_speed.append(speed)
    # baseline_latency = 1000/(sum(all_speed) / len(all_speed)) 
    # print(colored(f"[Autoregressive] average latency: {baseline_latency} ms", "red"))
    
    all_acceptance_rate = []
    all_speed = []
    for _ in tqdm(range(1), desc="TriForce Test"):
        acceptance_rate, speed,acc_rate_middle_list = TriForce(processor, graph_engine, inputs=inputs, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=False)
        all_acceptance_rate.append(acceptance_rate)
        all_speed.append(speed)
       
    method_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"average acceptance rate (NOT per token): {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
    print(colored(f"[TriForce] average latency: {method_latency} ms", "red"))
    # print(colored(f"[E2E Speedup]: {baseline_latency / method_latency}", "red"))

    # time1=time.time()
    # generate_ids=original_sampling(model=original_model, input_tokens=inputs, max_len=gen_len)
    # time2=time.time()
    # original_latency = 1000/(gen_len/(time2-time1))
    # print(colored(f"[original] average latency: {original_latency} ms", "red"))
