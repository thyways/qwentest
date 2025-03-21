# CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --prefill 124928 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6 --dataset 128k

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from termcolor import colored
from tqdm import tqdm
from model.modeling_qwen2_vl_new import Qwen2VLForConditionalGeneration
from model.modeling_qwen2_2b import Qwen2VLForConditionalGeneration_draft
from model.cache import FlashSimpleCache, RetrievalCache,OffloadingFlashSimpleCache,StreamingLLMEvictionCache
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info 
from utils.graph_infer import GraphInferenceEngine
from transformers import logging

#logging.set_verbosity_error()

#from sampling.decoding import TriForce
from sampling.new_decoding import TriForce
from utils.misc import print_config


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--draft_model_path', type=str, default="/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct")
    parser.add_argument('--target_model_path', type=str, default="/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct")
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--prefill', type=int, default=4096, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=40, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    #parser.add_argument('--dataset', type=str, default='gs', help='dataset')
    parser.add_argument('--temp', type=float, default=0.01, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--draft_cache_budget', type=int, default=512, help='draft cache budget')
    parser.add_argument('--chunk_size', type=int, default=64, help='chunk size')
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
    draft_model = Qwen2VLForConditionalGeneration_draft.from_pretrained(args.draft_model_path,torch_dtype = torch.float16,device_map="cuda:2")
    draft = draft_model.eval()
    print('Loading  target  model...')
    target_model = Qwen2VLForConditionalGeneration.from_pretrained(args.target_model_path,torch_dtype = torch.float16,device_map="cuda:2")
    target = target_model.eval()
    processor = AutoProcessor.from_pretrained(args.target_model_path)

    # Preparation for inference
    messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type":"image","image":"/data1/bks/liurunze/qwentest/image/dog.jpg"},
                                    {"type": "text", "text": "Describe the image"},
                                ],
                            }
                        ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
    inputs = inputs.to(target_model.device)
    
    ######## sampling parameters ########
    top_k = -1
    top_p = args.top_p
    temperature = args.temp

    prefill = args.prefill
    gen_len = args.gen_len
    gamma = args.gamma
    verbose = args.verbose
    chunk_size = args.chunk_size
    max_budget = args.budget

    print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=None, method="TriForce", spec_args={'budget': args.budget, 'chunk_size': chunk_size})
    #print_config( target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=None, method="TriForce", spec_args={'budget': args.budget, 'chunk_size': chunk_size})
    ####### cache init #######

    draft_cache_budget = args.draft_cache_budget
    recent_size = draft_cache_budget - 16 - gamma

    cache =FlashSimpleCache(target, inputs.input_ids.shape[1]+gen_len+32)
    graph_cache = RetrievalCache(target, max_budget=max_budget, prefill=inputs.input_ids.shape[1], gamma=gamma, chunk_size=chunk_size)
    draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

    # graph_engine = GraphInferenceEngine(target, cache, graph_cache, draft, draft_cache)
    # graph_engine.initialize_cuda_graph(gamma, probs=True, temperature=temperature, top_p=top_p)

    cache.print_status()
    graph_cache.print_status()
    #draft_cache.print_status()
    print(colored(f"tokenized_prompts length: {inputs.input_ids.shape[1]}", "green"))

    n_warmups = 1
    

# #Testing phase
#     all_speeds = []
#     all_outputs = []  # 存储所有生成结果

# for _ in tqdm(range(3), desc="Autoregressive Test"):
#     # 确保输入数据在正确设备上
    
    
#     # 执行生成并收集结果
#     speed, generated_ids = autoregressive_sampling(
#         processor, graph_engine, inputs,
#         max_len=gen_len, top_k=top_k,
#         top_p=top_p, temperature=temperature,
#         verbose=verbose
#     )
    
#     all_speeds.append(speed)
#     all_outputs.append(generated_ids)  # 保存所有生成结果
#     torch.cuda.synchronize()  # 确保准确计时

# # 计算延迟时使用所有结果的平均值
# avg_speed = sum(all_speeds) / len(all_speeds)
# baseline_latency = 1000 / avg_speed  # 转换为每token毫秒
# print(colored(f"[Autoregressive] Average latency: {baseline_latency:.2f} ms", "red"))

# # 处理所有生成结果
# for i, generated_ids in enumerate(all_outputs):
#     generated_ids_trimmed = [
#         out_ids[len(in_ids):] 
#         for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     print(colored(f"\nGeneration {i+1}:", "green"))
#     print(output_text)
    all_acceptance_rate = []
    all_speed = []
    for input_ids in tqdm(range(1), desc="TriForce Test"):
        #input_ids = input_ids.to(target.device)[:,:prefill]

        #acceptance_rate, speed,text = TriForce(processor, graph_engine=graph_engine,inputs=inputs, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=False )
        acceptance_rate, speed,text = TriForce(processor, target=target,cache=cache,graph_cache=graph_cache,inputs=inputs, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=False )
        all_acceptance_rate.append(acceptance_rate)
        all_speed.append(speed)

    method_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"average acceptance rate (NOT per token): {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
    print(colored(f"[TriForce] average latency: {method_latency} ms", "red"))
    print(text)