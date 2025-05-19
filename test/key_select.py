import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import json
from termcolor import colored
import torch
import decord
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import argparse
from model.modeling_qwen2_vl_sparsity import Qwen2VLForConditionalGeneration_target
from model.modeling_qwen2_2b import Qwen2VLForConditionalGeneration_draft
from model.modeling_qwen import Qwen2ForCausalLM
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.qwen2_load_video import process_vision_info_frame_idx
from utils.qwen2_load_video_all import process_vision_info_dynamic
from utils.new_decoding import TriForce, Autoregressive

from model.cache import FlashSimpleCache, RetrievalCache, StreamingLLMEvictionCache
from utils.graph_infer import GraphInferenceEngine

def build_messages_from_entry(entry: Dict, video_root: str) -> List[Dict]:
    return [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": str(Path(video_root) / entry["video_path"]),
                "total_pixels": 20000 * 28 * 28,
                "frame_idx": entry["frame_idx"]  # 传递关键帧索引
            },
            {
                "type": "text",
                "text": entry["question"]
            }
        ]
    }]

def load_dataset(path: str) -> List[Dict]:
    try:
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data 
            else:
                return [data]  
    except json.JSONDecodeError:
        with open(path) as f:
            return [json.loads(line) for line in f]

def main(args):
    # 初始化模型
    draft_model = Qwen2ForCausalLM.from_pretrained(args.draft_model_path, torch_dtype=torch.float16, device_map="cuda:1").eval()
    target_model = Qwen2VLForConditionalGeneration_target.from_pretrained(args.target_model_path, torch_dtype=torch.float16, device_map="cuda:1").eval()
    processor = AutoProcessor.from_pretrained(args.target_model_path)

    # 加载数据集
    dataset = load_dataset(args.dataset_path)

    # 性能统计
    total_acceptance_rate = 0
    total_baseline = 0
    total_triforce = 0
    valid_samples = 0

    for entry in tqdm(dataset, desc="Processing videos"):
        try:
            gen_len = args.gen_len

            layer_list = args.layer_list
            vision_token_ratio_list = args.vision_token_ratio_list

            gamma = args.gamma
            messages = build_messages_from_entry(entry, args.video_root)
            
            frame_idx = entry["frame_idx"][:args.nframes]
            image_inputs, video_inputs = process_vision_info_frame_idx(messages, frame_idx, args.nframes)
            image_inputs_all, video_inputs_all = process_vision_info(messages)

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(target_model.device)
            inputs_all = processor(
                text=[text],
                images=image_inputs_all,
                videos=video_inputs_all,
                padding=True,
                return_tensors="pt"
            ).to(target_model.device)
            print(colored(f"tokenized_prompts length: {inputs.input_ids.shape[1]}", "green"))
            print(colored(f"tokenized_prompts_all length: {inputs_all.input_ids.shape[1]}", "green"))

            draft_cache_budget = args.draft_cache_budget
            recent_size = draft_cache_budget - 16 - gamma

            cache =FlashSimpleCache(target_model)
            graph_cache = RetrievalCache(target_model)
            draft_cache = StreamingLLMEvictionCache(draft_model, start_size=16, recent_size=recent_size, gamma=gamma)
            
            engine = GraphInferenceEngine(target_model, cache, graph_cache, draft_model, draft_cache)
            engine.initialize_cuda_graph(args.gamma, probs=True, temperature=args.temp, top_p=args.top_p)

            baseline_speed = Autoregressive(processor, engine, inputs=inputs_all, 
                                         max_len=args.gen_len, top_p=args.top_p,
                                         temperature=args.temp, verbose=False)

            draft_cache_budget = args.draft_cache_budget
            recent_size = draft_cache_budget - 16 - gamma

            cache =FlashSimpleCache(target_model)
            graph_cache = RetrievalCache(target_model)
            draft_cache = StreamingLLMEvictionCache(draft_model, start_size=16, recent_size=recent_size, gamma=gamma)
            
            engine = GraphInferenceEngine(target_model, cache, graph_cache, draft_model, draft_cache)
            engine.initialize_cuda_graph(args.gamma, probs=True, temperature=args.temp, top_p=args.top_p)

            acceptance_rate, triforce_speed, _ = TriForce(processor, engine, inputs=inputs,
                                           gamma=args.gamma, max_len=args.gen_len,
                                           top_p=args.top_p, temperature=args.temp,verbose=False)
            total_acceptance_rate += acceptance_rate
            total_baseline += 1000 / baseline_speed
            total_triforce += 1000 / triforce_speed
            valid_samples += 1

        except Exception as e:
            print(f"Error processing {entry['id']}: {str(e)}")

    avg_acceptance_rate = total_acceptance_rate / valid_samples
    avg_baseline = total_baseline / valid_samples
    avg_triforce = total_triforce / valid_samples
    speedup = avg_baseline / avg_triforce

    print(f"\nResults on {valid_samples} samples:")
    print(f"Average acceptance rate: {avg_acceptance_rate:.2f}")
    print(f"Baseline latency: {avg_baseline:.2f} ms/token")
    print(f"TriForce latency: {avg_triforce:.2f} ms/token")
    print(f"Average speedup: {speedup:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/wmk/code/LongVideo/datasets/longvideobench/include_frame_idx.json", help="Path to LongVideoBench JSON file")
    parser.add_argument("--video_root", type=str, default="/home/wmk/code/data/LongVideoBench/videos", help="Root directory of video files")
    parser.add_argument("--nframes", type=int, default=2, help="Number of keyframes to use")
    parser.add_argument("--draft_model_path", type=str, default="/home/share/pyz/model_weight/Qwen2.5-0.5B-Instruct/")
    parser.add_argument("--target_model_path", type=str, default="/home/share/pyz/model_weight/Qwen2-VL-7B-Instruct/")
    parser.add_argument("--prefill", type=int, default=4096)
    parser.add_argument("--gamma", type=int, default=6)
    parser.add_argument("--temp", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--draft_cache_budget", type=int, default=1024)
    parser.add_argument('--layer_list', type=list, default=[8, 16, 24,], help='layer_list')
    parser.add_argument('--vision_token_ratio_list', type=list, default=[1, 0.5, 0.25, 0.125], help='vision_token_ratio_list')
    parser.add_argument('--gen_len', type=int, default=20, help='generation length')
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--chunk_size', type=int, default=64, help='chunk size')

    
    args = parser.parse_args()
    main(args)