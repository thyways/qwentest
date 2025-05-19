import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import sys
import time
import torch
import decord
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import argparse
from termcolor import colored

# 路径添加
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

# 模型和工具导入
from transformers import AutoProcessor, AutoTokenizer
from model.modeling_qwen2_vl_sparsity import Qwen2VLForConditionalGeneration_target
from model.modeling_qwen2_2b import Qwen2VLForConditionalGeneration_draft
from model.modeling_qwen import Qwen2ForCausalLM
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
                "frame_idx": entry["frame_idx"]
            },
            {"type": "text", "text": entry["question"]}
        ]
    }]


def load_dataset(path: str) -> List[Dict]:
    try:
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        with open(path) as f:
            return [json.loads(line) for line in f]


def sliding_window_generate(
    messages: List[Dict],
    image_inputs,
    video_inputs,
    model,
    tokenizer,
    processor,
    device,
    window_size: int = None,
    stride: int = None,
    gen_kwargs: dict = None,
) -> str:
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    L = len(input_ids)

    max_pos = model.config.max_position_embeddings
    window_size = window_size or max_pos
    stride = stride or window_size // 2
    gen_kwargs = gen_kwargs or {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    chunks = []
    for start in range(0, L, stride):
        end = min(start + window_size, L)
        window_ids = input_ids[start:end]
        text_piece = tokenizer.decode(window_ids, clean_up_tokenization_spaces=False)
        inputs = processor(
            text=[text_piece],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=window_size,
            return_tensors="pt",
            do_rescale=False,
            do_normalize=True,
        ).to(device)

        try:
            outs = model.generate(**inputs, **gen_kwargs)
        except RuntimeError as e:
            print(f"[Warning] window {start}:{end} 推理失败，跳过: {e}")
            torch.cuda.empty_cache()
            continue

        gen_ids = outs[0][len(window_ids):]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        chunks.append((start, gen_text))

        if end == L:
            break

    chunks.sort(key=lambda x: x[0])
    return "".join([txt for _, txt in chunks])


def main(args):
    draft_model = Qwen2ForCausalLM.from_pretrained(
        args.draft_model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    target_model = Qwen2VLForConditionalGeneration_target.from_pretrained(
        args.target_model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    processor = AutoProcessor.from_pretrained(args.target_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    dataset = load_dataset(args.dataset_path)

    total_acceptance, total_base, total_tri, valid = 0, 0, 0, 0

    for entry in tqdm(dataset, desc="Processing videos"):
        try:
            messages = build_messages_from_entry(entry, args.video_root)
            frame_idx = entry["frame_idx"][:args.nframes]
            img_in, vid_in = process_vision_info_frame_idx(messages, frame_idx, args.nframes)
            img_all, vid_all = process_vision_info(messages)

            tok_len = len(tokenizer.encode(processor.apply_chat_template(messages, False, True)))
            print(colored(f"Prompt tokens: {tok_len}" , "green"))

            answer = sliding_window_generate(
                messages, img_in, vid_in,
                target_model, tokenizer, processor,
                device=target_model.device,
                window_size=target_model.config.max_position_embeddings,
                stride=target_model.config.max_position_embeddings // 2,
                gen_kwargs={
                    "max_new_tokens": args.gen_len,
                    "temperature": args.temp,
                    "top_p": args.top_p,
                    "num_beams": 1,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "use_cache": True,
                }
            )

            cache = FlashSimpleCache(target_model)
            graph_cache = RetrievalCache(target_model)
            draft_cache = StreamingLLMEvictionCache(
                draft_model, start_size=16, recent_size=args.draft_cache_budget - 16 - args.gamma, gamma=args.gamma
            )
            engine = GraphInferenceEngine(target_model, cache, graph_cache, draft_model, draft_cache)

            base_speed = Autoregressive(processor, engine,
                                        inputs=None,
                                        max_len=args.gen_len, top_p=args.top_p,
                                        temperature=args.temp, verbose=False)

            cache = FlashSimpleCache(target_model)
            graph_cache = RetrievalCache(target_model)
            draft_cache = StreamingLLMEvictionCache(
                draft_model, start_size=16, recent_size=args.draft_cache_budget - 16 - args.gamma, gamma=args.gamma
            )
            engine = GraphInferenceEngine(target_model, cache, graph_cache, draft_model, draft_cache)

            acc, tri_speed, _ = TriForce(
                processor, engine, inputs=None,
                gamma=args.gamma, max_len=args.gen_len,
                top_p=args.top_p, temperature=args.temp, verbose=False
            )

            total_acceptance += acc
            total_base += 1000 / base_speed
            total_tri += 1000 / tri_speed
            valid += 1

        except Exception as e:
            print(f"Error processing {entry.get('id', 'unknown')}: {e}")
            torch.cuda.empty_cache()
            time.sleep(1)
            continue

    if valid > 0:
        print(f"\nResults on {valid} samples:")
        print(f"Average acceptance rate: {total_acceptance/valid:.2f}")
        print(f"Baseline latency: {total_base/valid:.2f} ms/token")
        print(f"TriForce latency: {total_tri/valid:.2f} ms/token")
        print(f"Average speedup: {(total_base/valid)/(total_tri/valid):.2f}x")

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
