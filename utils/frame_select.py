import argparse
import os
import json
import heapq
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP-based Video Keyframe Selection')
    parser.add_argument('--dataset_name', type=str, default='longvideobench', 
                       help='Dataset name (longvideobench or videomme)')
    parser.add_argument('--dataset_path', type=str, default='/home/wmk/code/data/LongVideoBench',
                       help='Path to dataset directory')
    parser.add_argument('--output_file', type=str, default='/home/wmk/code/LongVideo/selected_frame',
                       help='Output directory for selected frames')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for computation (cuda/cpu)')
    parser.add_argument('--max_num_frames', type=int, default=64,
                       help='Maximum number of keyframes to select')
    parser.add_argument('--ratio', type=int, default=1,
                       help='Sampling ratio for frame selection')
    parser.add_argument('--t1', type=float, default=0.8,
                       help='Threshold for mean difference')
    parser.add_argument('--t2', type=float, default=-100,
                       help='Threshold for standard deviation')
    parser.add_argument('--all_depth', type=int, default=5,
                       help='Maximum recursion depth')
    return parser.parse_args()

def load_dataset(args):
    if args.dataset_name == "longvideobench":
        label_path = os.path.join(args.dataset_path, 'lvb_val.json')
        video_path = os.path.join(args.dataset_path, 'videos')
    elif args.dataset_name == "videomme":
        label_path = os.path.join(args.dataset_path, 'videomme.json')
        video_path = os.path.join(args.dataset_path, 'data')
    else:
        raise ValueError("Invalid dataset name")
    
    with open(label_path, 'r') as f:
        datas = json.load(f)
    return datas, video_path

def clip_feature_extraction(args, datas, video_path):
    device = torch.device(args.device)
    model = CLIPModel.from_pretrained("/home/share/pyz/model_weight/clip-vit-large-patch14-336/").to(device)
    processor = CLIPProcessor.from_pretrained("/home/share/pyz/model_weight/clip-vit-large-patch14-336/")
    
    all_scores = []
    all_frames = []
    
    for data in tqdm(datas, desc="Extracting CLIP Features"):
        text = data['question']
        video_file = os.path.join(
            video_path, 
            data["video_path"] if args.dataset_name == "longvideobench" 
            else f"{data['videoID']}.mp4"
        )
        vr = VideoReader(video_file, ctx=cpu(0))
        fps = vr.get_avg_fps()
        frame_indices = range(0, len(vr), int(fps * args.ratio))
        
        # Text features
        text_inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        
        frame_scores = []
        frame_numbers = []
        
        for idx in tqdm(frame_indices, desc=f"Frames for {os.path.basename(video_file)}", leave=False):
            frame = Image.fromarray(vr[idx].asnumpy())
            image_inputs = processor(images=frame, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**image_inputs)
            sim = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
            frame_scores.append(sim.item())
            frame_numbers.append(idx)
        
        all_scores.append(frame_scores)
        all_frames.append(frame_numbers)
    
    return all_scores, all_frames

def recursive_selection(scores, frames, params):
    def _meanstd(scores_list, frames_list, depth):
        if depth >= params['all_depth'] or len(scores_list) <= params['n']:
            return [(scores_list, frames_list)]
        mean = np.mean(scores_list)
        std = np.std(scores_list)
        top_scores = heapq.nlargest(params['n'], scores_list)
        
        if (np.mean(top_scores) - mean) > params['t1'] and std > params['t2']:
            return [(scores_list, frames_list)]
        mid = len(scores_list) // 2
        left = _meanstd(scores_list[:mid], frames_list[:mid], depth+1)
        right = _meanstd(scores_list[mid:], frames_list[mid:], depth+1)
        return left + right
    
    params['n'] = max(1, params['max_num_frames'] // (2 ** params['all_depth']))
    segments = _meanstd(scores, frames, 0)
    
    selected = []
    per_segment = max(1, params['max_num_frames'] // len(segments))
    for seg_scores, seg_frames in segments:
        k = min(per_segment, len(seg_scores))
        top_idxs = heapq.nlargest(k, range(len(seg_scores)), key=lambda i: seg_scores[i])
        selected.extend([seg_frames[i] for i in top_idxs])
    
    return sorted(set(selected))

def main(args):
    # Load dataset
    datas, video_path = load_dataset(args)
    # Extract CLIP features
    all_scores, all_frames = clip_feature_extraction(args, datas, video_path)
    # Prepare output
    os.makedirs(args.output_file, exist_ok=True)
    output_path = os.path.join(args.output_file, f"{args.dataset_name}_clip_frames.json")
    
    results = []
    params = {
        'max_num_frames': args.max_num_frames,
        't1': args.t1,
        't2': args.t2,
        'all_depth': args.all_depth
    }
    
    for scores, frames in tqdm(zip(all_scores, all_frames), total=len(all_scores), desc="Selecting Keyframes"):
        scores_arr = np.array(scores)
        if len(scores_arr) >= args.max_num_frames:
            normalized = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-8)
            selected = recursive_selection(normalized.tolist(), frames, params)
        else:
            selected = frames
        results.append(selected)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved selected frames to {output_path}")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
