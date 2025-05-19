import argparse
import json
import os
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_anno_path', type=str, default='/home/wmk/code/LongVideo/datasets')
    parser.add_argument('--base_score_path', type=str, default='/home/wmk/code/LongVideo/selected_frame')
    parser.add_argument('--dataset_name', type=str, default='longvideobench')
    parser.add_argument('--score_type', type=str, default='longvideobench_clip_frames')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser_args()
    anno_path = os.path.join(args.base_anno_path, args.dataset_name,'include_frame_idx.json')
    score_path = os.path.join(args.base_score_path, f'{args.score_type}.json')

    with open(anno_path, 'r') as f:
        anno = json.load(f)

    with open(score_path, 'r') as f:
        score = json.load(f)
    for idx, data in enumerate(anno):
        data['frame_idx'] = score[idx]
    with open(anno_path, 'w') as f:
        json.dump(anno, f)