import argparse
import json
import os
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_num', type=int, default=64)
    parser.add_argument('--use_topk', type=lambda x:x.lower()=='true', default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser_args()
    path = '/home/wmk/code/LongVideo/datasets/longvideobench/include_frame_idx.json'
    with open(path, 'r') as f:
        data = json.load(f)
    for i in data:
        i['frame_num'] = args.frame_num
        i['use_topk'] = args.use_topk
    with open(path, 'w') as f:
        json.dump(data, f)