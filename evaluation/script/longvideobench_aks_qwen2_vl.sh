base_score_path=/home/wmk/code/LongVideo/selected_frame
score_type=longvideobench_clip_frames
dataset_name=longvideobench
base_anno_path=/home/wmk/code/LongVideo/datasets

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --score_type $score_type \
    --dataset_name $dataset_name \
    --base_anno_path $base_anno_path

frame_num=32
use_topk=True

python ./evaluation/insert_frame_num.py \
    --frame_num $frame_num \
    --use_topk $use_topk 

accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=/home/share/pyz/model_weight/Qwen2-VL-7B-Instruct/,use_topk=True,nframes=32 \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_7b \
    --output_path  /home/wmk/code/LongVideo/result/longvideobench/qwen2_vl_aks_32