#!/bin/bash
export WANDB_ENTITY=$1
export WANDB_PROJECT=$2

run_name="dpo-fts-llama2"
sft_model_checkpoint=$3
config_file="dpo/config_dpo.yaml"
output_dir="dpo/checkpoints/${run_name}"
data_path="dpo/data"

accelerate launch \
    --config_file $config_file dpo/run_dpo.py \
    --model_name_or_path=${sft_model_checkpoint} \
    --data_dir=$data_path \
    --use_local 1 \
    --output_dir $output_dir \
    --logging_steps 5 \
    --max_steps 1500 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --max_length 2048 \
    --report_to wandb \
    --eval_steps 200 \
    --run_name $run_name
