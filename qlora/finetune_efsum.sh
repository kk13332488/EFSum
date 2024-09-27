export OMP_NUM_THREADS=6
export WANDB_ENTITY=$1
export WANDB_PROJECT=$2
dataset_name='qlora/data/train.json'
base_model_name="meta-llama/Llama-2-7b-hf"
model_last_name="meta-llama-2-7b-hf"
model_path_to_be_saved="qlora/checkpoints/meta-llama-2-7b-hf"
export WANDB_NAME=$model_path_to_be_saved
accelerate launch  \
    --config_file qlora/accelerate_config.yaml qlora/qlora.py \
    --output_dir ${model_path_to_be_saved} \
    --model_name_or_path $base_model_name \
    --use_auth \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 100 \
    --save_total_limit 50 \
    --evaluation_strategy steps \
    --eval_dataset_size 1000 \
    --max_eval_samples 2000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 2048 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --max_steps 4000 \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset $dataset_name \
    --source_max_len 2048 \
    --target_max_len 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --eval_steps  100\
    --learning_rate 2e-4 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --dataset_format 'input-output'\
    --train_on_source False \
    --do_predict False \
    --report_to wandb \
    # --predict_with_generate True 