python train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --max_seq_length 1024 \ 
    --dataset_name timdettmers/openassistant-guanaco \
    --dataset_text_field text
    --double_quant \
    --quant_dtype nf4 \
    --bits 4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_memory_MB 15000 \ 
    --save_model_dir save_model_dir \
    --output_dir output \
    --optim paged_adamw_32bit
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --weight_decay 0.0\
    --learning_rate 2e-4 \
    --max_grad_norm 0.3 \
    --gradient_checkpointing \
    --do_train \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_steps 1\
    --save_steps 1\
    --save_total_limit 2\
