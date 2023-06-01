cd ../sd-scripts
accelerate launch --num_cpu_threads_per_process 1 train_db.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --dataset_config=/home/pawel/git/FineTuningSD/polo.toml \
    --output_dir=out_db_sd_scripts \
    --output_name=out_db_sd_scripts \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=1600 \
    --learning_rate=1e-6 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --cache_latents \
    --gradient_checkpointing
