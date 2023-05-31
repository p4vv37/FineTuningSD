cd ../sd-scripts
accelerate launch --num_cpu_threads_per_process 1 train_db.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --dataset_config=path/to/toml.toml \
    --output_dir=output_models_dir \
    --output_name=output_checkpoints_name \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=1600 \
    --learning_rate=1e-6 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --cache_latents \
    --gradient_checkpointing
