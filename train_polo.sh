cd ../sd-scripts
python train_db.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --dataset_config=/home/pawel/git/FineTuningSD/polo.toml \
    --output_dir=out_db_sd_scripts \
    --output_name=polo_1_5_low_lr \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=5000 \
    --learning_rate=1e-7 \
    --optimizer_type="AdamW8bit" \
    --mixed_precision="fp16" \
    --cache_latents \
    --gradient_checkpointing \
    --log_with="wandb" \
    --save_every_n_steps=1000
