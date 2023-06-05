cd ../sd-scripts
python train_textual_inversion.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --dataset_config=/home/pawel/git/FineTuningSD/squab.toml \
    --output_dir=out_db_sd_scripts \
    --output_name=squab_1_5_ti \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=15000 \
    --learning_rate=1e-6 \
    --optimizer_type="AdamW8bit" \
    --mixed_precision="fp16" \
    --cache_latents \
    --gradient_checkpointing \
    --log_with="wandb" \
    --save_every_n_steps=2000 \
    --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --token_string=houdinisquab --init_word=toy --num_vectors_per_token=16
