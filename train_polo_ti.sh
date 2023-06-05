cd ../sd-scripts
python train_textual_inversion.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --dataset_config=/home/pawel/git/FineTuningSD/polo.toml \
    --output_dir=out_db_sd_scripts \
    --output_name=polo_1_5_ti \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=10000 \
    --learning_rate=5e-6 \
    --optimizer_type="AdamW8bit" \
    --mixed_precision="fp16" \
    --cache_latents \
    --gradient_checkpointing \
    --log_with="wandb" \
    --save_every_n_steps=2000 \
    --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --token_string=pologt40 --init_word=car --num_vectors_per_token=16 \
    --use_object_template 
