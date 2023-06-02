export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./train_polo"
export OUTPUT_DIR="./out_model_lora_polo_txt"
export CLASS_DATA_DIR="./reg_polo"

python train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DATA_DIR\
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks car" \
  --class_prompt="a photo of car" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20000 \
  --validation_prompt="A photo of sks car on a race track" \
  --validation_epochs=40 \
  --seed="0" \
  --train_text_encoder
