export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./train_squab"
export OUTPUT_DIR="./out_model_lora"
export CLASS_DATA_DIR="./reg_squab/1_toy"

python train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DATA_DIR\
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks toy" \
  --class_prompt="a photo of toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --checkpointing_steps=200 \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=30000 \
  --validation_prompt="A photo of sks toy on a table" \
  --validation_epochs=40 \
  --seed="0" \
