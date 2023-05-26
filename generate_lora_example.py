from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionPipeline
import torch

lora_model_id = "out_model_lora/checkpoint-1500"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)
images = pipe("A picture of a  swimming under water, professional, highly detailed, in style of picasso ",
              num_inference_steps=60,
              negative_prompt="wood, table")
for num, image in enumerate(images.images):
    image.save(F"examples/dreambooth_lora/gen_400_{num}.png")