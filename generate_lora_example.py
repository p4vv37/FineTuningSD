from diffusers.loaders import AttnProcsLayers
from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch

lora_model_id = "out_model_lora_polo_txt"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
                                               safety_checker=None)
pipe = pipe.to("cuda")
# pipe.unet.load_attn_procs(lora_model_id)
pipe.load_lora_weights(lora_model_id)

images = pipe("A photo of sks car on a race track",
              num_inference_steps=60,
              negative_prompt="",
              #cross_attention_kwargs={"scale": 1.0},
              num_images_per_prompt=4
              ).images
for num, image in enumerate(images):
    image.save(F"examples/dreambooth_lora/gen_no_txt_{num}.png")