# FineTuningSD
Playing with fine tuning SD.
Object photo example:

![example](train_squab/20230524_101158.jpg)

Examples of source data:

1) LORA Dreambooth based on [example made by Huggingface](https://huggingface.co/docs/diffusers/main/en/training/lora)<br />
LR: 1e-4, train steps: 15000 time: 10h 39 min<br />
The network generates great images, but model is not able to generate new content, e.g. "toy swimming in water". It is a great generator of images that look exactly as training examples. It is able to generate something new if scale in cross_attention_kwargs would is lowered, but this is not possible, while the trained text encoder was loaded.
<br /> Results:<br /> ![example](examples/dreambooth_lora/2.png)![example](examples/dreambooth_lora/3.png)![example](examples/dreambooth_lora/gen0.png)

2) Same, but disabled text encoder fine-tuning<br />
LR: 5e-5, train steps: 19600 time: 12h 12 min <br />
prompt: "A photo of an sks toy flying in space, on orbit, professional, highly detailed, national geographic"<br />
Skipping the training of text encoder and loading only u-net makes it possible to generate some fun results. They are way less correct and more deformed. Maybe this is due to abstract nature of the toy, next step is to test it with something less... unusual.
<br /> Results:<br /> ![example](examples/dreambooth_lora/gen_no_txt1.png)
![example](examples/dreambooth_lora/gen_no_txt2.png)
![example](examples/dreambooth_lora/gen_no_txt3.png)
![example](examples/dreambooth_lora/gen_no_txt4.png)
