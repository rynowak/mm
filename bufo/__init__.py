"""bufo — Stable Diffusion LoRA fine-tuning on the *all-the-bufo* emoji set.

A self-contained sample that teaches LoRA fine-tuning of a pretrained diffusion
model. The data pipeline downloads the public bufo emoji corpus, the training
loop adapts a frozen Stable Diffusion UNet with low-rank adapters, and the
sampler generates novel bufos from text prompts.
"""
