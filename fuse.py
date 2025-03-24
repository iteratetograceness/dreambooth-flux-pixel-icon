"""
Script to fuse LoRA weights into the base Flux model and upload directly to HuggingFace.
Optimized for H100 GPU inference.
Created with the help of https://github.com/ezyang/codemcp
"""
from modal import Image, App, Secret

app = App("fuse-lora")
huggingface_secret = Secret.from_name("huggingface-secret")

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "peft",
        "huggingface_hub[hf_transfer]",
        "torch",
        "sentencepiece"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

@app.function(
    gpu="H100",
    image=image,
    secrets=[huggingface_secret],
    timeout=30 * 60,
)
def fuse_and_upload():
    import os
    import torch
    from diffusers import FluxPipeline
    from huggingface_hub import login
    
    # Login to HuggingFace
    login(token=os.environ["HF_TOKEN"])
    
    # Load base model with bfloat16 for H100
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # Enable VAE optimizations
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # Load LoRA weights
    pipe.load_lora_weights(
        "graceyun/lr_0.0002_steps_4200_rank_16_031325", 
        weight_name="pytorch_lora_weights.safetensors", 
        adapter_name="color_style"
    )
    
    # Fuse LoRA weights
    pipe.set_adapters(["color_style"])
    pipe.fuse_lora(lora_scale=1.0)
    
    # Unload LoRA weights
    pipe.unload_lora_weights()
    
    # Push to HuggingFace
    pipe.push_to_hub("graceyun/dotelier-color")
    
    return "Successfully fused and uploaded model"

@app.local_entrypoint()
def main():
    result = fuse_and_upload.remote()
    print(result)