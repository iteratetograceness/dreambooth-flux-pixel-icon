"""One-shot: publish a training checkpoint's LoRA from the dreambooth-flux
volume to a private HF repo, so serving images can bake it in by repo name.

Usage:
  modal run upload_lora.py --checkpoint-dir lr_0.0001_steps_1500_rank_16_bs_4/checkpoint-750 \
      --repo-id graceyun/dotelier-pixel-v2-ckpt750
"""
from modal import App, Image as ModalImage, Volume, Secret

app = App("dotelier-upload-lora")

image = ModalImage.debian_slim(python_version="3.12").pip_install("huggingface_hub==1.23.0")


@app.function(
    image=image,
    volumes={"/dreambooth-flux": Volume.from_name("dreambooth-flux")},
    secrets=[Secret.from_name("huggingface-secret")],
    timeout=600,
)
def upload(checkpoint_dir: str, repo_id: str) -> str:
    import os
    from huggingface_hub import HfApi

    src = f"/dreambooth-flux/{checkpoint_dir}/pytorch_lora_weights.safetensors"
    assert os.path.exists(src), f"missing {src}"

    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=src,
        path_in_repo="pytorch_lora_weights.safetensors",
        repo_id=repo_id,
    )
    readme = (
        f"---\nbase_model: black-forest-labs/FLUX.1-dev\ntags:\n- lora\n- flux\n---\n"
        f"# dotelier pixel-icon LoRA v2\n\n"
        f"Source: `dreambooth-flux` Modal volume, `{checkpoint_dir}`.\n"
        f"Recipe: lr 1e-4, batch 4, rank 16, guidance embedding 3.5, seed 42.\n"
        f"Serve with the training-matched template "
        f"`a PXCON, a 16-bit pixel art icon of {{prompt}}` at guidance 3.5, 28 steps.\n"
    )
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo_id)
    url = f"https://huggingface.co/{repo_id}"
    print(f"uploaded -> {url}")
    return url


@app.local_entrypoint()
def main(
    checkpoint_dir: str = "lr_0.0001_steps_1500_rank_16_bs_4/checkpoint-750",
    repo_id: str = "graceyun/dotelier-pixel-v2-ckpt750",
):
    print(upload.remote(checkpoint_dir, repo_id))
