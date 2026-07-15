"""
Evaluation harness for the dotelier pixel-icon fine-tune.

Generates a fixed prompt suite with fixed seeds across any combination of:
  - model source: fused repo (e.g. graceyun/dotelier-color) OR base FLUX.1-dev
    + a LoRA repo / local checkpoint dir on the dreambooth-flux volume
  - num_inference_steps / guidance_scale
  - prompt template (training-matched vs the deployed api.py template)

Outputs go to the `dotelier-eval` volume as PNGs plus an index.html contact
sheet per run, so runs are directly comparable side by side.

Usage:
  modal run eval.py                          # default comparison matrix on the fused prod model
  modal run eval.py --lora-repo graceyun/lr_0.0001_steps_1500_rank_16_bs_4
  modal run eval.py --checkpoint-dir lr_0.0001_steps_1500_rank_16_bs_4/checkpoint-750

Fetch results:
  modal volume get dotelier-eval <run_name>
"""
import json
from modal import App, Image as ModalImage, Volume, Secret

app = App("dotelier-eval")

image = (
    ModalImage.debian_slim(python_version="3.11")
    .pip_install(
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "peft",
        "sentencepiece",
        "torch",
        "huggingface_hub[hf_transfer]",
        "Pillow",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_CACHE": "/cache",
    })
)

eval_volume = Volume.from_name("dotelier-eval", create_if_missing=True)
train_volume = Volume.from_name("dreambooth-flux", create_if_missing=True)

# Fixed suite: first rows are close to the training distribution, later rows
# probe generalization (novel objects, composition, detail).
PROMPT_SUITE = [
    "a floppy disk",
    "a brown puppy",
    "a red mushroom",
    "a treasure chest",
    "a macintosh computer",
    "an hourglass",
    "a rocket ship launching",
    "a stack of pancakes with syrup",
    "a purple electric guitar",
    "a lighthouse at night",
    "a cup of coffee with steam",
    "a snail on a leaf",
]

# "prod" mirrors api.py's current generate_prompt (training-matched, no
# suffix); "white_bg" adds the background phrase most training captions
# carried; "legacy" is the old serving template kept for reference.
TEMPLATES = {
    "prod": "a {token}, a 16-bit pixel art icon of {prompt}",
    "white_bg": "a {token}, a 16-bit pixel art icon of {prompt}, on a white background",
    "legacy": "a {token} style icon of: {prompt}",
}

# what production actually serves today (api.py): base FLUX.1-dev + this LoRA,
# loaded unfused, 50 steps, guidance 7.5
PROD_LORA_REPO = "graceyun/lr_0.0002_steps_4200_rank_16_031325"

DEFAULT_SEED = 42
TOKEN = "PXCON"


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/eval": eval_volume,
        "/dreambooth-flux": train_volume,
    },
    secrets=[Secret.from_name("huggingface-secret")],
    timeout=2 * 60 * 60,
)
def evaluate(
    run_name: str,
    configs_json: str,
    model_repo: str = "graceyun/dotelier-color",
    lora_repo: str = "",
    checkpoint_dir: str = "",
):
    """Load one model source, then sweep configs over the prompt suite.

    configs_json: JSON list of {"steps": int, "guidance": float, "template": str}
    lora_repo / checkpoint_dir switch the source to base FLUX.1-dev + LoRA.
    """
    import os
    import time
    import torch
    from diffusers import FluxPipeline

    configs = json.loads(configs_json)

    use_lora = bool(lora_repo or checkpoint_dir)
    base = "black-forest-labs/FLUX.1-dev" if use_lora else model_repo
    print(f"loading {base} (lora_repo={lora_repo!r} checkpoint_dir={checkpoint_dir!r})")

    pipe = FluxPipeline.from_pretrained(base, torch_dtype=torch.bfloat16).to("cuda")

    if use_lora:
        if checkpoint_dir:
            path = f"/dreambooth-flux/{checkpoint_dir}"
            pipe.load_lora_weights(path)
        else:
            pipe.load_lora_weights(lora_repo, weight_name="pytorch_lora_weights.safetensors")

    out_root = f"/eval/{run_name}"
    os.makedirs(out_root, exist_ok=True)
    manifest = {
        "model_repo": model_repo,
        "lora_repo": lora_repo,
        "checkpoint_dir": checkpoint_dir,
        "configs": configs,
        "prompts": PROMPT_SUITE,
        "seed": DEFAULT_SEED,
        "results": [],
    }

    for cfg in configs:
        template = TEMPLATES[cfg["template"]]
        cfg_key = f"{cfg['template']}_s{cfg['steps']}_g{cfg['guidance']}"
        cfg_dir = f"{out_root}/{cfg_key}"
        os.makedirs(cfg_dir, exist_ok=True)

        for i, subject in enumerate(PROMPT_SUITE):
            prompt = template.format(token=TOKEN, prompt=subject)
            generator = torch.Generator("cuda").manual_seed(DEFAULT_SEED)
            start = time.time()
            img = pipe(
                prompt,
                num_inference_steps=cfg["steps"],
                guidance_scale=cfg["guidance"],
                height=1024,
                width=1024,
                generator=generator,
            ).images[0]
            elapsed = time.time() - start

            slug = subject.replace(" ", "_")[:40]
            fname = f"{i:02d}_{slug}.png"
            img.save(f"{cfg_dir}/{fname}")
            manifest["results"].append(
                {"config": cfg_key, "prompt": prompt, "file": f"{cfg_key}/{fname}", "seconds": round(elapsed, 2)}
            )
            print(f"[{cfg_key}] {subject}: {elapsed:.1f}s")

    with open(f"{out_root}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    _write_contact_sheet(out_root, [c for c in manifest["results"]], configs, run_name)
    eval_volume.commit()
    print(f"done — fetch with: modal volume get dotelier-eval {run_name}")
    return manifest


def _write_contact_sheet(out_root: str, results: list, configs: list, run_name: str):
    """rows = prompts, cols = configs — one glance shows the tradeoffs."""
    cfg_keys = [f"{c['template']}_s{c['steps']}_g{c['guidance']}" for c in configs]
    by_cfg = {k: [r for r in results if r["config"] == k] for k in cfg_keys}
    n_rows = max((len(v) for v in by_cfg.values()), default=0)

    cells = []
    header = "".join(f"<th>{k}</th>" for k in cfg_keys)
    for row in range(n_rows):
        tds = []
        for k in cfg_keys:
            rs = by_cfg[k]
            if row < len(rs):
                r = rs[row]
                tds.append(
                    f'<td><img src="{r["file"]}" width="256" loading="lazy">'
                    f'<div class="cap">{r["prompt"]}<br>{r["seconds"]}s</div></td>'
                )
            else:
                tds.append("<td></td>")
        cells.append(f"<tr>{''.join(tds)}</tr>")

    html = (
        f"<title>{run_name}</title>"
        "<style>body{font-family:monospace;background:#111;color:#eee}"
        "table{border-collapse:collapse}td,th{padding:6px;border:1px solid #333;vertical-align:top}"
        ".cap{max-width:256px;font-size:11px;color:#aaa}</style>"
        f"<h1>{run_name}</h1><table><tr>{header}</tr>{''.join(cells)}</table>"
    )
    with open(f"{out_root}/index.html", "w") as f:
        f.write(html)


@app.local_entrypoint()
def main(
    run_name: str = "",
    model_repo: str = "black-forest-labs/FLUX.1-dev",
    lora_repo: str = PROD_LORA_REPO,
    checkpoint_dir: str = "",
):
    # Default matrix: what prod serves today (50 steps / guidance 7.5 / prod
    # template on base+unfused LoRA) vs the guidance the LoRA was actually
    # trained at (3.5, via the training script's default guidance embedding)
    # and cheaper step counts.
    configs = [
        {"steps": 50, "guidance": 7.5, "template": "prod"},      # prod today
        {"steps": 50, "guidance": 3.5, "template": "prod"},      # on-distribution guidance
        {"steps": 28, "guidance": 3.5, "template": "prod"},      # cheap
        {"steps": 50, "guidance": 3.5, "template": "white_bg"},
    ]

    if not run_name:
        source = checkpoint_dir or lora_repo or model_repo
        run_name = "eval_" + source.replace("/", "_").replace("-", "_")

    manifest = evaluate.remote(
        run_name=run_name,
        configs_json=json.dumps(configs),
        model_repo=model_repo,
        lora_repo=lora_repo,
        checkpoint_dir=checkpoint_dir,
    )
    print(f"generated {len(manifest['results'])} images")
    print(f"fetch: modal volume get dotelier-eval {run_name}")
