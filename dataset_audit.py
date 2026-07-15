"""Audit the fine-tune dataset: image sizes, icon-to-frame ratio, pixel-grid
sharpness, captions. Writes a contact sheet + stats JSON to the dotelier-eval
volume under dataset_audit/.

  modal run dataset_audit.py
  modal run dataset_audit.py --dataset-name graceyun/dreambooth-pixels-v2
"""
import json
from modal import App, Image as ModalImage, Volume, Secret

app = App("dotelier-dataset-audit")

image = ModalImage.debian_slim(python_version="3.12").pip_install(
    "datasets==5.0.0", "Pillow", "numpy", "huggingface_hub==1.23.0"
)

eval_volume = Volume.from_name("dotelier-eval", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/eval": eval_volume},
    secrets=[Secret.from_name("huggingface-secret")],
    timeout=900,
)
def audit(dataset_name: str) -> dict:
    import os
    import numpy as np
    from datasets import load_dataset
    from PIL import Image, ImageDraw, ImageFont

    ds = load_dataset(dataset_name, split="train")
    out_dir = f"/eval/dataset_audit/{dataset_name.replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    stats = []
    thumbs = []
    for i, row in enumerate(ds):
        img = row["image"].convert("RGB")
        arr = np.asarray(img)
        # icon bounding box = non-near-white pixels
        mask = (arr < 245).any(axis=2)
        ys, xs = np.where(mask)
        if len(xs):
            bw, bh = (xs.max() - xs.min() + 1), (ys.max() - ys.min() + 1)
            fill = (bw * bh) / (arr.shape[0] * arr.shape[1])
        else:
            fill = 0.0
        # sharpness proxy: mean abs horizontal gradient on the luma channel,
        # restricted to the icon box (pixel art should have hard steps)
        luma = arr.mean(axis=2)
        grad = np.abs(np.diff(luma, axis=1))
        edge_strength = float(grad[grad > 8].mean()) if (grad > 8).any() else 0.0
        # unique-color count: crisp pixel art is low, anti-aliased art is high
        small = img.resize((128, 128), Image.NEAREST)
        ncolors = len(set(small.getdata()))
        caption = row.get("text", "")
        stats.append({
            "i": i, "size": list(img.size), "icon_fill": round(float(fill), 3),
            "edge_strength": round(edge_strength, 1), "ncolors_128": ncolors,
            "caption": caption,
        })
        thumbs.append((img.resize((160, 160), Image.LANCZOS), caption))

    # contact sheet: 7 cols
    cols = 7
    rows = (len(thumbs) + cols - 1) // cols
    CW, CH = 160, 190
    sheet = Image.new("RGB", (cols * CW, rows * CH), (17, 17, 17))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    for i, (t, cap) in enumerate(thumbs):
        x, y = (i % cols) * CW, (i // cols) * CH
        sheet.paste(t, (x, y))
        d.text((x + 2, y + 162), cap[:30], fill=(200, 200, 200), font=font)
        d.text((x + 2, y + 175), f"fill={stats[i]['icon_fill']} colors={stats[i]['ncolors_128']}",
               fill=(150, 190, 240), font=font)
    sheet.save(f"{out_dir}/contact_sheet.png", optimize=True)

    with open(f"{out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    eval_volume.commit()

    sizes = sorted({tuple(s["size"]) for s in stats})
    fills = [s["icon_fill"] for s in stats]
    summary = {
        "n": len(stats),
        "sizes": [list(s) for s in sizes],
        "icon_fill_min": min(fills), "icon_fill_max": max(fills),
        "icon_fill_mean": round(sum(fills) / len(fills), 3),
        "out_dir": out_dir,
    }
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(dataset_name: str = "graceyun/dreambooth-pixels"):
    audit.remote(dataset_name)
