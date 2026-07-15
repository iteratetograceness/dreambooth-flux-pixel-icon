"""Build graceyun/dreambooth-pixels-v2 (private): same 41 subjects/captions as
graceyun/dreambooth-pixels, but crisp and recomposed.

Why: the v1 1024px images carry anti-aliased edges (audit: hundreds of unique
colors vs ~20 in the older pristine 512px set) and the icon sits small in the
frame. The fine-tune faithfully learned the mush. Fix at the source:

  - where the subject also exists in graceyun/pixel-pngs-dreambooth (pristine,
    5-33 colors), use those pixels
  - crop to the icon bounding box, integer NEAREST-upscale to fill ~75-90% of
    a clean white 512 canvas (integer factors keep pixel cells uniform)
  - subjects only in v1 fall back to the 1024px image (cropped, NEAREST)
  - captions carried over from v1

  modal run dataset_v2.py            # build + push + audit summary
"""
import json
from modal import App, Image as ModalImage, Volume, Secret

app = App("dotelier-dataset-v2")

image = ModalImage.debian_slim(python_version="3.12").pip_install(
    "datasets==5.0.0", "Pillow", "numpy", "huggingface_hub==1.23.0"
)

eval_volume = Volume.from_name("dotelier-eval", create_if_missing=True)

CANVAS = 512
TARGET_FILL = 0.88  # linear fraction of canvas the icon's long side should reach


def _bbox(arr):
    import numpy as np
    mask = (arr < 245).any(axis=2)
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _cell_size(arr) -> int:
    """Estimate the logical pixel-cell size of upscaled pixel art via the most
    common horizontal+vertical run length of identical colors."""
    import numpy as np
    from collections import Counter

    counts = Counter()
    for axis in (0, 1):
        a = arr if axis == 1 else arr.transpose(1, 0, 2)
        for line in a[:: max(1, a.shape[0] // 64)]:
            changes = np.where((np.diff(line.astype(int), axis=0) != 0).any(axis=1))[0]
            runs = np.diff(np.concatenate([[-1], changes, [len(line) - 1]]))
            for r in runs:
                if 2 <= r <= 64:
                    counts[int(r)] += 1
    if not counts:
        return 1
    # the true cell size divides most runs; pick the mode
    return counts.most_common(1)[0][0]


def _unscale(img, cell: int):
    """Downsample upscaled pixel art to its logical grid by sampling the
    center of each cell — drops anti-aliased edge pixels entirely."""
    import numpy as np
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    ys = np.arange(cell // 2, h, cell)
    xs = np.arange(cell // 2, w, cell)
    from PIL import Image
    return Image.fromarray(arr[np.ix_(ys, xs)])


def _subject(caption: str) -> str:
    # normalize "a PXCON, a 16-bit pixel art icon of (a|an)? X" -> "x"
    s = caption.split(" icon of ", 1)[-1].strip().lower()
    for art in ("a ", "an ", "the "):
        if s.startswith(art):
            s = s[len(art):]
    return s.strip()


@app.function(
    image=image,
    volumes={"/eval": eval_volume},
    secrets=[Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def build(push_repo: str = "graceyun/dreambooth-pixels-v2") -> dict:
    import io
    import os
    import numpy as np
    from datasets import load_dataset, Dataset
    from PIL import Image

    v1 = load_dataset("graceyun/dreambooth-pixels", split="train")
    pristine = load_dataset("graceyun/pixel-pngs-dreambooth", split="train")

    pristine_by_subject = {}
    for row in pristine:
        pristine_by_subject[_subject(row["text"])] = row["image"].convert("RGB")

    out_rows = []
    report = []
    for row in v1:
        caption = row["text"]
        subj = _subject(caption)
        src = pristine_by_subject.get(subj)
        origin = "pristine-512"
        if src is None:
            src = row["image"].convert("RGB")
            origin = "v1-1024"

        arr = np.asarray(src)
        box = _bbox(arr)
        if box is None:
            report.append({"caption": caption, "origin": origin, "note": "EMPTY — skipped"})
            continue
        crop = src.crop(box)

        # recover the logical pixel grid (drops upscale AA), then re-upscale
        # by a clean integer factor so cells stay uniform
        cell = _cell_size(np.asarray(crop))
        logical = crop
        if cell >= 3:
            logical = _unscale(crop, cell)
        lw, lh = logical.size
        target = int(CANVAS * TARGET_FILL)
        long_side = max(lw, lh)
        if 8 <= long_side <= target:
            factor = max(1, round(target / long_side))
            while factor * long_side > CANVAS:
                factor -= 1
            factor = max(1, factor)
            crop = logical.resize((lw * factor, lh * factor), Image.NEAREST)
        else:
            # grid detection failed or icon already big: plain NEAREST to target
            w, h = crop.size
            scale = target / max(w, h)
            crop = crop.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)

        canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        cw, ch = crop.size
        canvas.paste(crop, ((CANVAS - cw) // 2, (CANVAS - ch) // 2))

        fill = (cw * ch) / (CANVAS * CANVAS)
        ncolors = len(set(canvas.resize((128, 128), Image.NEAREST).getdata()))
        out_rows.append({"image": canvas, "text": caption})
        report.append({"caption": caption, "origin": origin, "cell": cell,
                       "fill": round(fill, 3), "ncolors_128": ncolors})

    ds = Dataset.from_list(out_rows)
    ds.push_to_hub(push_repo, private=True)

    # save report + contact sheet to the eval volume
    out_dir = f"/eval/dataset_audit/{push_repo.replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/build_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cols = 7
    rows_n = (len(out_rows) + cols - 1) // cols
    CW = 160
    sheet = Image.new("RGB", (cols * CW, rows_n * CW), (17, 17, 17))
    for i, r in enumerate(out_rows):
        sheet.paste(r["image"].resize((CW, CW), Image.NEAREST), ((i % cols) * CW, (i // cols) * CW))
    sheet.save(f"{out_dir}/contact_sheet.png", optimize=True)
    eval_volume.commit()

    n_pristine = sum(1 for r in report if r["origin"] == "pristine-512")
    summary = {"n": len(out_rows), "from_pristine": n_pristine,
               "from_v1_1024": len(out_rows) - n_pristine, "repo": push_repo}
    print(json.dumps(summary, indent=2))
    for r in report:
        print(r)
    return summary


@app.local_entrypoint()
def main(push_repo: str = "graceyun/dreambooth-pixels-v2"):
    build.remote(push_repo)
