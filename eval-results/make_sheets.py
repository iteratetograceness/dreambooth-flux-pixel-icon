"""Build curated comparison grid PNGs from the downloaded dotelier-eval volume."""
import os, glob, json
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/volume"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/sheets"
os.makedirs(OUT, exist_ok=True)

CELL = 192
LABEL_H = 28
ROW_LABEL_W = 150

PROMPTS = json.load(open(glob.glob(f"{ROOT}/*/manifest.json")[0]))["prompts"]

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except Exception:
    FONT = FONT_SM = ImageFont.load_default()


def cell_img(run, cfg, idx):
    pattern = f"{ROOT}/{run}/{cfg}/{idx:02d}_*.png"
    fs = glob.glob(pattern)
    if not fs:
        return Image.new("RGB", (CELL, CELL), (40, 40, 40))
    img = Image.open(fs[0]).convert("RGB")
    return img.resize((CELL, CELL), Image.LANCZOS)


def build_grid(cols, rows, out_name, title):
    """cols: list of (label, run, cfg); rows: list of prompt indices"""
    W = ROW_LABEL_W + len(cols) * CELL
    H = LABEL_H * 2 + len(rows) * CELL
    sheet = Image.new("RGB", (W, H), (17, 17, 17))
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), title, fill=(255, 255, 255), font=FONT)
    for c, (label, _, _) in enumerate(cols):
        d.text((ROW_LABEL_W + c * CELL + 4, LABEL_H + 6), label, fill=(180, 220, 255), font=FONT_SM)
    for r, idx in enumerate(rows):
        y = LABEL_H * 2 + r * CELL
        # wrap prompt label
        words = PROMPTS[idx].split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 18:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for li, ln in enumerate(lines):
            d.text((6, y + CELL // 2 - 8 + li * 14), ln, fill=(200, 200, 200), font=FONT_SM)
        for c, (_, run, cfg) in enumerate(cols):
            sheet.paste(cell_img(run, cfg, idx), (ROW_LABEL_W + c * CELL, y))
    path = f"{OUT}/{out_name}"
    sheet.save(path, optimize=True)
    print(f"{path}: {os.path.getsize(path)/1e6:.2f} MB  ({W}x{H})")


BASE = "eval_graceyun_dotelier_color"
CKPT = lambda n: f"eval_lr_0.0001_steps_1500_rank_16_bs_4_checkpoint_{n}"
CKPTS = [250, 500, 750, 1000, 1250, 1500]

# Sheet 1: checkpoint sweep at the candidate serving config (train_match 28/3.5),
# old fused prod model at its old serving config as the leftmost reference.
cols1 = [("OLD MODEL 60/5.0", BASE, "deployed_s60_g5.0")] + [
    (f"ckpt-{n} 28/3.5", CKPT(n), "train_match_s28_g3.5") for n in CKPTS
]
build_grid(cols1, list(range(12)), "01_checkpoint_sweep_28steps.png",
           "Checkpoint sweep - new run (train-match template, 28 steps, guidance 3.5) vs old prod model (60 steps, g5.0)")

# Sheet 2: same sweep at 40 steps (does more compute change the ranking?)
cols2 = [("OLD MODEL 60/5.0", BASE, "deployed_s60_g5.0")] + [
    (f"ckpt-{n} 40/3.5", CKPT(n), "train_match_s40_g3.5") for n in CKPTS
]
build_grid(cols2, list(range(12)), "02_checkpoint_sweep_40steps.png",
           "Checkpoint sweep - train-match template, 40 steps, guidance 3.5")

# Sheet 3: settings matrix for candidate checkpoints 500/750/1000: template + steps
cols3 = []
for n in (500, 750, 1000):
    cols3 += [
        (f"ckpt-{n} legacy-tmpl 28", CKPT(n), "deployed_s28_g3.5"),
        (f"ckpt-{n} train-tmpl 28", CKPT(n), "train_match_s28_g3.5"),
        (f"ckpt-{n} train-tmpl 40", CKPT(n), "train_match_s40_g3.5"),
    ]
build_grid(cols3, list(range(12)), "03_settings_matrix_ckpt_500_750_1000.png",
           "Template & steps matrix - checkpoints 500/750/1000 (legacy template vs train-match, 28 vs 40 steps)")

# Sheet 4: old model settings check (steps reduction + template on old fused model)
cols4 = [
    ("60 steps g5.0 legacy", BASE, "deployed_s60_g5.0"),
    ("28 steps g3.5 legacy", BASE, "deployed_s28_g3.5"),
    ("28 steps g3.5 train-tmpl", BASE, "train_match_s28_g3.5"),
    ("40 steps g3.5 train-tmpl", BASE, "train_match_s40_g3.5"),
]
build_grid(cols4, list(range(12)), "04_old_model_settings.png",
           "Old fused prod model (graceyun/dotelier-color) - settings sweep")
