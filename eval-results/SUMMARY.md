# Eval results — checkpoint selection for the v2 fine-tune

Source data: `dotelier-eval` Modal volume (7 runs × 4 configs × 12 prompts,
seed 42, H100, the exact pinned serving stack — diffusers 0.39 / torch 2.13).
Runs cover the old fused prod model (`graceyun/dotelier-color`) and every
checkpoint of the new recipe (`lr_0.0001_steps_1500_rank_16_bs_4`, checkpoints
250–1500). Sheets here are downscaled composites built by `make_sheets.py`;
full-res PNGs stay on the volume.

Config key: `deployed_*` = legacy serving template (`a PXCON style icon of: X`),
`train_match_*` = training template (`a PXCON, a 16-bit pixel art icon of X`),
`s<steps>_g<guidance>`.

## Verdict: checkpoint-750, train-match template, 28 steps, guidance 3.5

**Confirms the earlier spot-check recommendation of checkpoint-750.**

Per-checkpoint grading (sheets 01/02, train-match template):

| ckpt | style | generalization | notes |
|------|-------|----------------|-------|
| 250  | weak — soft edges, bilinear blur | good | pixel grid not yet locked in |
| 500  | good | **fails**: "macintosh computer" renders as a white blob creature | coffee cup weak |
| **750** | **good, consistent 16-bit grid** | **best of the run** — all 12 subjects recognizable | best balance |
| 1000 | crisper | degrades: rocket → bird-like creature, snail → featureless blob | overfitting starts |
| 1250 | crisper | same failures as 1000 | |
| 1500 | crispest, most minimal | rocket still creature-ish; snail recovers | style ceiling, semantics worse than 750 |

The trend is the classic fidelity/generalization trade: style crispness rises
monotonically with steps, novel-subject semantics peak at 750 then decay.

## Secondary findings

1. **Template is the whole ballgame for the new run** (sheet 03): with the
   legacy template the new checkpoints emit smooth flat *vector* icons — no
   pixel grid at all. Only the train-match template produces pixel art.
   api.py on this branch already uses the train-match template.
2. **28 steps ≈ 40 steps ≈ 60 steps** at matched settings (sheets 01 vs 02,
   sheet 04): no visible quality gap on any prompt. Latency on H100:
   28 steps ≈ 5.2–5.9 s vs 60 steps ≈ 11.2–12.7 s per 1024px image — half the
   cost for free.
3. **Old fused model at 28/3.5 also holds up** (sheet 04) — if promotion of
   the new LoRA is deferred, dropping prod to 28 steps is still a safe win.
4. **Known regression to eyeball before promoting**: "a cup of coffee with
   steam" renders as a handle-less bowl on *all* new checkpoints (old model
   drew a proper cup). Everything else matches or beats the old model, and
   the new model actually keeps backgrounds white (old model floods the
   lighthouse background blue).
5. Guidance 3.5 (the value the LoRA was trained with) is used for all
   candidate configs; nothing here supports serving at 5.0+.

## Files

- `01_checkpoint_sweep_28steps.png` — old prod model vs ckpt 250→1500, 28 steps
- `02_checkpoint_sweep_40steps.png` — same at 40 steps
- `03_settings_matrix_ckpt_500_750_1000.png` — legacy vs train template, 28 vs 40
- `04_old_model_settings.png` — old fused model settings sweep
- `make_sheets.py` — rebuilds the sheets from a local copy of the volume

## Prod vs staging (added after the staging candidate was chosen)

`05_prod_vs_staging.png` — what prod serves today (old LoRA, 50 steps,
guidance 7.5, train-match template; from run `eval_prod_reference`) against
the staging candidate (ckpt-750, 28 steps, guidance 3.5), same prompts, seed
42, ~2.2x faster per image. Staging is crisper pixel art on 9/12 rows and
fixes two outright prod failures ("a snail on a leaf" renders as a flower in
prod; rocket sits on a blue blob). Prod is still better on the coffee cup
(handle) and arguably the lighthouse.

## Staging smoke test (2026-07-15)

`staging_smoke.png` — "a cactus in a terracotta pot" generated through the
deployed `dotelier-api-staging` container (B200, torch 2.13/CUDA 13 stack):
**3.07s** for 28 steps at 1024px (~9.5 it/s), no kernel issues, clean pixel
grid, white background. Staging endpoint answers with prod-identical auth
behavior (404 /, 401 unauthenticated /generate).

## Hill-climb round A: guidance sweep (2026-07-15, after owner preferred prod)

`06_guidance_sweep.png` — the v1-data checkpoints served at guidance 5.0/7.5
instead of 3.5. Higher guidance recovers much of prod's polish: saturated
colors, confident outlines, and ckpt-750 at g7.5 even draws the coffee cup
with handle + saucer. Trade-offs: night-scene backgrounds flood blue again
and a few rows (puppy, guitar) go soft/painterly at 7.5. Conclusion: the
dull/mushy look was partly a serving-guidance artifact, partly the v1
dataset's anti-aliased edges. The v2-dataset retrain should be evaluated at
g3.5 AND g5.0/7.5.

## Hill-climb diagnosis: the dataset (dataset_audit.py, dataset_v2.py)

- `graceyun/pixel-pngs-dreambooth` (31 imgs, 512px): pristine pixel art
  (5–33 unique colors) but icons fill only ~11–25% of frame.
- `graceyun/dreambooth-pixels` (41 imgs, 1024px, used by the v1 retrain):
  bigger icons, better captions, but many images carry anti-aliased edges
  (hundreds to thousands of unique colors) — the fine-tune learned the mush.
- `graceyun/dreambooth-pixels-v2` (pushed private): all 41 subjects rebuilt —
  pristine pixels where available (30/41), logical-grid re-snap via
  run-length cell detection, integer NEAREST upscale to ~60–85% frame fill,
  clean white 512 canvas, captions carried over.

## Hill-climb round 2: dataset v2.1 (62% fill) — the breakthrough (2026-07-16)

`08_dsv21_lr_00001.png` / `08_dsv21_lr_5e05.png` — retrained on the rebuilt
dataset (pristine pixels, icons at ~62% linear fill with real white margins),
lr sweep 1e-4 vs 5e-5.

- **Winner: lr 1e-4, checkpoint 750–1000, guidance 3.5, 28 steps** — crisp
  uniform pixel cells, white-ish backgrounds, all 12 subjects correct,
  coffee cup has a handle. Degradation starts at ckpt-1250.
- lr 5e-5 is prettier/more illustrated but keeps tinted backgrounds and
  scene shading longer; a "PXCON" caption renders into one image.
- Remaining nit: context-tinted backgrounds on environment-implying prompts
  (rocket→sky, snail→leaf-green). Root cause: training captions never say
  "on a white background". Serving-side fix under eval (white_bg template);
  training-side fix for next round: append the phrase to dataset captions.
