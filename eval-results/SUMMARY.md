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
