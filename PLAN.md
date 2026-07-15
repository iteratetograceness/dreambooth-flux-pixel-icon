# Improvement Plan — dotelier (dreambooth-flux-pixel-icon)

Companion to `REVIEW.md` (adversarial review findings). Phases are ordered by
dependency: nothing in Phase 2+ should ship until Phase 0 reconciles the repo
with what's actually deployed on Modal.

## Phase 0 — Reconcile deployed state — ✅ LARGELY DONE

The owner pushed the real serving iteration to `main` ("boop", 82d359c):
baked-in weights (base FLUX + unfused LoRA), B200, static in-file config.
PR #2 merged the review branch into it. Remaining:

- [ ] Confirm the deployed `dotelier-api` app matches the merged `main`
      exactly (redeploy from `main` to be sure), and note that redeploying
      picks up the Round 2 fixes (torch pin, fused LoRA, secret scoping,
      timeout) — see REVIEW.md Round 2.
- [x] Config source of truth: now in `api.py` inline config. Still open:
      `config.py`'s stale constants disagree (60/5 vs 50/7.5) — delete or
      unify (R2-10).
- [x] Leaked bypass secret rotated in Vercel + Modal (R2-4).
- [ ] **Decision (R2-1)**: B200 `buffer_containers=1` + 30-min scaledown ≈
      up to ~$9k/mo idle under steady light traffic. Confirm intentional or
      tune down.

## Phase 1 — Fine-tune evaluation (needs Modal for GPU; harness is ready)

`eval.py` (new) generates a fixed 12-prompt suite with fixed seeds into the
`dotelier-eval` volume with an HTML contact sheet per run.

- [ ] Baseline: `modal run eval.py` on the current fused model
      (`graceyun/dotelier-color`) — includes prod settings (60 steps, guidance
      5, deployed prompt template) vs cheaper settings and the
      training-matched template.
- [ ] Decide from the sheet: (a) does the train/inference prompt-template
      mismatch cost quality? (b) can steps drop from 60 → 28–40 (roughly
      halving H100 latency/cost)? (c) guidance: the LoRA was trained with the
      guidance **embedding at 3.5** (FLUX-dev is guidance-distilled and the
      launcher never passed `--guidance_scale`) but prod serves at 5.0 — the
      3.5 rows of the matrix are the on-distribution ones.
- [ ] Extend eval.py with a `--prod-stack` mode replicating api.py's exact
      serving stack (fuse_qkv + VAE autoquant + para-attn first-block cache)
      plus a cache-threshold sweep (0.0 / 0.06 / 0.08 / 0.12) — pixel art is
      worst-case content for residual caching and the current 0.12 threshold
      has never been quality-checked against uncached output.

## Phase 2 — Retraining (config fixed in this branch; run needs Modal)

The previous run trained with `train_batch_size=41` on a 41-image dataset for
4200 steps — full-batch gradient descent for ~4200 epochs. The new config in
`diffusers_lora_finetune.py`: batch 4, lr 1e-4 with 100-step warmup, 1500
steps (~146 epochs), rank 16, seed 42, checkpoints every 250 steps,
validation prompts to wandb re-enabled.

- [ ] **Re-process and re-upload the dataset first** (needs the original
      source images, which live on your machine): the `size = img.size`
      shadow bug (now fixed in preprocess.py) meant images were never
      standardized to 512, so training bilinear-blurred the pixel grid.
      While at it, consider raising `icon_scale` from 0.5 (icons occupy only
      ~25% of every training frame — the model learns "small centered sprite
      on white").
- [ ] `modal run diffusers_lora_finetune.py` (~1 H100-hour class job).
      Consider passing `--guidance_scale` explicitly to match the value you
      intend to serve at (see Phase 1c).
- [ ] Eval each checkpoint: `modal run eval.py --checkpoint-dir
      lr_0.0001_steps_1500_rank_16_bs_4/checkpoint-{250..1500}` and pick the
      best by contact sheet (style fidelity on in-domain rows, generalization
      on novel-object rows, no dataset regurgitation).
- [ ] If the winning checkpoint beats prod: update `fuse.py` to point at it,
      fuse, re-run Phase 1 eval on the fused output (fusing + autoquant +
      first-block-cache can shift quality), then swap the serving repo.
- [ ] Dataset follow-ups (optional, next round): captions audit, higher-res
      training (512 → 768/1024 to match 1024 serving), more diverse subjects.

## Phase 3 — API hardening (done in this branch; deploy after Phase 0)

Fixed in `api.py`:
- `num_outputs` bounded 1–4, `prompt` capped at 500 chars, `pixel_id` capped —
  previously an authenticated caller could request unbounded images/prompt on
  an H100.
- Bearer-token check now `hmac.compare_digest` (timing-safe).
- Modal Dict reads moved from import time (pydantic class defaults) to request
  time — config changes no longer require a redeploy to take effect, and the
  module imports without live Dicts.
- `HTTPException` no longer swallowed into a blanket 401; JWKS fetch and
  uploadthing PUT got timeouts; missing-`pipeline` check no longer throws
  `AttributeError`.

Still open (deliberate, needs decisions):
- [ ] Prompt template mismatch: api builds `"a {token} style icon of: X"`,
      training used `"a {token}, a 16-bit pixel art icon of X"`. Decide from
      Phase 1 eval, then update `generate_prompt` (or move the template into
      the style Dict so it ships with the style).
- [ ] `negative_prompt` is passed to `FluxPipeline` but FLUX.1-dev distilled
      guidance ignores it unless `true_cfg_scale > 1` (which doubles compute).
      Either drop it or consciously enable true CFG.
- [ ] Pin versions in the inference image (`diffusers`, `torch`, `torchao`,
      `para-attn`) — today a redeploy silently picks up whatever is latest.
- [ ] Origin-header check is advisory only (any non-browser client can set
      it); fine as CSRF defense-in-depth, but don't count it as auth. Consider
      per-user rate limiting keyed on the JWT `id`.
- [ ] JWT claims: enforce `aud`/`iss` and require `exp` once the values your
      auth server actually mints are known (currently any EdDSA token from the
      JWKS with an `id` claim passes, and exp-less tokens never expire).
- [ ] Drop unused secrets from `fastapi_app` (`huggingface-secret` — a
      write-capable token that can push the prod model — and
      `super-admin-user-id`) after confirming the deployed code doesn't use
      them.
- [ ] Set an explicit `timeout=` on `fastapi_app` (Modal default 300s is
      shorter than a cold-start + 60-step generate + upload chain), or move to
      spawn + poll.
- [ ] Pin `revision=<hf commit>` in `from_pretrained("graceyun/dotelier-color")`
      so a re-run of fuse.py (which still hardcodes the old overtrained LoRA)
      can't silently swap prod weights; make fuse.py take its LoRA source as a
      parameter.

## Phase 4 — Ops hygiene

- [ ] Delete or clearly quarantine dead weight: `training_script.py` (vendored
      copy of the diffusers example script; verified byte-identical to
      upstream at the pinned SHA except one autocast line — and the Modal job
      runs the checkout's copy, not this file).
- [ ] Revisit `scaledown_window=1200` (20 min of idle H100 ≈ real money) once
      cold-start time is measured; if cold start is the reason, measure how
      much the `@enter` warmup (full pipeline load + autoquant + 5-step
      generation) actually costs and consider memory snapshots / AOT path.
- [ ] Add a smoke test (`modal run api.py::PixelModel(dev=true)` path) and a
      CI job that at minimum compiles + deploys to a staging label.
- [ ] Document the secret/Dict/volume inventory in the README so the Modal
      side of the system is reconstructible from git.
