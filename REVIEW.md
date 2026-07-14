# Adversarial Review — dotelier (dreambooth-flux-pixel-icon)

Method: 4 independent review lenses (security, Modal correctness, ML/fine-tuning
methodology, ops/cost/drift), every finding then challenged by an adversarial
verifier against the actual code — 31 raw findings, 25 confirmed, 6 refuted.
Status legend: **fixed** = patched on this branch · **plan** = tracked in
`PLAN.md` (needs Modal access, a product decision, or an eval result first).

## Model quality (the fine-tune itself)

| # | Finding | Status |
|---|---------|--------|
| 1 | **Full-batch overtraining**: previous run used `train_batch_size=41` on the 41-image dataset for 4200 steps — full-batch gradient descent for ~4200 epochs. The deployed model was fused from this run. | fixed (new recipe: batch 4, lr 1e-4 + warmup, 1500 steps, checkpoints every 250) — rerun is plan Phase 2 |
| 2 | **Guidance mismatch**: FLUX.1-dev is guidance-distilled — guidance is a conditioning *input*. Training never passed `--guidance_scale`, so the LoRA adapted under the script default **3.5**; production serves at **5.0**. The model is systematically served off its training distribution. | plan (eval matrix probes 3.5 vs 5.0; pick from results) |
| 3 | **Prompt template mismatch**: training captions are `a PXCON, a 16-bit pixel art icon of X`; the API builds `a PXCON style icon of: X` — different token context, no "16-bit pixel art", a colon never seen in training. | plan (eval matrix A/Bs both templates) |
| 4 | **`preprocess.py` size-shadow bug**: `size = img.size` overwrote the 512×512 target, so dataset images were never standardized; training then bilinear-resized ~1024px images to 512, anti-aliasing the hard pixel edges that define the style — while the code deliberately used NEAREST to preserve them. | fixed (needs dataset re-process + re-upload before next training run) |
| 5 | **Composition prior**: every training image has the icon at 50% scale centered on white (~25% of the frame), so generations render small centered sprites and waste most of the 1024px output. | plan (recompose dataset at ~85–95% scale, retrain, compare) |
| 6 | **Resolution gap**: trained exclusively at 512, served exclusively at 1024 — pixel-art grid regularity is resolution-sensitive. | plan (eval; consider 1024 training or serve-512-upscale) |
| 7 | **No evaluation anywhere**: validation prompts were commented out of training; no checkpoint comparison; final (most overtrained) weights were fused and shipped. | fixed (`eval.py` harness + validation re-enabled + 250-step checkpoints) |
| 8 | **Eval/prod stack mismatch**: prod serves through fuse→autoquant VAE→para-attn first-block-cache (threshold 0.12 — aggressive; pixel art is worst-case content for residual caching). A bare-pipeline eval can't see what caching does to the pixel grid. | plan (add `--prod-stack` flag + threshold sweep to eval.py) |

## Security

| # | Finding | Status |
|---|---------|--------|
| 9 | JWT validation enforces only "valid EdDSA signature from any JWKS key + has `id` claim": `verify_aud` disabled, no issuer check, no required-claims list — a token with no `exp` never expires; any token minted by the same key for another purpose passes. | plan (needs the real aud/iss claim values your auth mints) |
| 10 | JWKS cache had no TTL — a rotated/revoked signing key kept validating for the life of a warm container. | fixed (15-min TTL) |
| 11 | Bearer token compared with `==` (timing-unsafe); unbounded `num_outputs`/`prompt` let any authorized caller buy arbitrary H100 time per request. | fixed (compare_digest; num_outputs 1–4, prompt ≤500 chars) |
| 12 | Internet-facing `fastapi_app` mounts secrets it never uses — including the **write-capable HF token** (same one that can push the prod model) and `super-admin-user-id` (referenced nowhere in the repo). Container compromise = model-supply-chain compromise. | plan (verify deployed code doesn't use them, then drop) |
| 13 | Origin-header gate is spoofable — but it sits in front of real bearer+JWT auth, so it's acceptable defense-in-depth. | refuted by verifier (no action) |

## Modal app correctness

| # | Finding | Status |
|---|---------|--------|
| 14 | `config.py` bootstrap is gutted: only `DEFAULT_STYLE` is written; every other key `api.py` hard-requires (steps, guidance, styles, origins, bypass secret) is commented out and `STYLE_CONFIGS = {}` — a fresh Modal workspace cannot boot, and the only working config lives in unversioned production Dicts. | plan Phase 0 (dump live Dicts → commit as source of truth; needs Modal access) |
| 15 | `negative_prompt` is a **silent no-op**: FluxPipeline only uses it when `true_cfg_scale > 1`, which is never passed. The per-style negative prompts in the styles Dict do nothing. | plan (drop it, or enable true CFG at ~2× compute — eval first) |
| 16 | Serving image installs `diffusers`/`torch`/`torchao`/`para-attn` fully unpinned, while unpickling a torchao `AUTOQUANT_CACHE` produced under a *pinned* environment — any redeploy can resolve incompatible versions and crash-loop every cold start (`@enter` re-raises). | plan (pin exact versions; regenerate quant cache under the pins) |
| 17 | `/generate` blocks synchronously through cold start (full pipeline load + autoquant + warmup gen + 60-step inference + uploads) under Modal's default 300s web timeout. | plan (explicit timeout or spawn+poll flow) |
| 18 | Modal Dict reads as pydantic class defaults froze config at deploy time and made the module unimportable without live Dicts. | fixed (request-time reads) |
| 19 | `pipeline` attribute check could `AttributeError`; `HTTPException` was remapped to a blanket 401; no HTTP timeouts on JWKS/upload calls. | fixed |

## Ops / cost / reproducibility

| # | Finding | Status |
|---|---------|--------|
| 20 | `aot.py` no longer even imports under current Modal client (`concurrency_limit` removed) — and it's the **only** writer of the quant cache prod depends on. | fixed (`max_containers`) |
| 21 | `optimize_transformer` referenced undefined `reference_outputs`/`max_diff_fused` — every multi-hour H100 compile run ended in a swallowed NameError and returned `success: False`; the numerical validation never ran. | fixed |
| 22 | AOT-compiled `.pt2` artifacts are consumed by nothing in the serving path — 3-hour H100 compile jobs produce dead weight (prod uses runtime autoquant + fbcache instead). | plan (wire in or explicitly quarantine as experiment) |
| 23 | `fuse.py` pushes over the exact HF repo prod loads with **no revision pin** — re-running it (it still hardcodes the old overtrained LoRA) silently replaces prod weights with no deploy and no rollback. | plan (pin `revision=` in api.py; parameterize fuse source) |
| 24 | `training_script.py` (77KB) is dead code — training runs the diffusers checkout's copy; edits here silently do nothing. `codemcp.toml` contradicts itself about which file is the training script. Verified byte-identical to upstream at the pinned SHA except one autocast line. | plan (delete) |
| 25 | Folder names drifted from actual sweep params (`steps_4200` folders, 4100 in config); non-idempotent `preprocess.py` overwrote sources in place and pushed to the hub on *import*. | fixed (derived dir names; processed/ subdir; `__main__` guard) |
| 26 | No CI, no tests, no smoke test — first execution of any change is a paid H100 cold start in prod. The `dev=true` canned-image path exists but nothing invokes it. | plan (import-check CI + staging deploy + dev-path smoke test) |
| 27 | 20-min H100 `scaledown_window` — deliberate latency/cost tradeoff, not a bug. | refuted by verifier (revisit only after cold-start is measured) |

## Refuted findings (kept for the record)

Six claims were killed in verification: the Origin-header and scaledown findings
above, plus four claims that described pre-fix code (import-time Dict defaults,
unbounded `num_outputs`, `AttributeError` guard, HTTPException-to-401) — those
defects were real but already patched on this branch before verification ran.
