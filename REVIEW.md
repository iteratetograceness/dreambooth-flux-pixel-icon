# Adversarial Review — dotelier (dreambooth-flux-pixel-icon)

Method: independent review lenses, every finding then challenged by an
adversarial verifier against the actual code.
Status legend: **fixed** = patched on this branch · **plan** = tracked in
`PLAN.md` (needs Modal access, a product decision, or an eval result first) ·
**decision** = deliberate tradeoff for the owner to confirm.

---

## ROUND 2 — audit of the new serving stack (post-"boop" push)

The owner's push replaced the serving architecture: weights baked into the
image at build time (base FLUX.1-dev + unfused LoRA, no more fused
`dotelier-color`), B200 GPU, static in-file config, pinned deps, torchao/
para-attn removed, training-matched prompt template. That rewrite resolved
several Round 1 findings outright (#3 template mismatch, #15 negative-prompt
no-op — dropped, #16 unpinned deps — mostly, #18 Dict reads, #17 partially).
17 raw Round 2 findings, 14 confirmed, 3 refuted.

| # | Finding | Status |
|---|---------|--------|
| R2-1 | **B200 idle cost**: `buffer_containers=1` + `scaledown_window=1800` on a ~$6.25/hr B200 — every isolated request warms 2 containers and each idles 30 min; worst-case steady light traffic approaches ~$9k/month of idle GPU. Web tier also keeps `buffer_containers=2`. | **decision** — deliberate latency tradeoff; confirm the math is worth it or drop buffer to 0 / shrink window |
| R2-2 | **`torch>=2.6.0` floating on B200**: default wheels for 2.6/2.7 lack Blackwell sm_100 kernels ("no kernel image available" → crash-looping billed B200); today it resolves a torch ~2 years newer than the pinned diffusers 0.31.0 was ever tested with. Works by accident. | fixed (pinned `torch==2.9.1`, cu128 wheel — verify one cold start before trusting a rebuild) |
| R2-3 | **guidance_scale 7.5** on guidance-distilled FLUX whose LoRA adapted at 3.5 — served ~2× off-distribution on every request; classic FLUX "burn" territory. Repo now carries three inconsistent values (3.5 trained, 5 in config.py, 7.5 served). | plan (eval matrix compares 7.5 vs 3.5 — pick from the contact sheet) |
| R2-4 | **Leaked bypass secret in git history** (commit 82d359c) — the Vercel protection-bypass value was pushed. | **resolved** — owner rotated it in Vercel and updated the Modal secret; old value is dead |
| R2-5 | **Templated prompt re-validated at 500 chars**: user prompts of 465–500 chars passed the API boundary then crashed inference (template adds 36 chars, InferenceConfig re-ran the validator). Introduced by Round 1's own hardening. | fixed (InferenceConfig overrides `prompt` unconstrained) |
| R2-6 | **Image bakes the full 58GB HF repo**: `snapshot_download` with no filters includes the 24GB single-file BFL checkpoints FluxPipeline never reads; no `revision=` pin either. | fixed (ignore_patterns; revision pin still TODO) |
| R2-7 | **LoRA served unfused**: rank-16 adapter matmuls ran on every projection, all 50 steps, every request — the old architecture served fused weights; the rewrite silently started paying this. | fixed (`fuse_lora()` + `unload_lora_weights()` at startup; numerically equivalent) |
| R2-8 | **Web tier 300s default timeout** vs synchronous cold-start + 50-step generate + serial uploads (persisted from Round 1 #17, worse now with 58GB weight streaming). | fixed stopgap (`timeout=900`); real fix = spawn + poll, in plan |
| R2-9 | **Unused secrets on the internet-facing web function** (write-capable HF token, super-admin-user-id) — persisted from Round 1 #12. | fixed (removed from `fastapi_app`; PixelModel's HF secret left — likely also removable now that weights are baked) |
| R2-10 | **Config drift inside the repo**: `config.py` (60 steps / guidance 5 / suffix system) is imported by nothing; `api.py`'s inline config (50 / 7.5) is what serves. The styles dict in api.py is decorative — token/suffix/negative_prompt are never applied; only a membership check uses it. | plan (single source of truth; let eval decide whether the suffix earns its place) |
| R2-11 | Warmup generation still uses the old prompt template and guidance 5.0 (serving uses 7.5) — harmless for warming kernels, but inconsistent. | plan (align with serving values when guidance is settled) |

Refuted in verification: 2 stale claims describing pre-fix eval.py, and 1
claim that requirements.txt drives any runtime (it doesn't — Modal images are
the spec; it's contributor convenience only).

---

## ROUND 1 — original audit (pre-rewrite architecture)

31 raw findings, 25 confirmed, 6 refuted. Note: findings #3, #12, #15, #16,
#17, #18 were subsequently resolved or superseded by the owner's serving
rewrite and the Round 2 fixes above; #14's "fresh environment can't boot"
concern is now moot (config lives in-file, not in Modal Dicts).

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
