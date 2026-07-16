# Improvement Plan — dotelier (dreambooth-flux-pixel-icon)

Companion to `REVIEW.md` (adversarial review findings). Phases are ordered by
dependency: nothing in Phase 2+ should ship until Phase 0 reconciles the repo
with what's actually deployed on Modal.

---

## OWNERSHIP MANDATE (2026-07-15)

The owner delegated the fine-tuning + inference-optimization loop end to end.
Goal: an optimized `dotelier-api` (better model, faster inference) to replace
what serves https://github.com/iteratetograceness/dotelier. Runner sessions
(fresh containers with Modal tokens) execute; this plan is their contract.

Pipeline:
1. **Dataset v2** — rebuild from the hub dataset `graceyun/dreambooth-pixels`
   (no local files needed): crop the huge white margins so icons fill ~90%
   of frame, standardize to crisp 512 with NEAREST, push as
   `graceyun/dreambooth-pixels-v2`. Fixes the two dataset findings (blur +
   tiny-centered-icon composition).
2. **Training** — refreshed stack (diffusers v0.39, torch 2.13), explicit
   `--guidance_scale=3.5`, dataset v2, checkpoints every 250 steps. Compare
   against the existing new-recipe run (trained on dataset v1) and the
   original prod LoRA.
3. **Model selection** — eval.py contact sheets across checkpoints; judge
   style fidelity (in-domain rows), generalization (novel-object rows), and
   frame usage. Commit curated sheets to the branch.
4. **Inference speed** — already done: fused LoRA, slimmer baked image, exact
   pins. From eval: pick lowest step count that holds quality (28 ≈ 5.8s vs
   60 ≈ 12.3s per image); align serving guidance with training (3.5).
   Optional second wave: torch.compile on the transformer, re-test
   first-block cache at a conservative threshold on the new stack.
5. **Staging deploy** — deploy the candidate as `dotelier-api-staging`
   (separate Modal app + endpoint label; NEVER overwrite the prod
   `dotelier-api` without explicit owner approval). Owner points the dotelier
   frontend at staging or hits it directly to compare.
   → current (2026-07-16, round 3): `api_staging.py` serves the dataset-v3
   retrain `graceyun/dotelier-pixel-v3-ckpt1000` (fused) at 28 steps /
   guidance 5.0 with the caption-matched template (eval-results
   10_dsv3_round3.png; smoke: rocket + guitar fixed).
   https://iteratetograceness--dotelier-api-staging.modal.run
6. **Promotion** — on owner's "promote": swap `LORA_REPO` + settings in
   api.py (copy the values from api_staging.py, but keep prod's
   buffer_containers/scaledown), deploy to prod, tag the release.

Standing rules for runners: inventory volumes before any GPU work (never redo
completed work), push results incrementally, no prod deploys, no fuse.py, no
PRs.

## Phase 0 — Reconcile deployed state — ✅ LARGELY DONE

The owner pushed the real serving iteration to `main` ("boop", 82d359c):
baked-in weights (base FLUX + unfused LoRA), B200, static in-file config.
PR #2 merged the review branch into it. Remaining:

- [ ] Confirm the deployed `dotelier-api` app matches the merged `main`
      exactly (redeploy from `main` to be sure), and note that redeploying
      picks up the Round 2 fixes (torch pin, fused LoRA, secret scoping,
      timeout) — see REVIEW.md Round 2.
      **2026-07-15 session finding:** prod is healthy (web tier answers,
      auth enforced) but v31 was deployed 2026-02-09 from commit `2ce7277`
      *with uncommitted changes* — that commit is not in this repo, so the
      running prod code is NOT reproducible from git. Promotion of staging
      would also fix this. Also: a redeploy from this branch would have
      failed until now — the refreshed pins (numpy 2.5.1) require python
      3.12 but the image added 3.11 (fixed on this branch).
- [x] Config source of truth: now in `api.py` inline config. Still open:
      `config.py`'s stale constants disagree (60/5 vs 50/7.5) — delete or
      unify (R2-10).
- [x] NOTE for all future sessions: the Modal Dicts (dotelier-inference-config,
      dotelier-styles, dotelier-http-config) are EXPECTED to be empty — Dict
      entries have short TTLs, which is exactly why the owner moved config
      in-file. Empty Dicts are NOT an outage signal. Never reintroduce Dicts
      for config; delete the three Dict objects when convenient to remove the
      confusion entirely.
- [x] Leaked bypass secret rotated in Vercel + Modal (R2-4).
- [ ] **Decision (R2-1)**: B200 `buffer_containers=1` + 30-min scaledown ≈
      up to ~$9k/mo idle under steady light traffic. Confirm intentional or
      tune down.

## Phase 1 — Fine-tune evaluation — ✅ DONE (2026-07-15, see eval-results/)

`eval.py` generates a fixed 12-prompt suite with fixed seeds into the
`dotelier-eval` volume with an HTML contact sheet per run. The volume now
holds runs for the old fused model, every new-run checkpoint (250–1500), and
`eval_prod_reference` (exactly what prod serves today: old LoRA, 50 steps,
guidance 7.5, train-match template). Curated downscaled sheets + grading are
committed in `eval-results/SUMMARY.md`.

- [x] Baseline on the old fused model (`graceyun/dotelier-color`).
- [x] Decisions from the sheets: (a) template — the new checkpoints only
      produce pixel art with the training-matched template (legacy template
      yields smooth vector icons); api.py already uses train-match. (b) steps
      28 ≈ 40 ≈ 60 visually; 28 halves latency (~5.2–5.9s vs ~11.2–12.7s per
      1024px image on H100). (c) guidance 3.5 (on-distribution) is the pick.
- [ ] `--prod-stack` mode (fuse_qkv + autoquant + para-attn cache sweep) —
      dropped for now: the "boop" rewrite removed torchao/para-attn from
      serving, so bare pipeline + fuse_lora IS the prod stack today.

## Phase 2 — Retraining (config fixed in this branch; run needs Modal)

The previous run trained with `train_batch_size=41` on a 41-image dataset for
4200 steps — full-batch gradient descent for ~4200 epochs. The new config in
`diffusers_lora_finetune.py`: batch 4, lr 1e-4 with 100-step warmup, 1500
steps (~146 epochs), rank 16, seed 42, checkpoints every 250 steps,
validation prompts to wandb re-enabled.

- [x] Training run done: checkpoints on the `dreambooth-flux` volume under
      `lr_0.0001_steps_1500_rank_16_bs_4/` (final also pushed to hub).
- [x] Eval of every checkpoint (250–1500) — **checkpoint-750 wins**: style
      locked in, best generalization; 1000+ overfit (rocket → creature,
      snail → blob), 250 still blurry, 500 fails "macintosh computer".
      Details in `eval-results/SUMMARY.md`.
- [x] Winning LoRA published to a private HF repo:
      `graceyun/dotelier-pixel-v2-ckpt750` (`upload_lora.py`). No fuse.py —
      serving fuses at startup (`fuse_lora()`), same as prod.
- [x] Dataset v2 built (2026-07-16, `dataset_v2.py`): pristine pixels via
      logical-grid re-snap + recompose. v2.0 (88% fill) failed — erased the
      white-bg signal, model drifted to illustration style then overfit.
      v2.1 (62% fill) is the shipped version (`graceyun/dreambooth-pixels-v2`).
- [x] Retrained on v2.1, lr sweep 1e-4 vs 5e-5 (dirs `dsv21_lr_*` on the
      volume). Winner: **lr 1e-4 checkpoint-1000**, published as
      `graceyun/dotelier-pixel-v21-ckpt1000`. Serve with the chunky-8-bit
      template + guidance 5.0 (template shootout, eval-results 09).
- [ ] Round-3 retrain (optional): bake the style into captions
      ("an 8-bit pixel art icon of X, thick black outline, flat colors, on a
      white background") so the long serving template isn't needed; also
      targets the occasional soft-focus sample and the handle-less coffee cup.

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
- [x] Prompt template mismatch: resolved — api.py already serves the
      training-matched template, and Phase 1 eval confirmed it's mandatory
      for the new checkpoints (legacy template produces non-pixel output).
- [ ] `negative_prompt` is passed to `FluxPipeline` but FLUX.1-dev distilled
      guidance ignores it unless `true_cfg_scale > 1` (which doubles compute).
      Either drop it or consciously enable true CFG.
- [x] Pin versions in the inference image — done (exact pins, python 3.12;
      torchao/para-attn no longer in the stack).
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
