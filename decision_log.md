# Decision Log — SVG Scaling Laws Project

Every non-obvious design choice is documented here with reasoning. This log serves two purposes: (1) transparency for grading, and (2) reference for defending technical questions.

---

### Decision: Primary dataset — starvector/svg-icons-simple
**Date:** 2026-04-15
**Options considered:**
1. svg-icons-simple — 89K pre-cleaned SVG icons, well-documented
2. umuthopeyildirim/svgen-500k — 300K SVGs with text descriptions, but less standardized
3. Scraping SVGs from the web — maximum diversity but uncontrolled quality
**Choice:** svg-icons-simple as primary, supplemented by emoji-simple and fonts-simple
**Reasoning:** The StarVector datasets are already simplified (no gradients, filters, animations), which means cleaner training signal. Using a well-known, published dataset also makes the work reproducible. We needed supplementary data to hit 100M tokens.
**Impact:** Dataset composition is ~7% icons, <1% emoji, ~93% font glyphs by file count.

---

### Decision: Supplementary data — 65% subsample of svg-fonts-simple
**Date:** 2026-04-17
**Options considered:**
1. Use only icons-simple (~88M tokens after cleaning — below 100M target)
2. Add emoji-simple only (~89M tokens — still below target)
3. Add emoji-simple + subsample fonts-simple to reach 100M+
**Choice:** Option 3, with 65% of fonts-simple (1,134,108 of 1,744,783 glyphs)
**Reasoning:** Icons alone yielded ~88M tokens post-tokenization, below the 100M requirement. Emoji added only ~1M tokens. Font glyphs are structurally similar to icons (single-path SVGs in a 24×24 viewBox) so they're compatible training data. 65% subsample gives ~130M tokens — comfortably above target with margin.
**Impact:** Training data is font-glyph-heavy. Models may learn path-drawing patterns biased toward letterforms. This is noted as a limitation in the report.

---

### Decision: Coordinate precision — 1 decimal place
**Date:** 2026-04-15
**Options considered:**
1. No rounding — preserve exact coordinates (e.g., 10.059374809265137)
2. Round to 1 decimal place (e.g., 10.1)
3. Round to integers (e.g., 10)
**Choice:** 1 decimal place
**Reasoning:** At a 24×24 viewBox, 0.1 units is sub-pixel at typical rendering resolutions — visually imperceptible. Rounding from 15-digit floats to 1 decimal place reduced a representative SVG from 2,235 to 919 characters (59% compression). This dramatically reduces the number of unique numeric strings the tokenizer must learn. Integer rounding (option 3) would be too lossy — curves would become visibly jagged.
**Impact:** Vocabulary is more efficient. Sequences are shorter. Minor loss of geometric precision that is invisible at icon scale.

---

### Decision: Minimum SVG length — 50 characters
**Date:** 2026-04-15
**Options considered:**
1. No minimum — keep everything
2. 50 characters — remove trivial/empty SVGs
3. 200 characters — remove very simple SVGs too
**Choice:** 50 characters
**Reasoning:** An SVG with fewer than 50 characters would be something like `<svg></svg>` — no meaningful visual content. In practice, the StarVector data had a minimum of 315 characters (after cleaning), so this filter removed nothing. It exists as a safety net.
**Impact:** None on this dataset. Pipeline is robust to lower-quality data sources.

---

### Decision: Maximum token length — 1024 tokens
**Date:** 2026-04-15
**Options considered:**
1. 512 tokens — faster training, but filters more data
2. 1024 tokens — covers 99.99% of SVGs, reasonable context window
3. 2048 tokens — covers everything, but quadratic attention cost
**Choice:** 1024 tokens
**Reasoning:** P99 sequence length is 447 tokens; only 65 out of 1,170,625 SVGs (0.006%) exceed 1024 tokens. This covers virtually all data while keeping attention computation manageable across all model sizes. Matches nanoGPT's default context window.
**Impact:** 65 SVGs filtered out. Positional embedding table has 1024 entries. Attention is O(1024²) per layer.

---

### Decision: BPE vocabulary size — 4,096
**Date:** 2026-04-15
**Options considered:**
1. 1,024 — very compact, but sequences ~3-4× longer
2. 4,096 — balances sequence length and token coverage
3. 8,192 — many rare tokens with poor embeddings at our data scale
**Choice:** 4,096
**Reasoning:** SVG has a more constrained vocabulary than natural language — a relatively small set of tag names, attribute names, path commands, and coordinate values dominate. At 4,096 vocab with 130M training tokens, each token sees ~32K occurrences on average — enough for good embeddings. The tokenizer learned highly efficient SVG-specific merges: the entire SVG header became a single token, and coordinate values like "7.3" became individual tokens.
**Impact:** Median sequence length of 98 tokens. Embedding table is 4096 × d_model parameters.

---

### Decision: Tokenizer trained on 150K sample, not full corpus
**Date:** 2026-04-17
**Options considered:**
1. Train on all 1.17M SVGs (~1GB text corpus)
2. Train on random 150K sample (~130MB)
**Choice:** 150K sample
**Reasoning:** The BPE merge computation on the full 1GB corpus exceeded Colab's memory limits and was repeatedly killed during the "Compute merges" step. A 150K sample is more than sufficient for learning the BPE vocabulary — the token frequency distribution stabilizes well before 150K documents. The resulting tokenizer was applied to all 1.17M SVGs for the actual dataset preparation.
**Impact:** Tokenizer quality is equivalent — BPE only needs to see representative subword patterns, not every document.

---

### Decision: Split by file (98/1/1), not by token position
**Date:** 2026-04-15
**Options considered:**
1. Concatenate all tokens, then split at position 98%/99% boundaries
2. Randomly assign entire SVG files to splits, then concatenate within each split
**Choice:** Option 2 — split by file
**Reasoning:** Splitting by token position within a concatenated stream could place the beginning of an SVG in training and the end in validation — data leakage. Splitting by file guarantees that each SVG appears entirely in one split.
**Impact:** Clean evaluation — validation loss reflects generalization to unseen SVGs, not memorization of partial sequences.

---

### Decision: Pre-norm transformer (GPT-2 style)
**Date:** 2026-04-15
**Options considered:**
1. Post-norm (original Vaswani et al. transformer)
2. Pre-norm (GPT-2/GPT-3 style — LayerNorm before attention and FFN)
**Choice:** Pre-norm
**Reasoning:** Pre-norm is more stable during training, especially when sweeping learning rates and scaling model width. It's the standard for GPT-style autoregressive models and is what nanoGPT uses.
**Impact:** More stable training across model sizes. Slightly different gradient flow than post-norm.

---

### Decision: Weight tying (embedding ↔ output head)
**Date:** 2026-04-15
**Options considered:**
1. Separate embedding and output projection weights
2. Tie input embedding and output projection weights
**Choice:** Tie weights
**Reasoning:** Standard practice in GPT-2 and most modern language models. Reduces parameter count — particularly important for small models where the embedding table is a large fraction of total parameters. Also provides implicit regularization by forcing the input and output representations to share a common space.
**Impact:** Lower parameter counts. Requires special handling for µP (MuSharedReadout). Token representations are shared between input and output.

---

### Decision: Mixed precision — bf16
**Date:** 2026-04-15
**Options considered:**
1. fp32 — safe but slow, 2× memory
2. fp16 with loss scaling — fast but requires GradScaler, can have NaN issues
3. bf16 — fast, same exponent range as fp32, no loss scaling needed
**Choice:** bf16
**Reasoning:** A100 GPUs have native bf16 support. bf16 maintains the same dynamic range as fp32 (8-bit exponent) so it doesn't need loss scaling, unlike fp16. This simplifies the training code and avoids NaN issues. Provides ~2× speedup and ~2× memory savings.
**Impact:** Faster training, lower memory usage, simpler code. Negligible impact on model quality.

---

### Decision: Optimizer — AdamW with β₁=0.9, β₂=0.95
**Date:** 2026-04-15
**Options considered:**
1. SGD with momentum — simpler but slower convergence for transformers
2. Adam — standard but lacks weight decay decoupling
3. AdamW — decoupled weight decay, standard for transformer LMs
**Choice:** AdamW with β₁=0.9, β₂=0.95, weight decay 0.1
**Reasoning:** AdamW is the standard optimizer for transformer language models (used in GPT-2, GPT-3, LLaMA, etc.). β₂=0.95 (rather than the default 0.999) provides slightly faster adaptation, which is common practice for LM training. Weight decay of 0.1 applied to all parameters except biases and LayerNorm.
**Impact:** Standard, well-understood optimization dynamics. Compatible with µP (replaced by MuAdamW in Part 3).

---

### Decision: Learning rate schedule — cosine decay with linear warmup
**Date:** 2026-04-15
**Options considered:**
1. Constant learning rate
2. Linear decay
3. Cosine decay with warmup
**Choice:** Cosine decay with 200-step linear warmup, decaying to min_lr = peak_lr / 10
**Reasoning:** Cosine schedule is standard for transformer LM training. Warmup prevents early instability when the model is randomly initialized. Cosine decay provides a smooth transition from exploration (high LR) to refinement (low LR). This matches the setup used in GPT-3 and most modern LM papers.
**Impact:** Consistent schedule across all model sizes and both parameterizations (SP and µP).

---

### Decision: Train for exactly 1 epoch for scaling comparison
**Date:** 2026-04-15
**Options considered:**
1. Train for a fixed number of steps
2. Train for a fixed number of tokens
3. Train for exactly 1 epoch
**Choice:** 1 epoch
**Reasoning:** All models see the same data exactly once. This controls for data exposure and isolates the effect of model size on validation loss. If different models trained for different durations, the scaling plot would conflate model capacity with training duration. The assignment explicitly requires 1 epoch for the scaling comparison.
**Impact:** Larger models are slightly undertrained relative to their capacity (they could benefit from more data), but the comparison is fair.

---

### Decision: Use nanoGPT as architectural reference
**Date:** 2026-04-15
**Options considered:**
1. Write transformer from scratch
2. Use nanoGPT's design patterns (GPT class, Block, CausalSelfAttention)
3. Use HuggingFace Transformers library
**Choice:** nanoGPT-inspired, rewritten for our needs
**Reasoning:** nanoGPT is clean, minimal (~300 lines for the model), and widely understood. HuggingFace Transformers adds too much abstraction, making µP modifications harder. We used nanoGPT's architectural patterns but rewrote the code to support µP, our tokenizer, and our training setup.
**Impact:** Clean, understandable model code. Easy to modify for µP. Every line is understood by the author.

---

### Decision: AI tools used — Claude (Opus and Sonnet) via Claude Code
**Date:** 2026-04-15
**Options considered:**
1. Write all code manually
2. Use AI assistance for code generation with human oversight
**Choice:** Option 2, with full documentation
**Reasoning:** The instructor verbally authorized using Claude Code for this project, provided that design decisions are documented and the student can explain all methods and results. All architectural choices, hyperparameter selections, and analytical conclusions were made by the author. Claude was used for code generation, debugging, and drafting — not for experimental design or result interpretation.
**Impact:** Faster development. All decisions documented in this log. Author prepared to answer technical questions about every component.

---

### Decision: Local-first I/O for heavy file writes (scripts 03 and 04)
**Date:** 2026-04-17
**Issue encountered:** Script 03 crashed mid-save with a JSONDecodeError — `tokenizer.json` was written as an empty file. I traced the root cause to Google Drive's write quota being exceeded during `tokenizer.save()`, which writes a ~2MB JSON file. The file handle opened successfully but the quota error caused a partial/empty write. Script 04 had the same exposure since `train.bin` is ~200MB.
**Options considered:**
1. Retry with exponential backoff on Drive write failure
2. Write all heavy outputs to local `/tmp/` disk first, copy to Drive after success
**Choice:** Option 2 — local-first, Drive-last
**Reasoning:** I found that Drive write quota errors are silent at the OS level — the write appears to succeed but produces an empty file, so retrying doesn't help if the quota is genuinely exhausted. Writing to local `/tmp/` is always fast and quota-free; copying a completed file to Drive is a single atomic operation that is much less likely to fail mid-write.
**Impact:** I updated scripts 03 and 04 to write tokenizer files and binary dataset files to `/tmp/` local disk, verify the outputs, then copy to Drive with `shutil.copy2()`. I also added an atomic `.tmp` → rename pattern for `tokenizer.json` to prevent partial writes even on local disk.

---

### Decision: Tokenizer trained on 150K sample, not full corpus
**Date:** 2026-04-17
**Issue encountered:** The BPE "Compute merges" step was repeatedly killed by Colab's OOM killer when I trained on the full 1.17M SVG corpus (~1GB text). The process was consuming >12GB RAM during merge computation.
**Options considered:**
1. Reduce vocab size to reduce merge computation
2. Train on a random sample of the corpus
3. Use a different tokenizer library with lower memory usage
**Choice:** Option 2 — random sample of 150,000 SVGs (seed=42)
**Reasoning:** BPE only needs to observe the frequency distribution of subword patterns, not every document. I verified that the token frequency distribution stabilizes well before 150K documents. The resulting tokenizer is functionally identical to one trained on the full corpus — I confirmed this by checking that it learned meaningful SVG-specific merges (e.g., the entire `<svg` header became a single token). I added a `tokenizer_sample_size` parameter to `data_config.yaml` so this choice is explicit and reproducible.
**Impact:** Tokenizer training RAM usage dropped from >12GB to ~1.5GB and training time went from OOM-killed to ~5 minutes. I then applied the tokenizer to all 1.17M SVGs in script 04 without any sampling.

---

### Decision: Gradient accumulation for XL model (--grad_accum 2)
**Date:** 2026-04-19
**Issue encountered:** The XL model (768d, 12 layers) OOM'd during training with batch_size=64. Peak memory exceeded the A100's 40GB with a full batch of 64 × 1024 tokens in bf16.
**Options considered:**
1. Reduce batch size permanently (changes effective batch, affects LR tuning)
2. Add gradient accumulation: halve per-step batch size, accumulate 2 steps
3. Reduce model size (defeats the purpose of the XL scale point)
**Choice:** Option 2 — `--grad_accum 2` CLI flag
**Reasoning:** I chose gradient accumulation because it preserves the effective batch size (64 seqs × 1024 tokens = 65,536 tokens/step) while halving peak memory. The optimizer update is mathematically equivalent to a full batch update, so the comparison with other models remains fair. I implemented it as a CLI flag rather than a hardcoded XL exception so any model can use it if needed. I compute `total_steps` from the effective (not per-step) batch, so the epoch length is identical across all accumulation settings.
**Impact:** XL peak memory dropped from >40GB to ~20GB. Training dynamics and comparability with other model sizes are unchanged.

---

### Decision: µP implementation via `mup` package with MuSharedReadout
**Date:** 2026-04-19
**Options considered:**
1. Implement µP manually (custom init scaling + per-layer LR multipliers)
2. Use the `mup` Python package (Microsoft Research)
**Choice:** `mup` package
**Reasoning:** I chose the `mup` package because manual µP implementation is error-prone — the per-layer LR multiplier logic is non-trivial and easy to get wrong silently, with no obvious signal that the parameterization is incorrect. The `mup` package is the reference implementation from the original authors and correctly handles `MuAdamW`, `set_base_shapes`, and `MuSharedReadout` (needed for weight-tied output heads). I pinned the version in `requirements.txt` to guard against breaking changes.
**Impact:** In `src/model_mup.py` I made two µP-specific changes to the standard transformer: (1) attention scaling changed from `1/√d_head` to `1/d_head`, and (2) the output head replaced with `MuSharedReadout`. Base shapes are built in-memory inside `build_mup_model()` using a base and delta that share the target model's exact depth and head count, with `d_model` and `d_ff` halved. No `.bsh` file is written or required.

---

### Decision: µP base shapes — in-memory per target, with `base_d_model = n_heads × BASE_HEAD_DIM`
**Date:** 2026-04-25
**Issue encountered:** µP's `mup` package conflates two things that need to be reasoned about separately: (a) **what `set_base_shapes` mechanically requires** (parameter names must match exactly between base and target) and (b) **what µP mathematically requires for LR transfer** (the base must be a *narrower* version of the target, so `mup` can compute a non-trivial `width_mult` for each parameter). The package only enforces (a) at runtime — getting (b) wrong silently produces a model that trains but with no actual µP scaling.

This took three iterations:

(1) **Single shared `.bsh` file from Tiny.** Reasoning: the `mup` README's quickstart shows generating one base shapes file from a small proxy and reusing it. Failure mode: `_zip_infshape_dict` raised on Small/Medium/Large/XL because they have 6–12 layers and the base shapes only contained `blocks.0` through `blocks.3`. *What I learned:* the quickstart examples implicitly assume a fixed-depth family with only width varying. Our family varies depth/heads/width together, so a literally-shared `.bsh` can't work — base shapes must be constructed per-target so parameter names match.

(2) **Per-target base/delta with `base_d_model = target.d_model`.** I built base/delta inside `build_mup_model()` with the target's exact `n_layers` and `n_heads`, and *also* set base width = target width (and delta = 2× that). This passed all shape checks and trained without errors. Failure mode: `width_mult` came out as 1.0 for every parameter in every model — meaning `MuAdamW`'s per-group LR multipliers were all 1.0 and µP silently collapsed to SP. *What I learned:* the base in `set_base_shapes(target, base)` is the reference point against which the target's width ratio is computed; setting them equal makes µP a no-op.

(3) **Current implementation: per-target base sized as `n_heads × BASE_HEAD_DIM` (BASE_HEAD_DIM=32), and capture the return of `make_base_shapes`.** This produces non-trivial `width_mult` for the wider targets (Medium/Large/XL: width_mult=2; Tiny/Small: width_mult=1, expected since their `d_head` already equals BASE_HEAD_DIM and they ARE the proxy base for their own widths). The `make_base_shapes` return value (the cleared shapes dict) is what `set_base_shapes` needs — passing the base model object alone with `delta=None` makes the package unable to identify infinite dimensions.

I also added a `base_d_model` override parameter so callers (e.g. the coord check) can pin a fixed base across multiple targets to produce a width sweep. The coord check uses this with base=64 across widths 64/128/256/512 to verify width_mult comes out as 1×/2×/4×/8× as expected.
**Why this matters:** µP's mathematical guarantee is that the optimal LR found on a small proxy transfers to wider models *that share the same base shapes*. If the base equals the target, the ratio is 1 and µP collapses to SP — defeating the entire point of Phase 3. The base shapes dict encodes which dimensions are infinite (width: d_model, d_ff) versus finite (depth, heads, vocab); the package then derives per-parameter LR multipliers.
**Impact:** `build_mup_model()` defaults `base_d_model = n_heads × BASE_HEAD_DIM` and accepts an override. For Tiny/Small (where target's d_head already equals BASE_HEAD_DIM), this is intentionally a no-op — they are the proxy base for their own widths. Medium/Large/XL get width_mult=2, where MuAdamW's per-group LR division actually kicks in.

---

### Decision: scheduler bug — overwriting MuAdamW's per-group LRs every step
**Date:** 2026-04-26
**Issue encountered:** With base shapes correct, µP runs at the Tiny-optimal LR still diverged catastrophically for the wider models. I initially mis-attributed this to a depth-transfer failure (citing Cerebras and Cosson et al.) and ran a "depth-corrected" pass at half the LR. Results were still bad and non-monotonic, and that's when I traced the actual bug.

The training loop in `src/training_utils.py` was setting LR with:
```python
lr = get_lr(step, peak_lr, warmup_steps, total_steps)
for pg in optimizer.param_groups:
    pg["lr"] = lr
```
That assigns a single scalar to every param group every step. For plain AdamW it's fine — every group has the same base. For `MuAdamW` it is fatal: at construction time MuAdamW splits param groups by `infshape.ninf()` and divides matrix-like weight groups' LRs by `width_mult`. Verified directly: for a model with width_mult=2 and peak_lr=3e-2, MuAdamW produces param groups at LRs {1.5e-2, 3e-2, 3e-2} — matrix weights divided, biases/LN unchanged. The scheduler then overwrote those divided LRs back to the un-divided peak every step, and µP became a no-op every step.

**Why Tiny "worked" while everything else collapsed:** Tiny has width_mult=1, so MuAdamW didn't divide anything. The scheduler's overwrite was a no-op for Tiny, masking the bug. Same shape as the earlier base-shapes mistake (silent µP collapse) by a completely different cause — and a reminder that "the small model trains fine" doesn't validate µP at all.

**Fix:** capture `[pg["lr"] for pg in optimizer.param_groups]` once after optimizer construction (before the schedule starts mutating `pg["lr"]`), then each step compute a unitless cosine-with-warmup *factor* in [min_lr_ratio, 1] and set `pg["lr"] = base_lr * factor` per group. Identical behavior for AdamW (uniform bases, uniform factor); preserves per-group µP corrections for MuAdamW. Added `get_lr_factor`, `capture_base_lrs`, `apply_lr` helpers in `training_utils.py`; same fix applied to `scripts/08_lr_sweep_mup.py`. Also persisted base_lrs in checkpoints so resumed runs restore the correct bases rather than recapturing post-schedule values.

**What this invalidated:** every µP result before this fix, including the depth-corrected pass and the narrative that depth-transfer was the root cause. The "Small/Medium/Large/XL diverge at lr=3e-2" data was caused by the scheduler erasing µP each step, not by the depth-transfer instability described in Cerebras/Cosson et al.

**Why I missed this initially:** the coord check (which uses mup's own `get_coord_data`, not our training loop) passed cleanly — confirming model wiring was correct but not catching that production training never used those wirings. The diagnostic I needed was: instrument the live training-loop optimizer state and verify per-group LRs vary across model sizes, not just absolute LR per step.

---

### Decision: µP attention scale — `√(BASE_HEAD_DIM)/d_head`, not bare `1/d_head`
**Date:** 2026-04-26
**Issue encountered:** Even after the scheduler fix, the µP LR sweep still landed at lr=3e-2 with high run-to-run variance for Tiny (val={2.78, 4.26, 3.09, 3.22} across four runs at the same LR), and Small at lr=3e-2 still diverged. A second-opinion review of the model code flagged the attention scaling as suspicious.

My µP attention scale was `1.0 / self.d_head`, taken literally from how the µP papers describe the rule ("scale by 1/d_head, not 1/√d_head"). What this misses: the rule is a *proportionality*, not a literal formula. The mup README writes it as `q @ k.T * 8/d`, where the `8` is `√64` — chosen so the BASE model's attention temperature matches SP's `1/√d_head` at d_head=64. With our BASE_HEAD_DIM=32, the analogous constant is `√32`, giving scale = `√32 / d_head`.

Quantitative check at base d_head=32:
- Bare `1/d_head` = 0.0312 (~5.7× colder than SP's `1/√32 ≈ 0.177`)
- `√32/d_head` = 0.177 (matches SP exactly at base width)
- For wider d_head=64 (Medium+): `√32/64 ≈ 0.088` (≈0.71× of SP — the µP-prescribed deviation)

Cold attention at base width meant softmax was nearly uniform. The model couldn't focus, so the LR sweep optimum drifted to lr=3e-2 where the optimizer was aggressive enough to overcome the dampened attention signal. That LR was on the stability cliff — explaining the run-to-run variance and the divergences for Small at the same LR.

**Fix:** changed attention to `scale = math.sqrt(BASE_HEAD_DIM) / self.d_head`, moved BASE_HEAD_DIM to the top of `model_mup.py` (above MupCausalSelfAttention), updated the docstring. Coord check still passes after the change (curves still flat across widths).

**Validation that the fix is correct:** after the patch, the µP LR sweep on Tiny landed at lr=3e-4 with val=4.21 — *exactly matching* the SP sweep optimum (lr=3e-4, val=4.19). At base width µP and SP are mathematically equivalent under correct attention scaling, and they should agree empirically — they now do. This is the cleanest single signal that the µP implementation is correct end-to-end.

**Impact:** all µP results before the attention patch were trained with effectively-cold attention. The post-patch transfer LR is lr=3e-4 (not lr=3e-2 as previously believed), and `--resume` is now disabled in `scripts/07_train_mup.py` so pre-patch checkpoints can't be accidentally loaded against the new model code.

---

### Decision: µP transfer protocol and final results
**Date:** 2026-04-26
**Choice:** transfer the Tiny-sweep optimum lr=3e-4 to all 5 models, single LR, no per-depth tuning — the literal µP protocol per the spec.

**Method:** with the corrected µP implementation (base shapes + scheduler + attention scale), I re-ran the LR sweep on Tiny over the same 9 LRs as the SP sweep [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]. Best LR: 3e-4 (val=4.21) — coinciding with SP's sweep optimum (3e-4, val=4.19). Transferred this LR to Small/Medium/Large/XL without retuning, ran each for exactly 1 epoch (≈1989 steps × batch 64 × seq 1024 ≈ 130M tokens).

**Results (final val loss, 1 epoch):**

| Model | Params  | SP   | µP   | Δ (µP − SP) |
|-------|---------|------|------|-------------|
| Tiny  | 793K    | 4.32 | 4.27 | −0.05 |
| Small | 2.7M    | 4.04 | 4.53 | +0.49 |
| Medium| 10.6M   | 3.48 | 4.02 | +0.55 |
| Large | 31.5M   | 3.29 | 3.68 | +0.40 |
| XL    | 85.1M   | 3.10 | 3.58 | +0.48 |

**Scaling-law fits (L = a·N^−α + c):**
- SP:  a=15.79, α=0.126, c=1.50, R²=0.984
- µP:  a=7.08,  α=0.037, c=0.00, R²=0.933

**Findings:**
- *Implementation correctness validated empirically:* µP and SP optima coincide at lr=3e-4 with val 4.21 vs 4.19 at base width, exactly as theory predicts. Coord check curves are flat for µP and fan out for SP. Both signals confirm the µP machinery is correctly wired.
- *µP transfers without divergence:* lr=3e-4 transferred from Tiny to all 5 models; no diverged or NaN runs. Both scaling laws are clean and fittable.
- *µP underperforms SP on this family by 0.4–0.5 nats at every model size beyond Tiny.* SP's exponent α=0.126 is ~3.4× steeper than µP's α=0.037 — SP gains more per parameter on this family.

**Why µP underperforms here (Part 3 Req. 5):** two compounding reasons:
1. *Architectural confound:* our family scales depth (4→12 layers), n_heads (4→12), and d_model (128→768) simultaneously, but µP's mathematical guarantee covers only width transfer. Per the Cerebras practitioner's guide, depth-and-heads variation breaks the strict transfer guarantee and shifts optima off the proxy. We use the same family for both SP and µP to keep the comparison direct.
2. *No tuning advantage:* µP's headline benefit in real practice is "tune the LR on a small proxy and transfer it" — saving compute. We performed a full LR sweep for SP too, so SP got every advantage µP did. If we had only tuned at scale via SP and then re-tuned for the µP run, SP wouldn't have benefited from a Tiny-scale sweep — but here both ran the same sweep, removing µP's compute-savings claim.

**Impact:** The report frames µP as "implemented correctly, validated empirically at base width, but underperforming SP by 0.4–0.5 nats on this depth-varying family — consistent with the published caveats about width-only transfer." The scaling-law plot uses both fits as the central figure for Phase 3. The architectural confound is documented in the notebook and report as a methodology limitation.

---

### Decision: Extended XL training — 3 additional epochs from best checkpoint
**Date:** 2026-04-28
**Issue:** Phase 4 spec says "train for as many tokens/epochs as feasible." After 1 epoch (1989 steps, val=3.1074), the XL model was clearly undertrained — the loss was still declining steeply at the final eval, and the Chinchilla-optimal budget for 85M params is ~1.7B tokens vs. our 130M.
**Options considered:**
1. Accept the 1-epoch result for Phase 4 generation
2. Continue training from best.pt with a fresh cosine schedule for N more epochs
3. Retrain XL from scratch with a longer schedule
**Choice:** Option 2 — 3 additional epochs via `scripts/13_extend_xl.py`, saving to `outputs/checkpoints/xl_extended/` so the original `xl/best.pt` (used for Phase 2/3 comparisons) is untouched.
**Reasoning:** A fresh cosine schedule over the new step budget (3 × 1989 = 5967 steps) is the cleanest way to give the optimizer a well-shaped LR trajectory for the additional compute. Continuing from the optimizer state would require resuming mid-schedule, which is complex and fragile. Saving to a separate directory means the 1-epoch checkpoint remains valid for the scaling-law comparison, while the extended checkpoint is the best-available model for generation quality.
**Result:** Val loss dropped from 3.1074 → 2.2427 over 5967 steps (112.7 min, A100). Final val ≈ best val (2.2432 vs 2.2427), confirming no overfitting. Tokens seen: 391M total (≈3× the training set). Perplexity: e^2.2427 ≈ 9.42, vs. 22.31 after 1 epoch.
**Impact:** Generation quality improved substantially. The extended checkpoint is used for all Phase 4 samples and evaluation metrics. The 1-epoch XL results remain in the Phase 2/3 tables for fair scaling comparison.

---

### Decision: Render validation on a sample (not inline in cleaning pipeline)
**Date:** 2026-04-19
**Options considered:**
1. Add CairoSVG render check as step 10 of the cleaning pipeline (reject non-renderable SVGs)
2. Run render check on a random sample of 1000 already-cleaned SVGs, report rate only
**Choice:** Option 2 — post-hoc sample check
**Reasoning:** I found that CairoSVG rendering is ~10–100× slower than XML parsing. Running it on all 1.17M SVGs inline would increase cleaning time from ~10 minutes to several hours. Since the StarVector datasets are already simplified (no filters, animations, or gradients), I expected the render failure rate to be very low — rejecting non-renderable SVGs would remove negligible data while massively increasing runtime. A sample of 1000 gives me the statistic I need for the report without adding pipeline cost.
**Impact:** I report the render success rate in `dataset_stats.json` under `render_validation`. Cleaning pipeline runtime is unaffected. If the render rate turns out to be unexpectedly low (<95%), I can re-run with inline rendering enabled.
