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

### Decision: µP base shapes — in-memory, with Tiny widths as the proxy base
**Date:** 2026-04-25
**Issue encountered:** This took three iterations because µP's `mup` package conflates two things that need to be reasoned about separately: (a) **what `set_base_shapes` mechanically requires** (parameter names must match exactly between base and target) and (b) **what µP mathematically requires for LR transfer** (the base must be a *narrower* version of the target so that `mup` can compute a nontrivial width ratio). The package only enforces (a) at runtime — getting (b) wrong silently produces a model that trains but with no actual µP scaling.

(1) **First attempt: single shared `.bsh` file from Tiny.** Reasoning: the `mup` README and quickstart show you generating one base shapes file from a small "proxy" model and reusing it, so I built one `.bsh` from Tiny (4 layers, d=128) and tried to apply it to all 5 targets. Failure mode: `_zip_infshape_dict` raised on Small/Medium/Large/XL because they have 6–12 layers and the base shapes only contained `blocks.0` through `blocks.3`. *What I learned:* the package's quickstart examples implicitly assume a fixed-depth family — only width varies. Our blueprint table varies depth, heads, and width together, so a literally-shared `.bsh` can't work without restructuring the architecture.

(2) **Second attempt: per-target in-memory base/delta with target's own widths.** Reasoning: to satisfy the name-matching requirement I built base/delta inside `build_mup_model()` with the target's exact `n_layers` and `n_heads`. I needed the base and delta to differ in width (so `mup` could identify the infinite dimension) but didn't initially understand that the base width *also* matters relative to the target — I set base = target's widths and delta = 2× target's widths. This passed all shape checks and trained without errors. Failure mode: Medium and Large produced val loss *worse than Tiny*, breaking the scaling law. *What I learned:* the base in `set_base_shapes(target, base)` isn't just a shape-checking template — it's the reference point against which the target's width ratio is computed. Setting base = target makes the ratio 1:1, which means `MuAdamW`'s per-parameter LR multipliers all equal 1, which means µP silently collapses to SP. The LR found on the Tiny sweep (lr=0.03) is fine for an 800K-param SP model but ~10× too high for a 10M-param SP model — explaining why training got stuck. Diagnosing this required going back to the README and tracing what `set_base_shapes` actually does internally.

(3) **Current implementation: Tiny widths as the proxy base for every target.** Each target builds its own base/delta in-memory, but base/delta widths are hardcoded to Tiny's (d=128, d_ff=512) and (d=256, d_ff=1024). This satisfies both requirements at once: parameter names match the target (because depth/heads come from the target's config) and the width ratio between target and base is meaningful (3× for Medium, 6× for XL), so `mup`'s per-parameter LR multipliers produce real scaling. The Tiny sweep is now genuinely a sweep on the proxy base, and its optimum transfers to wider models as µP intends.
**Why this matters:** µP's mathematical guarantee is that the optimal LR found on a small "proxy" model transfers to wider models that share the **same base shapes**. The base shapes file encodes which dimensions are infinite (width: d_model, d_ff) versus finite (depth, heads). `set_base_shapes(target, base)` then computes the width ratio and produces per-parameter LR multipliers. If the base equals the target, the ratio is 1 and µP becomes equivalent to SP — defeating the entire point of Phase 3.
**Options considered:**
1. Save a single `.bsh` and reuse it (failed — depth mismatch)
2. Build base = target (failed — no scaling correction applied)
3. Build base in-memory using Tiny's widths but matching the target's depth/heads
**Choice:** Option 3
**Reasoning:** This preserves µP's LR transfer property while satisfying `mup`'s name-matching requirement. Each target model gets its own base/delta pair built fresh, but they share the same proxy width (Tiny's d=128, d_ff=512). For the Tiny target this is a no-op; for wider targets, `mup` correctly identifies the width dimensions as infinite and applies the right per-parameter LR scaling. n_heads in the base/delta is adjusted if it doesn't divide BASE_D_MODEL=128 (n_heads doesn't appear in any parameter shape, so this only affects whether the model can be instantiated, not the µP behavior).
**Impact:** `build_mup_model()` now defines `BASE_D_MODEL=128, BASE_D_FF=512` as module-level constants and uses these to construct base/delta for every target. The LR sweep runs on Tiny (which IS the proxy base width), and the resulting LR transfers correctly to Small, Medium, Large, and XL via µP's per-parameter scaling.

---

### Decision: Render validation on a sample (not inline in cleaning pipeline)
**Date:** 2026-04-19
**Options considered:**
1. Add CairoSVG render check as step 10 of the cleaning pipeline (reject non-renderable SVGs)
2. Run render check on a random sample of 1000 already-cleaned SVGs, report rate only
**Choice:** Option 2 — post-hoc sample check
**Reasoning:** I found that CairoSVG rendering is ~10–100× slower than XML parsing. Running it on all 1.17M SVGs inline would increase cleaning time from ~10 minutes to several hours. Since the StarVector datasets are already simplified (no filters, animations, or gradients), I expected the render failure rate to be very low — rejecting non-renderable SVGs would remove negligible data while massively increasing runtime. A sample of 1000 gives me the statistic I need for the report without adding pipeline cost.
**Impact:** I report the render success rate in `dataset_stats.json` under `render_validation`. Cleaning pipeline runtime is unaffected. If the render rate turns out to be unexpectedly low (<95%), I can re-run with inline rendering enabled.
