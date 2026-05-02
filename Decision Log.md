# Decision Log - SVG Scaling Laws Project

Notes on design decisions and bugs I ran into. Mostly so I can answer questions about this stuff later, also so the grader can see what was deliberate vs. what I stumbled into.

I'm only writing things down that aren't obvious from the code itself. Standard transformer setup choices (AdamW, cosine schedule, etc.) are at the top because they were quick decisions; the longer entries near the bottom are the ones where I actually screwed up at some point.

---

### Primary dataset: starvector/svg-icons-simple
2026-04-15

Used svg-icons-simple as the main dataset. About 89K SVGs, already preprocessed by the StarVector folks, well-documented. Considered umuthopeyildirim/svgen-500k briefly but it's less standardized and I didn't want to spend the first day debugging dataset format quirks. Web scraping was off the table for obvious reasons.

Icons alone don't get to 100M tokens, so I added emoji-simple and fonts-simple as supplementary data. End composition by file count is roughly 7% icons, ~1% emoji, 93% fonts. The font-heaviness comes back to bite me in Phase 4 (see the rebalancing entry).

### Coordinate precision: 1 decimal place
2026-04-15

Round coordinates to 1 decimal. At a 24x24 viewBox 0.1 units is sub-pixel, so visually nothing changes. One representative SVG went from 2235 chars to 919 chars after rounding (59% smaller), which means many fewer unique numeric strings for the tokenizer to learn. Integer rounding would be too lossy - curves visibly jagger.

### Min SVG length: 50 chars
2026-04-15

Just a sanity check. Nothing in the StarVector data is even close to under 50 chars (min after cleaning is 315), so this filter removes nothing. I left it in for safety in case I ever swap in a noisier dataset.

### Max token length: 1024
2026-04-15

P99 sequence length is 447 tokens. Only 65 of 1.17M SVGs exceed 1024 tokens (0.006%). Same context window nanoGPT uses. Anything bigger would burn quadratic compute on attention with no real coverage gain.

### BPE vocab size: 4096
2026-04-15

SVG vocabulary is much more constrained than English. A handful of tag names, attribute names, path commands, and number strings dominate. With vocab=4096 and 130M training tokens, every token sees ~32K occurrences on average, plenty for stable embeddings. The tokenizer ended up learning impressively efficient merges - the entire SVG header collapses to a single token (id 1024 in the original tokenizer), and number strings like "7.3" get their own tokens. Median sequence comes out around 98 tokens.

I tried 1024 vocab briefly and sequences blew up to 3-4x longer; tried 8192 and a lot of the rare tokens looked like noise.

### Train tokenizer on a 150K sample
2026-04-17

This was a forced choice, not a design one. Training BPE on the full 1.17M corpus (~1GB text) consistently OOM'd Colab's 12GB during the "Compute merges" step - it kept getting killed and I lost a couple hours trying to figure out why. A 150K sample (seed=42) is much smaller (~130MB) and the resulting tokenizer is functionally identical, since BPE only needs the frequency distribution of subword patterns to converge. I checked that it learned the same characteristic merges (header-as-one-token, etc.) as a sanity check.

The full 1.17M corpus is then tokenized using this tokenizer in script 04, no sampling there.

### Split by file, not by token position
2026-04-15

Concatenating everything and splitting at token-position boundaries is a leakage trap: the start of an SVG would end up in train and the end in val. Splitting by file means each SVG appears entirely in one split, full stop. 98/1/1.

### Pre-norm, weight tying
2026-04-15

Pre-norm because it's more stable when sweeping LR or scaling width. Weight-tied embedding/output head because GPT-2 does it and the embedding table is a big chunk of params at small scales, so tying it down is helpful. The weight tying does mean I have to use `MuSharedReadout` rather than a plain linear in the muP variant, more on that below.

### bf16 mixed precision
2026-04-15

A100s have native bf16. bf16 keeps fp32's 8-bit exponent so I don't need GradScaler or worry about underflow like with fp16. About 2x speedup, 2x less memory. No quality regression I can detect.

### AdamW, beta2=0.95, weight decay 0.1
2026-04-15

Standard for LM training, same as GPT-3 and friends. beta2=0.95 (rather than 0.999) gives slightly faster adaptation, common practice. Weight decay applied to weight matrices only - not biases, not LN params.

### Cosine schedule with linear warmup
2026-04-15

200 steps of warmup, then cosine decay to 0.1x peak. Same recipe basically every modern LM uses. Warmup matters because random init + high LR diverges fast otherwise.

### Train each model for exactly 1 epoch
2026-04-15

The assignment requires this for the scaling comparison and it makes sense: every model sees the same data once, so the only thing that varies between them is capacity. Larger models are undertrained relative to capacity (Chinchilla-optimal for XL would be ~1.7B tokens, we have 130M), but the comparison is fair across sizes. The undertraining shows up later in the steepness of the scaling exponent.

### nanoGPT as architectural reference
2026-04-15

I started from nanoGPT's structure (Block, CausalSelfAttention, the GPT class) but rewrote everything for our needs. HuggingFace Transformers was the alternative - too much abstraction, would've made muP modifications painful. My code is short (about 300 lines for the model) and I understand every line, which is the main thing for being able to defend it.

### AI tools (Claude Code) used with documentation
2026-04-15

The instructor verbally OK'd Claude Code as long as the design decisions are documented and I can explain the work. Architectural choices and analytical conclusions are mine; Claude was useful for code generation and debugging. This log is part of that documentation requirement.

---

### Local-first I/O for heavy file writes
2026-04-17

Script 03 crashed mid-save with a JSONDecodeError at one point. The `tokenizer.json` was written as an empty file. Took me a while to figure out why - it turned out Drive's write quota was being exceeded silently during `tokenizer.save()`. The file handle opens fine, the write appears to succeed, and you get a 0-byte file. Retrying doesn't help because the quota is genuinely exhausted.

Script 04's `train.bin` is ~200MB so it had the same exposure.

Fix: write everything heavy to /tmp first, verify it, then copy to Drive with `shutil.copy2()`. Atomic .tmp→rename pattern on `tokenizer.json` to make local writes safe too.

### Gradient accumulation for XL
2026-04-19

XL (768d, 12 layers) OOMs on A100 40GB with batch_size=64 in bf16. Halving the batch and using `--grad_accum 2` keeps the effective batch the same (65,536 tokens/step), which means total_steps and the LR schedule stay identical to the other models, just peak memory drops from 40+GB to ~20GB. I made it a CLI flag rather than an XL-specific hack so any model can use it. `total_steps` is computed from effective batch, not per-step batch, so epoch length is the same regardless of accumulation setting.

---

### muP via the `mup` package, not manual
2026-04-19

I went with the Microsoft `mup` package rather than implementing muP myself. Manual muP is the kind of thing where you can have a subtly wrong per-layer LR multiplier and the model still trains, just without the muP property. There's no obvious failure signal, which is the worst kind of bug. The `mup` package handles `MuAdamW`, `set_base_shapes`, and `MuSharedReadout` (which I need because of weight tying). Pinned the version in requirements.txt.

In `src/model_mup.py` I changed two things from the SP transformer:
1. Attention scaling from `1/sqrt(d_head)` to `sqrt(BASE_HEAD_DIM)/d_head` (this is the muP rule with the proportionality constant - more on this in the attention-scale entry below, this got messed up the first time)
2. Output head replaced with `MuSharedReadout`.

Base shapes are constructed in-memory inside `build_mup_model()` per target. No `.bsh` file is written.

### muP base shapes - took me three tries to get this right
2026-04-25

This is the longest entry in the log because I burned the most time on it. The mup package conflates two things:

- What `set_base_shapes` mechanically requires: parameter names match between base and target.
- What muP mathematically requires: the base must be a *narrower* version of the target so width_mult comes out greater than 1.

The package only checks the first one. If you get the second one wrong, training proceeds without errors but muP silently does nothing.

**Try 1: shared `.bsh` file from Tiny.** The mup README's quickstart shows generating one base-shapes file from a small proxy and reusing it. So I tried that. `_zip_infshape_dict` raised on Small/Medium/Large/XL because the base shapes only had `blocks.0` through `blocks.3` (Tiny is 4 layers) but Small etc. have 6-12 layers, so the param names don't match. Lesson: the quickstart implicitly assumes a fixed-depth family with only width varying. Our family changes depth too, so a literally-shared .bsh is impossible.

**Try 2: per-target base/delta with base_d_model = target.d_model.** I built base/delta inside `build_mup_model()` with the target's exact n_layers and n_heads. But I also set base width = target width (and delta = 2x). All shape checks pass, training proceeds, no errors. Then I noticed every parameter had width_mult=1.0, which means MuAdamW's per-group LR multipliers were all 1.0 and muP collapsed to SP. Lesson: the base in `set_base_shapes(target, base)` is the reference for the width ratio. If base == target, the ratio is 1, and you've defeated the entire point.

**Try 3 (current):** per-target base sized as `n_heads * BASE_HEAD_DIM` (BASE_HEAD_DIM=32), and capture the dict that `make_base_shapes` returns. Now Medium/Large/XL get width_mult=2 (their d_head is 64, base d_head is 32). Tiny/Small still get width_mult=1, which is fine because their d_head already equals BASE_HEAD_DIM, so they ARE the proxy base for their own widths. The `make_base_shapes` return value is what `set_base_shapes` actually consumes; passing the base model object alone doesn't work because the package can't infer infinite dimensions without seeing the delta.

I also added a `base_d_model` override so the coord check can pin a fixed base across multiple targets. The coord check uses base=64 across widths 64/128/256/512 to verify width_mult comes out as 1x/2x/4x/8x.

### muP scheduler bug - a different silent collapse
2026-04-26

After the base shapes were finally right, muP runs at the Tiny-optimal LR still diverged catastrophically for the wider models. I wasted a few hours initially mis-attributing this to the depth-transfer failure described in the Cerebras and Cosson papers. I even ran a "depth-corrected" pass at half the LR. Results were still bad and non-monotonic in size, which doesn't match anything depth-transfer would predict, and that's when I started looking for an actual bug rather than a theory failure.

The training loop in `src/training_utils.py` was doing this:

```python
lr = get_lr(step, peak_lr, warmup_steps, total_steps)
for pg in optimizer.param_groups:
    pg["lr"] = lr
```

For plain AdamW this is fine - all groups have the same base LR, you set them all to the same value, no problem. For MuAdamW it's fatal. At construction time MuAdamW splits param groups by `infshape.ninf()` and divides matrix weight groups' LRs by width_mult. So for a model with width_mult=2 and peak_lr=3e-2, MuAdamW's groups are at {1.5e-2, 3e-2, 3e-2}: matrix weights divided, biases/LN unchanged. Then the scheduler comes along every step and overwrites all of them to 3e-2. muP became a no-op every single step.

Why I missed this: Tiny has width_mult=1, so MuAdamW didn't divide anything for Tiny, and the scheduler's overwrite was a no-op. So Tiny "worked" and the wider models didn't, which I initially interpreted as depth transfer failing. Same shape of bug as the base-shapes mistake (silent muP collapse), completely different cause.

The fix is to capture per-group base LRs once after optimizer construction, then each step compute a unitless cosine-with-warmup factor in [min_lr_ratio, 1] and multiply: `pg["lr"] = base_lr * factor` per group. For AdamW (uniform bases) this is equivalent to the old behavior; for MuAdamW it preserves the divisions. Helpers added in `training_utils.py`: `get_lr_factor`, `capture_base_lrs`, `apply_lr`. Same fix in `scripts/08_lr_sweep_mup.py`. Also persist base_lrs in checkpoints so resumed runs restore correct bases instead of recapturing post-schedule values.

This invalidated every muP result I had at that point, including the depth-corrected pass. The narrative I'd been building about depth-transfer being the culprit was wrong.

The reason the coord check didn't catch this: it uses mup's own `get_coord_data`, not our training loop. So the model wiring was correct, the coord check confirmed that, but the production training loop never used those wirings properly. Diagnostic I should've run earlier: instrument the live optimizer state in training, not just check absolute LR per step.

### muP attention scale - third silent muP collapse
2026-04-26

After the scheduler fix, the muP LR sweep was *still* odd. Tiny landed at lr=3e-2 with high run-to-run variance (val={2.78, 4.26, 3.09, 3.22} across four identical runs), and Small at lr=3e-2 still diverged. A second-opinion code review flagged the attention scaling as suspicious.

What I had: `scale = 1.0 / self.d_head`, taken literally from how the muP papers describe the rule ("scale by 1/d_head, not 1/sqrt(d_head)"). What I missed: the rule is a proportionality. You need a constant in front. The mup README writes attention as `q @ k.T * 8/d`, where `8` is `sqrt(64)` - chosen so that at the base (d_head=64) the temperature equals SP's `1/sqrt(d_head)`. Our BASE_HEAD_DIM is 32, so the right constant is `sqrt(32)`, giving `scale = sqrt(32) / d_head`.

Numerically at base d_head=32:
- Bare 1/d_head = 0.0312, about 5.7x colder than SP's 1/sqrt(32) ≈ 0.177.
- sqrt(32)/d_head = 0.177, matches SP exactly.

So at base width attention was 5.7x colder than it should be. Softmax was nearly uniform, the model couldn't focus, and the LR sweep optimum drifted up to lr=3e-2 where the optimizer was aggressive enough to overcome the dampened signal. lr=3e-2 sits on the stability cliff, hence the variance and the divergences for Small.

Fix: `scale = math.sqrt(BASE_HEAD_DIM) / self.d_head`. Coord check still passes after the change.

How I know this is correct: after the patch, the muP LR sweep on Tiny landed at lr=3e-4 with val=4.21, *exactly* matching SP's sweep optimum (lr=3e-4, val=4.19). At base width muP and SP should be equivalent under correct attention scaling, and they should agree empirically. They do now. This is the cleanest single signal that muP is correctly wired end-to-end.

Pre-patch checkpoints were trained with cold attention so they're not loadable against the new model code. I disabled `--resume` in `scripts/07_train_mup.py` to prevent accidentally mixing them.

---

### muP transfer protocol and final results
2026-04-26

Per the spec: transfer the Tiny-sweep optimum lr=3e-4 to all 5 models, no per-size retuning.

Method: with the corrected muP (base shapes + scheduler + attention all fixed), I re-ran the LR sweep on Tiny over the same 9 LRs as the SP sweep [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]. Best LR is 3e-4 (val=4.21), which matches SP's optimum (3e-4, val=4.19). Transferred this to Small/Medium/Large/XL without retuning, ran each for 1 epoch (~1989 steps, batch 64, seq 1024, ~130M tokens).

Final val losses:

| Model  | Params | SP   | muP  | Delta (muP - SP) |
|--------|--------|------|------|------------------|
| Tiny   | 793K   | 4.32 | 4.27 | -0.05            |
| Small  | 2.7M   | 4.04 | 4.07 | +0.03            |
| Medium | 10.6M  | 3.48 | 4.02 | +0.55            |
| Large  | 31.5M  | 3.29 | 3.68 | +0.40            |
| XL     | 85.1M  | 3.10 | 3.58 | +0.48            |

Power-law fits (L = a * N^-alpha + c):
- SP:  a=15.79, alpha=0.126, c=1.50, R^2=0.984
- muP: a=7.08,  alpha=0.037, c=0.00, R^2=0.933

Things that are right:
- muP and SP agree at base width (Tiny: 4.27 vs 4.32). The empirical correctness check passes.
- LR transfer didn't blow up. No diverged or NaN runs at any size.
- Both fits are clean.

Things that are weird:
- muP underperforms SP by 0.4-0.5 nats at every size beyond Tiny. SP's exponent is ~3.4x steeper.

Why muP underperforms here: two things probably both contribute.

1. *Architectural confound.* My family scales depth (4→12 layers), heads (4→12), and width (128→768) all at once. muP's guarantee is for width transfer specifically; varying depth and heads breaks the strict guarantee and shifts the optimal LR off the Tiny sweep. The Cerebras practitioner's guide notes this. I kept the same family for SP and muP to keep the comparison apples-to-apples, but it does mean muP isn't being tested in its ideal setting.

2. *No tuning advantage for muP.* muP's main practical benefit is "tune on a small proxy and transfer," saving compute. But here SP got a full sweep too, so SP got every advantage muP did. If we'd only swept at the small scale and only ever re-trained, muP would have looked like the better choice (no retuning needed).

This goes in the report as "muP implemented correctly, validated empirically at base width, but underperforms by 0.4-0.5 nats on this depth-varying family - consistent with published caveats about width-only transfer." Both fits are the central Phase 3 figure. The architectural confound is acknowledged as a methodology limitation.

I tried to disentangle these factors with a width-only follow-up experiment (see below).

---

### Extended XL training (3 more epochs)
2026-04-28

After 1 epoch, XL is at val=3.1074 and the loss curve is still trending steeply down. Phase 4 spec says "as many tokens/epochs as feasible," so I continued.

Three options:
1. Accept 1-epoch XL for Phase 4.
2. Continue from best.pt with a fresh cosine schedule.
3. Retrain from scratch with a longer schedule.

Picked (2). Fresh cosine over the new step budget (3 * 1989 = 5967 steps) gives the optimizer a clean LR trajectory; resuming mid-schedule from the previous run's optimizer state would be complicated and fragile. Saved to `outputs/checkpoints/xl_extended/` to leave the original `xl/best.pt` untouched for Phase 2/3.

Result: val 3.1074 → 2.2427 over 5967 steps (112.7 min on A100). Final ≈ best (2.2432 vs 2.2427), no overfitting. ~391M total tokens seen, about 3x the training set. Test perplexity dropped from 22.31 to 9.31.

Generation quality improves substantially with the extended checkpoint. The 1-epoch result still appears in Phase 2/3 tables for the scaling comparison.

### Unconditional generation prompt: inject token ID 1024
2026-04-29

This was a fun one. The original generation script used `prompt="<svg"` and got 0% render rate. I tried using the full SVG header string instead, also 0%. Looking at a raw sample output: `<svg L10.6 14.4 C...`. The model is emitting coordinate data inside the opening tag's attribute area, not after it.

Root cause: BPE token 1024 is the *entire* string `<svg xmlns="..." viewBox="0 0 24 24" ...><path ... d="M`, including the opening of the path's d attribute. That's the token that appears at position 1 of every training sequence. Feeding `"<svg"` as a text prompt encodes to a different sequence of tokens because the tokenizer breaks the incomplete tag boundary differently, so the model is fed something it never saw during training and continues generating mid-attribute. Even feeding the full header string has this problem - re-encoding through BPE doesn't deterministically produce the same token IDs the model saw, because BPE encoding is context-sensitive.

Fix: inject `[BOS_ID, 1024]` directly as `prompt_ids`, bypassing the tokenizer's encode step entirely. Added a `prompt_ids` parameter to `generate_one()` that short-circuits encoding when provided. Prefix-conditioned generation still goes through `tokenizer.encode()` since the prompts there are intentionally partial and matching exact training boundaries matters less.

Result: unconditional render rate jumped from 0% to 80% (1-epoch XL) / 73% (extended). The text-encoding versions stayed at 0%.

Lesson: never re-encode the unconditional prompt through the tokenizer.

---

### Width-only model family (clean muP test)
2026-04-30

The Phase 3 µP-vs-SP gap could plausibly be from the depth/heads confound (my family changes all three). I wanted to isolate width transfer, so I made a second model family with everything except width fixed: depth=6, heads=4, d_model varying 128/192/256/384/512. With base_d_model fixed at 4*32=128, w_xs is the literal muP base (width_mult=1) and the others get clean width_mult of 1.5x/2x/3x/4x.

Did the same protocol: SP LR sweep on w_xs, muP LR sweep on w_xs, transfer best LR to all 5, train 1 epoch each.

- SP best: lr=3e-4 (val=4.2485)
- muP best: lr=3e-4 (val=4.2586) ← matches SP at base, confirms correctness
- SP scaling: alpha=0.096, R^2=0.989 (clean)
- muP scaling: alpha=0.008, R^2=0.283 (essentially flat)

So muP still underperforms even with no depth/heads variation. The confound was *not* the dominant cause.

What I think is actually going on: MuAdamW divides hidden-layer LRs by width_mult. At w_xl (width_mult=4) the effective hidden LR is 7.5e-5. The SP sweep showed that even lr=1e-4 gives val=4.98 (nowhere near optimal 4.25 at lr=3e-4), so 7.5e-5 is deep in the undertrained zone. With ~2000 steps and cosine decay, wider muP models just don't get enough updates at their reduced LR to converge. This has nothing to do with depth/heads - it's a step-budget vs. effective-LR mismatch.

So the report version of the muP narrative gets refined: the depth confound is not the only thing, and probably not even the dominant thing, in our single-epoch setup. Both factors contribute, but the LR-budget effect likely dominates.

Scripts: `06b_lr_sweep_width_only.py`, `08b_lr_sweep_mup_width_only.py`, `14_width_only_scaling.py`. WIDTH_ONLY_CONFIGS added to `src/model.py`.

---

### Rebalanced corpus for Phase 4
2026-05-01

Looking at unconditional samples from XL (both 1-epoch and extended), basically every output is a thin black single-path letterform shape. Which makes sense - 93% of training is font glyphs, so the model learned a single-stroke font-glyph prior. Technically valid SVGs, but not what the spec means by "coherent icons/shapes."

Decided to retrain XL on a more balanced corpus, keeping all Phase 2/3 results untouched. Phase 2/3 are scaling-law experiments where consistency matters more than diversity; Phase 4 is a "best generation" experiment where diversity matters more. Different deliverables, different corpora is defensible as long as I report both.

New mix in `configs/data_config_balanced.yaml`:
- icons-simple: all ~80K
- emoji-simple: all ~4K
- fonts-simple: 20% subsample, ~200K (deliberately kept low to dilute font dominance)
- stack-simple: capped at 50K initially

First run got 64.9M training tokens, well below 100M target. I considered upping the fonts fraction, but rejected that since the whole point of the rebalance is *less* fonts. Instead bumped `stack_max_samples` from 50K to 300K, which adds about 57M tokens. New total: 122M tokens (above target). All `*_balanced` outputs are in separate directories so the original Phase 2/3 paths are never touched.

Then I had to fix two tokenizer training bugs (next entry).

After running everything, the cleaned corpus had 716K SVGs - all duplicates removed (2.4%), 0 invalid XML, 0 too-short. Median seq length 130 tokens, P95 554, P99 1166. Vocab still 4096 (BPE, ByteLevel).

But that wasn't quite enough either. See "non-simplified svg-stack" below.

### Tokenizer bugs on the balanced corpus
2026-05-01

Two bugs in `src/tokenizer_utils.py` that compounded to make the BPE trainer hang at "Compute merges 0 / 4096" on the balanced corpus. Phase 1 worked fine, balanced didn't. Took an embarrassingly long time to figure out.

**Bug 1: Colab /tmp size.** The temp file from `train_tokenizer()` was 230+ MB. Colab /tmp is often only a few hundred MB total, so the file was getting silently truncated. Fix: write to /content (Colab local disk, ~100 GB) when available, fall back to system temp otherwise.

**Bug 2: long single-line "words".** With `ByteLevel(use_regex=False)`, each input line becomes one pre-tokenized "word." Stack SVGs are often 2-10 KB minified single lines. BPE merge cost scales as word_length * num_words * vocab_size, so giant words make merge computation effectively never finish. Phase 1 worked because icons/fonts averaged ~500 chars/SVG. Fix: hard-wrap each SVG to 512-char lines before writing to the temp file. Total bytes are preserved (no content lost), the trainer sees ~932K word chunks instead of ~666K giant words, and merge computation completed in 2:57 (comparable to Phase 1's 2:21).

Why this is OK: BPE learns merges over byte pairs. Wrapping changes only the chunking the trainer iterates over, not the bytes themselves. The resulting tokenizer encodes full-length SVGs identically at inference.

I should also note: I had previously tried a `>`-based split that I claimed was a fix. It wasn't - it had no evidence behind it, didn't fix the stall, and I reverted it before the actual fix. Documenting both attempts here so the failure mode (length-times-count product, not data structure) is on record.

### Multi-element SVGs from non-simplified svg-stack
2026-05-01

Even after rebalancing, generated samples were all `<path>` only. No circles, no rects, no groups. Why? Every StarVector "simplified" dataset converts every SVG to a single path during preprocessing. So the model literally never saw `<circle>` etc. during training. The (v1) balanced corpus didn't help because it just rearranged the same single-path data.

Checked the *non-simplified* StarVector datasets. svg-emoji has 70% multi-element, svg-stack has 52%. Also looked at umuthopeyildirim/svgen-500k - all single-path, useless for this.

Streamed 100K multi-element SVGs from non-simplified svg-stack, filtered for SVGs containing one of `<circle`, `<rect`, `<g `, `<ellipse`, `<polygon`, `<line` and under 5000 chars. After cleaning, 98,832 survived. Appended to the balanced cleaned.jsonl, taking total to 815K SVGs.

Final corpus: 815K SVGs, 155M training tokens. By source: 42% fonts, 36% stack-simple, 12% stack-multi, 10% icons, <1% emoji. Tokenizer was retrained on this combined corpus. It now learns dedicated tokens for `circle`, `rect`, `<g `, `ellipse`, and `fill="#`, which is exactly what we wanted.

Trap I fell into: Google Drive FUSE mount silently drops appended data. I lost about 2 hours to this. The fix was to copy the cleaned.jsonl to local disk, append there, then copy back to Drive.

Outcome on the balanced XL (4 epochs on 155M tokens):
- val loss 1.86, perplexity 6.47 (vs 9.31 for the extended-original 4-epoch XL on 130M tokens)
- prefix render rate 33.3% (vs 6.7% extended-original, 20% 1-epoch original)
- unconditional render rate 8.3% (vs 73% extended-original, 80% 1-epoch original)

The unconditional drop is real but expected: longer/more-complex balanced SVGs frequently hit the 1024-token cap before they can close `</svg>`. The samples that *do* render include colored fills and multi-element groupings, which the original model literally couldn't produce.

### Generation prompt for the balanced tokenizer
2026-05-01

The balanced tokenizer learned different BPE merges than the original. I had to redo the prompt-injection investigation from scratch. Token 467 in the balanced tokenizer is the analogue of original-tokenizer's token 1024 - it merges the SVG header plus `<path ... d="M` into one chunk.

Tried multi-token prompts that ended at the `>` boundary of the SVG opening tag (the idea was to let the model choose what element comes next, rather than forcing `<path>`). Specifically `[1, 465, 463, 1560]`. Got 7-token outputs uniformly: the model immediately emits EOS because it never saw that exact token sequence during training, since every training example continues into a path token. So it has nothing to learn from at that boundary.

So the ideal "let the model pick the element" prompt didn't work. Settled on `[1, 467]` for unconditional, which matches training distribution but means every unconditional sample starts with a path. For prefix-conditioned, `tokenizer.encode()` works fine because the prefixes are intentionally partial - the model is conditioning on the actual text content rather than picking up a specific token boundary, so context-sensitivity of BPE encoding doesn't bite as hard there.

This is a fundamental coupling between training data and generation prompts under aggressive BPE merging. If your tokenizer merges the header with the first element, you can't easily prompt for a different element type without going out of distribution. Worth flagging in the discussion.

### Show both original and balanced in the report
2026-05-01

Three XL checkpoints I could plausibly call "the result":
- 1-epoch original (font-heavy 130M, val 3.11, perplexity 22.31, 80% / 20%)
- Extended original (4 epochs from same checkpoint, val 2.24, perplexity 9.31, 73% / 6.7%)
- Balanced (4 epochs, 155M, val 1.86, perplexity 6.47, 8.3% / 33.3%)

Decision: include both originals *and* the balanced in the report rather than picking one as "primary." Neither is strictly better. The original has a higher unconditional render rate but it's only producing one kind of output (single-path glyphs); the balanced produces more diverse output but more frequently runs out of context window. The actual finding is the tradeoff itself - training data composition is the dominant factor in generation quality, more so than capacity or training duration.

Report Table 8 puts them side by side. Figures show the original's unconditional grid (since that's where the original "wins") and the balanced's prefix grid (since that's where the balanced wins). The narrative explains the validity-vs-diversity tradeoff explicitly.

---

### Render validation: sample, not inline
2026-04-19

CairoSVG rendering is 10-100x slower than XML parsing. Running it inline in the cleaning pipeline would push cleaning from ~10 min to several hours. Since the StarVector data is already simplified (no filters, animations, gradients), I expected the render failure rate to be very low, in which case rejecting non-renderable SVGs would remove almost nothing while making the pipeline much slower.

So instead I render-check a random 1000-SVG sample after cleaning and report the rate in `dataset_stats.json` under `render_validation`. If the rate ever turned out to be unexpectedly low (under 95%) I'd switch to inline rendering, but it's been fine.
