# SVG Scaling Laws — Complete Project Blueprint

## CS-GY 6923 Optional Project | Due: May 1, 2026

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [Repository Structure](#2-repository-structure)
3. [Compute Strategy (Colab Pro + A100)](#3-compute-strategy)
4. [Phase 1: Data Pipeline](#4-phase-1-data-pipeline-15)
5. [Phase 2: Transformer Scaling Study](#5-phase-2-transformer-scaling-study-35)
6. [Phase 3: µP Scaling and Extrapolation](#6-phase-3-µp-scaling-and-extrapolation-25)
7. [Phase 4: Generation and Evaluation](#7-phase-4-generation-and-evaluation-15)
8. [Phase 5: Analysis and Report](#8-phase-5-analysis-and-report-10)
9. [Key Concepts You Must Understand](#9-key-concepts-you-must-understand)
10. [Risk Mitigation & Fallback Plans](#10-risk-mitigation--fallback-plans)
11. [Timeline](#11-timeline)
12. [Decision Log Template](#12-decision-log)

---

## 1. Project Overview & Goals

**One-sentence summary:** Train decoder-only transformers at 5+ scales on SVG code, fit power-law scaling curves, compare standard vs. µP parameterization, and generate/evaluate SVG samples.

**Why this is interesting (for the report intro):**
- SVG is a structured, non-linguistic domain — unlike natural language, it has strict syntax (valid XML), a coordinate system, and outputs can be visually rendered
- Scaling laws (Kaplan et al., 2020) were established for natural language — do they hold for structured code?
- µP (Yang et al., 2022) promises zero-shot hyperparameter transfer — does it work in practice on a non-NL domain?
- Generated outputs are visually inspectable — we can literally see what the model learned

**What we're measuring:**
- How validation loss decreases as model size increases (the scaling law)
- Whether µP gives better scaling than fixed learning rate
- Whether generated SVGs are valid XML, renderable, and visually coherent

---

## 2. Repository Structure

```
svg-scaling-laws/
├── README.md                    # Setup instructions, how to run
├── requirements.txt             # All dependencies
├── decision_log.md              # Every design decision documented
│
├── configs/
│   ├── data_config.yaml         # Data pipeline parameters
│   ├── model_configs.yaml       # All 5 model architectures
│   └── training_config.yaml     # Training hyperparameters
│
├── scripts/
│   ├── 01_download_data.py      # Download from HuggingFace
│   ├── 02_clean_normalize.py    # SVG cleaning pipeline
│   ├── 03_train_tokenizer.py    # BPE tokenizer training
│   ├── 04_prepare_dataset.py    # Tokenize, split, save as binary
│   ├── 05_train_model.py        # Main training script (SP)
│   ├── 06_lr_sweep.py           # Learning rate sweep
│   ├── 07_train_mup.py          # Training with µP
│   ├── 08_lr_sweep_mup.py       # LR sweep for µP
│   ├── 09_generate_samples.py   # Sample generation
│   ├── 10_evaluate.py           # Evaluation metrics
│   └── 11_plot_results.py       # All plots for the report
│
├── src/
│   ├── __init__.py
│   ├── model.py                 # Transformer model definition
│   ├── model_mup.py             # µP version of the model
│   ├── dataset.py               # PyTorch Dataset class
│   ├── tokenizer_utils.py       # Tokenizer helpers
│   ├── svg_utils.py             # SVG cleaning/validation/rendering
│   ├── training_utils.py        # Training loop, logging, checkpointing
│   └── scaling_law.py           # Power law fitting utilities
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb           # Run Phase 1 end-to-end
│   ├── 02_scaling_study.ipynb           # Run Phase 2 (training + plots)
│   ├── 03_mup_study.ipynb               # Run Phase 3
│   ├── 04_generation.ipynb              # Run Phase 4
│   └── 05_all_plots_and_analysis.ipynb  # Final plots for report
│
└── outputs/                     # Generated during runs
    ├── data/                    # Processed data files
    ├── tokenizer/               # Saved tokenizer
    ├── checkpoints/             # Model checkpoints
    ├── logs/                    # Training logs (CSV)
    ├── samples/                 # Generated SVG samples
    ├── plots/                   # All figures for report
    └── report/                  # Final PDF report
```

**Why this structure:**
- `scripts/` are numbered so the execution order is obvious
- `src/` contains reusable modules imported by scripts
- `notebooks/` wrap the scripts for Colab with GPU access
- `configs/` centralizes all hyperparameters (easy to change, easy to document)
- Separation of scripts and notebooks means Claude Code builds the logic, notebooks just orchestrate

---

## 3. Compute Strategy

### Hardware: Colab Pro with A100 (40GB VRAM)

**Memory estimates per model:**

| Model  | ~Params | Est. Memory (train, fp32) | Est. Memory (train, mixed precision) |
|--------|---------|---------------------------|--------------------------------------|
| Tiny   | ~1M     | ~0.5 GB                   | ~0.3 GB                              |
| Small  | ~3M     | ~1 GB                     | ~0.6 GB                              |
| Medium | ~10M    | ~3 GB                     | ~1.5 GB                              |
| Large  | ~30M    | ~7 GB                     | ~4 GB                                |
| XL     | ~88M    | ~20 GB                    | ~10 GB                               |

All models fit comfortably on A100. We'll use **mixed precision (fp16/bf16)** to save memory and get ~2x speedup.

**Time estimates (A100, 100M tokens, 1 epoch):**

| Model  | Est. time/epoch |
|--------|-----------------|
| Tiny   | ~10 min         |
| Small  | ~15 min         |
| Medium | ~30 min         |
| Large  | ~60 min         |
| XL     | ~2-3 hours      |

**Total GPU time budget:**
- LR sweep (7 runs × Tiny): ~70 min
- 5 models × 1 epoch (SP): ~4-5 hours
- LR sweep (7 runs × Tiny, µP): ~70 min
- 5 models × 1 epoch (µP): ~4-5 hours
- Best model extended training: ~3-5 hours
- Miscellaneous (debugging, generation): ~2 hours
- **Total: ~16-20 hours of A100 time**

**Colab Pro strategy:**
- Use "Connect to a runtime" → GPU → A100 (may need to retry if unavailable)
- Save checkpoints to Google Drive so you don't lose progress on disconnect
- Use `torch.compile()` if PyTorch 2.0+ for additional speedup
- Each notebook should be runnable independently (resume from saved state)

### Checkpoint & Resume Strategy

Every training script must:
1. Save checkpoints every N steps to Google Drive
2. Log metrics to CSV files (not just stdout)
3. Support `--resume` flag to continue from last checkpoint
4. Save the best model (lowest val loss) separately

```python
# Mount Google Drive at the start of every notebook
from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/svg-scaling-laws/'
```

---

## 4. Phase 1: Data Pipeline (15%)

### 4.1 Datasets

**Primary: `starvector/svg-icons-simple`**
- ~89,370 simplified SVG icons
- Already cleaned by StarVector team (no gradients, filters, animations)
- Load with: `load_dataset("starvector/svg-icons-simple")`

**Supplementary (if needed for 100M token target):**
- `starvector/svg-emoji-simple` — 14.5 MB of emoji SVGs
- `starvector/svg-fonts-simple` — 2.38 GB of font glyphs (subsample)
- `starvector/svg-stack-simple` — 3.87 GB of diverse SVGs (subsample)

**Strategy:** Start with icons-simple. After tokenization, count tokens. If under 100M, add emoji-simple. If still under, subsample from fonts-simple until we hit 100M. Document exactly what we used and how much.

### 4.2 SVG Cleaning Pipeline

Each SVG goes through these steps IN ORDER:

```
Raw SVG
  → Strip XML comments (<!-- ... -->)
  → Strip <?xml?> processing instructions
  → Strip <metadata>, <desc>, <title> blocks
  → Extract <svg>...</svg> (discard anything outside)
  → Round decimal numbers to 1 decimal place
  → Collapse whitespace (multiple spaces → single space)
  → Validate: is it valid XML? (lxml.etree.fromstring)
  → Length filter: discard if < 50 characters
  → Deduplicate by MD5 hash
  → Store cleaned SVG
```

**Design decisions to document:**

| Decision | Choice | Justification |
|----------|--------|---------------|
| Coordinate precision | 1 decimal place | Reduces unique number tokens dramatically. At icon scale (24x24 viewBox), 0.1px precision is sub-pixel and visually imperceptible. |
| Min length | 50 characters | SVGs shorter than this are typically empty or trivial (just an `<svg></svg>` wrapper). |
| Max token length | 1024 tokens | Keeps context window manageable for all model sizes. The median SVG icon is much shorter. Removes only very complex outliers. |
| Deduplication | MD5 hash of cleaned SVG | Many icon sets contain duplicates. Training on duplicates wastes compute and inflates metrics. |
| XML validation | lxml strict parsing | Ensures 100% of training data is syntactically valid. Model should never see broken SVG. |

### 4.3 Tokenizer

**Type:** BPE (Byte Pair Encoding) via HuggingFace `tokenizers` library

**Vocabulary size:** 4096

**Why 4096:**
- The SVG domain has a relatively constrained vocabulary compared to natural language
- Common patterns: `<svg`, `<path`, `<circle`, `<rect`, `<g`, `fill=`, `d="M`, `viewBox=`, hex colors `#FFFFFF`, coordinate numbers
- 4096 is enough to learn all common SVG substrings as single tokens
- Smaller vocab (1K) → sequences are too long, model needs more capacity for syntax
- Larger vocab (8K+) → many rare tokens with poor embeddings given our data size
- 4096 = 2^12, which is clean for embedding table sizing

**Special tokens:**
- `<PAD>` (id=0): padding for batch collation
- `<BOS>` (id=1): beginning of sequence, prepended to every SVG
- `<EOS>` (id=2): end of sequence, appended to every SVG
- `<UNK>` (id=3): unknown (BPE should rarely produce this)

**Pre-tokenizer:** ByteLevel (treats raw bytes, handles all Unicode)

**What to show in report:**
- Example tokenizations of 3-4 SVGs at different complexities
- Token frequency distribution (Zipf-like plot)
- Most common tokens (should be SVG-specific like `<path`, `fill`)

### 4.4 Data Splitting and Binary Preparation

**Split:** 98% train / 1% val / 1% test **by file** (not by token position)

**Why by file:** If we split by token position within a concatenated file, the model could see the beginning of an SVG in training and the end in validation — that's data leakage. Splitting by file means entire SVGs are in one split only.

**Binary format for training:**
```python
# After tokenization, concatenate all token IDs with <BOS>/<EOS> separators:
# <BOS> [svg1_tokens] <EOS> <BOS> [svg2_tokens] <EOS> ...
# Save as numpy memmap for memory-efficient loading during training
# This is the same approach nanoGPT uses
```

**Token count check:** After preparing the training split, count total tokens. Print a clear message:
```
Training tokens: 142,387,291 ✓ (target: 100M minimum)
```

### 4.5 Statistics to Compute and Save

```python
stats = {
    "datasets_used": ["svg-icons-simple", "svg-emoji-simple"],  # list what we used
    "files_before_cleaning": 89370,
    "files_after_cleaning": 85201,      # example
    "files_removed_reasons": {
        "failed_normalize": 234,
        "too_short": 1892,
        "invalid_xml": 143,
        "duplicates": 1900,
    },
    "vocab_size": 4096,
    "train_files": 83497,
    "val_files": 852,
    "test_files": 852,
    "train_tokens": 142387291,
    "val_tokens": 1453921,
    "test_tokens": 1447832,
    "sequence_length_stats": {
        "mean": 187.3,
        "median": 142,
        "p95": 512,
        "p99": 891,
        "max": 1024,
    },
}
```

### 4.6 Plots to Generate

1. **Sequence length histogram** — X: token count per SVG, Y: frequency. Log scale on Y.
2. **Example SVGs at different complexity levels** — Render 3 SVGs: simple (~50 tokens), medium (~200 tokens), complex (~800 tokens). Show both the SVG code snippet and the rendered image side by side.
3. **Token frequency distribution** — Zipf plot of token frequencies.

---

## 5. Phase 2: Transformer Scaling Study (35%)

### 5.1 Model Architecture

**Type:** Decoder-only Transformer (GPT-style)

This is the same architecture as GPT-2/GPT-3, just smaller. Key components:
- Token embedding layer (vocab_size → d_model)
- Learned positional embeddings (max_seq_len → d_model)
- N transformer blocks, each containing:
  - Multi-head causal self-attention
  - Layer normalization (pre-norm style, like GPT-2)
  - Feed-forward network (d_model → d_ff → d_model, with GELU activation)
  - Residual connections
- Final layer norm → linear projection to vocab_size → softmax

**Model configurations:**

| Name   | d_model | n_layers | n_heads | d_ff  | ~Params | head_dim |
|--------|---------|----------|---------|-------|---------|----------|
| Tiny   | 128     | 4        | 4       | 512   | ~1M     | 32       |
| Small  | 192     | 6        | 6       | 768   | ~3M     | 32       |
| Medium | 384     | 6        | 6       | 1536  | ~10M    | 64       |
| Large  | 512     | 10       | 8       | 2048  | ~30M    | 64       |
| XL     | 768     | 12       | 12      | 3072  | ~88M    | 64       |

**Exact parameter count formula:**
```
Embedding:     vocab_size × d_model + max_seq_len × d_model
Per block:     4 × d_model² + 2 × d_model × d_ff + biases + layer_norm
Output head:   d_model × vocab_size (often weight-tied with embedding)
Total:         Embedding + n_layers × per_block + output_head
```

We should weight-tie the embedding and output head (standard practice, saves params).

**Context window:** 1024 tokens (matches our max token length filter)

### 5.2 Training Setup

**Optimizer:** AdamW
- β1 = 0.9, β2 = 0.95 (standard for LMs)
- Weight decay = 0.1 (applied to all params except biases and layer norms)
- Gradient clipping: max_norm = 1.0

**Learning rate schedule:** Cosine decay with linear warmup
```
warmup_steps = 200  (or ~1% of total steps, whichever is larger)
After warmup: cosine decay from peak LR to min_lr = peak_lr / 10
```

**Batch size:** Fixed across all models, measured in tokens
- Suggestion: 64 sequences × 1024 tokens = 65,536 tokens per batch
- If memory allows on A100, can go to 128 × 1024 = 131,072
- Use gradient accumulation if needed

**Mixed precision:** bf16 on A100 (more stable than fp16, no loss scaling needed)

**Epochs:** Exactly 1 for the scaling comparison (all models see the same data once)

### 5.3 Learning Rate Sweep (on Tiny model)

**Why:** The learning rate is the most impactful hyperparameter. Too low = slow convergence. Too high = training diverges. We find the sweet spot on the cheapest model.

**Protocol:**
```python
lr_values = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]  # 7 values, log-spaced
```

For each LR:
1. Train Tiny model for 1 epoch (or a fixed number of steps, e.g., 5000)
2. Record final validation loss
3. Plot: LR (log scale) vs. validation loss → pick the LR at the minimum

**Expected result:** A U-shaped curve. Too-low LRs barely train, too-high LRs diverge. The optimal is typically around 1e-3 to 3e-3 for models this size.

**IMPORTANT:** Use this SAME learning rate for ALL model sizes in Part 2. This is the "standard parameterization" baseline. The whole point of Part 3 is to show that this approach is suboptimal — the best LR for Tiny is not the best for XL.

### 5.4 Scaling Law Fitting

After training all 5 models for 1 epoch, we have 5 data points:
```
(N_tiny, L_tiny), (N_small, L_small), ..., (N_xl, L_xl)
```
where N = parameter count, L = validation loss.

**Fit the power law:** L = a · N^(-α) + c

- `a` = scaling coefficient
- `α` = scaling exponent (the key number — how fast loss drops with scale)
- `c` = irreducible loss floor (entropy of the data)

**Fitting method:** Use `scipy.optimize.curve_fit` with initial guesses:
```python
from scipy.optimize import curve_fit
import numpy as np

def power_law(N, a, alpha, c):
    return a * N**(-alpha) + c

# Initial guesses
p0 = [10.0, 0.5, 1.0]
# Bounds: a > 0, 0 < alpha < 2, c > 0
bounds = ([0, 0, 0], [np.inf, 2, np.inf])

popt, pcov = curve_fit(power_law, param_counts, val_losses, p0=p0, bounds=bounds)
a, alpha, c = popt
```

**What to report:**
- Fitted α value (for natural language, Kaplan et al. found α ≈ 0.076; ours may differ)
- R² of the fit
- Plot with data points + fitted curve on log-log axes
- Discussion: is α similar to NL? What does that mean about SVG complexity?

### 5.5 Additional Metrics to Track

For EVERY training run, log to CSV at regular intervals (e.g., every 100 steps):
```csv
step, train_loss, val_loss, learning_rate, tokens_seen, wall_time_sec, gpu_memory_mb
```

**Plots to generate:**
1. **Scaling plot** — log(params) vs. val_loss with power law fit (the main result)
2. **Training curves** — step vs. train_loss for all 5 models overlaid
3. **LR sweep plot** — log(LR) vs. final val_loss (U-shaped curve)
4. **Throughput table** — model name, params, tokens/sec, GPU memory, wall time

---

## 6. Phase 3: µP Scaling and Extrapolation (25%)

### 6.1 What is µP and Why It Matters

**The problem:** In standard parameterization (SP), the optimal learning rate changes as you scale up the model width. The LR you found for Tiny is probably too high for XL — it may cause instability or suboptimal convergence. This means you'd need to re-tune LR for every model size, which is expensive.

**The solution (µP):** Maximal Update Parameterization changes how layers are initialized and how learning rates are scaled per-layer, so that the optimal LR transfers across model widths. You tune once on a small model, and it works for larger models — "zero-shot hyperparameter transfer."

**Key changes µP makes:**
1. **Initialization:** Output layer weights initialized with 1/width scaling instead of 1/√width
2. **Learning rate per layer:** Different layers get different effective LRs based on width
3. **Attention scaling:** Changes from 1/√d_head to 1/d_head
4. **Embedding LR:** Input embeddings get a different LR multiplier

### 6.2 Implementation with `mup` Package

```python
# Install: pip install mup

from mup import MuReadout, MuSharedReadout, make_base_shapes, set_base_shapes
from mup.optim import MuAdamW

# Step 1: Define a "base model" (smallest width) and "delta model" (slightly wider)
base_model = TransformerLM(d_model=128, ...)   # base width
delta_model = TransformerLM(d_model=256, ...)  # any wider width

# Step 2: Save base shapes
base_shapes = make_base_shapes(base_model, delta_model, savefile="base_shapes.bsh")

# Step 3: For any target width, set base shapes then create optimizer
target_model = TransformerLM(d_model=768, ...)
set_base_shapes(target_model, base_shapes)

# Step 4: Use MuAdamW instead of AdamW
optimizer = MuAdamW(target_model.parameters(), lr=optimal_lr_from_sweep)
```

**Model modifications needed for µP:**
- Replace final `nn.Linear` output layer with `MuReadout`
- If weight-tying embedding and output, use `MuSharedReadout`
- Attention: use `1/d` scaling instead of `1/√d` (the mup package may handle this, but verify)
- The `mup` package docs have a specific "µP for Transformers" section — follow it

### 6.3 µP Experimental Protocol

1. **Modify model code** to support µP (create `model_mup.py`)
2. **Generate base shapes** from Tiny (d_model=128) and a delta model (d_model=256)
3. **LR sweep on Tiny µP model:** Same protocol as Part 2 (7 LRs, log-spaced)
4. **Train all 5 sizes** with the optimal µP LR (same data, same epochs)
5. **Compare:**
   - Plot SP scaling curve and µP scaling curve on the SAME graph
   - Fit power law to both
   - Compare α values
   - Show LR sweep comparison (SP vs µP on small model)

### 6.4 Expected Results

- For small models (Tiny, Small): SP and µP should perform similarly (LR from Tiny is near-optimal for Tiny by definition)
- For larger models (Large, XL): µP should show lower validation loss because the LR is better adapted to the width
- The gap should widen with scale
- µP scaling curve should have a steeper α (better scaling exponent)

If µP does NOT help: that's also a valid result! Document it and discuss why (maybe the width range isn't large enough, or the 1-epoch training doesn't show the difference).

### 6.5 Scaling Law Extrapolation

Using whichever scaling law fit is better (SP or µP):

1. **Predict loss for 10× larger model:** If XL has ~88M params, predict loss at ~880M params
   ```python
   predicted_loss = a * (880e6)**(-alpha) + c
   ```

2. **Confidence interval:** Use the covariance matrix from `curve_fit`:
   ```python
   # pcov is the covariance matrix from curve_fit
   # Use it to compute prediction uncertainty
   from scipy.optimize import curve_fit
   popt, pcov = curve_fit(...)
   perr = np.sqrt(np.diag(pcov))  # 1-sigma parameter uncertainties
   ```

3. **Discussion points:**
   - Power laws are empirical — they may break down at scales we haven't tested
   - Chinchilla (Hoffmann et al., 2022) showed that scaling laws also depend on data — 100M tokens might not be enough data for an 880M model
   - The prediction is an extrapolation 10× beyond our data — high uncertainty is expected and honest

---

## 7. Phase 4: Generation and Evaluation (15%)

### 7.1 Best Model Selection

Choose the model with the best (lowest) validation loss from Parts 2-3. This is likely:
- XL model with µP parameterization
- Optionally train for additional epochs (2-3) to improve quality

### 7.2 Extended Training

- Train the best model for as many epochs as feasible (track val loss — stop if it plateaus or increases)
- Try light hyperparameter tuning: dropout (0.0, 0.1), weight decay (0.01, 0.1)
- Save the best checkpoint by val loss

### 7.3 Sample Generation

**Unconditional generation (10+ samples):**
```python
# Start with: <BOS> <svg
# Let the model generate tokens autoregressively until <EOS> or max_length
prompt_tokens = tokenizer.encode("<svg").ids
generated = model.generate(prompt_tokens, max_length=1024, ...)
```

**Prefix-conditioned generation (5+ samples):**
```python
prefixes = [
    # Partial face: circle + one eye
    '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="black"/><circle cx="9" cy="10" r="1" fill="black"/>',

    # Open path
    '<svg viewBox="0 0 24 24"><path d="M2 12 L12 2 L22 12',

    # Group with one shape
    '<svg viewBox="0 0 24 24"><g fill="red"><rect x="4" y="4" width="6" height="6"/>',

    # Star beginning
    '<svg viewBox="0 0 24 24"><polygon points="12,2 15,9',

    # Nested structure
    '<svg viewBox="0 0 24 24"><g transform="translate(12,12)"><circle r="10" fill="none" stroke="blue"/>',
]
```

**Sampling strategies to try:**
- Temperature: 0.5, 0.8, 1.0, 1.2
- Top-k: 50
- Top-p (nucleus): 0.9, 0.95
- Compare: low temperature (sharp, conservative) vs. high temperature (creative, noisy)

### 7.4 Evaluation Metrics

```python
def evaluate_samples(generated_svgs):
    results = {
        "total": len(generated_svgs),
        "xml_valid": 0,       # parses with lxml
        "svg_renderable": 0,  # renders with CairoSVG without error
        "has_svg_root": 0,    # root element is <svg>
        "tags_closed": 0,     # all tags properly closed
    }

    for svg in generated_svgs:
        # XML validity
        try:
            tree = etree.fromstring(svg.encode())
            results["xml_valid"] += 1
            if tree.tag.endswith("svg") or tree.tag == "svg":
                results["has_svg_root"] += 1
        except:
            continue

        # Render test
        try:
            cairosvg.svg2png(bytestring=svg.encode())
            results["svg_renderable"] += 1
        except:
            pass

    # Compute rates
    for key in ["xml_valid", "svg_renderable", "has_svg_root"]:
        results[f"{key}_rate"] = results[key] / results["total"]

    return results
```

**Test set perplexity:**
```python
perplexity = torch.exp(average_cross_entropy_loss_on_test_set)
```

### 7.5 Visualization

1. **Sample grid:** 4×3 grid of rendered unconditional samples
2. **Temperature comparison:** Same prompt, 4 temperatures, show rendered results
3. **Prefix completion:** Show prefix code → completion code → rendered result (3 columns)

---

## 8. Phase 5: Analysis and Report (10%)

### 8.1 Report Structure

The PDF should follow the structure in the assignment:

| Section | Pages | Content |
|---------|-------|---------|
| Introduction | 0.5-1 | Why SVG scaling laws matter, approach overview |
| Data | 1-1.5 | Pipeline description, tokenizer details, stats, rendered examples |
| Methods | 1.5-2 | Architecture table, training setup, LR schedule, µP explanation |
| Results | 2-3 | Scaling plots, LR sweeps, SP vs µP comparison, extrapolation, samples |
| Discussion | 1-2 | Key insights, design decisions, limitations, future work |
| Conclusion | 0.5 | Summary of findings |
| References | - | Kaplan 2020, Hoffmann 2022, Yang 2022, Rodriguez 2023 |
| Appendix | - | Extra samples, detailed tables, code snippets |

### 8.2 Key Discussion Points

**Scaling insights:**
- Compare our α to Kaplan et al.'s α ≈ 0.076 for NL
- If α is different: SVG might be "easier" (higher α, steeper curve) because it's more structured, or "harder" (lower α) because visual coherence requires more capacity
- Discuss the irreducible loss floor c — what does it represent for SVG?

**µP insights:**
- At what model size did SP vs µP diverge?
- Was the µP optimal LR higher or lower than the SP optimal LR?
- Did µP change the scaling exponent or just shift the curve?

**SVG-specific observations:**
- Did small models learn basic syntax (valid XML) but produce visual noise?
- Did larger models learn spatial coherence (symmetry, centering)?
- Were there "phase transitions" — sudden jumps in capability at certain scales?

---

## 9. Key Concepts You Must Understand

### For defending your work, you should be able to explain:

**1. Scaling Laws**
- "As you increase model size, validation loss decreases following a power law: L = a·N^(-α) + c"
- α tells you how efficiently the model uses additional parameters
- c is the floor — the best any model could do (related to inherent randomness/entropy in data)
- Kaplan et al. found this for natural language; we're testing if it holds for SVG

**2. BPE Tokenization**
- Start with characters, iteratively merge most common pairs
- "We chose 4096 because SVG has a constrained vocabulary — fewer unique patterns than English — so a smaller vocab captures the important subwords while keeping sequences short"

**3. Why learning rate matters at different scales**
- In standard parameterization, wider layers have larger weight matrices
- Same LR → different effective update magnitudes at different widths
- A LR good for a 128-dim model applies too-large updates to a 768-dim model's weights

**4. What µP does**
- Adjusts initialization and per-layer learning rates so the effective update magnitude stays constant regardless of width
- "It's a reparameterization that makes the optimal LR invariant to model width"
- Key practical benefit: tune once on small model, transfer to large model

**5. Cosine LR schedule with warmup**
- Warmup: linearly increase LR from 0 to peak over first N steps (prevents early instability)
- Cosine decay: smoothly decrease LR following a cosine curve (avoids abrupt drops)
- Better than constant LR because early training needs high LR (explore), late training needs low LR (refine)

**6. Why we train for exactly 1 epoch for scaling comparison**
- Controls for data — every model sees the same data the same number of times
- Isolates the effect of model size from training duration
- If some models trained longer, the scaling plot would conflate two variables

**7. Power law fitting**
- We use nonlinear least squares (curve_fit) to find a, α, c
- Plot on log-log scale: a power law appears as a straight line
- R² tells us how well the power law fits our data

---

## 10. Risk Mitigation & Fallback Plans

| Risk | Mitigation |
|------|-----------|
| Can't reach 100M training tokens from icons-simple alone | Add emoji-simple and subsample from fonts-simple. Track exact counts. |
| A100 unavailable on Colab | Fall back to V100 or T4. Scale down XL model if needed (reduce d_model to 512). Document this choice. |
| Colab disconnects during training | Checkpoint every 500 steps to Google Drive. All scripts support `--resume`. |
| `mup` package has breaking changes or compatibility issues | Pin exact version in requirements.txt. If mup fails, implement µP manually (it's just init scaling + LR multipliers per layer). |
| Training diverges for large models | Reduce LR by 3×, increase warmup. Add gradient clipping (max_norm=1.0). |
| Generated SVGs are all garbage | This is expected at small scales — document it as a finding. Focus analysis on what the model DID learn (syntax, basic shapes). |
| Power law fit is poor (low R²) | Could mean we need more model sizes, or the relationship isn't a clean power law. Try alternative functional forms. Report honestly. |

---

## 11. Timeline

**Assuming start: April 15 (today) → Due: May 1**

| Day | Task | Who |
|-----|------|-----|
| Apr 15 | Build Phase 1 code (data pipeline) | Claude Code |
| Apr 15 | Run Phase 1 on Colab, verify stats | You |
| Apr 16 | Build Phase 2 code (model + training) | Claude Code |
| Apr 16-17 | Run LR sweep (Tiny, ~70 min) | Colab |
| Apr 17-18 | Train 5 models SP (4-5 hours GPU) | Colab |
| Apr 18 | Build Phase 3 code (µP) | Claude Code |
| Apr 18-19 | Run µP LR sweep + 5 models (4-5 hours) | Colab |
| Apr 20 | Build Phase 4 code (generation + eval) | Claude Code |
| Apr 20-21 | Train best model extended + generate samples | Colab |
| Apr 22-23 | Generate all plots, analyze results | Claude Chat |
| Apr 24-25 | Draft report | Claude Chat + You |
| Apr 26-27 | Revise report, study concepts | You |
| Apr 28-30 | Buffer for issues, final polish | Everyone |

**Key principle:** Start GPU runs as early as possible. Code writing can happen in parallel while models train.

---

## 12. Decision Log

Use this format throughout the project. Every non-obvious choice gets an entry.

```markdown
### Decision: [Short title]
**Date:** YYYY-MM-DD
**Options considered:**
1. Option A — pros/cons
2. Option B — pros/cons
**Choice:** Option B
**Reasoning:** [Why this option]
**Impact:** [What this affects downstream]
```

### Initial Decisions (pre-implementation):

### Decision: Vocabulary size = 4096
**Date:** 2026-04-14
**Options considered:**
1. 1024 — Very compact vocab, but SVG sequences would be very long (~3-4× longer)
2. 4096 — Good balance of sequence length and token coverage for SVG-specific patterns
3. 8192 — Many rare tokens would have poor embeddings with only 100M training tokens
**Choice:** 4096
**Reasoning:** SVG has fewer unique patterns than natural language. 4096 captures all common SVG tags, attributes, hex colors, and coordinate patterns while keeping sequence lengths manageable. At 100M training tokens, each of 4096 tokens appears ~24K times on average — sufficient for learning good embeddings.
**Impact:** Determines sequence lengths, model context window, and embedding table size.

### Decision: Max sequence length = 1024 tokens
**Date:** 2026-04-14
**Options considered:**
1. 512 — Faster training, but would filter out many icons
2. 1024 — Covers ~99% of icons, reasonable context window
3. 2048 — Covers everything, but quadratic attention cost is high for small models
**Choice:** 1024
**Reasoning:** After tokenization with 4096 BPE vocab, the vast majority of simplified SVG icons fit within 1024 tokens. This keeps attention computation manageable and matches well with nanoGPT's default setup.
**Impact:** Determines positional embedding size, attention memory cost, and which SVGs are filtered out.

### Decision: Use nanoGPT as reference (modified)
**Date:** 2026-04-14
**Options considered:**
1. Build transformer from scratch — full control, but more bugs
2. Use nanoGPT as base — well-tested, clean code, widely known
3. Use HuggingFace Transformers — too much abstraction, harder to modify for µP
**Choice:** nanoGPT-inspired, rewritten for our needs
**Reasoning:** nanoGPT is clean, minimal, and easy to understand. We'll use its design patterns (GPT class, Block class, CausalSelfAttention) but rewrite to support µP and our specific needs. This way we understand every line.
**Impact:** Model code structure, familiarity for debugging.

### Decision: Weight tying (embedding ↔ output head)
**Date:** 2026-04-14
**Options considered:**
1. Separate embedding and output weights — more params, more flexibility
2. Tie weights — fewer params, standard practice, regularization effect
**Choice:** Tie weights
**Reasoning:** Standard in GPT-2 and most modern LMs. Reduces parameter count (especially significant for small models where the embedding table is a large fraction of total params). Also provides implicit regularization.
**Impact:** Parameter counts will be lower. Need special handling for µP (use MuSharedReadout).

### Decision: Pre-norm (LayerNorm before attention/FFN)
**Date:** 2026-04-14
**Options considered:**
1. Post-norm (original transformer) — less stable, needs careful LR tuning
2. Pre-norm (GPT-2 style) — more stable training, easier to scale
**Choice:** Pre-norm
**Reasoning:** Pre-norm is more stable for training, especially important when we're sweeping LRs and scaling up. It's the standard for GPT-style models.
**Impact:** Model architecture, stability of training.

### Decision: Mixed precision with bf16
**Date:** 2026-04-14
**Options considered:**
1. fp32 — safe but slow, uses 2× memory
2. fp16 — fast but needs loss scaling, can have NaN issues
3. bf16 — fast, same dynamic range as fp32, no loss scaling needed
**Choice:** bf16
**Reasoning:** A100 has native bf16 support. bf16 has the same exponent range as fp32 so it doesn't need loss scaling (unlike fp16). Simpler code, fewer training issues.
**Impact:** ~2× faster training, ~2× less memory, enables larger batch sizes.

---

## Appendix: References

1. Kaplan et al. (2020). "Scaling Laws for Neural Language Models." https://arxiv.org/abs/2001.08361
2. Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." https://arxiv.org/abs/2203.15556
3. Yang et al. (2022). "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." https://arxiv.org/abs/2203.09789
4. Rodriguez et al. (2023). "StarVector: Generating Scalable Vector Graphics Code from Images and Text." https://arxiv.org/abs/2312.11556
