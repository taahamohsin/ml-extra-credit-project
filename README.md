# SVG Scaling Laws

CS-GY 6923 Optional Project, NYU Tandon. Due May 1, 2026.

Train decoder-only transformers at 5 scales on SVG code, fit power-law scaling curves, compare standard parameterization (SP) vs muP, and generate/evaluate SVG samples from the best model. There's also a supplementary width-only muP comparison and a balanced-corpus retrain for Phase 4 generation.

---

## Setup

Recommended: Google Colab Pro with A100. All notebooks mount Google Drive and persist outputs there.

```bash
pip install -r requirements.txt

# cairosvg needs system Cairo libraries (Colab installs these automatically)
# On macOS locally:
brew install cairo pango gdk-pixbuf libffi
```

---

## Running the Experiments

Each phase has a notebook. Run them roughly in order. Phases 1-4 are the main spec deliverables; the `b` notebooks (1b, 3b, 4b) are supplementary work.

### Phase 1: Data pipeline (`notebooks/01_data_pipeline.ipynb`)

Downloads SVG datasets from HuggingFace, cleans and normalizes them, trains a BPE tokenizer, writes binary train/val/test splits.

Scripts called: `01_download_data.py`, `02_clean_normalize.py`, `03_train_tokenizer.py`, `04_prepare_dataset.py`

### Phase 2: Scaling study (`notebooks/02_scaling_study.ipynb`)

LR sweep on Tiny, trains all 5 SP model sizes for 1 epoch each, fits the power-law scaling curve.

Scripts called: `05_train_model.py`, `06_lr_sweep.py`, `11_plot_results.py`

### Phase 3: muP comparison (`notebooks/03_mup_study.ipynb`)

Trains all 5 model sizes under muP with the same LR sweep protocol, fits the muP scaling curve, runs the muP coordinate check for correctness validation.

Scripts called: `07_train_mup.py`, `08_lr_sweep_mup.py`, `09_coord_check_mup.py`

### Phase 4: Generation and evaluation (`notebooks/04_generation.ipynb`)

Continues XL training for 3 additional epochs, generates unconditional and prefix-conditioned samples from both the 1-epoch and extended checkpoints, evaluates XML validity and render rate.

Scripts called: `13_extend_xl.py`, `10_generate_samples.py`, `11_evaluate_samples.py`, `12_plot_samples.py`

### Phase 1b (supplementary): Balanced data pipeline (`notebooks/01b_data_pipeline_balanced.ipynb`)

Same pipeline as Phase 1 but with a different mix to address the font-glyph dominance of the original corpus (about 93% fonts). Outputs go to `*_balanced` directories so the original Phase 1 outputs aren't touched. The notebook walks through download, clean, tokenize, binary splits for the balanced corpus end-to-end.

Scripts called: `01b_download_balanced.py`, `02_clean_normalize.py`, `03_train_tokenizer.py`, `04_prepare_dataset.py` (with `--config configs/data_config_balanced.yaml`)

### Phase 3b (supplementary): Width-only muP comparison (`notebooks/03b_mup_width_only.ipynb`)

Phase 3's muP-vs-SP comparison is confounded by depth and head count varying alongside width. This notebook re-runs the comparison on a clean width-only family (depth=6, heads=4 fixed across all sizes, only `d_model` and `d_ff` vary).

Scripts called: `06b_lr_sweep_width_only.py`, `08b_lr_sweep_mup_width_only.py`, `05_train_model.py` and `07_train_mup.py` with `--config_family width_only`, `14_width_only_scaling.py`

### Phase 4b (supplementary): Generation on balanced corpus (`notebooks/04b_generation_balanced.ipynb`)

Retrains XL for 4 epochs on the balanced corpus, generates samples, evaluates. The notebook also includes a Step 2b cell that streams 100K multi-element SVGs from the non-simplified `starvector/svg-stack` and appends them to the cleaned corpus before tokenizer retraining; the final balanced training set is 815K SVGs / 155M tokens. This is the corpus the balanced XL was actually trained on.

Scripts called: `01b_download_balanced.py`, `02_clean_normalize.py`, `03_train_tokenizer.py`, `04_prepare_dataset.py`, `05_train_model.py`, `10_generate_samples.py`, `11_evaluate_samples.py`, `12_plot_samples.py`

---

## Scripts

| Script | Purpose |
|---|---|
| `01_download_data.py` | Download SVG datasets (icons + emoji + fonts) |
| `01b_download_balanced.py` | Download balanced 4-source corpus (icons + emoji + fonts subsample + stack) |
| `02_clean_normalize.py` | Clean, normalize, deduplicate SVGs |
| `03_train_tokenizer.py` | Train BPE tokenizer (vocab 4096) |
| `04_prepare_dataset.py` | Tokenize corpus, split 98/1/1, write binary |
| `05_train_model.py` | Train one SP model. Accepts `--config_family default\|width_only`, `--epochs`, `--ckpt_dir`, `--result_suffix`, `--data_config` |
| `06_lr_sweep.py` | SP LR sweep on Tiny |
| `06b_lr_sweep_width_only.py` | SP LR sweep on w_xs (width-only family) |
| `07_train_mup.py` | Train one muP model. Accepts the same `--config_family` and isolation flags as `05_train_model.py` |
| `08_lr_sweep_mup.py` | muP LR sweep on Tiny |
| `08b_lr_sweep_mup_width_only.py` | muP LR sweep on w_xs (width-only family) |
| `09_coord_check_mup.py` | muP coordinate check (validates muP wiring) |
| `10_generate_samples.py` | Generate unconditional and prefix samples. `--prompt_ids_override` for non-default tokenizers |
| `11_evaluate_samples.py` | Evaluate XML validity, render rate, test perplexity |
| `12_plot_samples.py` | Sample grids and temperature comparison plots |
| `13_extend_xl.py` | Continue XL training for additional epochs |
| `14_width_only_scaling.py` | Fit and plot SP-vs-muP scaling curves for the width-only family |

---

## Model Sizes

### Default family (Phases 2-4)

| Name | d_model | n_layers | n_heads | d_ff | Non-emb params |
|---|---|---|---|---|---|
| Tiny | 128 | 4 | 4 | 512 | 793K |
| Small | 192 | 6 | 6 | 768 | 2.7M |
| Medium | 384 | 6 | 6 | 1536 | 10.6M |
| Large | 512 | 10 | 8 | 2048 | 31.5M |
| XL | 768 | 12 | 12 | 3072 | 85.1M |

### Width-only family (Phase 3b only)

Depth and heads fixed across all sizes so muP's width transfer guarantee applies cleanly.

| Name | d_model | n_layers | n_heads | d_ff | Non-emb params |
|---|---|---|---|---|---|
| w_xs | 128 | 6 | 4 | 512 | 1.19M |
| w_small | 192 | 6 | 4 | 768 | 2.67M |
| w_medium | 256 | 6 | 4 | 1024 | 4.74M |
| w_large | 384 | 6 | 4 | 1536 | 10.65M |
| w_xl | 512 | 6 | 4 | 2048 | 18.92M |

Under muP with `base_d_model = 4 * 32 = 128`, w_xs is the literal base (width_mult=1) and the others get unambiguous width_mult values 1.5x / 2x / 3x / 4x.

---

## Key Hyperparameters

- Optimizer: AdamW, beta1=0.9, beta2=0.95, weight_decay=0.1
- LR schedule: cosine with 200-step linear warmup, min_lr = peak_lr / 10
- Batch size: 64 sequences x 1024 tokens = 65536 tokens/step (XL uses `--grad_accum 2` to halve peak memory)
- Mixed precision: bf16
- Selected LR (both SP and muP, both families): 3e-4

---

## Outputs

All outputs go under `outputs/` (symlinked to Google Drive on Colab, gitignored).

Phases 1-4 (original corpus):
- `outputs/checkpoints/xl/` -- 1-epoch and extended XL checkpoints
- `outputs/data/binary/` -- train.bin, val.bin, test.bin
- `outputs/tokenizer/` -- tokenizer.json
- `outputs/samples_1epoch/`, `outputs/samples_extended/` -- generated SVGs from the original-corpus XL
- `outputs/logs/` -- training CSVs and `result_*.json` files
- `outputs/plots/` -- figures for the report

Phases 1b/4b (balanced corpus):
- `outputs/checkpoints/xl_balanced/xl/` -- balanced XL checkpoint
- `outputs/data/{raw_balanced,cleaned_balanced,binary_balanced}/` -- balanced corpus at each pipeline stage
- `outputs/tokenizer_balanced/` -- balanced tokenizer
- `outputs/samples_balanced/` -- generated SVGs from the balanced XL
- `outputs/logs/result_xl_balanced.json` -- balanced training result
- `outputs/balanced_prompt_ids.json` -- discovered token IDs for unconditional prompts on the balanced tokenizer

Phase 3b (width-only):
- `outputs/checkpoints_width_only/{sp,mup}/` -- SP and muP checkpoints per width
- `outputs/logs/result_{sp,mup}_w_*_width_only.json` -- per-model results
- `outputs/logs/lr_sweep_width_only_{sp,mup}.json` -- sweep results
- `outputs/plots/scaling_law_width_only.png`, `outputs/plots/lr_sweep_width_only*.png`

---

## Documentation

- `decision_log.md`: chronological record of design decisions, bugs encountered, and how I diagnosed them. The longer entries are mostly muP-related (base shapes, scheduler interaction with MuAdamW, attention scale).
- `report_final.tex`: the writeup. References figures from `outputs/plots/`.

---

## References

1. Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361
2. Hoffmann et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). arXiv:2203.15556
3. Yang et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. arXiv:2203.09789
4. Rodriguez et al. (2023). StarVector: Generating Scalable Vector Graphics Code from Images and Text. arXiv:2312.11556
