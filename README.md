# SVG Scaling Laws

CS-GY 6923 Optional Project, NYU Tandon. Due May 1, 2026.

Train decoder-only transformers at 5 scales on SVG code, fit power-law scaling curves, compare standard vs. muP parameterization, and generate/evaluate SVG samples from the best model.

---

## Setup

**Recommended:** Google Colab Pro with A100 GPU. All notebooks are designed to run on Colab with outputs persisted to Google Drive.

```bash
pip install -r requirements.txt

# cairosvg needs system Cairo libraries (Colab installs these automatically)
# On macOS locally:
brew install cairo pango gdk-pixbuf libffi
```

---

## Running the Experiments

Each phase has a corresponding notebook. Run them in order.

**Phase 1 -- Data pipeline** (`notebooks/01_data_pipeline.ipynb`)

Downloads SVG datasets from HuggingFace, cleans and normalizes SVGs, trains a BPE tokenizer, and writes binary train/val/test splits.

Scripts called: `01_download_data.py`, `02_clean_normalize.py`, `03_train_tokenizer.py`, `04_prepare_dataset.py`

**Phase 2 -- Scaling study** (`notebooks/02_scaling_study.ipynb`)

Runs a learning rate sweep on the Tiny model, trains all 5 SP model sizes for 1 epoch, fits the power-law scaling curve.

Scripts called: `05_train_model.py`, `06_lr_sweep.py`

**Phase 3 -- muP comparison** (`notebooks/03_mup_study.ipynb`)

Trains all 5 model sizes under muP with the same LR sweep protocol, fits the muP scaling curve, runs the coordinate check for muP correctness.

Scripts called: `07_train_mup.py`, `08_lr_sweep_mup.py`, `09_coord_check_mup.py`

**Phase 4 -- Generation and evaluation** (`notebooks/04_generation.ipynb`)

Continues XL training for 3 additional epochs, generates samples from both the 1-epoch and extended checkpoints, evaluates XML validity and render rate, plots sample grids.

Scripts called: `13_extend_xl.py`, `10_generate_samples.py`, `11_evaluate_samples.py`, `12_plot_samples.py`

---

## Scripts

| Script | Purpose |
|---|---|
| `01_download_data.py` | Download SVG datasets from HuggingFace |
| `02_clean_normalize.py` | Clean, normalize, deduplicate SVGs |
| `03_train_tokenizer.py` | Train BPE tokenizer (vocab size 4096) |
| `04_prepare_dataset.py` | Tokenize corpus, split 98/1/1, write binary |
| `05_train_model.py` | Train one SP model for 1 epoch |
| `06_lr_sweep.py` | SP learning rate sweep on Tiny |
| `07_train_mup.py` | Train one muP model for 1 epoch |
| `08_lr_sweep_mup.py` | muP learning rate sweep on Tiny |
| `09_coord_check_mup.py` | muP coordinate check (correctness validation) |
| `10_generate_samples.py` | Generate unconditional and prefix SVG samples |
| `11_evaluate_samples.py` | Evaluate XML validity, render rate, perplexity |
| `12_plot_samples.py` | Plot sample grids and temperature comparison |
| `13_extend_xl.py` | Continue XL training for additional epochs |

---

## Model Sizes

| Name | d_model | n_layers | n_heads | d_ff | Non-emb params |
|---|---|---|---|---|---|
| Tiny | 128 | 4 | 4 | 512 | 793K |
| Small | 192 | 6 | 6 | 768 | 2.7M |
| Medium | 384 | 6 | 6 | 1536 | 10.6M |
| Large | 512 | 10 | 8 | 2048 | 31.5M |
| XL | 768 | 12 | 12 | 3072 | 85.1M |

---

## Key Hyperparameters

- Optimizer: AdamW, beta1=0.9, beta2=0.95, weight_decay=0.1
- LR schedule: cosine with 200-step linear warmup, min_lr = peak_lr / 10
- Batch size: 64 sequences x 1024 tokens = 65536 tokens/step
- Mixed precision: bf16
- Selected LR (both SP and muP): 3e-4

---

## Outputs

All outputs go under `outputs/` (symlinked to Google Drive on Colab, gitignored):

- `outputs/checkpoints/` -- model checkpoints (.pt files)
- `outputs/logs/` -- training CSVs and evaluation JSON files
- `outputs/samples_1epoch/` -- generated SVGs from the 1-epoch XL checkpoint
- `outputs/samples_extended/` -- generated SVGs from the extended XL checkpoint
- `outputs/plots/` -- all figures for the report
- `outputs/tokenizer/` -- tokenizer.json
- `outputs/data/binary/` -- train.bin, val.bin, test.bin

---

## References

1. Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361
2. Hoffmann et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556
3. Yang et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. arXiv:2203.09789
4. Rodriguez et al. (2023). StarVector: Generating Scalable Vector Graphics Code from Images and Text. arXiv:2312.11556
