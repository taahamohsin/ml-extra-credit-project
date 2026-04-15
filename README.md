# SVG Scaling Laws

CS-GY 6923 Optional Project — NYU Tandon  
Due: May 1, 2026

**One-sentence summary:** Train decoder-only transformers at 5 scales on SVG code, fit power-law scaling curves, compare standard vs. µP parameterization, and generate/evaluate SVG samples.

---

## Quick Start (Colab)

1. Upload this repo to GitHub (or Google Drive)
2. Open `notebooks/01_data_pipeline.ipynb` in Google Colab
3. Set `REPO_URL` in Cell 1 to your GitHub URL
4. Run all cells top to bottom
5. Follow with `notebooks/02_scaling_study.ipynb`, etc.

**Recommended runtime:** Colab Pro with A100 GPU

---

## Repository Structure

```
svg-scaling-laws/
├── README.md
├── requirements.txt
├── decision_log.md          # Design decisions (see blueprint Section 12)
│
├── configs/
│   ├── data_config.yaml     # Data pipeline parameters
│   ├── model_configs.yaml   # Model architectures (Phase 2)
│   └── training_config.yaml # Training hyperparameters (Phase 2)
│
├── scripts/
│   ├── 01_download_data.py      # Download from HuggingFace
│   ├── 02_clean_normalize.py    # SVG cleaning pipeline
│   ├── 03_train_tokenizer.py    # BPE tokenizer training
│   ├── 04_prepare_dataset.py    # Tokenize, split, save as binary
│   ├── 05_train_model.py        # Main training script (SP)    [Phase 2]
│   ├── 06_lr_sweep.py           # LR sweep                     [Phase 2]
│   ├── 07_train_mup.py          # Training with µP             [Phase 3]
│   ├── 08_lr_sweep_mup.py       # LR sweep for µP             [Phase 3]
│   ├── 09_generate_samples.py   # Sample generation            [Phase 4]
│   ├── 10_evaluate.py           # Evaluation metrics           [Phase 4]
│   └── 11_plot_results.py       # All plots for report         [Phase 5]
│
├── src/
│   ├── __init__.py
│   ├── svg_utils.py         # SVG cleaning, validation, rendering
│   ├── tokenizer_utils.py   # BPE tokenizer helpers
│   ├── model.py             # Transformer (Phase 2)
│   ├── model_mup.py         # µP Transformer (Phase 3)
│   ├── dataset.py           # PyTorch Dataset (Phase 2)
│   ├── training_utils.py    # Training loop, checkpointing (Phase 2)
│   └── scaling_law.py       # Power law fitting (Phase 2)
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb          # Phase 1 end-to-end
│   ├── 02_scaling_study.ipynb          # Phase 2
│   ├── 03_mup_study.ipynb              # Phase 3
│   ├── 04_generation.ipynb             # Phase 4
│   └── 05_all_plots_and_analysis.ipynb # Phase 5
│
└── outputs/                 # Generated during runs (gitignored)
    ├── data/
    │   ├── raw/             # Raw JSONL files from HuggingFace
    │   ├── cleaned/         # Cleaned SVG JSONL
    │   └── binary/          # train.bin, val.bin, test.bin (uint16 memmaps)
    ├── tokenizer/           # tokenizer.json
    ├── checkpoints/         # Model checkpoints (.pt files)
    ├── logs/                # Training logs (CSV)
    ├── samples/             # Generated SVG samples
    ├── plots/               # All figures for the report
    └── report/              # Final PDF
```

---

## Phase 1: Data Pipeline

**Scripts:** `01` → `02` → `03` → `04`  
**Notebook:** `notebooks/01_data_pipeline.ipynb`

### SVG Cleaning Pipeline

Every raw SVG passes through these steps in order:

1. Strip XML comments (`<!-- ... -->`)
2. Strip `<?xml?>` processing instructions
3. Strip `<metadata>`, `<desc>`, `<title>` blocks
4. Extract `<svg>...</svg>` (discard anything outside)
5. Round decimal numbers to 1 decimal place
6. Collapse whitespace
7. Validate as valid XML (`lxml` strict parsing)
8. Length filter: discard if < 50 characters
9. MD5 deduplication

### Tokenizer

- Type: BPE via HuggingFace `tokenizers`
- Vocab size: 4096
- Pre-tokenizer: ByteLevel
- Special tokens: `<PAD>`(0), `<BOS>`(1), `<EOS>`(2), `<UNK>`(3)
- Every encoded sequence is automatically wrapped with BOS/EOS

### Dataset

- 98% train / 1% val / 1% test, split **by file** (no data leakage)
- Stored as `numpy uint16` memmap files (`train.bin`, `val.bin`, `test.bin`)
- Same format as [nanoGPT](https://github.com/karpathy/nanoGPT)
- Target: ≥ 100M training tokens

---

## Running Locally (without Colab)

```bash
# Install dependencies
pip install -r requirements.txt

# System dependency for cairosvg (macOS)
brew install cairo pango gdk-pixbuf libffi

# Run Phase 1 scripts in order
python scripts/01_download_data.py
python scripts/02_clean_normalize.py
python scripts/03_train_tokenizer.py
python scripts/04_prepare_dataset.py
```

All scripts accept `--config configs/data_config.yaml` (default).

---

## Design Decisions

See `decision_log.md` and `PROJECT_BLUEPRINT.md` Section 12 for all documented choices.

Key decisions:
- **Vocab size 4096:** Balances sequence length vs. token coverage for SVG's constrained vocabulary
- **Max seq length 1024:** Covers ~99% of simplified icons; keeps attention cost manageable
- **Split by file:** Prevents data leakage across train/val/test
- **bf16 mixed precision:** Native A100 support, no loss scaling needed vs fp16

---

## References

1. Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361
2. Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.* arXiv:2203.15556
3. Yang et al. (2022). *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* arXiv:2203.09789
4. Rodriguez et al. (2023). *StarVector: Generating Scalable Vector Graphics Code from Images and Text.* arXiv:2312.11556
