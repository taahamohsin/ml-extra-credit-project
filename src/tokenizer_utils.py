"""
tokenizer_utils.py
------------------
BPE tokenizer training and utility wrappers for the SVG dataset.

Tokenizer design:
  - Type:         BPE (Byte Pair Encoding)
  - Vocab size:   4096
  - Pre-tokenizer: ByteLevel (handles all Unicode via raw bytes)
  - Special tokens:
      <PAD>  id=0  — padding
      <BOS>  id=1  — beginning of sequence
      <EOS>  id=2  — end of sequence
      <UNK>  id=3  — unknown
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


# ---------------------------------------------------------------------------
# Special token definitions (order must match ids)
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(
    svg_texts: list[str],
    vocab_size: int = 4096,
    save_dir: Optional[str | Path] = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer on a list of SVG strings.

    Parameters
    ----------
    svg_texts : list[str]
        The cleaned SVG strings to train on.
    vocab_size : int
        Target vocabulary size (default 4096).
    save_dir : str or Path, optional
        If provided, saves tokenizer.json to this directory.

    Returns
    -------
    Tokenizer
        A trained HuggingFace tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    # Write corpus to a temp file and use train_from_files, which is what
    # the HuggingFace BpeTrainer is optimized for. train_from_iterator with
    # very long single-line documents (our SVGs are one line each after
    # whitespace collapsing) causes the trainer to see far fewer "words" than
    # expected, resulting in a tiny vocabulary.
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
        for text in svg_texts:
            tmp.write(text + "\n")

    try:
        tokenizer.train([tmp_path], trainer=trainer)
    finally:
        os.unlink(tmp_path)

    # Add post-processing so encode() automatically wraps with BOS/EOS
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, BOS_ID),
            (EOS_TOKEN, EOS_ID),
        ],
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_dir / "tokenizer.json"))
        print(f"Tokenizer saved to {save_dir / 'tokenizer.json'}")

    return tokenizer


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    """Load a previously saved tokenizer from directory."""
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"No tokenizer.json found at {path}")
    return Tokenizer.from_file(str(path))


# ---------------------------------------------------------------------------
# Encoding / decoding helpers
# ---------------------------------------------------------------------------

def encode(tokenizer: Tokenizer, svg: str, add_special_tokens: bool = True) -> list[int]:
    """Encode a single SVG string to a list of token ids."""
    enc = tokenizer.encode(svg)
    return enc.ids


def decode(tokenizer: Tokenizer, ids: list[int], skip_special_tokens: bool = True) -> str:
    """Decode a list of token ids back to a string."""
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def token_length(tokenizer: Tokenizer, svg: str) -> int:
    """Return the number of tokens for a given SVG string (includes BOS/EOS)."""
    return len(encode(tokenizer, svg))


# ---------------------------------------------------------------------------
# Tokenizer statistics
# ---------------------------------------------------------------------------

def compute_tokenizer_stats(
    tokenizer: Tokenizer,
    svg_texts: list[str],
    max_token_length: int = 1024,
) -> dict:
    """
    Compute token-length statistics over a list of SVG strings.

    Returns a dict with: mean, median, p95, p99, max, and the count of
    SVGs that exceed max_token_length.
    """
    import numpy as np

    lengths = [token_length(tokenizer, svg) for svg in svg_texts]
    lengths_arr = np.array(lengths)

    return {
        "count": len(lengths),
        "mean": float(np.mean(lengths_arr)),
        "median": float(np.median(lengths_arr)),
        "p95": float(np.percentile(lengths_arr, 95)),
        "p99": float(np.percentile(lengths_arr, 99)),
        "max": int(np.max(lengths_arr)),
        "min": int(np.min(lengths_arr)),
        "exceeds_max_length": int(np.sum(lengths_arr > max_token_length)),
        "exceeds_max_length_pct": float(np.mean(lengths_arr > max_token_length) * 100),
    }


def get_top_tokens(tokenizer: Tokenizer, n: int = 50) -> list[tuple[str, int]]:
    """
    Return the top-n most common tokens by id (not by corpus frequency).
    This simply lists vocab entries 4..n+4 (skipping special tokens).
    For actual corpus frequency analysis, use compute_token_frequencies().
    """
    vocab = tokenizer.get_vocab()
    # Sort by id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    # Skip special tokens (ids 0-3)
    return [(tok, idx) for tok, idx in sorted_vocab if idx >= len(SPECIAL_TOKENS)][:n]


def compute_token_frequencies(
    tokenizer: Tokenizer,
    svg_texts: list[str],
) -> dict[int, int]:
    """
    Count how many times each token id appears across all SVG texts.
    Returns {token_id: count}.
    """
    from collections import Counter
    counts: Counter = Counter()
    for svg in svg_texts:
        ids = encode(tokenizer, svg, add_special_tokens=False)
        counts.update(ids)
    return dict(counts)
