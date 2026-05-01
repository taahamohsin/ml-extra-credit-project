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


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


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
    if save_dir is not None:
        stale = Path(save_dir) / "tokenizer.json"
        if stale.exists():
            stale.unlink()
            print(f"Deleted stale tokenizer: {stale}")

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    # use_regex=False disables the GPT-2 regex splitter, which collapses SVG
    # symbol/punctuation runs into ~88 unique word types and caps vocab at 161.
    # SVG has no linguistic word boundaries — we want BPE to learn merges
    # freely across the full byte stream of each line.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        # Seed the vocab with all 256 possible bytes so the base alphabet is
        # always complete, regardless of which bytes appear in this corpus.
        initial_alphabet=ByteLevel.alphabet(),
    )

    # train_from_iterator on long single-line SVGs produces a tiny vocab
    # because the HuggingFace BPE trainer batches by line, exhausting merges
    # at vocab_size=161. File-based training avoids this.
    import tempfile, os
    print(f"Writing {len(svg_texts):,} SVGs to temp file for tokenizer training ...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
        for text in svg_texts:
            tmp.write(text + "\n")
    print(f"Temp file: {tmp_path}  ({os.path.getsize(tmp_path) / 1e6:.1f} MB)")

    try:
        print("Training tokenizer from file ...")
        tokenizer.train([tmp_path], trainer=trainer)
    except Exception as e:
        os.unlink(tmp_path)
        raise RuntimeError(f"Tokenizer training failed: {e}") from e

    os.unlink(tmp_path)
    print("Temp file deleted.")

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
        save_path = save_dir / "tokenizer.json"
        # Atomic write: rename avoids partial writes on Google Drive
        tmp_save = save_path.with_suffix(".tmp")
        try:
            tokenizer.save(str(tmp_save))
            tmp_save.replace(save_path)
            print(f"Tokenizer saved to {save_path}")
            import json as _json
            with open(save_path) as _f:
                _json.load(_f)
            print(f"  Verified: {save_path.stat().st_size / 1e3:.1f} KB, valid JSON")
        except Exception as e:
            if tmp_save.exists():
                tmp_save.unlink()
            raise RuntimeError(f"Failed to save tokenizer to {save_path}: {e}") from e

    return tokenizer


def load_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    """Load a previously saved tokenizer from directory."""
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"No tokenizer.json found at {path}")
    return Tokenizer.from_file(str(path))


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
    """Return vocab entries 4..n+4 by id (skipping special tokens 0-3)."""
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
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
