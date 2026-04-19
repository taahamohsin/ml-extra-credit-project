"""
dataset.py
----------
PyTorch Dataset that reads the numpy uint16 memmap binary files produced by
04_prepare_dataset.py and returns (input, target) sequence pairs for language
model training.

Each item is a randomly-sampled contiguous block of `max_seq_len` tokens:
  input  = tokens[i : i + max_seq_len]       (shifted left by 1 vs target)
  target = tokens[i + 1 : i + max_seq_len + 1]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SVGTokenDataset(Dataset):
    """
    Streams random fixed-length windows from a binary token file.

    Parameters
    ----------
    bin_path   : path to a .bin file (numpy uint16 memmap)
    seq_len    : context window length (should match model's max_seq_len)
    num_samples: virtual epoch size — how many items __len__ reports.
                 Set to (total_tokens // seq_len) for ~1 epoch coverage.
    seed       : optional seed for reproducible validation sampling
    """

    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int = 1024,
        num_samples: int | None = None,
        seed: int | None = None,
    ):
        self.bin_path = Path(bin_path)
        self.seq_len  = seq_len

        self.data = np.memmap(str(self.bin_path), dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

        # Need at least seq_len + 1 tokens to produce one (input, target) pair
        assert self.n_tokens > seq_len, (
            f"{self.bin_path} has only {self.n_tokens} tokens; "
            f"need > {seq_len}"
        )

        self.max_start = self.n_tokens - seq_len - 1

        if num_samples is None:
            self.num_samples = self.max_start + 1
        else:
            self.num_samples = num_samples

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample a random starting position (ignore idx — each call is random)
        start = int(self.rng.integers(0, self.max_start + 1))
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        x = chunk[:-1]  # (seq_len,)
        y = chunk[1:]   # (seq_len,)
        return x, y


def make_datasets(
    binary_dir: str | Path,
    seq_len: int = 1024,
    train_samples: int | None = None,
    val_samples: int = 512,
) -> tuple[SVGTokenDataset, SVGTokenDataset]:
    """
    Convenience factory: returns (train_dataset, val_dataset).

    val_samples is fixed so validation is reproducible across runs.
    """
    binary_dir = Path(binary_dir)
    train_ds = SVGTokenDataset(
        binary_dir / "train.bin",
        seq_len=seq_len,
        num_samples=train_samples,
    )
    val_ds = SVGTokenDataset(
        binary_dir / "val.bin",
        seq_len=seq_len,
        num_samples=val_samples,
        seed=0,  # fixed seed for reproducible val
    )
    return train_ds, val_ds
