"""Char/byte-level data loaders.

Two datasets are supported:

* `tinyshakespeare` — Karpathy's char-rnn corpus, ~1.1M Python-string chars,
  vocab ~65. Loaded as Unicode chars. Used for architectural validation
  (the four-stage CSA implementation, v1 of this repo).

* `enwiki8` — the first 100MB of an English Wikipedia dump, the standard
  Hutter Prize / char-LM benchmark. Loaded as raw bytes with a compact
  byte-vocab (~200 values), 90M / 5M / 5M train / val / test split. Used
  for the small-scale CSA crossover study.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch

DATA_DIR = Path(__file__).parent / "data"

TINY_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINY_FILE = DATA_DIR / "tinyshakespeare.txt"

ENWIKI8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIKI8_ZIP = DATA_DIR / "enwik8.zip"
ENWIKI8_FILE = DATA_DIR / "enwik8"
ENWIKI8_TOTAL = 100_000_000  # standard truncation
ENWIKI8_TRAIN = 90_000_000
ENWIKI8_VAL = 5_000_000
ENWIKI8_TEST = ENWIKI8_TOTAL - ENWIKI8_TRAIN - ENWIKI8_VAL


@dataclass
class CharDataset:
    train: torch.Tensor
    val: torch.Tensor
    vocab_size: int
    stoi: dict           # encode unit -> token id   (unit is str for tinyshakespeare, int(byte) for enwiki8)
    itos: dict           # token id -> encode unit
    test: torch.Tensor | None = None   # enwiki8 has a test split; tinyshakespeare doesn't

    def encode(self, s) -> torch.Tensor:
        # s may be str (tinyshakespeare) or bytes (enwiki8)
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        units = [self.itos[int(i)] for i in ids]
        if units and isinstance(units[0], int):
            return bytes(units).decode("utf-8", errors="replace")
        return "".join(units)


# ---------------------------------------------------------------------------
# tinyshakespeare
# ---------------------------------------------------------------------------


def _download_tinyshakespeare() -> str:
    DATA_DIR.mkdir(exist_ok=True)
    if not TINY_FILE.exists():
        r = requests.get(TINY_URL, timeout=30)
        r.raise_for_status()
        TINY_FILE.write_text(r.text, encoding="utf-8")
    return TINY_FILE.read_text(encoding="utf-8")


def _load_tinyshakespeare(split_frac: float = 0.9) -> CharDataset:
    text = _download_tinyshakespeare()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(split_frac * len(data))
    return CharDataset(
        train=torch.from_numpy(data[:n]),
        val=torch.from_numpy(data[n:]),
        vocab_size=len(chars),
        stoi=stoi,
        itos=itos,
    )


# ---------------------------------------------------------------------------
# enwiki8
# ---------------------------------------------------------------------------


def _download_enwiki8() -> bytes:
    DATA_DIR.mkdir(exist_ok=True)
    if not ENWIKI8_FILE.exists():
        if not ENWIKI8_ZIP.exists():
            print(f"[data] downloading enwik8 from {ENWIKI8_URL} (~35 MB)")
            r = requests.get(ENWIKI8_URL, timeout=120, stream=True)
            r.raise_for_status()
            with ENWIKI8_ZIP.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print("[data] unzipping enwik8")
        with zipfile.ZipFile(ENWIKI8_ZIP) as z:
            with z.open("enwik8") as src, ENWIKI8_FILE.open("wb") as dst:
                dst.write(src.read())
    raw = ENWIKI8_FILE.read_bytes()[:ENWIKI8_TOTAL]
    assert len(raw) == ENWIKI8_TOTAL, f"expected {ENWIKI8_TOTAL} bytes, got {len(raw)}"
    return raw


def _load_enwiki8() -> CharDataset:
    raw = _download_enwiki8()
    byte_values = sorted(set(raw))                          # ints 0..255 that actually appear
    stoi = {b: i for i, b in enumerate(byte_values)}
    itos = {i: b for b, i in stoi.items()}
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    # Remap bytes -> token ids using a vectorized lookup table.
    lut = np.full(256, -1, dtype=np.int64)
    for b, i in stoi.items():
        lut[b] = i
    ids = lut[arr]
    assert (ids >= 0).all(), "unmapped byte in enwik8 (shouldn't happen)"
    return CharDataset(
        train=torch.from_numpy(ids[:ENWIKI8_TRAIN]),
        val=torch.from_numpy(ids[ENWIKI8_TRAIN : ENWIKI8_TRAIN + ENWIKI8_VAL]),
        test=torch.from_numpy(ids[ENWIKI8_TRAIN + ENWIKI8_VAL :]),
        vocab_size=len(byte_values),
        stoi=stoi,
        itos=itos,
    )


# ---------------------------------------------------------------------------
# dispatch + sampling
# ---------------------------------------------------------------------------


def load(dataset: str = "tinyshakespeare") -> CharDataset:
    if dataset == "tinyshakespeare":
        return _load_tinyshakespeare()
    if dataset == "enwiki8":
        return _load_enwiki8()
    raise ValueError(f"unknown dataset: {dataset!r}")


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (x, y) where y is x shifted by one position."""
    ix = torch.randint(
        0, len(data) - block_size - 1, (batch_size,), generator=generator
    )
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
