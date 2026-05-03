"""TinyShakespeare char-level data loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"


def download() -> str:
    DATA_DIR.mkdir(exist_ok=True)
    if not DATA_FILE.exists():
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        DATA_FILE.write_text(r.text, encoding="utf-8")
    return DATA_FILE.read_text(encoding="utf-8")


@dataclass
class CharDataset:
    train: torch.Tensor
    val: torch.Tensor
    vocab_size: int
    stoi: dict[str, int]
    itos: dict[int, str]

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load(split_frac: float = 0.9) -> CharDataset:
    text = download()
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
