"""Char/byte-level data loaders.

Two datasets are supported:

* `tinyshakespeare` — Karpathy's char-rnn corpus, ~1.1M Python-string chars,
  vocab ~65. Loaded as Unicode chars. Used for architectural validation
  (the four-stage CSA implementation, v1 of this repo).

* `enwiki8` — the first 100MB of an English Wikipedia dump, the standard
  Hutter Prize / char-LM benchmark. Loaded as raw bytes with a compact
  byte-vocab (~200 values), 90M / 5M / 5M train / val / test split. Used
  for the small-scale CSA crossover study.

* `pg19` — Project Gutenberg books (Rae et al. 2019), the standard long-range
  language-modelling benchmark. Same byte-level treatment and the same
  90M / 5M / 5M split sizes as enwiki8, so bpc numbers are directly
  comparable. Books average ~400KB, so every training window stays inside a
  single document even at 64K context — unlike enwiki8, where 41% of bytes
  at 16K sit in an article shorter than the window.

Document boundaries are tracked for pg19 and used by `get_batch` and by the
evaluation sweeps, so no window ever straddles two books.
"""
from __future__ import annotations

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
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

# PG-19. Served as one plain-text file per book from a public GCS bucket, so no
# `datasets` dependency is needed. Split sizes deliberately mirror enwiki8.
PG19_BUCKET = "deepmind-gutenberg"
PG19_LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
PG19_FILE_URL = "https://storage.googleapis.com/{bucket}/{name}"
PG19_DIR = DATA_DIR / "pg19"
PG19_SUBSET_SEED = 1337
PG19_BUDGETS = {"train": 90_000_000, "validation": 5_000_000, "test": 5_000_000}
PG19_SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


@dataclass
class CharDataset:
    train: torch.Tensor
    val: torch.Tensor
    vocab_size: int
    stoi: dict           # encode unit -> token id   (unit is str for tinyshakespeare, int(byte) for enwiki8)
    itos: dict           # token id -> encode unit
    test: torch.Tensor | None = None   # enwiki8 has a test split; tinyshakespeare doesn't
    # Per-split document start offsets, when the corpus has real document
    # boundaries (pg19). None means "one undifferentiated stream", which is how
    # tinyshakespeare and enwiki8 have always been treated.
    doc_starts: dict[str, torch.Tensor] | None = None

    def starts_for(self, split: str) -> torch.Tensor | None:
        return None if self.doc_starts is None else self.doc_starts.get(split)

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
# pg19
# ---------------------------------------------------------------------------


def _pg19_list(split: str) -> list[tuple[str, int]]:
    """List (object_name, size) under a PG-19 split prefix, sorted by book id.

    Sorted so the subset selected for a given budget is deterministic: the same
    books land in the same split on every machine.
    """
    names: list[tuple[str, int]] = []
    token = None
    while True:
        params = {"prefix": f"{split}/", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(PG19_LIST_URL.format(bucket=PG19_BUCKET), params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("items", []):
            if item["name"].endswith(".txt"):
                names.append((item["name"], int(item["size"])))
        token = payload.get("nextPageToken")
        if not token:
            break

    def book_id(entry: tuple[str, int]) -> int:
        return int(entry[0].rsplit("/", 1)[1].removesuffix(".txt"))

    return sorted(names, key=book_id)


def _pg19_fetch_split(split: str, budget: int) -> tuple[bytes, list[int]]:
    """Download whole books until `budget` bytes are covered.

    Returns the concatenated corpus and the byte offset each book starts at.
    Books are never truncated: a window sampled inside a book is then always
    real text, and the final book is kept whole even if it overshoots.
    """
    listing = _pg19_list(split)
    # Take a *random* subset, not the lowest book ids: Gutenberg ids correlate
    # with how long a text has been in the archive, so the first N by id is a
    # systematically old, canonical sample (id 10 is the King James Bible, 4.4MB
    # on its own). Seeded, so the subset is still reproducible.
    rng = np.random.default_rng(PG19_SUBSET_SEED)
    listing = [listing[i] for i in rng.permutation(len(listing))]
    chosen: list[tuple[str, int]] = []
    total = 0
    for name, size in listing:
        chosen.append((name, size))
        total += size
        if total >= budget:
            break
    if total < budget:
        raise RuntimeError(f"pg19 {split}: only {total:,} bytes available, need {budget:,}")

    print(f"[data] pg19 {split}: fetching {len(chosen)} books (~{total/1e6:.0f}MB)")

    def fetch(entry: tuple[str, int]) -> bytes:
        name, _ = entry
        r = requests.get(PG19_FILE_URL.format(bucket=PG19_BUCKET, name=name), timeout=120)
        r.raise_for_status()
        return r.content

    with ThreadPoolExecutor(max_workers=16) as pool:
        books = list(pool.map(fetch, chosen))

    starts: list[int] = []
    offset = 0
    for b in books:
        starts.append(offset)
        offset += len(b)
    return b"".join(books), starts


def _download_pg19() -> tuple[dict[str, bytes], dict[str, list[int]]]:
    """Fetch (and cache) the PG-19 subset. Cache is keyed by budget."""
    PG19_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = PG19_DIR / "meta.json"
    cached_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    raw: dict[str, bytes] = {}
    starts: dict[str, list[int]] = {}
    for split, budget in PG19_BUDGETS.items():
        blob = PG19_DIR / f"{split}.bin"
        if blob.exists() and cached_meta.get(split, {}).get("budget") == budget:
            raw[split] = blob.read_bytes()
            starts[split] = cached_meta[split]["starts"]
            continue
        data_bytes, split_starts = _pg19_fetch_split(split, budget)
        blob.write_bytes(data_bytes)
        raw[split] = data_bytes
        starts[split] = split_starts
        cached_meta[split] = {"budget": budget, "starts": split_starts,
                              "bytes": len(data_bytes)}
        meta_file.write_text(json.dumps(cached_meta))
    return raw, starts


def _load_pg19() -> CharDataset:
    raw, starts = _download_pg19()
    # Vocab over the union of splits, as enwiki8 does, so no split can contain
    # a byte the lookup table has never seen.
    byte_values = sorted(set().union(*(set(v) for v in raw.values())))
    stoi = {b: i for i, b in enumerate(byte_values)}
    itos = {i: b for b, i in stoi.items()}
    lut = np.full(256, -1, dtype=np.int64)
    for b, i in stoi.items():
        lut[b] = i

    tensors: dict[str, torch.Tensor] = {}
    for split, blob in raw.items():
        ids = lut[np.frombuffer(blob, dtype=np.uint8).astype(np.int64)]
        assert (ids >= 0).all(), f"unmapped byte in pg19 {split}"
        tensors[PG19_SPLIT_NAMES[split]] = torch.from_numpy(ids)

    return CharDataset(
        train=tensors["train"],
        val=tensors["val"],
        test=tensors["test"],
        vocab_size=len(byte_values),
        stoi=stoi,
        itos=itos,
        doc_starts={
            PG19_SPLIT_NAMES[k]: torch.tensor(v, dtype=torch.long)
            for k, v in starts.items()
        },
    )


# ---------------------------------------------------------------------------
# dispatch + sampling
# ---------------------------------------------------------------------------


def load(dataset: str = "tinyshakespeare") -> CharDataset:
    if dataset == "tinyshakespeare":
        return _load_tinyshakespeare()
    if dataset == "enwiki8":
        return _load_enwiki8()
    if dataset == "pg19":
        return _load_pg19()
    raise ValueError(f"unknown dataset: {dataset!r}")


class DocumentSampler:
    """Uniform sampling of windows that lie entirely inside one document.

    Without this, a window straddles document boundaries and part of its
    context is unrelated text — which silently caps how much long-range signal
    the data can carry. (On enwiki8 at 16K context, 41% of corpus bytes sit in
    an article shorter than the window.)

    Sampling is uniform over *valid start positions*, not over documents, so
    long books are not under-weighted relative to short ones.
    """

    def __init__(self, doc_starts: torch.Tensor, total: int, block_size: int) -> None:
        ends = torch.cat([doc_starts[1:], torch.tensor([total])])
        # A document contributes (length - block_size - 1) legal start offsets.
        counts = (ends - doc_starts - block_size - 1).clamp_min(0)
        keep = counts > 0
        self.starts = doc_starts[keep]
        self.counts = counts[keep]
        self.cumulative = torch.cumsum(self.counts, dim=0)
        self.total_windows = int(self.cumulative[-1]) if len(self.cumulative) else 0
        if self.total_windows == 0:
            raise ValueError(
                f"no document is long enough for block_size={block_size}"
            )

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        r = torch.randint(0, self.total_windows, (n,), generator=generator)
        doc = torch.searchsorted(self.cumulative, r, right=True)
        prior = torch.where(doc > 0, self.cumulative[(doc - 1).clamp_min(0)], torch.zeros_like(r))
        return self.starts[doc] + (r - prior)

    def eval_windows(self, block_size: int) -> list[int]:
        """Deterministic non-overlapping windows, restarted at every document."""
        out: list[int] = []
        for start, count in zip(self.starts.tolist(), self.counts.tolist()):
            out.extend(range(start, start + count, block_size))
        return out


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    sampler: DocumentSampler | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (x, y) where y is x shifted by one position.

    With `sampler`, every window lies inside a single document.
    """
    if sampler is not None:
        ix = sampler.sample(batch_size, generator)
    else:
        ix = torch.randint(
            0, len(data) - block_size - 1, (batch_size,), generator=generator
        )
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
