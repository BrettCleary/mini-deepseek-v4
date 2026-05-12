"""Smoke tests for the dataset loaders.

Both datasets are cached after first download, so re-runs are fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402


def test_tinyshakespeare_load() -> None:
    ds = data.load("tinyshakespeare")
    assert ds.vocab_size == 65, f"expected vocab=65, got {ds.vocab_size}"
    assert ds.test is None
    # tinyshakespeare is ~1.1M chars; default split is 0.9.
    total = len(ds.train) + len(ds.val)
    assert total > 1_000_000 and total < 1_200_000
    # Round-trip a small string.
    text = "First Citizen:\n"
    ids = ds.encode(text)
    assert ds.decode(ids) == text


def test_enwiki8_load() -> None:
    ds = data.load("enwiki8")
    # Splits are exact.
    assert len(ds.train) == data.ENWIKI8_TRAIN
    assert len(ds.val) == data.ENWIKI8_VAL
    assert ds.test is not None and len(ds.test) == data.ENWIKI8_TEST
    # Byte vocab: 256 max, typically ~200 distinct bytes appear.
    assert 150 <= ds.vocab_size <= 256, f"unexpected vocab size {ds.vocab_size}"
    # itos values are ints (byte values), stoi keys are ints too.
    sample_id = int(ds.train[0].item())
    assert isinstance(ds.itos[sample_id], int)
    # Round-trip: encode "the " (a near-certain substring of enwiki8) and decode.
    s = b"the "
    ids = ds.encode(s)
    assert ds.decode(ids) == s.decode("utf-8")


def test_get_batch_shapes() -> None:
    ds = data.load("tinyshakespeare")
    x, y = data.get_batch(ds.train, block_size=128, batch_size=4, device=torch.device("cpu"))
    assert x.shape == (4, 128) and y.shape == (4, 128)
    # y is x shifted by one — equivalent positions in the source must match.
    # We can't verify that here without re-deriving offsets, but we can check
    # that values are within vocab range.
    assert 0 <= int(x.min()) and int(x.max()) < ds.vocab_size
    assert 0 <= int(y.min()) and int(y.max()) < ds.vocab_size


def _run_all() -> None:
    tests = [test_tinyshakespeare_load, test_enwiki8_load, test_get_batch_shapes]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
