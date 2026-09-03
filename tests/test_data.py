"""Smoke tests for the dataset loaders.

Both datasets are cached after first download, so re-runs are fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
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


# ---------------------------------------------------------------------------
# Document-aware sampling (pg19). A window that straddles two documents puts
# unrelated text into the "context" whose value long-context experiments are
# trying to measure.
# ---------------------------------------------------------------------------


def _sampler(lengths, block_size):
    starts, off = [], 0
    for n in lengths:
        starts.append(off)
        off += n
    return data.DocumentSampler(torch.tensor(starts), off, block_size), starts, off


def test_sampler_never_crosses_a_document_boundary():
    sp, starts, total = _sampler([100, 20, 50], block_size=30)
    bounds = list(zip(starts, starts[1:] + [total]))
    g = torch.Generator().manual_seed(0)
    for i in sp.sample(5000, g).tolist():
        assert any(lo <= i and i + 31 <= hi for lo, hi in bounds), i


def test_sampler_drops_documents_shorter_than_the_window():
    sp, _, _ = _sampler([100, 20, 50], block_size=30)
    # The 20-byte document cannot hold a 30-token window.
    assert sp.starts.tolist() == [0, 120]
    assert sp.counts.tolist() == [69, 19]


def test_sampler_is_uniform_over_windows_not_documents():
    """Long books must not be under-weighted relative to short ones."""
    sp, _, _ = _sampler([1000, 100], block_size=30)
    g = torch.Generator().manual_seed(1)
    ix = sp.sample(40000, g)
    frac_first = (ix < 1000).float().mean().item()
    expected = 969 / (969 + 69)
    assert abs(frac_first - expected) < 0.02, (frac_first, expected)


def test_eval_windows_restart_at_every_document():
    sp, _, _ = _sampler([100, 20, 50], block_size=30)
    assert sp.eval_windows(30) == [0, 30, 60, 120]


def test_sampler_rejects_block_size_no_document_can_hold():
    with pytest.raises(ValueError):
        _sampler([10, 20], block_size=64)


# ---------------------------------------------------------------------------
# Byte vocab with the untrainable tail folded (pg19).
# ---------------------------------------------------------------------------


def test_build_byte_vocab_folds_the_rare_tail():
    # 'a' x 500, 'b' x 200, 'c' x 3  -> 'c' is below threshold and folds.
    train = b"a" * 500 + b"b" * 200 + b"c" * 3
    lut, stoi, itos, vocab_size = data.build_byte_vocab(train, min_count=100)
    assert vocab_size == 3                      # a, b, <rare>
    assert set(stoi) == {ord("a"), ord("b")}
    rare = vocab_size - 1
    assert lut[ord("c")] == rare
    assert lut[ord("a")] != rare and lut[ord("b")] != rare


def test_unseen_bytes_map_to_rare_not_to_an_untrained_row():
    """The actual bug: 4 byte values appeared in pg19's test split but never in
    train, so their embedding rows were pure init at evaluation time."""
    train = b"x" * 1000
    lut, _, _, vocab_size = data.build_byte_vocab(train, min_count=10)
    rare = vocab_size - 1
    for b in (0, 7, 200, 255):
        assert lut[b] == rare
    assert lut[ord("x")] == 0


def test_vocab_ids_are_contiguous_and_cover_all_bytes():
    train = bytes(range(256)) * 50
    lut, stoi, itos, vocab_size = data.build_byte_vocab(train, min_count=10)
    assert vocab_size == 257                     # all 256 kept, plus <rare>
    assert sorted(stoi.values()) == list(range(256))
    assert lut.min() >= 0 and lut.max() < vocab_size
    assert set(itos) == set(range(vocab_size))


def test_min_count_of_one_keeps_everything_seen():
    train = b"ab" + b"c"
    _, stoi, _, vocab_size = data.build_byte_vocab(train, min_count=1)
    assert set(stoi) == {ord("a"), ord("b"), ord("c")}
    assert vocab_size == 4                       # + <rare> for unseen bytes
