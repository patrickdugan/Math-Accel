"""The accelerated code must give bit-identical results to the vendored reference."""

import numpy as np
import pytest

import mathaccel.kernels as K
from mathaccel import CompressedBasis
from mathaccel.reference import cbasis_ref as REF

PRIMES = [7681, 263167, 1032193, 1048573]  # the last two are near 2^20, the packing limit


def _rand(rng, rows, cols, hi, dtype=np.float64):
    return rng.integers(0, hi, size=(rows, cols)).astype(dtype)


@pytest.fixture(params=[True, False], ids=["numba", "numpy-fallback"])
def backend(request, monkeypatch):
    if request.param and not K.HAVE_NUMBA:
        pytest.skip("numba not installed")
    if not request.param:
        monkeypatch.setattr(K, "HAVE_NUMBA", False)
    return request.param


@pytest.mark.parametrize("cols", [1, 2, 7, 64, 513])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_pack_layout_identical(backend, cols, dtype):
    rng = np.random.default_rng(cols)
    a = _rand(rng, 37, cols, 1 << 20, dtype)
    assert np.array_equal(K.pack_rows(a), REF.PackedBlock(a).data)


@pytest.mark.parametrize("cols", [1, 2, 7, 64, 513])
def test_unpack_and_gather(backend, cols):
    rng = np.random.default_rng(100 + cols)
    a = _rand(rng, 41, cols, 1 << 20)
    data = K.pack_rows(a)
    ref = REF.PackedBlock(a)
    for s, e in [(0, 41), (3, 17), (40, 41), (5, 5), (30, 99)]:
        got = K.unpack_rows(data, s, e, cols)
        want = ref[s:e] if e > s else np.zeros((0, cols))
        assert got.dtype == np.float64 and np.array_equal(got, want)
    for sel in [np.arange(cols), np.arange(cols)[::-1], rng.permutation(cols)[: max(1, cols // 3)]]:
        assert np.array_equal(K.gather_cols(data, 2, 30, sel), ref[2:30][:, sel])


@pytest.mark.parametrize("p", PRIMES)
def test_mod_inplace(backend, p):
    rng = np.random.default_rng(p)
    V = rng.integers(-(1 << 52), 1 << 52, size=(97, 211)).astype(np.float64)
    want = np.mod(V, p)
    got = K.mod_inplace(V.copy(), p)
    assert np.array_equal(got, want)


def _batches(rng, n, p, nbatch, rows):
    """Random batches with rank deficiency: some rows are combinations of earlier ones mod p."""
    seen = []
    for b in range(nbatch):
        V = rng.integers(0, p, size=(rows, n)).astype(np.int64)
        V[:, rng.random(n) < 0.15] = 0  # some all-zero columns
        if seen:
            for i in range(0, rows, 3):
                a, c = seen[rng.integers(len(seen))], seen[rng.integers(len(seen))]
                V[i] = (int(rng.integers(p)) * a + int(rng.integers(p)) * c) % p
        if b == nbatch - 1:
            V[-2:] = 0  # zero rows
        seen.extend(list(V))
        yield V.astype(np.float64)


def _same_basis(a, b):
    assert a.rank == b.rank
    assert np.array_equal(a.P, b.P) and np.array_equal(a.Q, b.Q)
    assert len(a.blocks) == len(b.blocks)
    for Ca, Cb in zip(a.blocks, b.blocks):
        assert Ca.shape == Cb.shape
        A = REF._chunk64(Ca, 0, Ca.shape[0]) if isinstance(Ca, REF.PackedBlock) else Ca.astype(np.float64)
        B = Cb[0:Cb.shape[0]] if hasattr(Cb, "rows64") else Cb.astype(np.float64)
        assert np.array_equal(A, B)
    assert a.stored() == b.stored()


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("pack", [True, False])
@pytest.mark.parametrize("n,rows,nbatch,sub", [(60, 9, 8, 4), (300, 40, 6, 16), (1200, 120, 5, 64)])
def test_compressed_basis_identical(backend, p, pack, n, rows, nbatch, sub):
    rng = np.random.default_rng(n * 7 + rows + p % 97)
    ref = REF.CompressedBasis(n, p, sub=sub, pack=pack)
    acc = CompressedBasis(n, p, sub=sub, pack=pack)
    for V in _batches(rng, n, p, nbatch, rows):
        assert ref.add(V.copy()) == acc.add(V.copy())
        _same_basis(ref, acc)
        probe = rng.integers(0, p, size=(5, n)).astype(np.float64)
        assert np.array_equal(ref.reduce(probe), acc.reduce(probe))


def test_row_indexing_matches_reference(backend):
    rng = np.random.default_rng(5)
    a = _rand(rng, 23, 101, 1 << 20)
    ref, acc = REF.PackedBlock(a), __import__("mathaccel").PackedBlock(a)
    for i in [0, 7, 22, -1]:
        assert np.array_equal(ref[i], acc[i])
    assert np.array_equal(ref[4:19], acc[4:19])
