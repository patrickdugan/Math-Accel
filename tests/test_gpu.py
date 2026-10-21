"""The GPU reduction must equal the reference reduction exactly (skipped without a CUDA GPU)."""

import numpy as np
import pytest

from mathaccel import gpu
from mathaccel.reference import cbasis_ref as REF

pytestmark = pytest.mark.skipif(not gpu.available(), reason="no CUDA GPU / CuPy")


def _basis(rng, n, p, nbatch, rows):
    b = REF.CompressedBasis(n, p, sub=8, pack=True)
    for _ in range(nbatch):
        V = rng.integers(0, p, size=(rows, n)).astype(np.float64)
        V[::4] = 0
        b.add(V)
    return b


@pytest.mark.parametrize("p", [7681, 263167, 1032193, 1048573])
@pytest.mark.parametrize("n,rows,nbatch", [(37, 5, 4), (301, 40, 5), (1203, 150, 4)])
@pytest.mark.parametrize("tile_cols,rows_chunk", [(16, 16), (64, 32), (16384, 8192)])
def test_gpu_reduce_equals_reference(p, n, rows, nbatch, tile_cols, rows_chunk):
    rng = np.random.default_rng(n + rows + p % 101 + tile_cols)
    b = _basis(rng, n, p, nbatch, rows)
    red = gpu.GPUReducer(p, tile_cols=tile_cols, rows_chunk=rows_chunk)
    for m in (1, 7, 33):
        V = rng.integers(0, p, size=(m, n)).astype(np.float64)
        assert np.array_equal(red.reduce(b, V), b.reduce(V.copy()))


def test_limb_count():
    assert gpu.GPUReducer(127).L == 1
    assert gpu.GPUReducer(16381).L == 2
    assert gpu.GPUReducer(263167).L == 3


@pytest.mark.parametrize("pack_p", [263167, 1048573])
def test_basis_with_gpu_backend_identical(pack_p, monkeypatch):
    """A CompressedBasis whose reduce() runs on the GPU builds the same basis as the reference."""
    from mathaccel import CompressedBasis

    monkeypatch.setattr(CompressedBasis, "use_gpu", True)
    monkeypatch.setattr(CompressedBasis, "gpu_min_stored", 1)
    rng = np.random.default_rng(pack_p % 1000)
    n, rows = 400, 37
    ref = REF.CompressedBasis(n, pack_p, sub=16, pack=True)
    acc = CompressedBasis(n, pack_p, sub=16, pack=True)
    for _ in range(9):
        V = rng.integers(0, pack_p, size=(rows, n)).astype(np.float64)
        V[::5] = 0
        assert ref.add(V.copy()) == acc.add(V.copy())
        assert np.array_equal(ref.P, acc.P) and np.array_equal(ref.Q, acc.Q)
        probe = rng.integers(0, pack_p, size=(6, n)).astype(np.float64)
        assert np.array_equal(ref.reduce(probe), acc.reduce(probe))


@pytest.mark.parametrize('tile_cols,rows_chunk', [(16, 16), (64, 32)])
def test_gpu_async_buffer_reuse(tile_cols, rows_chunk):
    """Reuse both slots across chunks, blocks, odd final tiles, and calls."""
    rng = np.random.default_rng(410)
    b = _basis(rng, 317, 263167, 6, 31)
    original = [block.data.copy() for block in b.blocks]
    red = gpu.GPUReducer(b.p, tile_cols, rows_chunk)
    # A non-default stream must order custom cuBLAS calls with unpack/fold.
    stream = gpu.cp.cuda.Stream(non_blocking=True)
    for m in (17, 3, 33):
        V = rng.integers(0, b.p, (m, b.n)).astype(np.float64)
        expected = b.reduce(V.copy())
        with stream:
            actual = red.reduce(b, V)
        assert np.array_equal(actual, expected)
    for block, before in zip(b.blocks, original):
        assert np.array_equal(block.data, before)


@pytest.mark.parametrize('empty', ['rows', 'columns', 'basis'])
def test_gpu_empty_reduction(empty):
    rng = np.random.default_rng(77)
    b = REF.CompressedBasis(19, 7681, pack=True)
    if empty == 'columns':
        b.add(np.eye(b.n, dtype=np.float64))
    elif empty == 'rows':
        b.add(rng.integers(0, b.p, (7, b.n)).astype(np.float64))
    V = rng.integers(0, b.p, (0 if empty == 'rows' else 3, b.n)).astype(np.float64)
    assert np.array_equal(gpu.GPUReducer(b.p).reduce(b, V), b.reduce(V.copy()))


@pytest.mark.parametrize('p', [127, 16381, 1048573])
def test_gpu_maximum_exact_inner_dimension(p):
    """The largest permitted int32 sum and final padded chunk stay exact."""
    rank, q, m = gpu.K_EXACT, 17, 2
    b = REF.CompressedBasis(rank + q, p, pack=True)
    b.P = np.arange(rank, dtype=np.int64)
    b.Q = np.arange(rank, rank + q, dtype=np.int64)
    b.blocks = [REF.PackedBlock(np.full((rank, q), p - 1, dtype=np.float64))]
    V = np.full((m, b.n), p - 1, dtype=np.float64)
    expected = np.full((m, q), (p - 1 - rank) % p, dtype=np.float64)
    assert np.array_equal(gpu.GPUReducer(p).reduce(b, V), expected)


def test_gpu_strided_probe_and_permuted_columns():
    rng = np.random.default_rng(139)
    b = _basis(rng, 117, 1048573, 3, 17)
    # P and Q need not be contiguous ranges or in natural column order.
    permutation = rng.permutation(b.n)
    b.P, b.Q = permutation[b.P], permutation[b.Q]
    V = rng.integers(0, b.p, (7, b.n * 2)).astype(np.float64)[:, ::-2]
    before = V.copy()
    red = gpu.GPUReducer(b.p, tile_cols=16, rows_chunk=16)
    assert np.array_equal(red.reduce(b, V), b.reduce(V.copy()))
    assert np.array_equal(V, before)
