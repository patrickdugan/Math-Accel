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
