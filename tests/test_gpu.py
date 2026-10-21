"""GPU reductions and packed rewrites must match the reference exactly (CUDA required)."""

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


@pytest.mark.parametrize("p", [127, 7681, 263167, 1048573])
@pytest.mark.parametrize("tile_cols,rewrite_rows", [(16, 16), (64, 32), (16384, 512)])
@pytest.mark.parametrize("cache_rhs", [False, True])
def test_gpu_rewrite_packed_bytes(p, tile_cols, rewrite_rows, cache_rhs):
    from mathaccel import PackedBlock
    from mathaccel.reference.exact_ref import submul

    rng = np.random.default_rng(p + tile_cols)
    n, k = 117, 20
    permutation = rng.permutation(n)
    pc, keep = permutation[:k], permutation[k:]
    # Strided RHS, permuted gathers, and odd source and output widths.
    Rk = rng.integers(0, p, (k, len(keep) * 2)).astype(np.float64)[:, ::-2]
    values = [rng.integers(0, p, (rows, n)).astype(np.float64) for rows in (1, 17, 39)]
    blocks = [PackedBlock(v) for v in values]
    originals = [block.data.copy() for block in blocks]
    old_blocks = list(blocks)
    expected = []
    for v in values:
        out = np.ascontiguousarray(v[:, keep])
        submul(out, np.ascontiguousarray(v[:, pc]), Rk, p)
        expected.append(REF.PackedBlock(out))
    red = gpu.GPUReducer(p, tile_cols=tile_cols, rewrite_rows=rewrite_rows)
    if not cache_rhs:
        red.rewrite_cache_bytes = 32  # force tiled regeneration without large allocations
    # Initialize on default stream, then use the same reducer on a non-default stream.
    stream = gpu.cp.cuda.Stream(non_blocking=True)
    with stream:
        red.rewrite(blocks, pc, keep, Rk)
    for actual, ref in zip(blocks, expected):
        assert actual.rows == ref.rows and actual.ncols == ref.ncols
        assert np.array_equal(actual.data, ref.data)
    # Rewriting returns new buffers and never changes the source arrays.
    for block, before in zip(old_blocks, originals):
        assert np.array_equal(block.data, before)


@pytest.mark.parametrize("p", [127, 16381, 1048573])
def test_gpu_rewrite_maximum_inner_dimension(p):
    from mathaccel import PackedBlock

    k, q, rows = gpu.K_EXACT, 17, 2
    pc = np.arange(k, dtype=np.int64)
    keep = np.arange(k, k + q, dtype=np.int64)
    blocks = [PackedBlock(np.full((rows, k + q), p - 1, dtype=np.float64))]
    Rk = np.full((k, q), p - 1, dtype=np.float64)
    gpu.GPUReducer(p, tile_cols=16).rewrite(blocks, pc, keep, Rk)
    expected = REF.PackedBlock(np.full((rows, q), (p - 1 - k) % p, dtype=np.float64))
    assert np.array_equal(blocks[0].data, expected.data)


def test_gpu_rewrite_empty_output_and_no_blocks():
    from mathaccel import PackedBlock

    red = gpu.GPUReducer(7681)
    red.rewrite([], np.array([0]), np.array([], dtype=np.int64), np.empty((1, 0)))
    blocks = [PackedBlock(np.ones((3, 7), dtype=np.float64))]
    red.rewrite(blocks, np.arange(7), np.array([], dtype=np.int64), np.empty((7, 0)))
    assert blocks[0].data.shape == (3, 0, 5)
    assert blocks[0].rows == 3 and blocks[0].ncols == 0


def test_gpu_add_dispatch_and_storage_identical(monkeypatch):
    from mathaccel import CompressedBasis

    monkeypatch.setattr(CompressedBasis, "use_gpu", True)
    monkeypatch.setattr(CompressedBasis, "gpu_min_stored", 1)
    calls = []
    original_rewrite = gpu.GPUReducer.rewrite

    def track(self, blocks, pc, keep, Rk):
        calls.append(len(pc))
        return original_rewrite(self, blocks, pc, keep, Rk)

    monkeypatch.setattr(gpu.GPUReducer, "rewrite", track)
    rng = np.random.default_rng(673)
    ref = REF.CompressedBasis(173, 1048573, sub=16, pack=True)
    acc = CompressedBasis(173, 1048573, sub=16, pack=True)
    for _ in range(12):
        V = rng.integers(0, acc.p, (23, acc.n)).astype(np.float64)
        V[::5] = 0
        V[1] = V[2]
        assert acc.add(V.copy()) == ref.add(V.copy())
        assert np.array_equal(acc.P, ref.P) and np.array_equal(acc.Q, ref.Q)
        assert len(acc.blocks) == len(ref.blocks)
        for actual, expected in zip(acc.blocks, ref.blocks):
            assert actual.ncols == expected.ncols and actual.rows == expected.rows
            assert np.array_equal(actual.data, expected.data)
    assert calls, "add() never selected the GPU rewrite"
    assert len(acc.Q) == 0


def test_gpu_unavailable_uses_cpu_rewrite(monkeypatch):
    from mathaccel import CompressedBasis

    monkeypatch.setattr(CompressedBasis, "use_gpu", True)
    monkeypatch.setattr(CompressedBasis, "gpu_min_stored", 1)
    monkeypatch.setattr(gpu, "available", lambda: False)
    rng = np.random.default_rng(781)
    ref = REF.CompressedBasis(67, 7681, pack=True)
    acc = CompressedBasis(67, 7681, pack=True)
    for _ in range(3):
        V = rng.integers(0, acc.p, (13, acc.n)).astype(np.float64)
        assert acc.add(V.copy()) == ref.add(V.copy())
        for actual, expected in zip(acc.blocks, ref.blocks):
            assert np.array_equal(actual.data, expected.data)


def test_gpu_add_large_old_rank_uses_cpu_reduce_gpu_rewrite(monkeypatch):
    from mathaccel import CompressedBasis, PackedBlock

    monkeypatch.setattr(CompressedBasis, "use_gpu", True)
    monkeypatch.setattr(CompressedBasis, "gpu_min_stored", 1)
    rank, p = gpu.K_EXACT + 1, 263167
    ref = REF.CompressedBasis(rank + 2, p, pack=True)
    acc = CompressedBasis(rank + 2, p, pack=True)
    values = np.full((rank, 2), p - 1, dtype=np.float64)
    for b, block_type in ((ref, REF.PackedBlock), (acc, PackedBlock)):
        b.P = np.arange(rank, dtype=np.int64)
        b.Q = np.arange(rank, rank + 2, dtype=np.int64)
        b.blocks = [block_type(values)]
    calls = []
    original_rewrite = gpu.GPUReducer.rewrite

    def rewrite(self, blocks, pc, keep, Rk):
        calls.append(len(pc))
        return original_rewrite(self, blocks, pc, keep, Rk)

    def forbidden_reduce(*args):
        raise AssertionError("rank beyond int32 bound must reduce on CPU")

    monkeypatch.setattr(gpu.GPUReducer, "reduce", forbidden_reduce)
    monkeypatch.setattr(gpu.GPUReducer, "rewrite", rewrite)
    V = np.full((1, acc.n), p - 1, dtype=np.float64)
    V[0, -1] -= 1
    assert acc.add(V.copy()) == ref.add(V.copy()) == 1
    assert calls == [1]  # rewrite's inner dimension is the new pivot count
    assert np.array_equal(acc.P, ref.P) and np.array_equal(acc.Q, ref.Q)
    for actual, expected in zip(acc.blocks, ref.blocks):
        assert np.array_equal(actual.data, expected.data)


def test_gpu_rewrite_exception_restores_stream(monkeypatch):
    from mathaccel import PackedBlock
    from mathaccel.reference.exact_ref import submul

    p = 263167
    rng = np.random.default_rng(339)
    values = rng.integers(0, p, (17, 67)).astype(np.float64)
    pc, keep = np.arange(7), np.arange(7, 67)
    Rk = rng.integers(0, p, (7, 60)).astype(np.float64)
    blocks = [PackedBlock(values)]
    before = blocks[0].data.copy()
    red = gpu.GPUReducer(p, tile_cols=16, rewrite_rows=16)
    old_stream = gpu.cublas.getStream(red._handle)
    original_pack = red._rewrite_pack

    def fail(*args):
        raise RuntimeError("injected fold failure")

    monkeypatch.setattr(red, "_rewrite_pack", fail)
    stream = gpu.cp.cuda.Stream(non_blocking=True)
    with stream, pytest.raises(RuntimeError, match="injected fold failure"):
        red.rewrite(blocks, pc, keep, Rk)
    assert gpu.cublas.getStream(red._handle) == old_stream
    assert np.array_equal(blocks[0].data, before)
    monkeypatch.setattr(red, "_rewrite_pack", original_pack)
    with stream:
        red.rewrite(blocks, pc, keep, Rk)
    expected = values[:, keep].copy()
    submul(expected, values[:, pc].copy(), Rk, p)
    assert np.array_equal(blocks[0].data, REF.PackedBlock(expected).data)
