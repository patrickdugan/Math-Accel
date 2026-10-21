"""Reproducible GPU reduce microbenchmark; uses synthetic packed RREF storage.

Set CUPY_CACHE_DIR, TEMP and TMP to a drive with space. --source can point to
an earlier gpu.py for a paired comparison on exactly the same generated inputs.
Times include host staging, transfers, kernels, and the returned NumPy result.
The optional cProfile table measures host call time, not individual GPU kernels.
"""
from __future__ import annotations

import argparse
import cProfile
import importlib.util
import linecache
from pathlib import Path
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathaccel import CompressedBasis, PackedBlock
from mathaccel import gpu


def packed_basis(rng, rank, cols, block_rows, p):
    """Generate deterministic synthetic RREF storage without dense float64 copies."""
    b = CompressedBasis(rank + cols, p, pack=True)
    b.P = np.arange(rank, dtype=np.int64)
    b.Q = np.arange(rank, b.n, dtype=np.int64)
    # Generate packed storage directly, without a large temporary float64 basis.
    pairs = (cols + 1) // 2
    for s in range(0, rank, block_rows):
        rows = min(block_rows, rank - s)
        lo = rng.integers(0, p, (rows, pairs), dtype=np.uint64)
        hi = rng.integers(0, p, (rows, pairs), dtype=np.uint64)
        if cols % 2:
            hi[:, -1] = 0
        words = lo | (hi << 20)
        block = object.__new__(PackedBlock)
        block.rows, block.ncols = rows, cols
        block.data = np.empty((rows, pairs, 5), dtype=np.uint8)
        for byte in range(5):
            block.data[:, :, byte] = (words >> (8 * byte)).astype(np.uint8)
        b.blocks.append(block)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank', type=int, default=6144)
    ap.add_argument('--cols', type=int, default=55039, help='non-pivot columns')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--block', type=int, default=128)
    ap.add_argument('--tile-cols', type=int, default=16384)
    ap.add_argument('--rows-chunk', type=int, default=8192)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--line-profile', action='store_true', help='host elapsed time by reduce() line')
    ap.add_argument('--check', action='store_true', help='compare with exact CPU reduction')
    args = ap.parse_args()
    if min(args.rank, args.cols, args.batch, args.block, args.repeats,
           args.tile_cols, args.rows_chunk) < 1:
        ap.error('dimensions, block, and repeats must be positive')
    backend = gpu
    if args.source:
        spec = importlib.util.spec_from_file_location('gpu_benchmark_source', args.source)
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
    cp = backend.cp
    rng = np.random.default_rng(2026)
    p = 263167
    b = packed_basis(rng, args.rank, args.cols, args.block, p)
    V = rng.integers(0, p, (args.batch, b.n), dtype=np.int32).astype(np.float64)
    reducer = backend.GPUReducer(p, args.tile_cols, args.rows_chunk)
    print(f'source={args.source or "mathaccel.gpu"} rank={args.rank} q={args.cols} '
          f'batch={args.batch} tile={args.tile_cols} chunk={args.rows_chunk} '
          f'packed_MB={sum(c.data.nbytes for c in b.blocks)/1e6:.1f}', flush=True)
    result = reducer.reduce(b, V)  # warm CUDA context, kernels, and memory pools
    if args.check:
        expected = b._reduce_cpu(V)
        assert np.array_equal(result, expected), 'GPU result differs from CPU'
        print('exact_cpu_match=True', flush=True)
        del expected
    timings = []
    for _ in range(args.repeats):
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        actual = reducer.reduce(b, V)
        cp.cuda.get_current_stream().synchronize()
        timings.append(time.perf_counter() - start)
        assert np.array_equal(actual, result), 'repeated result changed'
    print(f'seconds={timings} median={np.median(timings):.6f}', flush=True)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        reducer.reduce(b, V)
        cp.cuda.get_current_stream().synchronize()
        profiler.disable()
        pstats.Stats(profiler).strip_dirs().sort_stats('tottime').print_stats(25)
    if args.line_profile:
        code = reducer.reduce.__func__.__code__
        elapsed = {}
        last = [None, 0.0]

        def trace(frame, event, arg):
            if frame.f_code is not code:
                return None
            now = time.perf_counter()
            if event in ('line', 'return'):
                if last[0] is not None:
                    elapsed[last[0]] = elapsed.get(last[0], 0.0) + now - last[1]
                last[:] = [frame.f_lineno, now]
            return trace

        sys.settrace(trace)
        try:
            reducer.reduce(b, V)
        finally:
            sys.settrace(None)
        print('Host elapsed by line (GPU waits are charged to their synchronization line):')
        for lineno, seconds in sorted(elapsed.items(), key=lambda item: -item[1])[:15]:
            source = linecache.getline(code.co_filename, lineno).strip()
            print(f'{seconds:.6f}s line {lineno}: {source}')


if __name__ == '__main__':
    main()
