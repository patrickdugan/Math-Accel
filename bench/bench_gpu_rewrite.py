"""Paired CPU/GPU microbenchmark of add()'s packed-block rewrite only.

Uses identical synthetic RREF storage and RHS for both backends. Copying the list
of immutable input blocks is outside timing; output allocations, transfers, GEMMs,
folding, and packing are included. Each GPU result is compared byte-for-byte with
the CPU result. This is not an end-to-end add() or closure benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathaccel import CompressedBasis
from mathaccel import gpu
from bench_gpu_reduce import packed_basis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank', type=int, default=2048, help='old basis rows')
    ap.add_argument('--cols', type=int, default=16383, help='old non-pivot width')
    ap.add_argument('--pivots', type=int, default=256, help='new pivot count')
    ap.add_argument('--block', type=int, default=128)
    ap.add_argument('--tile-cols', type=int, default=16384)
    ap.add_argument('--rewrite-rows', type=int, default=512)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--cpu-repeats', type=int, default=3)
    args = ap.parse_args()
    if min(vars(args).values()) < 1 or args.pivots >= args.cols:
        ap.error('all dimensions and repeats must be positive; pivots must be less than cols')
    rng = np.random.default_rng(2026)
    p = 263167
    original = packed_basis(rng, args.rank, args.cols, args.block, p)
    pc = np.sort(rng.choice(args.cols, args.pivots, replace=False)).astype(np.int64)
    keep = np.setdiff1d(np.arange(args.cols, dtype=np.int64), pc)
    Rk = rng.integers(0, p, (args.pivots, len(keep)), dtype=np.int32).astype(np.float64)
    red = gpu.GPUReducer(p, tile_cols=args.tile_cols, rewrite_rows=args.rewrite_rows)
    print(f'rank={args.rank} old_q={args.cols} pivots={args.pivots} new_q={len(keep)} '
          f'block={args.block} tile={args.tile_cols} rewrite_rows={args.rewrite_rows} '
          f'packed_MB={sum(c.data.nbytes for c in original.blocks)/1e6:.1f}', flush=True)

    def run(which):
        b = CompressedBasis(original.n, p, sub=512, pack=True)
        b.blocks = list(original.blocks)  # neither rewrite mutates the input blocks
        gpu.cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        if which == 'cpu':
            b._rewrite_cpu(pc, keep, Rk)
        else:
            red.rewrite(b.blocks, pc, keep, Rk)
        gpu.cp.cuda.get_current_stream().synchronize()
        return b.blocks, time.perf_counter() - start

    # Warm Numba, BLAS, and CUDA with small valid matrices before timing either backend.
    warm = packed_basis(rng, 17, 67, 17, p)
    wp = np.arange(7, dtype=np.int64)
    wk = np.arange(7, 67, dtype=np.int64)
    wr = np.zeros((7, 60), dtype=np.float64)
    wb = CompressedBasis(warm.n, p, pack=True)
    wb.blocks = list(warm.blocks)
    wb._rewrite_cpu(wp, wk, wr)
    red.rewrite(warm.blocks, wp, wk, wr)
    expected = None
    cpu_times = []
    for _ in range(args.cpu_repeats):
        expected = None  # release the preceding result before allocating another
        expected, dt = run('cpu')
        cpu_times.append(dt)
    actual = None
    gpu_times = []
    for trial in range(args.repeats + 1):
        actual = None
        actual, dt = run('gpu')
        for result, reference in zip(actual, expected):
            assert result.ncols == reference.ncols and result.rows == reference.rows
            assert np.array_equal(result.data, reference.data), 'packed result differs from CPU'
        if trial:  # one full-size GPU warmup for shape-specific allocations
            gpu_times.append(dt)
    cpu_median, gpu_median = np.median(cpu_times), np.median(gpu_times)
    print(f'cpu_seconds={cpu_times} median={cpu_median:.6f}', flush=True)
    print(f'gpu_seconds={gpu_times} median={gpu_median:.6f}', flush=True)
    print(f'packed_bytes_identical=True rewrite_speedup={cpu_median/gpu_median:.3f}x', flush=True)


if __name__ == '__main__':
    main()
