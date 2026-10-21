# Math-Accel

Exact modular linear algebra for **degree-bounded closures**: the linear-algebra core of Macaulay,
XL and polynomial-calculus style computations over a prime field $\mathbb F_p$. Every result is
exact. Speed comes from data layout and parallel kernels, never from approximation.

The code comes from the closure engines of a research project on index calculus for elliptic
curves over prime fields (`semaev-index-calculus`). There a closure is a subspace $W_D$ of a
large polynomial algebra, grown by multiplying by tower variables and reducing until it is closed
or contains 1. Those runs reach $10^5$ columns and take hours to days on one machine, so the
engine's speed decides which questions can be asked.

## What is here (v0.2 CPU; v0.3 GPU development)

- **`mathaccel.CompressedBasis`**: an exact row-reduced basis mod $p$ that stores only the
  non-pivot columns.
  - It holds $r(n-r)$ entries for $r$ rows, at most $n^2/4$.
  - Storage is float32 (exact for $p<2^{24}$), or packed at 2.5 bytes per entry, two residues per
    40-bit word (exact for $p\le2^{20}$).
- **`mathaccel.kernels`**: Numba-parallel versions of the hot spots:
  - packing and unpacking;
  - a fused *gather-unpack*, which decodes only the columns a step needs, with no full unpack and
    no copy;
  - an exact parallel $x \bmod p$.

  Without Numba, the same functions run in NumPy with the same results.
- **`mathaccel.reference`**: the original implementations, vendored verbatim. They are the ground
  truth.

**CPU exactness model.** Arithmetic is float64 on integers in $[0,p)$. Every matrix product is cut
along its inner dimension into chunks of at most
$k_{\max}(p)=\lfloor(2^{53}-p)/(p-1)^2\rfloor-1$ terms, so each partial sum is an integer below
$2^{53}$, and is reduced mod $p$ between chunks.

## Correctness first

`tests/test_equivalence.py` requires **bit-identical** results against the reference:
- identical packed bytes;
- identical pivots, column sets, stored blocks, reductions and return values over randomized
  sequences of rank-deficient additions;
- for $p$ = 7681, 263167, 1032193 and 1048573, in both storage modes, with Numba and with the NumPy
  fallback;
- the streaming reduction at $p=16777213<2^{24}$, where $k_{\max}(p)=31$ forces a reduction mod $p$
  every 31 terms, with buffers from one block to the whole basis (116 tests).

```bash
python -m venv .venv && .venv/Scripts/pip install numpy numba scipy pytest   # or .venv/bin/pip
python -m pytest -q tests
```

## Speed

**Profile of the reference engine.** One 15360-column closure was measured: isogeny tower, $m=5$,
$D=7$, $p=263167$, 250 s under cProfile. It used about 18% of a 12-thread CPU, because about half
its time went to single-threaded memory work.

| Where the time went | Share |
|---|---|
| unpacking and repacking 2.5-byte residues | ~36% |
| array copies | ~11% |
| matrix products and mod-$p$ reduction | ~42% |
| other | ~11% |

**Against the reference.** All runs use the same closure driver on the same machine (i5-11400H,
6 cores and 12 threads, 6 threads each).

| Closure | Reference | v0.1 | v0.2 | Result |
|---|---|---|---|---|
| isogeny $m=5$, $D=6$, 7937 columns, run to completion | 51.7 s | 29.6 s | — | rank 3695 for all |
| isogeny $m=5$, $D=7$, 15360 columns, run to completion | 972.7 s | 445.9 s (2.2×) | **191.3 s (5.1×)** | refuted ($1\in W_D$, rank 15360) for all |
| isogeny $m=5$, $D=7$, first 240 s | rank 5056 | rank 7203 | — | — |

What v0.1 changed: parallel packing and unpacking, and a fused column gather in place of full
unpacks and copies.

What v0.2 changed in `reduce()`:
- it streams the rows of consecutive blocks into one buffer of adaptive size, so there are fewer
  and larger exact matrix products, each at most $k_{\max}(p)$ terms;
- it accumulates in place with BLAS `dgemm` and $\beta=1$, with no temporary and no coefficient
  copies (SciPy, optional).

`add()` uses the same in-place update.

For comparison, the research project's dense kernel needed 1576 s for the same $D=7$ closure.

`bench/bench_chain.py` reproduces these. It needs the research repository's closure driver.

## Experimental GPU reduction and block rewrite

`mathaccel.gpu.GPUReducer` implements exact packed-basis reduction with CuPy and cuBLAS
int8 tensor-core products. It also rewrites the existing packed blocks in `add()` on the GPU.
It is **off by default**. With a compatible CuPy/CUDA installation:

```python
from mathaccel import CompressedBasis

CompressedBasis.use_gpu = True
basis = CompressedBasis(n, p, pack=True)
```

The backend is selected once the packed basis stores at least `gpu_min_stored` entries
(default 2,000,000). Each GPU product supports inner dimension at most 133144: the old rank
for reduction, or the new pivot count for rewriting. A phase above that limit uses the CPU.
It uses up to three 7-bit limbs per residue and nine int8-to-int32 products. Every int32
accumulator is exact at that bound; the weighted nine-term sum stays below 2^56 in int64. A final kernel reduces it mod p,
gathers the probe's non-pivot columns, and subtracts the product exactly.

Packed rows are copied directly into reusable pinned buffers. A separate CUDA stream uploads
the next chunk while the preceding chunk's GPU work runs; events protect host and device
buffers from premature reuse. There are two buffers when the rank spans multiple row chunks,
and one otherwise. The basis still crosses PCIe once per `reduce()` call. The rewrite
uploads each packed row chunk once more in `add()`; the two phases do not yet share uploads.

**Full closure result before these transfer/fold changes.** On the 61183-column isogeny
h=2, m=6, D=7 closure, CPU v0.2 reached rank 6118 in 240 seconds; the original GPU path
reached 5463. The optimized path has **not yet been remeasured on the full closure**.

**GPU reduction microbenchmark, 2026-10-21.** RTX 3050 Laptop, p=263167, batch 512,
rank 6144, 55039 non-pivot columns, 845.4 MB of packed basis in 128-row blocks, and
16384-column tiles. These are deterministic synthetic RREF inputs, not a closure run.
Medians include host staging, transfers, kernels, and the returned NumPy result, after one
warmup and over five repeats:

| GPU implementation | Row chunk | Median per reduction |
|---|---|---|
| Original (`bdab8eb`) | 8192 | 3.801 s |
| Pinned staging and fused gather/subtract/fold | 8192 (unchanged default) | **1.337 s (2.84×)** |
| Same optimization, smaller chunks for overlap | 2048 | 1.330 s |

The optimized full-sized result matched the exact CPU reducer. That revision passed all
**164 tests**, including 48 GPU cases covering equality with the reference, async buffer reuse,
non-default streams, strided probes, empty dimensions, and the maximum exact inner dimension. The
vendored reference is unchanged.

Profiles attribute about 0.54 s to the old host gather and 0.91 s to its final NumPy
subtraction/modulo in one representative call; GPU waits are charged to the synchronized
download, so these profiles do not isolate individual device kernels. Raw measurements and
profiles are in `bench/results/gpu_reduce_*.txt`.

The research lane continued running during these measurements; Math-Accel used one CPU
thread. These timings establish a reduction microbenchmark improvement, not a full-closure
speedup or a reason to enable GPU by default. Version/release promotion awaits that comparison.

To reproduce in PowerShell on this machine (keep temporary files and CuPy caches on D:):

```powershell
$env:CUPY_CACHE_DIR = 'D:/Temp/cupy_cache'
$env:TEMP = 'D:/Temp'
$env:TMP = 'D:/Temp'
$env:NUMBA_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
git show bdab8eb:mathaccel/gpu.py | Set-Content -Encoding utf8 D:/Temp/gpu_before.py
.venv/Scripts/python.exe bench/bench_gpu_reduce.py --source D:/Temp/gpu_before.py --repeats 5
.venv/Scripts/python.exe bench/bench_gpu_reduce.py --repeats 5 --check
.venv/Scripts/python.exe bench/bench_gpu_reduce.py --rows-chunk 2048 --repeats 5
```

Use `--profile` or `--line-profile` for host timings. Smaller dimensions can be supplied
with `--rank`, `--cols`, and `--batch` for a low-memory correctness check.

### GPU rewrite in add()

The old-block update `blk[:, keep] - blk[:, pc] @ Rk` now runs on the GPU when the packed
basis meets the same opt-in threshold. Pivot gathers, exact limb GEMMs, modular subtraction,
remaining-column gathers, and 20-bit packing all run on the device. Results download directly
as packed bytes, preserving each block's row count, byte layout, and zero padding.

The rewrite stages at most `rewrite_rows` old rows at a time (default 512) in pinned memory,
and replaces old blocks progressively to avoid a second full host basis. It caches the RHS
limbs across all blocks when they fit `rewrite_cache_bytes` (default 256 MiB); larger RHS
matrices use bounded column tiles regenerated per row chunk. The new-row RREF and packing
of the newly appended block remain on the CPU.

**Paired rewrite microbenchmark, 2026-10-21.** Same RTX 3050 Laptop, p=263167, one CPU thread,
128-row stored blocks, 16384-column tiles, and rewrite row cap 512. CPU and GPU get identical
synthetic inputs; timings include allocations, transfers, products, modular arithmetic, and
packing. Every GPU result matches the CPU's packed bytes.

| Old rank | Old stored columns | New pivots | CPU rewrite median | GPU rewrite median | Speedup |
|---|---|---|---|---|---|
| 2048 | 16383 | 256 | 0.803 s (3 runs) | 0.144 s (3 runs) | 5.58× |
| 6144 | 55039 | 512 | 17.851 s (3 runs) | 1.805 s (5 runs) | **9.89×** |

This compares the block rewrite alone, excluding reduction, RREF, and packing the new block;
it is not a whole-`add()` or closure speedup. The research job continued running throughout.
The default GPU setting and development version are unchanged. Raw timings are in
`bench/results/gpu_rewrite_small.txt` and `gpu_rewrite_large.txt`.

Using the same D: cache/temp paths and one-thread environment shown above:

```powershell
.venv/Scripts/python.exe bench/bench_gpu_rewrite.py
.venv/Scripts/python.exe bench/bench_gpu_rewrite.py --rank 6144 --cols 55039 --pivots 512 --cpu-repeats 3 --repeats 5
```

All **196 tests pass**, including 80 GPU cases. Coverage includes byte identity through
rank-deficient and full-rank additions, permuted pivots and columns, strided RHS matrices,
row and column tails, both RHS cache modes, 1/2/3-limb arithmetic at the maximum exact inner
dimension, non-default CUDA streams, exception cleanup/retry, and CPU reduction above the
GPU rank limit followed by GPU rewriting. The vendored reference is unchanged.

## Roadmap

1. **CPU.**
   - Fuse the mod-$p$ steps into the matrix-product loop.
   - Parallelize the small recursive RREF and the gathers of pivot and non-pivot columns.
2. **GPU, exactly.** The opt-in reduction and rewrite above are implemented. Next:
   - rerun the paired 240-second full-closure comparison when the active research lane is idle;
   - reduce repeated basis transfers across batches and measure the small-basis crossover.
3. **A closure driver** independent of the research code: generic operators (multiplication
   tables in layers), checkpoint and resume, and a validation harness to run against any new
   backend before use.

## License

MIT. See `LICENSE`.

## Authors

Patrick Dugan and Claude (Anthropic).
