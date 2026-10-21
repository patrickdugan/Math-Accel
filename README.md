# Math-Accel

Exact modular linear algebra for **degree-bounded closures**: the linear-algebra core of Macaulay,
XL and polynomial-calculus style computations over a prime field $\mathbb F_p$. Every result is
exact. Speed comes from data layout and parallel kernels, never from approximation.

The code comes from the closure engines of a research project on index calculus for elliptic
curves over prime fields (`semaev-index-calculus`). There a closure is a subspace $W_D$ of a
large polynomial algebra, grown by multiplying by tower variables and reducing until it is closed
or contains 1. Those runs reach $10^5$ columns and take hours to days on one machine, so the
engine's speed decides which questions can be asked.

## What is here (v0.2)

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

**Exactness model.** All arithmetic is float64 on integers in $[0,p)$. Every matrix product is cut
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

## Roadmap

1. **CPU.**
   - Fuse the mod-$p$ steps into the matrix-product loop.
   - Parallelize the small recursive RREF and the gathers of pivot and non-pivot columns.
2. **GPU, exactly.** Consumer GPUs run float64 at 1/32 to 1/64 of the float32 rate, so the float64
   trick does not transfer. The plan is **int8 tensor cores with limb splitting**:
   - write each residue mod $p<2^{21}$ as three 7-bit limbs;
   - form a product as 9 int8 matrix products with exact int32 accumulation, exact for inner
     dimensions up to about $2^{17}$;
   - recombine mod $p$ on the device.

   With 4 GB of VRAM the basis does not fit for large closures, so blocks are streamed. That pays
   only for the matrix-product share of the time, which the CPU work above should make the larger
   share.
3. **A closure driver** independent of the research code: generic operators (multiplication
   tables in layers), checkpoint and resume, and a validation harness to run against any new
   backend before use.

## License

MIT. See `LICENSE`.

## Authors

Patrick Dugan and Claude (Anthropic).
