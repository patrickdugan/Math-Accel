# Handoff for Codex (2026-10-21)

Two threads of work are open. Both are on Patrick's Windows PC: i5-11400H, 6 cores and 12
threads, about 24 GB of RAM of which about 12 GB is usable, and an RTX 3050 Laptop GPU with 4 GB.
Drive C: is **full (0 bytes free)**, so keep every file on D:.

## 1. Math-Accel (this repository, public, MIT)

An exact mod-p linear-algebra engine for degree-bounded closures. It comes from the
`CompressedBasis` of the private research repo `patrickdugan/semaev-index-calculus`
(`semaev_ic/cbasis.py`, `closure_compressed.py`).

**Rule.** Every change must stay **bit-identical** to `mathaccel/reference/` (the original code,
vendored verbatim). Run the tests before each commit.

```bash
cd D:/Math-Accel
CUPY_CACHE_DIR=D:/Temp/cupy_cache .venv/Scripts/python.exe -m pytest -q tests   # 155 pass
```

- `.venv` has numpy 2.5.4, numba 0.68, scipy 1.18 and cupy-cuda12x 14.2. Set `CUPY_CACHE_DIR` to a
  folder on D:: the default is on C:, which is full.

**Status.**
- **v0.1** (pushed): Numba-parallel pack and unpack, a fused column gather, and an exact parallel
  mod p.
- **v0.2** (pushed): `reduce()` streams rows across blocks into an adaptive buffer and accumulates
  in place with SciPy `dgemm` (beta = 1). A full 15360-column closure takes 972.7 s with the
  reference and **191.3 s** with v0.2 (5.1×), same result. README has the table.
- **v0.3 (GPU, written and tested, pushed with this handoff):**
  - `mathaccel/gpu.py` holds `GPUReducer`, an exact mod-p product on int8 tensor cores.
  - Each residue is split into 3 limbs of 7 bits, giving 9 cuBLAS `gemmEx` int8→int32 products
    (exact for inner dimension up to 133144), then a fused kernel folds them mod p.
  - The basis is uploaded once per call, by column tiles, and unpacked into limbs on the GPU.
  - It is opt in: `CompressedBasis.use_gpu = True`, active for packed bases storing at least
    `gpu_min_stored` entries.
  - `tests/test_gpu.py` (39 tests) covers exact equality with the reference.
  - **Unmeasured:** its speed. A benchmark was running at handoff:
    `bench/bench_chain.py --repo D:/semaev-nb/semaev_ic accel isogeny 2 6 7 --seconds 240 [--gpu]`
    compares the rank reached in 240 s, CPU against GPU. Run both and record the numbers in the
    README.
- Measured on this card: int8 GEMM about 42 TOPS, fp16 about 10 TFLOPS, fp32 about 3.5 TFLOPS.
  PCIe is 5.2 GB/s up and 4.4 GB/s down with pinned memory, and 2.1 GB/s pageable.
  `bench/probe_*.py` reproduce these.

**Next steps, in order.**
1. Finish the v0.3 benchmark. If the GPU wins, bump the version, update the README and push.
2. **Pinned memory and overlap** in `GPUReducer.reduce`. Host chunks are pageable now (2.1 GB/s);
   use pinned buffers and double-buffer the upload against the GEMMs.
3. **Move `add()`'s rewrite** of every stored block (`out = blk[:, keep] - blk[:, pc] @ Rk`) to
   the GPU with the same limb GEMM. It costs as many flops as `reduce()`. Better still, fuse it with
   the next `reduce()` pass so the basis crosses PCIe once per batch.
4. CPU leftovers: a parallel recursive RREF, and fusing the mod-p steps into the product loop.
5. A standalone closure driver and pip packaging.

**Profiles** are in `bench/results/`. Before v0.1, half the time went to single-threaded packing
and copies. After v0.2, about 65% goes to matrix products, which is why the GPU path is the next
lever.

## 2. The research experiments (repo `semaev-index-calculus`, branch `claude/nifty-brown-nu441z`)

A coordinating Claude session writes handoff notes in `semaev_ic/handoff/`. Follow the newest one
addressed to the Windows PC. The latest is **`to_windows_pc_worker_part2.md`**.

**Rules**:
- do not modify code, seeds, primes or caps;
- run only the cells the note assigns;
- report summary lines verbatim, without interpretation.

**Worktrees.** Never commit or pull in a worktree whose lane is running: git's autostash orphans
the lane's log file.
- `D:\semaev-nb` is lane `subhanallah`;
- `D:\semaev-nb2` is lane `subhanallah-b`.

**Environment.**
- Python: `PYTHON=C:/Users/patri/semaev-index-calculus/.venv/Scripts/python.exe`. Do not install
  into that venv.
- Git Bash has no `hostname -s`, so put a shim first on PATH:
  - `PATH=/d/Temp/shim:$PATH` makes `hostname -s` print `subhanallah`;
  - `PATH=/d/Temp/shimb:$PATH` makes it print `subhanallah-b`.

**Running now:** `isogeny:3:5` in `D:\semaev-nb`. It is Experiment 10, PKM's isogeny tower, at
h = 3, m = 5, D = 7, with the compressed engine, cap 110000 and 12 threads. It was launched with:

```bash
cd D:/semaev-nb && PATH=/d/Temp/shim:$PATH PYTHON=C:/Users/patri/semaev-index-calculus/.venv/Scripts/python.exe \
  nohup bash semaev_ic/run_chain_cells.sh --sync --engine compressed --cap 110000 --threads 12 isogeny:3:5 > /dev/null 2>&1 &
```

- At handoff, its first target was at rank about 28.3k of 106970. The queue was shrinking, so the
  closure is probably close to saturating.
- It uses about 7 GB, and checkpoints every 30 min to `semaev_ic/checkpoints/`.
- If it stops, rerun the same command: it resumes from the checkpoint.
- Watch with `tail -f D:/semaev-nb/semaev_ic/results/chain_worker_subhanallah.log`.

**When it logs `cells done`:**
1. Add its summary line `isogeny k=3 m=5: D_* = [...] (..., compressed)` verbatim to
   `semaev_ic/handoff/from_windows_pc_worker_exp10.md`. Commit from the idle worktree
   `D:\semaev-nb2`, then pull and push.
2. Resume the paused h = 2 lanes (their checkpoints are kept), each with 6 threads and cap 72000:

   ```bash
   cd D:/semaev-nb  && PATH=/d/Temp/shim:$PATH  PYTHON=... nohup bash semaev_ic/run_chain_cells.sh --sync --engine compressed --cap 72000 --threads 6 isogeny:2:6 > /dev/null 2>&1 &
   cd D:/semaev-nb2 && PATH=/d/Temp/shimb:$PATH PYTHON=... nohup bash semaev_ic/run_chain_cells.sh --sync --engine compressed --cap 72000 --threads 6 isogeny:4:4 isogeny:2:7 > /dev/null 2>&1 &
   ```

3. When both lanes are done, put their summary lines in the same report file.

**Already reported** (dense engine, cap 30000):

| Cell | D_* |
|---|---|
| isogeny k=2 m=3 | [6, 6, 6] |
| isogeny k=2 m=4 | [6, 6, 6] |
| isogeny k=2 m=5 | [7, 7, 7] |
| isogeny k=2 m=6 | ['>6', '>6', '>6'] |
| isogeny k=2 m=7 | ['>5', '>5', '>5'] |
| isogeny k=4 m=4 | ['>5', '>5', '>5'] |

With the compressed engine at cap 72000, isogeny k=4 m=4 gave ['>6', '>6', '>6'].

**Paused cells:**
- Experiment 7's `dickson:2:6`: two targets capped at D = 6, the third stopped after D = 4.
- `isogeny:2:6`: its first target was at rank about 19k of 61183 at D = 7.
- `isogeny:2:7`: its first target was not refuted at D = 6 (rank 12888); the second was in
  progress.

**The paper** (`main` of `semaev-index-calculus`, `paper/paper.md`, revision 5, with Claude as
co-author) is finished. A Mythos session is polishing it. Don't edit it unless asked.

## Update: the GPU benchmark result (just in)

On isogeny h=2, m=6, D=7 (61183 columns), in the same 240 s window:

| Engine | Rank reached |
|---|---|
| CPU v0.2 | 6118 |
| GPU path | 5463 |

The GPU path is **not faster yet** at this size. The likely causes, to check first:
- Pageable host-to-GPU copies (2.1 GB/s), assembled with `np.concatenate`.
- The whole basis is re-uploaded on every `reduce()` call.
- `add()` still runs on the CPU.
- Small bases pay the per-tile setup cost.

Profile `GPUReducer.reduce` before going further (next steps 2–3 above). Raising
`gpu_min_stored` would keep small bases on the CPU. The GPU path stays off by default
(`use_gpu = False`), so nothing depends on it.


## Codex continuation: GPU staging and fold (2026-10-21, Asia/Taipei)

The GPU path has been profiled and optimized locally. The running research lane was left
untouched; no research code, parameters, checkpoints, or Git state were changed. Its log was
still progressing (rank 19827/106970 at the last status check). No competing full closure was
launched. Math-Accel checks used one CPU thread, with TEMP, TMP, and CUPY_CACHE_DIR on D:.

Changes:
- `mathaccel/gpu.py`: direct copies from packed blocks into pinned staging buffers, a separate
  upload stream, and two event-protected slots when the rank spans multiple chunks. This removes
  concatenate/pageable staging and permits upload/GEMM overlap.
- The fold kernel now gathers V on Q, subtracts, and returns float64 residues. The entire
  nine-term weighted int64 sum is below 2^56, so it needs only one modulo. This removes the
  full host Q gather, int64-to-float64 conversion, and final NumPy modulo.
- Custom cuBLAS calls follow the caller's CUDA stream; buffer lifetimes are protected through
  asynchronous work and exception cleanup. The previous cuBLAS stream is restored.
- `bench/bench_gpu_reduce.py` generates deterministic synthetic packed RREF inputs and supports
  `--source` for old/new comparisons, `--check` against the exact CPU reducer, and host profiles.
- `tests/test_gpu.py`: async slot reuse, odd final tiles, non-default streams, immutable inputs,
  strided/permuted probes, empty dimensions, and rank 133144 for 1/2/3 limbs.

Results: p=263167, batch=512, rank=6144, q=55039 (61183 total columns), packed basis=845.4 MB,
block rows=128, tile columns=16384. Medians of five warmed repetitions:

| Backend | Row chunk | Seconds per reduction |
|---|---|---|
| Original GPU at bdab8eb | 8192 | 3.800567 |
| Optimized GPU | 8192 (default unchanged) | 1.336728 |
| Optimized GPU | 2048 | 1.330063 |

This is a **2.84x reduction microbenchmark improvement**, not a closure speedup. The optimized
full-sized output matched the exact CPU reducer. CPU-side profiles of the old path measured
about 0.54 s for gathering Q and 0.91 s for final subtraction/modulo in one representative call.
GPU work is asynchronous, so its time is included in download waits, not isolated by cProfile.
Raw profiles/timings are `bench/results/gpu_reduce_*.txt`; README includes reproduction commands.
The research lane was active throughout, so timings are not from an isolated machine.

Validation: **164 passed** (116 CPU/reference tests + 48 GPU tests), in 44.04 s with one CPU
thread. The reference directory is unchanged. CuPy emits a CUDA_PATH detection warning but
loads and runs the actual GPU tests successfully.

**Next:**
1. When the active research lane is idle, rerun the original paired CPU/GPU 240-second closure
   benchmark at the same thread count, capture logs, and decide whether GPU wins overall.
   GPU stays opt-in (`use_gpu=False` by default); no version/release promotion is justified yet.
2. Move `add()`'s block rewrite onto the GPU, then consider retaining/reusing basis transfers
   across reduction and rewrite. No rewrite changes were made in this continuation.
3. Keep following the research lane's original handoff independently; report only its exact
   summary lines when it finishes. Do not pull/commit in a running lane's worktree.


## Codex continuation: GPU add() block rewrite (2026-10-21, Asia/Taipei)

The preceding commit `8d496a0` was pushed to `patrickdugan/Math-Accel` main as requested.
The public remote's ownership and push permission were verified against the connected GitHub
account. The research job remained running untouched throughout this continuation. No research
code, processes, Git state, parameters, or checkpoints were changed; its log continued advancing.

**Implemented:** `GPUReducer.rewrite(blocks, pc, keep, Rk)` replaces each old stored block with
`(blk[:, keep] - blk[:, pc] @ Rk) mod p`, with GPU pivot gathers, exact int8 limb GEMMs,
modular subtraction, retained-column gathers, and packing into the reference's 20-bit format.
Packed bytes download directly. Each packed row chunk is uploaded once per rewrite; row cap
`rewrite_rows` defaults to 512. Blocks retain their boundaries and are replaced progressively.
The RHS limb cache defaults to 256 MiB; larger RHS data uses bounded column tiles regenerated
per row chunk. The kernel uses the same int32 and int64 exactness bounds as reduction.

`CompressedBasis.add()` selects this path under the existing `use_gpu` opt-in and
`gpu_min_stored` threshold. The CPU rewrite was extracted unchanged into `_rewrite_cpu`.
A GPU product's inner dimension must be <=133144: old rank for reduce(), new pivot count for
rewrite. Each phase falls back to CPU independently above that bound. RREF of the new rows and
packing the newly appended block are still CPU work. Reduction and rewrite still upload the
basis separately; sharing those transfers is future work.

**Validation:** 196 passed (116 CPU/reference + 80 GPU cases), 38.54 s with one CPU thread.
Coverage adds direct packed-byte comparisons, multiple chunks, odd widths, permuted gathers,
strided RHS, both cache paths, 1/2/3-limb maximum-inner-dimension cases, non-default streams,
full-rank/rank-deficient add() sequences, missing-GPU fallback, exception cleanup/retry, and
CPU reduction above the GPU rank limit with a GPU rewrite afterwards. Reference files are
unchanged. The shared synthetic fixture was also checked against exact CPU reduction.

**Paired rewrite-only microbenchmarks:** RTX 3050 Laptop, p=263167, block rows 128,
tile columns 16384, rewrite row cap 512. One CPU thread; the research job was active.

| Old rank | Old q | New pivots | CPU median | GPU median | Speedup |
|---|---|---|---|---|---|
| 2048 | 16383 | 256 | 0.802509 s (3 runs) | 0.143853 s (3 runs) | 5.579x |
| 6144 | 55039 | 512 | 17.850501 s (3 runs) | 1.804926 s (5 runs) | 9.890x |

All measured GPU outputs matched CPU packed bytes. Allocations, transfers, products, modulo,
and packing are timed; reduction, RREF, and new-block packing are excluded. Raw logs are
`bench/results/gpu_rewrite_{small,large}.txt`; `bench/bench_gpu_rewrite.py` reproduces them.
README has commands and caveats. These are not whole-add() or closure speedups. GPU remains
off by default and the development version is unchanged.

**Next:** keep the research lane untouched. When it is idle, rerun the paired full closure
benchmark to measure the combined GPU reduce + rewrite path before release promotion.
The next implementation opportunity is sharing basis uploads across those two phases or
retaining useful data on device. Follow the separate research handoff for eventual summary
reporting; do not pull/commit in its running worktree.
