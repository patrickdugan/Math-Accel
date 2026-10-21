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
