"""Benchmark: one chain closure of semaev-index-calculus, with the reference basis or Math-Accel's.

The closure driver is `closure_compressed` from the research repository (path given by
--repo). The only change between the two runs is the basis class it uses:
- `ref` is the original `cbasis.CompressedBasis`;
- `accel` is `mathaccel.CompressedBasis`.

Usage:
    python bench/bench_chain.py --repo D:/semaev-nb/semaev_ic {ref|accel} TOWER K M D [--seconds S]

Prints the wall time, the rank reached and whether 1 is in the closure, plus the log lines every
`--every` seconds. With --seconds the closure stops after S seconds (the driver's stop_after
hook), using a throwaway checkpoint in a temp directory.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("which", choices=["ref", "accel"])
ap.add_argument("tower")
ap.add_argument("k", type=int)
ap.add_argument("m", type=int)
ap.add_argument("D", type=int)
ap.add_argument("--seconds", type=float, default=None)
ap.add_argument("--gpu", action="store_true", help="accel only: GPU reduce() (mathaccel.gpu)")
a = ap.parse_args()

sys.path.insert(0, a.repo)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import closure_compressed as CC  # noqa: E402
import run_chain as RC  # noqa: E402
from chain import ChainAlgebra, chain_generators  # noqa: E402
from chain_engine import EngineView  # noqa: E402

if a.which == "accel":
    import mathaccel

    CC.CompressedBasis = mathaccel.CompressedBasis
    mathaccel.CompressedBasis.use_gpu = a.gpu
    print(f"mathaccel {mathaccel.__version__}, numba={mathaccel.HAVE_NUMBA}, gpu={a.gpu}", flush=True)

p, E, X = RC.cell_setup(a.tower, a.k, 2026)
alg = ChainAlgebra(a.tower, p, a.m, a.k, a.D, RC.AUX.get(a.tower))
view = EngineView(alg)
todo, _, _ = RC.draw_targets(E, X, a.tower, a.k, a.m, 2026, 1)
G = chain_generators(alg, E, todo[0][0])
tmp = Path(tempfile.mkdtemp(prefix="mabench"))
ck = tmp / "bench.ckpt"
t0 = time.time()
r = CC.closure_compressed(view, G, a.D, symmetric=False, batch=RC.CBATCH, pack=p <= 2 ** 20,
                          checkpoint=str(ck) if a.seconds else None, stop_after=a.seconds,
                          log=lambda s: print(s, flush=True))
wall = time.time() - t0
print(f"RESULT {a.which}{'+gpu' if a.gpu else ''} {a.tower} k={a.k} m={a.m} D={a.D} n={r.get('n')} rank={r.get('rank')} "
      f"refuted={r.get('one_in_W')} interrupted={r.get('interrupted', False)} wall={wall:.1f}s", flush=True)
for f in tmp.glob("*"):
    f.unlink()
tmp.rmdir()
