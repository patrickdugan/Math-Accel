"""Profile closure_compressed on one chain closure, the way run_chain.py --engine compressed calls it.

Usage: python bench/profile_closure.py PATH_TO_semaev_ic TOWER K M D SECONDS [--accel]
Runs one target's closure at degree D for SECONDS (closure_compressed's stop_after hook), with a
throwaway checkpoint in the temp directory, and prints the cProfile tables.
"""
import cProfile
import io
import pstats
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, sys.argv.pop(1))  # first argument: the research repository's semaev_ic
import run_chain as RC  # noqa: E402
from chain import ChainAlgebra, chain_generators  # noqa: E402
from chain_engine import EngineView  # noqa: E402
import closure_compressed as CC  # noqa: E402

ACCEL = "--accel" in sys.argv
if ACCEL:
    sys.argv.remove("--accel")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import mathaccel

    CC.CompressedBasis = mathaccel.CompressedBasis
closure_compressed = CC.closure_compressed

tower, k, m, D, secs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
p, E, X = RC.cell_setup(tower, k, 2026)
alg = ChainAlgebra(tower, p, m, k, D, RC.AUX.get(tower))
view = EngineView(alg)
todo, _, _ = RC.draw_targets(E, X, tower, k, m, 2026, 1)
G = chain_generators(alg, E, todo[0][0])
ck = Path(tempfile.gettempdir()) / "mathaccel_prof.ckpt"
for f in (ck, Path(str(ck) + ".tmp")):
    if f.exists():
        f.unlink()
prof = cProfile.Profile()
t0 = time.time()
prof.enable()
r = closure_compressed(view, G, D, symmetric=False, batch=RC.CBATCH, pack=p <= 2 ** 20,
                       checkpoint=str(ck), stop_after=secs, log=lambda s: print(s, flush=True))
prof.disable()
wall = time.time() - t0
print(f"wall {wall:.1f}s rank {r.get('rank')} n {r.get('n')} interrupted {r.get('interrupted')}")
for key in ("tottime", "cumulative"):
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats(key).print_stats(22)
    print(f"===== by {key}\n" + s.getvalue())
for f in (ck, Path(str(ck) + ".tmp")):
    if f.exists():
        f.unlink()
