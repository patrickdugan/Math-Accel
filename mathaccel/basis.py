"""Exact row-reduced bases mod p that store only the non-pivot columns (accelerated).

This is the compressed basis of `semaev-index-calculus` (`semaev_ic/cbasis.py`, vendored unchanged
in `mathaccel.reference`). It keeps the same object, the same algorithm and the same order of
operations, and it gives bit-identical results; `tests/test_equivalence.py` checks this. Only the
data movement differs:
- packed rows are unpacked by a parallel kernel;
- `add()` gathers just the columns it needs from each packed block, with no full unpack and no
  copy;
- reductions mod p run in parallel.

The basis is kept in reduced row echelon form on the current non-pivot columns Q: r rows by
(n - r) stored entries, at most n^2 / 4. Each `add()` eliminates the new pivots from the old rows
and drops those columns.

Exactness:
- float64 matrix products, with the inner dimension cut at `kmax(p)` so that every partial sum is
  an integer below 2^53;
- storage either float32 (exact for p < 2^24) or packed, two residues per 5 bytes (exact for
  p <= 2^20).
"""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .reference.exact_ref import kmax, rref

PACK_BITS = K.PACK_BITS


def submul(V: np.ndarray, A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """V <- (V - A @ B) mod p, exactly. V, A, B float64 with entries in [0, p)."""
    Kd = A.shape[1]
    if Kd == 0 or V.shape[0] == 0:
        return V
    step = kmax(p)
    for s in range(0, Kd, step):
        K.gemm_sub(V, A[:, s:s + step], np.ascontiguousarray(B[s:s + step]))
        K.mod_inplace(V, p)
    return V


class PackedBlock:
    """rows x ncols residues < 2^20, two per five bytes (the reference layout). Indexing by a row
    (int) or a slice gives float64 rows on the stored columns."""

    __slots__ = ("data", "rows", "ncols")

    def __init__(self, a: np.ndarray):
        self.rows, self.ncols = a.shape
        self.data = K.pack_rows(a)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.ncols)

    @property
    def size(self) -> int:
        return self.rows * self.ncols

    def rows64(self, s: int, e: int) -> np.ndarray:
        return K.unpack_rows(self.data, s, e, self.ncols)

    def gather(self, s: int, e: int, cols: np.ndarray) -> np.ndarray:
        return K.gather_cols(self.data, s, e, cols)

    def __getitem__(self, s):
        if isinstance(s, (int, np.integer)):
            i = int(s)
            if i < 0:
                i += self.rows
            return self.rows64(i, i + 1)[0]
        if isinstance(s, slice):
            a, b, step = s.indices(self.rows)
            if step == 1:
                return self.rows64(a, b)
        return np.vstack([self[int(i)] for i in range(self.rows)[s]])


def _rows64(C, s: int, e: int) -> np.ndarray:
    if isinstance(C, PackedBlock):
        return C.rows64(s, e)
    return C[s:e].astype(np.float64)


def _gather64(C, s: int, e: int, cols: np.ndarray) -> np.ndarray:
    if isinstance(C, PackedBlock):
        return C.gather(s, e, cols)
    return np.ascontiguousarray(C[s:e][:, cols], dtype=np.float64)


class CompressedBasis:
    def __init__(self, n: int, p: int, sub: int = 512, pack: bool = False):
        assert p < 2 ** 24
        assert not pack or p <= 2 ** PACK_BITS, "packed storage holds residues below 2^20"
        self.p, self.n, self.sub, self.pack = p, n, sub, pack
        self.Q = np.arange(n, dtype=np.int64)  # non-pivot columns, in storage order
        self.P = np.zeros(0, dtype=np.int64)  # pivot columns, in row order
        self.blocks: list = []  # float32 arrays or PackedBlocks, rows aligned with P, columns with Q

    def _store(self, a: np.ndarray):
        return PackedBlock(a) if self.pack else a.astype(np.float32)

    @property
    def rank(self) -> int:
        return len(self.P)

    #: memory for one streamed chunk of basis rows in reduce(), in bytes (float64)
    chunk_bytes = 256 * 2 ** 20

    def reduce(self, V: np.ndarray) -> np.ndarray:
        """V (float64, entries in [0, p), full width n) reduced against the basis, on columns Q.

        The rows of all blocks are streamed in order into one buffer of `chunk` rows (at least
        `sub`, at most kmax(p) and the rank, and about `chunk_bytes` of float64), and each full buffer is
        subtracted with one in-place product: W <- W - coef[:, rows] @ buffer. Every product is one
        inner sum of at most kmax(p) terms, and W is reduced mod p before the running count of
        terms would exceed kmax(p), so every partial sum stays an integer below 2^53. The result
        is the reference's exactly; only the grouping of the terms differs.
        """
        p = self.p
        W = np.ascontiguousarray(V[:, self.Q])
        if not self.rank:
            return W
        q = W.shape[1]
        coefT = np.ascontiguousarray(V[:, self.P].T)  # rank x batch, so coefT[a:b].T is F-ordered
        km = kmax(p)
        chunk = int(min(km, self.rank, max(self.sub, self.chunk_bytes // (8 * max(q, 1)))))
        buf = np.empty((chunk, q), dtype=np.float64)
        state = [0, 0, 0]  # rows filled in buf, basis row of buf[0], terms since the last mod

        def flush():
            filled, start, acc = state
            if not filled:
                return
            if acc + filled > km:
                K.mod_inplace(W, p)
                acc = 0
            K.gemm_sub(W, coefT[start:start + filled].T, buf[:filled])
            state[:] = [0, start + filled, acc + filled]

        for C in self.blocks:
            s, rows = 0, C.shape[0]
            while s < rows:
                take = min(chunk - state[0], rows - s)
                if isinstance(C, PackedBlock):
                    K.unpack_into(C.data, s, s + take, C.ncols, buf, state[0])
                else:
                    buf[state[0]:state[0] + take] = C[s:s + take]
                state[0] += take
                s += take
                if state[0] == chunk:
                    flush()
        flush()
        K.mod_inplace(W, p)
        return W

    def add(self, V: np.ndarray) -> int:
        """Add the rows of V; return the number of new independent rows."""
        p = self.p
        W = self.reduce(V)
        W = W[W.any(axis=1)]
        if W.shape[0] == 0:
            return 0
        R, pc = rref(W, p)
        del W
        if not pc:
            return 0
        pc = np.array(pc, dtype=np.int64)
        keep = np.ones(len(self.Q), dtype=bool)
        keep[pc] = False
        keep_idx = np.flatnonzero(keep)
        Rk = np.ascontiguousarray(R[:, keep])
        del R
        for i, C in enumerate(self.blocks):
            nkeep = len(keep_idx)
            if self.pack:
                chunks = []
            else:
                new = np.empty((C.shape[0], nkeep), dtype=np.float32)
            for s in range(0, C.shape[0], self.sub):
                e = min(s + self.sub, C.shape[0])
                out = _gather64(C, s, e, keep_idx)
                submul(out, _gather64(C, s, e, pc), Rk, p)
                if self.pack:
                    chunks.append(out)
                else:
                    new[s:e] = out
            if self.pack:
                new = PackedBlock(chunks[0] if len(chunks) == 1 else np.vstack(chunks))
                del chunks
            self.blocks[i] = new
            del C
        self.blocks.append(self._store(Rk))
        self.P = np.concatenate([self.P, self.Q[pc]])
        self.Q = self.Q[keep]
        return len(pc)

    def stored(self) -> int:
        return sum(C.size for C in self.blocks)
