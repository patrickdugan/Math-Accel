"""Reference, vendored verbatim from semaev-index-calculus (semaev_ic/cbasis.py) except for the
import line. A row-reduced basis that stores only the non-pivot columns (for rank computations only).

closure_fast.Basis keeps every row at full width n, so a rank-r basis costs r * n entries. When
r is a large fraction of n, almost all of that is the identity block. CompressedBasis keeps the
RREF rows on the current non-pivot columns Q only, r * (n - r) entries, at most n^2 / 4. Every
add() eliminates the new pivots from the old rows and drops those columns, so all stored blocks
always share the same column set Q.

Exactness as in closure_fast: float64 GEMMs with the inner dimension cut at kmax(p), float32
storage (exact for p < 2^24).

pack=True stores the blocks as PackedBlock instead: two residues per five bytes (40-bit words,
little-endian), exact for p <= 2^20. That is 2.5 bytes per entry against 4, so the peak
r (n - r) of a closure drops from n^2 (two sigma-halves of ~95000 columns at k = 12: 18 GB)
to 11.3 GB. Rows are unpacked to float64 chunk by chunk when they are used, exactly where the
float32 blocks were converted anyway, so the arithmetic is unchanged.
"""

from __future__ import annotations

import numpy as np

from .exact_ref import kmax, rref, submul

PACK_BITS = 20
PACK_MASK = np.uint64((1 << PACK_BITS) - 1)


class PackedBlock:
    """rows x ncols residues < 2^20, two per five bytes. Indexing by a row (int) or a slice gives
    float64 rows on the stored columns, like indexing the float32 block it replaces."""

    __slots__ = ("data", "rows", "ncols")

    def __init__(self, a: np.ndarray):
        rows, ncols = a.shape
        c2 = (ncols + 1) // 2
        v = np.zeros((rows, 2 * c2), dtype=np.uint64)
        v[:, :ncols] = a
        w = v[:, 0::2] | (v[:, 1::2] << np.uint64(PACK_BITS))
        self.data = np.ascontiguousarray(w.view(np.uint8).reshape(rows, c2, 8)[:, :, :5])
        self.rows, self.ncols = rows, ncols

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.ncols)

    @property
    def size(self) -> int:
        return self.rows * self.ncols

    def __getitem__(self, s):
        b = self.data[s]
        single = b.ndim == 2
        if single:
            b = b[None]
        rows, c2 = b.shape[0], b.shape[1]
        full = np.zeros((rows, c2, 8), dtype=np.uint8)
        full[:, :, :5] = b
        w = full.view(np.uint64).reshape(rows, c2)
        out = np.empty((rows, 2 * c2), dtype=np.float64)
        out[:, 0::2] = (w & PACK_MASK).astype(np.float64)
        out[:, 1::2] = (w >> np.uint64(PACK_BITS)).astype(np.float64)
        out = out[:, :self.ncols]
        return out[0] if single else out


def _chunk64(C, s: int, e: int) -> np.ndarray:
    """Rows s:e of a block as float64."""
    if isinstance(C, PackedBlock):
        return C[s:e]
    return C[s:e].astype(np.float64)


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

    def reduce(self, V: np.ndarray) -> np.ndarray:
        """V (float64, entries in [0, p), full width n) reduced against the basis, on columns Q."""
        p = self.p
        W = V[:, self.Q].copy()
        if not self.rank:
            return W
        coef = V[:, self.P]
        km, acc, off = kmax(p), 0, 0
        # one sub-block's GEMM is a single inner sum, so it must not exceed kmax(p)
        # (kmax = 517 for p ~ 2^22, but 31 for p ~ 2^24)
        step = min(self.sub, km)
        for C in self.blocks:
            for s in range(0, C.shape[0], step):
                blk = _chunk64(C, s, s + step)
                if acc + blk.shape[0] > km:
                    np.mod(W, p, out=W)
                    acc = 0
                W -= coef[:, off + s:off + s + blk.shape[0]] @ blk
                acc += blk.shape[0]
            off += C.shape[0]
        np.mod(W, p, out=W)
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
        Rk = np.ascontiguousarray(R[:, keep])
        del R
        for i, C in enumerate(self.blocks):
            nkeep = int(keep.sum())
            if self.pack:
                chunks = []
            else:
                new = np.empty((C.shape[0], nkeep), dtype=np.float32)
            for s in range(0, C.shape[0], self.sub):
                blk = _chunk64(C, s, s + self.sub)
                out = np.ascontiguousarray(blk[:, keep])
                submul(out, blk[:, pc], Rk, p)
                if self.pack:
                    chunks.append(out)
                else:
                    new[s:s + self.sub] = out
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
