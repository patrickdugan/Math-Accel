"""Reference exact kernels, vendored verbatim from semaev-index-calculus (semaev_ic/closure_fast.py):
kmax, submul, _rref_small, rref. Used as the ground truth in the equivalence tests."""

from __future__ import annotations

import numpy as np

EXACT = 2 ** 53


def kmax(p: int) -> int:
    """Largest inner dimension K with K * (p - 1)^2 + p < 2^53 (at least 1)."""
    return max(1, (EXACT - p) // ((p - 1) ** 2) - 1)


def submul(V: np.ndarray, A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """V <- (V - A @ B) mod p, exactly. V, A, B float64 with entries in [0, p)."""
    K = A.shape[1]
    if K == 0 or V.shape[0] == 0:
        return V
    step = kmax(p)
    for s in range(0, K, step):
        V -= A[:, s:s + step] @ B[s:s + step]
        np.mod(V, p, out=V)
    return V


def _rref_small(V: np.ndarray, p: int):
    """RREF of a few rows by a pivot loop. V is overwritten. Returns (rows, pivot columns)."""
    V = V[V.any(axis=1)]
    nr = V.shape[0]
    piv: list[int] = []
    done, c = 0, 0
    while done < nr:
        nzc = np.flatnonzero(V[done:, c:].any(axis=0))
        if len(nzc) == 0:
            break
        c += int(nzc[0])
        r = done + int(np.flatnonzero(V[done:, c])[0])
        if r != done:
            V[[done, r]] = V[[r, done]]
        V[done, c:] = np.mod(V[done, c:] * pow(int(V[done, c]), -1, p), p)
        col = V[:, c].copy()
        col[done] = 0
        nz = np.flatnonzero(col)
        if len(nz):
            V[nz, c:] = np.mod(V[nz, c:] - np.outer(col[nz], V[done, c:]), p)
        piv.append(c)
        done += 1
        c += 1
    return V[:done], piv


def rref(V: np.ndarray, p: int, leaf: int = 48):
    """RREF over F_p of V (float64 in [0, p)), by splitting rows: echelonize the top half, reduce
    the bottom half against it with one GEMM, recurse, then clear the bottom pivots from the top.
    Returns (R, piv): R[:, piv] = identity and each row is zero left of its pivot."""
    if V.shape[0] <= leaf:
        return _rref_small(V.copy(), p)
    h = V.shape[0] // 2
    R1, p1 = rref(V[:h], p, leaf)
    V2 = V[h:].copy()
    if p1:
        submul(V2, V2[:, p1], R1, p)
    R2, p2 = rref(V2, p, leaf)
    if p2 and len(p1):
        submul(R1, R1[:, p2], R2, p)
    if not p1:
        return R2, p2
    if not p2:
        return R1, p1
    return np.vstack([R1, R2]), p1 + p2
