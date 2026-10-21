"""Multithreaded kernels for the hot spots of the compressed-basis closure.

A profile of a 15360-column closure (isogeny tower, m = 5, D = 7; README) found about half the time
in single-threaded memory work:
- unpacking and repacking the 2.5-byte residue storage, about 36%;
- array copies, about 11%.
The matrix products themselves were about 42%.

These kernels do the same work in parallel over rows, with Numba:
- `pack_rows` and `unpack_rows`, with exactly the reference byte layout: two residues below 2^20
  per 40-bit little-endian word, stored as 5 bytes;
- `gather_cols`, which unpacks only the requested columns, so a caller never unpacks a whole
  block just to slice it;
- `mod_inplace`, an exact x mod p for float64 integers below 2^53.

If Numba is not installed, every function falls back to NumPy code with the same results.
"""

from __future__ import annotations



import numpy as np

PACK_BITS = 20
_MASK = (1 << PACK_BITS) - 1

try:  # pragma: no cover - exercised in both configurations by the tests
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

try:  # pragma: no cover
    from scipy.linalg.blas import dgemm as _dgemm
except ImportError:  # pragma: no cover
    _dgemm = None


if HAVE_NUMBA:

    @njit(parallel=True, cache=True)
    def _pack(a, ncols, c2):
        rows = a.shape[0]
        data = np.empty((rows, c2, 5), np.uint8)
        sh = np.uint64(PACK_BITS)
        for r in prange(rows):
            for j in range(c2):
                c = 2 * j
                v0 = np.uint64(a[r, c])
                v1 = np.uint64(0)
                if c + 1 < ncols:
                    v1 = np.uint64(a[r, c + 1])
                w = v0 | (v1 << sh)
                data[r, j, 0] = np.uint8(w & np.uint64(255))
                data[r, j, 1] = np.uint8((w >> np.uint64(8)) & np.uint64(255))
                data[r, j, 2] = np.uint8((w >> np.uint64(16)) & np.uint64(255))
                data[r, j, 3] = np.uint8((w >> np.uint64(24)) & np.uint64(255))
                data[r, j, 4] = np.uint8((w >> np.uint64(32)) & np.uint64(255))
        return data

    @njit(parallel=True, cache=True)
    def _unpack(data, s, e, ncols):
        rows = e - s
        c2 = data.shape[1]
        out = np.empty((rows, ncols), np.float64)
        mask = np.uint64(_MASK)
        sh = np.uint64(PACK_BITS)
        for r in prange(rows):
            for j in range(c2):
                w = (np.uint64(data[s + r, j, 0]) | (np.uint64(data[s + r, j, 1]) << np.uint64(8))
                     | (np.uint64(data[s + r, j, 2]) << np.uint64(16))
                     | (np.uint64(data[s + r, j, 3]) << np.uint64(24))
                     | (np.uint64(data[s + r, j, 4]) << np.uint64(32)))
                c = 2 * j
                out[r, c] = np.float64(w & mask)
                if c + 1 < ncols:
                    out[r, c + 1] = np.float64(w >> sh)
        return out

    @njit(parallel=True, cache=True)
    def _unpack_into(data, s, e, ncols, out, r0):
        rows = e - s
        c2 = data.shape[1]
        mask = np.uint64(_MASK)
        sh = np.uint64(PACK_BITS)
        for r in prange(rows):
            for j in range(c2):
                w = (np.uint64(data[s + r, j, 0]) | (np.uint64(data[s + r, j, 1]) << np.uint64(8))
                     | (np.uint64(data[s + r, j, 2]) << np.uint64(16))
                     | (np.uint64(data[s + r, j, 3]) << np.uint64(24))
                     | (np.uint64(data[s + r, j, 4]) << np.uint64(32)))
                c = 2 * j
                out[r0 + r, c] = np.float64(w & mask)
                if c + 1 < ncols:
                    out[r0 + r, c + 1] = np.float64(w >> sh)

    @njit(parallel=True, cache=True)
    def _gather(data, s, e, cols):
        rows = e - s
        k = cols.shape[0]
        out = np.empty((rows, k), np.float64)
        mask = np.uint64(_MASK)
        for r in prange(rows):
            for i in range(k):
                c = cols[i]
                j = c >> 1
                w = (np.uint64(data[s + r, j, 0]) | (np.uint64(data[s + r, j, 1]) << np.uint64(8))
                     | (np.uint64(data[s + r, j, 2]) << np.uint64(16))
                     | (np.uint64(data[s + r, j, 3]) << np.uint64(24))
                     | (np.uint64(data[s + r, j, 4]) << np.uint64(32)))
                if c & 1:
                    w = w >> np.uint64(PACK_BITS)
                out[r, i] = np.float64(w & mask)
        return out

    @njit(parallel=True, cache=True)
    def _mod(V, p):
        rows, cols = V.shape
        for i in prange(rows):
            for j in range(cols):
                x = V[i, j]
                # q * p and x - q * p are integers below 2^53, so exact; floor(x / p) may be off
                # by one near a multiple of p, which the two corrections fix
                r = x - p * np.floor(x / p)
                if r < 0.0:
                    r += p
                elif r >= p:
                    r -= p
                V[i, j] = r
        return V


def pack_rows(a: np.ndarray) -> np.ndarray:
    """Residues in [0, 2^20) (any float or integer dtype, rows x ncols) -> (rows, ceil(ncols/2), 5)
    uint8, the reference PackedBlock layout."""
    rows, ncols = a.shape
    c2 = (ncols + 1) // 2
    if HAVE_NUMBA and rows and ncols:
        return _pack(np.ascontiguousarray(a), ncols, c2)
    v = np.zeros((rows, 2 * c2), dtype=np.uint64)
    v[:, :ncols] = a
    w = v[:, 0::2] | (v[:, 1::2] << np.uint64(PACK_BITS))
    return np.ascontiguousarray(w.view(np.uint8).reshape(rows, c2, 8)[:, :, :5])


def _words(data: np.ndarray) -> np.ndarray:
    full = np.zeros(data.shape[:2] + (8,), dtype=np.uint8)
    full[:, :, :5] = data
    return full.view(np.uint64).reshape(data.shape[:2])


def unpack_into(data: np.ndarray, s: int, e: int, ncols: int, out: np.ndarray, r0: int) -> None:
    """Write rows s:e of packed data into out[r0:r0 + (e - s)] (float64, C-contiguous)."""
    e = min(e, data.shape[0])
    if e <= s:
        return
    if HAVE_NUMBA:
        _unpack_into(data, s, e, ncols, out, r0)
    else:
        out[r0:r0 + (e - s)] = unpack_rows(data, s, e, ncols)


def gemm_sub(C: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """C <- C - A @ B in place, for C-contiguous float64 C (m x n) and B (k x n); A is m x k.

    With SciPy this is one BLAS dgemm with beta = 1 on the transposed problem
    C^T <- C^T - B^T A^T, so no temporary of C's size is made. A is used without a copy when it is
    F-contiguous, for instance the transpose of a C-contiguous k x m array. Without SciPy it is
    C -= A @ B.
    """
    if A.shape[1] == 0 or C.size == 0:
        return C
    if _dgemm is not None and C.flags.c_contiguous and B.flags.c_contiguous:
        _dgemm(-1.0, B.T, A, beta=1.0, c=C.T, trans_a=0, trans_b=1, overwrite_c=True)
        return C
    C -= A @ B
    return C


def unpack_rows(data: np.ndarray, s: int, e: int, ncols: int) -> np.ndarray:
    """Rows s:e of packed data as float64 (e - s, ncols)."""
    e = min(e, data.shape[0])
    if e <= s:
        return np.zeros((0, ncols))
    if HAVE_NUMBA:
        return _unpack(data, s, e, ncols)
    w = _words(data[s:e])
    out = np.empty((e - s, 2 * data.shape[1]), dtype=np.float64)
    out[:, 0::2] = (w & np.uint64(_MASK)).astype(np.float64)
    out[:, 1::2] = (w >> np.uint64(PACK_BITS)).astype(np.float64)
    return out[:, :ncols]


def gather_cols(data: np.ndarray, s: int, e: int, cols: np.ndarray) -> np.ndarray:
    """Rows s:e, columns cols (int64, in the given order) of packed data, as float64."""
    e = min(e, data.shape[0])
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    if e <= s:
        return np.zeros((0, len(cols)))
    if HAVE_NUMBA:
        return _gather(data, s, e, cols)
    w = _words(data[s:e])[:, cols >> 1]
    w = np.where(cols & 1, w >> np.uint64(PACK_BITS), w)
    return (w & np.uint64(_MASK)).astype(np.float64)


def mod_inplace(V: np.ndarray, p: int) -> np.ndarray:
    """V <- V mod p in [0, p), exactly, for float64 V holding integers below 2^53."""
    if HAVE_NUMBA and V.ndim == 2 and V.flags.c_contiguous and V.size > 4096:
        return _mod(V, float(p))
    np.mod(V, p, out=V)
    return V
