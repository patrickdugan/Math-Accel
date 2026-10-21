"""Exact mod-p matrix products on NVIDIA tensor cores, via int8 limb splitting (CuPy + cuBLAS).

Consumer GPUs run float64 at 1/32 to 1/64 of the float32 rate, so the CPU's exact-float64 method
does not transfer. Instead, each residue v < p < 2^21 is written in base 2^7,

    v = l_0 + 2^7 l_1 + 2^14 l_2,  0 <= l_i <= 127,

so every limb fits in an int8. Then A @ B = sum over (i, j) of 2^(7(i+j)) (A_i @ B_j). Each A_i @ B_j
is one cuBLAS int8 GEMM with int32 accumulation (cublasGemmEx, CUBLAS_COMPUTE_32I, tensor cores).
It is exact while 127^2 * K < 2^31, that is for inner dimension K <= 133144. A fused kernel then
forms sum_{i,j} (2^(7(i+j)) mod p) * C_ij mod p in int64. Every term is below 2^52, so the result is
exact.

`GPUReducer.reduce(basis, V)` computes the same W as `CompressedBasis.reduce` (packed storage)
with this product. W is tiled by columns. For each tile, the tile's slice of the packed basis is
uploaded once, unpacked on the GPU straight into limbs, and multiplied in int32 into nine
accumulators that persist across row chunks; one fold kernel per tile reduces them mod p. The
basis crosses PCIe once per call. GPU memory stays near
(9 * 4 * m + 3 * rows_chunk) * tile_cols bytes plus the packed chunk.
"""

from __future__ import annotations

import os

import numpy as np

try:  # pragma: no cover
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(os.environ.get("TEMP", "."), "cupy_cache"))
    import cupy as cp
    from cupy.cuda import cublas

    HAVE_CUPY = True
except Exception:  # pragma: no cover
    HAVE_CUPY = False

LIMB_BITS = 7
K_EXACT = (2 ** 31 - 1) // (127 * 127)  # 133144: int32 sums of int8 limb products stay exact
_CUDA_R_8I, _CUDA_R_32I = 3, 10

_UNPACK_SRC = r"""
extern "C" __global__
void unpack_limbs_T(const unsigned char* data, int rows, int pairs, int ncols, int L,
                    int ldk, int ldq, signed char* out) {
    // data: rows x pairs x 5 bytes (two 20-bit residues per 40-bit word, little-endian).
    // out[l][c][r] (l < L, c < ldq, r < ldk): limb l of the residue in row r, column c.
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * pairs;
    if (t >= total) return;
    int r = (int)(t / pairs), j = (int)(t % pairs);
    const unsigned char* b = data + ((long long)r * pairs + j) * 5;
    unsigned long long w = (unsigned long long)b[0] | ((unsigned long long)b[1] << 8)
        | ((unsigned long long)b[2] << 16) | ((unsigned long long)b[3] << 24)
        | ((unsigned long long)b[4] << 32);
    for (int h = 0; h < 2; h++) {
        int c = 2 * j + h;
        if (c >= ncols) break;
        unsigned int v = (unsigned int)((h ? (w >> 20) : w) & 0xFFFFFull);
        for (int l = 0; l < L; l++) {
            out[((long long)l * ldq + c) * ldk + r] = (signed char)((v >> (7 * l)) & 127u);
        }
    }
}
"""

_FOLD_SRC = r"""
extern "C" __global__
void fold_mod(const int* C, int L, long long plane, const long long* cw, long long p,
              long long* out, long long n) {
    // out[x] = sum_{i,j} cw[i + j] * C[(i * L + j) * plane + x] mod p, for x < n.
    long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    long long s = 0;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++) {
            s += cw[i + j] * (long long)C[(long long)(i * L + j) * plane + x];
            s %= p;
        }
    out[x] = s;
}
"""


def available() -> bool:
    if not HAVE_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _pad(x: int, a: int = 16) -> int:
    return (x + a - 1) // a * a


class GPUReducer:
    def __init__(self, p: int, tile_cols: int = 16384, rows_chunk: int = 8192):
        assert HAVE_CUPY, "cupy is not installed"
        assert p < 2 ** (3 * LIMB_BITS), "three 7-bit limbs hold residues below 2^21"
        self.p = p
        self.L = max(1, -(-int(p - 1).bit_length() // LIMB_BITS))
        self.tile_cols = _pad(tile_cols)
        self.rows_chunk = _pad(min(rows_chunk, K_EXACT))
        self.cw = cp.asarray([pow(2, LIMB_BITS * t, p) for t in range(2 * self.L - 1)], dtype=cp.int64)
        self._unpack = cp.RawKernel(_UNPACK_SRC, "unpack_limbs_T")
        self._fold = cp.RawKernel(_FOLD_SRC, "fold_mod")
        self._handle = cp.cuda.device.get_cublas_handle()
        self._one = np.array(1, np.int32)

    # ---------------------------------------------------------------- limbs on the host
    def _limbs_host(self, M: np.ndarray, rows_pad: int, cols_pad: int) -> "cp.ndarray":
        """int8 limbs [L, rows_pad, cols_pad] of the residue matrix M (zero padded)."""
        Mi = np.zeros((rows_pad, cols_pad), dtype=np.int32)
        Mi[:M.shape[0], :M.shape[1]] = M
        out = np.empty((self.L, rows_pad, cols_pad), dtype=np.int8)
        for l in range(self.L):
            out[l] = (Mi >> (LIMB_BITS * l)) & 127
        return cp.asarray(out)

    def _gemm_acc(self, A_ptr: int, lda: int, Bt_ptr: int, ldb: int, C_ptr: int, m: int, n: int, k: int):
        """C (m x n, int32, row-major) += A (m x k) @ Bt.T (Bt is n x k); row-major int8 with leading
        dimensions lda, ldb (the "TN" layout of cuBLAS's int8 kernels)."""
        cublas.gemmEx(self._handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, n, m, k,
                      self._one.ctypes.data, Bt_ptr, _CUDA_R_8I, ldb, A_ptr, _CUDA_R_8I, lda,
                      self._one.ctypes.data, C_ptr, _CUDA_R_32I, n, 72, -1)

    # ---------------------------------------------------------------- the reduction
    def reduce(self, basis, V: np.ndarray) -> np.ndarray:
        """Exactly CompressedBasis.reduce(V) for a packed basis, with the products on the GPU."""
        p, L = self.p, self.L
        W = np.ascontiguousarray(V[:, basis.Q])
        if not basis.rank:
            return W
        # any block with the packed layout (mathaccel's or the reference PackedBlock)
        assert all(getattr(getattr(C, "data", None), "ndim", 0) == 3 and C.data.shape[2] == 5
                   for C in basis.blocks), "GPU path needs packed blocks"
        m, q, rank = W.shape[0], W.shape[1], basis.rank
        assert rank <= K_EXACT, "fold more often for ranks above 133144 (not needed below)"
        m_pad, rank_pad = _pad(m), _pad(rank)
        A = self._limbs_host(V[:, basis.P], m_pad, rank_pad)  # [L, m_pad, rank_pad]
        # row index of each block's first row, in P order
        starts = np.cumsum([0] + [C.rows for C in basis.blocks])
        kc = self.rows_chunk
        tile = self.tile_cols
        result = np.empty_like(W)
        for c0 in range(0, q, tile):
            c1 = min(q, c0 + tile)
            if c0 % 2:
                raise AssertionError("tiles must start at an even column (pairs of residues)")
            j0, j1 = c0 // 2, (c1 + 1) // 2
            ncols = c1 - c0
            qt_pad = _pad(ncols)
            Cacc = cp.zeros((L * L, m_pad, qt_pad), dtype=cp.int32)
            limbs = cp.zeros((L, qt_pad, kc), dtype=cp.int8)
            for s in range(0, rank, kc):
                e = min(rank, s + kc)
                # host: rows s:e of the basis, columns c0:c1, as packed bytes (rows x pairs x 5)
                parts = []
                for b, C in enumerate(basis.blocks):
                    bs, be = starts[b], starts[b + 1]
                    lo, hi = max(s, bs), min(e, be)
                    if lo < hi:
                        parts.append(C.data[lo - bs:hi - bs, j0:j1])
                chunk = np.ascontiguousarray(parts[0] if len(parts) == 1 else np.concatenate(parts))
                d = cp.asarray(chunk)
                limbs.fill(0)
                rows = e - s
                total = rows * (j1 - j0)
                self._unpack(((total + 255) // 256,), (256,),
                             (d, np.int32(rows), np.int32(j1 - j0), np.int32(ncols), np.int32(L),
                              np.int32(kc), np.int32(qt_pad), limbs))
                k = _pad(rows, 16)
                for i in range(L):
                    for j in range(L):
                        A_ptr = A[i].data.ptr + s  # column offset s in a row-major m_pad x rank_pad
                        self._gemm_acc(A_ptr, rank_pad, limbs[j].data.ptr, kc,
                                       Cacc[i * L + j].data.ptr, m_pad, qt_pad, k)
                del d
            out = cp.empty((m_pad, qt_pad), dtype=cp.int64)
            n = m_pad * qt_pad
            self._fold(((n + 255) // 256,), (256,),
                       (Cacc, np.int32(L), np.int64(m_pad * qt_pad), self.cw, np.int64(p), out, np.int64(n)))
            prod = cp.asnumpy(out[:m, :ncols]).astype(np.float64)
            result[:, c0:c1] = np.mod(W[:, c0:c1] - prod, p)
            del Cacc, limbs, out
        return result
