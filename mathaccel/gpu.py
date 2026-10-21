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
accumulators that persist across row chunks; one fold kernel per tile reduces them mod p,
gathers V on the non-pivot columns, and subtracts the product. The
basis crosses PCIe once per call. GPU memory stays near
(9 * 4 * m + 3 * rows_chunk) * tile_cols bytes plus two packed chunks,
the probe V (float64), its coefficient limbs, and the current output tile.
Two pinned host buffers and a separate upload stream overlap staging and transfers
with the preceding chunk's unpack and GEMMs. Events protect both host and device
buffers from reuse while a transfer or an unpack is still reading them.
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
              const double* V, long long vcols, const long long* Q, int c0,
              int ncols, int ldq, double* out, long long n) {
    long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    long long row = x / ncols, col = x % ncols;
    long long offset = row * ldq + col;
    long long s = 0;
    // L <= 3, cw < p < 2^21 and C <= 2^31-1, so the entire
    // nine-term sum is below 2^56. One int64 modulo suffices.
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            s += cw[i + j] * (long long)C[(long long)(i * L + j) * plane + offset];
    long long v = (long long)V[row * vcols + Q[c0 + col]];
    long long residue = v - s % p;
    out[x] = (double)(residue < 0 ? residue + p : residue);
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
        self._ready = cp.cuda.Event(disable_timing=True)
        self._ready.record()

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
        m, q, rank = V.shape[0], len(basis.Q), basis.rank
        if not rank or not m or not q:
            return np.ascontiguousarray(V[:, basis.Q])
        # any block with the packed layout (mathaccel's or the reference PackedBlock)
        assert all(getattr(getattr(C, "data", None), "ndim", 0) == 3 and C.data.shape[2] == 5
                   for C in basis.blocks), "GPU path needs packed blocks"
        assert rank <= K_EXACT, "fold more often for ranks above 133144 (not needed below)"
        m_pad, rank_pad = _pad(m), _pad(rank)
        A = self._limbs_host(V[:, basis.P], m_pad, rank_pad)  # [L, m_pad, rank_pad]
        # row index of each block's first row, in P order
        starts = np.cumsum([0] + [C.rows for C in basis.blocks])
        kc = self.rows_chunk
        tile = self.tile_cols
        result = np.empty((m, q), dtype=V.dtype)
        # Fold gathers Q and subtracts on device, avoiding a full host gather,
        # int64-to-float64 conversion, and NumPy modulo of every output entry.
        dV = cp.asarray(V, dtype=cp.float64, order="C")
        dQ = cp.asarray(basis.Q, dtype=cp.int64)
        compute = cp.cuda.get_current_stream()
        compute.wait_event(self._ready)
        upload = cp.cuda.Stream(non_blocking=True)
        # Bind custom cuBLAS calls to the caller's stream as well as RawKernels.
        previous_stream = cublas.getStream(self._handle)
        # Retain pinned allocations until all async readers have completed. Two
        # slots suffice: host staging/upload of chunk N can overlap GEMMs N-1.
        slots = min(2, (rank + kc - 1) // kc)
        capacity = min(rank, kc) * ((min(q, tile) + 1) // 2) * 5
        pinned = [cp.cuda.alloc_pinned_memory(capacity) for _ in range(slots)]
        host = [np.frombuffer(buf, dtype=np.uint8) for buf in pinned]
        device = [cp.empty(capacity, dtype=cp.uint8) for _ in range(slots)]
        uploaded = [cp.cuda.Event(disable_timing=True) for _ in range(slots)]
        consumed = [cp.cuda.Event(disable_timing=True) for _ in range(slots)]
        used = [False] * slots
        # Device allocations may reuse storage previously touched on compute.
        allocated = cp.cuda.Event(disable_timing=True)
        allocated.record(compute)
        upload.wait_event(allocated)
        try:
            cublas.setStream(self._handle, compute.ptr)
            for c0 in range(0, q, tile):
                c1 = min(q, c0 + tile)
                j0, j1 = c0 // 2, (c1 + 1) // 2
                ncols = c1 - c0
                qt_pad = _pad(ncols)
                Cacc = cp.zeros((L * L, m_pad, qt_pad), dtype=cp.int32)
                limbs = cp.zeros((L, qt_pad, kc), dtype=cp.int8)
                for chunk_index, s in enumerate(range(0, rank, kc)):
                    e = min(rank, s + kc)
                    rows, pairs = e - s, j1 - j0
                    slot = chunk_index % slots
                    if used[slot]:
                        # Host writes must wait for the previous DMA read.
                        uploaded[slot].synchronize()
                    size = rows * pairs * 5
                    chunk = host[slot][:size].reshape(rows, pairs, 5)
                    # Copy directly from strided blocks into pinned storage;
                    # no concatenate temporary or pageable staging allocation.
                    offset = 0
                    for b, C in enumerate(basis.blocks):
                        bs, be = starts[b], starts[b + 1]
                        lo, hi = max(s, bs), min(e, be)
                        if lo < hi:
                            count = hi - lo
                            chunk[offset:offset + count] = C.data[lo - bs:hi - bs, j0:j1]
                            offset += count
                    d = device[slot][:size].reshape(rows, pairs, 5)
                    with upload:
                        if used[slot]:
                            # Device writes must wait for the previous unpack.
                            upload.wait_event(consumed[slot])
                        d.set(chunk, stream=upload)
                        uploaded[slot].record(upload)
                    used[slot] = True
                    compute.wait_event(uploaded[slot])
                    limbs.fill(0)
                    total = rows * pairs
                    self._unpack(((total + 255) // 256,), (256,),
                                 (d, np.int32(rows), np.int32(pairs), np.int32(ncols), np.int32(L),
                                  np.int32(kc), np.int32(qt_pad), limbs))
                    consumed[slot].record(compute)
                    k = _pad(rows, 16)
                    for i in range(L):
                        for j in range(L):
                            A_ptr = A[i].data.ptr + s
                            self._gemm_acc(A_ptr, rank_pad, limbs[j].data.ptr, kc,
                                           Cacc[i * L + j].data.ptr, m_pad, qt_pad, k)
                out = cp.empty((m, ncols), dtype=cp.float64)
                n = m * ncols
                self._fold(((n + 255) // 256,), (256,),
                           (Cacc, np.int32(L), np.int64(m_pad * qt_pad), self.cw, np.int64(p),
                            dV, np.int64(V.shape[1]), dQ, np.int32(c0), np.int32(ncols),
                            np.int32(qt_pad), out, np.int64(n)))
                result[:, c0:c1] = cp.asnumpy(out)
                del Cacc, limbs, out
            return result
        finally:
            # Also on exceptions, don't release pinned/device buffers while an
            # upload or unpack still uses them. Stream order covers all GEMMs.
            upload.synchronize()
            compute.synchronize()
            cublas.setStream(self._handle, previous_stream)
