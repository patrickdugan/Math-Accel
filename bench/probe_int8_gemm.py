"""Probe cuBLAS int8 GEMM (int8 x int8 -> int32 on tensor cores) through CuPy: correctness and speed.

C (m x n, int32, row-major) = A (m x k, int8) @ B (k x n, int8), computed as the column-major
product C^T = B^T A^T. With "TN" the second factor is passed transposed, so B is stored as
Bt (n x k, row-major), the layout cuBLAS's IMMA kernels prefer.
Usage: python bench/probe_int8_gemm.py
"""
import time

import cupy as cp
import numpy as np
from cupy.cuda import cublas, runtime

CUDA_R_8I, CUDA_R_32I = 3, 10
COMPUTE_32I = getattr(cublas, "CUBLAS_COMPUTE_32I", 72)
ALGO = getattr(cublas, "CUBLAS_GEMM_DEFAULT_TENSOR_OP", 99)


def int8_matmul_tn(A: cp.ndarray, Bt: cp.ndarray, C: cp.ndarray) -> cp.ndarray:
    """C = A @ Bt.T, with A (m x k) and Bt (n x k) int8 row-major and C (m x n) int32."""
    m, k = A.shape
    n = Bt.shape[0]
    handle = cp.cuda.device.get_cublas_handle()
    one, zero = np.array(1, np.int32), np.array(0, np.int32)
    # column-major view: C^T (n x m) = op(Bt^T-as-colmajor) ... = Bt (n x k) @ A^T (k x m)
    cublas.gemmEx(handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, n, m, k,
                  one.ctypes.data, Bt.data.ptr, CUDA_R_8I, k, A.data.ptr, CUDA_R_8I, k,
                  zero.ctypes.data, C.data.ptr, CUDA_R_32I, n, COMPUTE_32I, ALGO)
    return C


rng = np.random.default_rng(0)
for (m, k, n) in [(64, 96, 128), (512, 2048, 16384)]:
    A = rng.integers(0, 128, (m, k), dtype=np.int8)
    B = rng.integers(0, 128, (k, n), dtype=np.int8)
    dA, dBt = cp.asarray(A), cp.asarray(np.ascontiguousarray(B.T))
    dC = cp.empty((m, n), dtype=cp.int32)
    int8_matmul_tn(dA, dBt, dC)
    cp.cuda.Device().synchronize()
    ok = np.array_equal(cp.asnumpy(dC), A.astype(np.int64) @ B.astype(np.int64))
    reps = 20 if m > 100 else 3
    t0 = time.perf_counter()
    for _ in range(reps):
        int8_matmul_tn(dA, dBt, dC)
    cp.cuda.Device().synchronize()
    dt = (time.perf_counter() - t0) / reps
    print(f"m={m} k={k} n={n}: exact={ok}  {dt * 1e3:.2f} ms  {2 * m * k * n / dt / 1e12:.1f} TOPS")
