"""GPU throughput probe: int8 gemmEx with different algorithms and shapes, and float32/float16 baselines.
Usage: python bench/probe_gpu_throughput.py
"""
import time

import cupy as cp
import numpy as np
from cupy.cuda import cublas

CUDA_R_8I, CUDA_R_32I = 3, 10
COMPUTE_32I = getattr(cublas, "CUBLAS_COMPUTE_32I", 72)


def int8_tn(A, Bt, C, algo):
    m, k = A.shape
    n = Bt.shape[0]
    h = cp.cuda.device.get_cublas_handle()
    one, zero = np.array(1, np.int32), np.array(0, np.int32)
    cublas.gemmEx(h, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, n, m, k, one.ctypes.data, Bt.data.ptr,
                  CUDA_R_8I, k, A.data.ptr, CUDA_R_8I, k, zero.ctypes.data, C.data.ptr, CUDA_R_32I, n,
                  COMPUTE_32I, algo)


def timeit(f, reps=10):
    f()
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / reps


rng = np.random.default_rng(1)
for (m, k, n) in [(512, 4096, 16384), (2048, 4096, 16384), (4096, 4096, 8192)]:
    A = cp.asarray(rng.integers(0, 128, (m, k), dtype=np.int8))
    Bt = cp.asarray(rng.integers(0, 128, (n, k), dtype=np.int8))
    C = cp.empty((m, n), dtype=cp.int32)
    for name, algo in [("default", -1), ("tensor_op", 99)]:
        dt = timeit(lambda: int8_tn(A, Bt, C, algo))
        print(f"int8 {name:9s} m={m} k={k} n={n}: {dt * 1e3:7.2f} ms {2 * m * k * n / dt / 1e12:5.1f} TOPS")
    Af, Bf = A.astype(cp.float32), Bt.T.astype(cp.float32)
    dt = timeit(lambda: Af @ Bf)
    print(f"fp32           m={m} k={k} n={n}: {dt * 1e3:7.2f} ms {2 * m * k * n / dt / 1e12:5.1f} TFLOPS")
    Ah, Bh = A.astype(cp.float16), Bt.T.astype(cp.float16)
    dt = timeit(lambda: Ah @ Bh)
    print(f"fp16           m={m} k={k} n={n}: {dt * 1e3:7.2f} ms {2 * m * k * n / dt / 1e12:5.1f} TFLOPS")
    del A, Bt, C, Af, Bf, Ah, Bh
    cp.get_default_memory_pool().free_all_blocks()
