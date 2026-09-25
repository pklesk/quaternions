r"""
Module for timing comparison of qmatmul's CUDA-based backends: 
direct vs. proposed-algorithm, each via numba-cuda and via CuPy (cuBLAS-backed). Runs a
single fixed-size quaternion matrix-matrix product (configurable via the module-level
constants below; defaults to the dense-layer dimensions from the QCNN example) with
a warm-up call per approach followed by repeated, timed runs, reporting the median.

Link to project repository
--------------------------
`https://github.com/pklesk/quaternions <https://github.com/pklesk/quaternions>`_
"""

__author__ = ["Przemysław Klęsk", "Aleksandr Cariow"]
__email__ = ["pklesk@zut.edu.pl", "alexandr.tariov@zut.edu.pl"]

import numpy as np
import time
import qmatmul as qmm
import qmatmul_cupy as qmmcp

LINE_SEPARATOR = 256 * "="

M, N, P = 100, 100000, 100
SEED = 3
RANGE = 2.0 # each random real/imaginary part drawn uniformly from (-RANGE, RANGE)      
DTYPE = np.float64
REPETITIONS = 11

np.random.seed(SEED)

if __name__ == "__main__":
    print(f"QUATERNIONS -> CUPY EXAMPLE... [M: {M}, N: {N}, P: {P}, dtype: {DTYPE}, repetitions: {REPETITIONS}]")
    print(LINE_SEPARATOR)
        
    A = qmm.qmatrand(M, N, -RANGE, RANGE, dtype=DTYPE)
    B = qmm.qmatrand(N, P, -RANGE, RANGE, dtype=DTYPE)
    
    approaches = {
        "direct_numba_cuda": lambda: qmm.dot(A, B, approach="direct_numba_cuda"),
        "algo_numba_cuda": lambda: qmm.dot(A, B, approach="algo_numba_cuda"),
        "direct_cupy": lambda: qmmcp.qmatmul_direct_cupy(A, B),
        "algo_cupy": lambda: qmmcp.qmatmul_algo_cupy(A, B),
    }
        
    # warm-up: one small-problem call per approach, BEFORE any timing
    print("WARM-UPS...")
    qmm.dot(A, B, approach="direct_numba_cuda")
    qmm.dot(A, B, approach="algo_numba_cuda")
    qmmcp.qmatmul_direct_cupy(A, B)
    qmmcp.qmatmul_algo_cupy(A, B)
    print("WARM-UPS DONE.")
    
    # timing: several repetitions, report median
    print(f"QMATMUL TIMINGS...")
    n_reps = REPETITIONS
    for name, fn in approaches.items():
        times = []
        for _ in range(n_reps):
            t1 = time.time()
            fn()
            t2 = time.time()
            times.append(t2 - t1)
        times.sort()
        median = times[n_reps // 2]
        print(f"{name} -> median: {median * 1000:.3f} ms  (min: {min(times) * 1000:.3f} ms, max: {max(times) * 1000:.3f} ms)")
    print("QMATMUL TIMINGS DONE.")
    
    print(LINE_SEPARATOR)
    print("QUATERNIONS -> CUPY EXAMPLE DONE.")
    