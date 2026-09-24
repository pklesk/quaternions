"""
Numerical accuracy experiment for the qmatmul package.

Measures the relative error of float32 computations against a float64 reference,
for both the direct (definition-based) and algo (proposed fast algorithm)
approaches, CUDA backend. A single deterministic run per problem size is used
(no repetitions or warm-up, as this is an accuracy study, not a timing study).

Primary metrics: relative_frobenius_error (global accuracy), and max_abs_error
together with max_combined_error (worst-case accuracy; max_combined_error uses
the same atol/rtol tolerance already used for the np.allclose correctness checks
in main.py). max_relative_error, rms_error and min_abs_ref are auxiliary metrics
characterizing the element-wise error distribution.

Link to project repository
--------------------------
https://github.com/pklesk/quaternions
"""

__author__ = ["Przemysław Klęsk", "Aleksandr Cariow"]
__email__ = ["pklesk@zut.edu.pl", "alexandr.tariov@zut.edu.pl"]

import numpy as np
from qmatmul import (
    qmatrand,
    qmatmul_direct_numpy_st,
    qmatmul_direct_numba_cuda_float32,
    qmatmul_direct_numba_cuda_float64,
    qmatmul_algo_numba_cuda_float32,
    qmatmul_algo_numba_cuda_float64,
)

# experiment settings
SEED = 1    # intentionally different from SEED = 0 used in main.py
RANGE = 2.0   # same value range as the main efficiency experiments
ATOL = 1e-2   # same atol already used for np.allclose(rtol=1e-5, atol=1e-2) in main.py
RTOL = 1e-5   # same rtol already used for np.allclose(rtol=1e-5, atol=1e-2) in main.py
SIZES = [
    (100, 100, 100),
    (100, 300, 200),
    (300, 300, 300),
    (1000, 1000, 1000),
    (1000, 3000, 2000),
    (3000, 3000, 3000),
]

LINE_SEPARATOR = 256 * "="

def relative_frobenius_error(C_approx, C_ref_f64):
    """Relative error in Frobenius norm: ||C_approx - C_ref|| / ||C_ref||."""
    diff = C_approx.astype(np.float64) - C_ref_f64
    return np.linalg.norm(diff) / np.linalg.norm(C_ref_f64)

def max_relative_error(C_approx, C_ref_f64, eps=1e-15):
    """Element-wise relative error: max_i |approx_i - ref_i| / (|ref_i| + eps)."""
    diff = np.abs(C_approx.astype(np.float64) - C_ref_f64)
    denom = np.abs(C_ref_f64) + eps
    return np.max(diff / denom)

def rms_error(C_approx, C_ref_f64):
    """Root-mean-square absolute error: ||C_approx - C_ref||_F / sqrt(n)."""
    diff = C_approx.astype(np.float64) - C_ref_f64
    return np.sqrt(np.mean(diff ** 2))

def min_abs_ref(C_ref_f64):
    """Smallest |reference value| present in this run."""
    return np.min(np.abs(C_ref_f64))

def max_abs_error(C_approx, C_ref_f64):
    """Maximum absolute error: max_i |approx_i - ref_i|."""
    diff = np.abs(C_approx.astype(np.float64) - C_ref_f64)
    return np.max(diff)

def max_combined_error(C_approx, C_ref_f64, atol=ATOL, rtol=RTOL):
    """Maximum elementwise error relative to the np.allclose tolerance boundary:
    max_i |approx_i - ref_i| / (atol + rtol * |ref_i|). A value <= 1 means every
    element would pass np.allclose(rtol=rtol, atol=atol)."""
    diff = np.abs(C_approx.astype(np.float64) - C_ref_f64)
    denom = atol + rtol * np.abs(C_ref_f64)
    return np.max(diff / denom)


if __name__ == "__main__":
    print(f"NUMERICAL ACCURACY EXPERIMENT [seed: {SEED}, range: {RANGE}, atol: {ATOL}, rtol: {RTOL}]...")
    print(LINE_SEPARATOR)

    results = []

    for (M, N, P) in SIZES:
        print(f"M: {M}, N: {N}, P: {P} (M * N * P = {M * N * P:.1e})")

        np.random.seed(SEED)
        A64 = qmatrand(M, N, -RANGE, RANGE, dtype=np.float64)
        B64 = qmatrand(N, P, -RANGE, RANGE, dtype=np.float64)
        A32 = A64.astype(np.float32)
        B32 = B64.astype(np.float32)

        # ground-truth reference: direct formula, float64, numpy
        C_ref = qmatmul_direct_numpy_st(A64, B64).astype(np.float64)

        # float32 runs (CUDA backend) - the ones of interest
        C_direct_f32 = qmatmul_direct_numba_cuda_float32(A32, B32)
        C_algo_f32 = qmatmul_algo_numba_cuda_float32(A32, B32)

        # float64 CUDA runs - sanity check (should be close to machine epsilon)
        C_direct_f64 = qmatmul_direct_numba_cuda_float64(A64, B64)
        C_algo_f64 = qmatmul_algo_numba_cuda_float64(A64, B64)

        row = {
            "M": M, "N": N, "P": P,
            "MIN_ABS_REF": min_abs_ref(C_ref),

            "DIRECT_F32_REL_FROB": relative_frobenius_error(C_direct_f32, C_ref),
            "DIRECT_F32_RMS": rms_error(C_direct_f32, C_ref),
            "DIRECT_F32_MAXABS": max_abs_error(C_direct_f32, C_ref),
            "DIRECT_F32_MAXCOMB": max_combined_error(C_direct_f32, C_ref),
            "DIRECT_F32_REL_MAX_ABS": max_relative_error(C_direct_f32, C_ref),

            "ALGO_F32_REL_FROB": relative_frobenius_error(C_algo_f32, C_ref),
            "ALGO_F32_RMS": rms_error(C_algo_f32, C_ref),
            "ALGO_F32_MAXABS": max_abs_error(C_algo_f32, C_ref),
            "ALGO_F32_MAXCOMB": max_combined_error(C_algo_f32, C_ref),
            "ALGO_F32_REL_MAX_ABS": max_relative_error(C_algo_f32, C_ref),

            "DIRECT_F64_REL_FROB": relative_frobenius_error(C_direct_f64, C_ref),
            "ALGO_F64_REL_FROB": relative_frobenius_error(C_algo_f64, C_ref),
        }
        results.append(row)

        print(f"MIN |C_REF| (THIS RUN): {row['MIN_ABS_REF']:.3e}")
        print(f"DIRECT, FLOAT32 -> REL_FROB: {row['DIRECT_F32_REL_FROB']:.3e}, "
              f"RMS: {row['DIRECT_F32_RMS']:.3e}, "
              f"MAX_ABS: {row['DIRECT_F32_MAXABS']:.3e}, "
              f"MAX_COMBINED: {row['DIRECT_F32_MAXCOMB']:.3e}, "
              f"REL_MAX_ABS [naive]: {row['DIRECT_F32_REL_MAX_ABS']:.3e}")
        print(f"ALGO,   FLOAT32 -> REL_FROB: {row['ALGO_F32_REL_FROB']:.3e}, "
              f"RMS: {row['ALGO_F32_RMS']:.3e}, "
              f"MAX_ABS: {row['ALGO_F32_MAXABS']:.3e}, "
              f"MAX_COMBINED: {row['ALGO_F32_MAXCOMB']:.3e}, "
              f"REL_MAX_ABS [naive]: {row['ALGO_F32_REL_MAX_ABS']:.3e}")
        print(f"DIRECT, FLOAT64 -> REL_FROB: {row['DIRECT_F64_REL_FROB']:.3e} (SANITY CHECK)")
        print(f"ALGO,   FLOAT64 -> REL_FROB: {row['ALGO_F64_REL_FROB']:.3e} (SANITY CHECK)")
        print(LINE_SEPARATOR)

    print("SUMMARY - RELATIVE FROBENIUS ERROR:")
    print(f"{'M,N,P':<20}{'DIRECT_F32_REL_FROB':>28}{'ALGO_F32_REL_FROB':>28}{'DIRECT_F64_REL_FROB':>28}{'ALGO_F64_REL_FROB':>28}")
    for row in results:
        size_str = f"{row['M']}, {row['N']}, {row['P']}"
        print(f"{size_str:<20}{row['DIRECT_F32_REL_FROB']:>28.3e}{row['ALGO_F32_REL_FROB']:>28.3e}{row['DIRECT_F64_REL_FROB']:>28.3e}{row['ALGO_F64_REL_FROB']:>28.3e}")

    print()
    print("SUMMARY - WORST-CASE ERROR (FLOAT32):")
    print(f"{'M,N,P':<20}{'DIRECT_F32_MAXABS':>28}{'ALGO_F32_MAXABS':>28}{'DIRECT_F32_MAXCOMB':>28}{'ALGO_F32_MAXCOMB':>28}")
    for row in results:
        size_str = f"{row['M']}, {row['N']}, {row['P']}"
        print(f"{size_str:<20}{row['DIRECT_F32_MAXABS']:>28.3e}{row['ALGO_F32_MAXABS']:>28.3e}{row['DIRECT_F32_MAXCOMB']:>28.3e}{row['ALGO_F32_MAXCOMB']:>28.3e}")

    print()
    print("SUMMARY - AUXILIARY (ELEMENT-WISE RELATIVE ERROR MECHANISM):")
    print(f"{'M,N,P':<20}{'MIN_ABS_REF':>28}{'DIRECT_F32_REL_MAX_ABS':>28}{'ALGO_F32_REL_MAX_ABS':>28}")
    for row in results:
        size_str = f"{row['M']}, {row['N']}, {row['P']}"
        print(f"{size_str:<20}{row['MIN_ABS_REF']:>28.3e}{row['DIRECT_F32_REL_MAX_ABS']:>28.3e}{row['ALGO_F32_REL_MAX_ABS']:>28.3e}")

    print(LINE_SEPARATOR)
    print(f"NUMERICAL ACCURACY EXPERIMENT DONE.")
    