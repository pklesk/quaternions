r"""
Module providing `direct_cupy` / `algo_cupy` computational approaches for
quaternion matrix-matrix multiplication. It is meant for additional research / efficiency tests
on environments where CuPy is available (especially with tensor cores support).  
Kept purposely separate from `qmatmul.py`, but containing twin-like counterparts of functions present there.  
"""

import time
import cupy as cp
import qmatmul as qmm

A_BLOCKS_SIGNS = qmm.A_BLOCKS_SIGNS
A_BLOCKS_PARTS = qmm.A_BLOCKS_PARTS

def stack_cupy(E):
    """(for internal use) cupy counterpart of `qmatmul.stack`: (R, S, 4) device array -> stacked (4*R, S) device array."""
    R, S, _ = E.shape
    E4 = cp.empty((4 * R, S), dtype=E.dtype)
    E4[:R] = E[:, :, 0]
    E4[R:2 * R] = E[:, :, 1]
    E4[2 * R:3 * R] = E[:, :, 2]
    E4[3 * R:] = E[:, :, 3]
    return E4

def a44_cupy(A, a_blocks_signs=A_BLOCKS_SIGNS, a_blocks_parts=A_BLOCKS_PARTS):
    """(for internal use) cupy counterpart of `qmatmul.a44`: (M, N, 4) device array -> transformation (4*M, 4*N) device array.
    Reuses the A_BLOCKS_SIGNS / A_BLOCKS_PARTS constants imported from qmatmul (plain Python-int indexing into
    them is fine here, since the 4x4 outer loop runs on the host)."""
    M, N, _ = A.shape
    A44 = cp.empty((4 * M, 4 * N), dtype=A.dtype)
    for i in range(4):
        iM = i * M
        for j in range(4):
            jN = j * N
            A44[iM:iM + M, jN:jN + N] = int(a_blocks_signs[i, j]) * A[:, :, int(a_blocks_parts[i, j])]
    return A44

def had4_cupy(E4):
    """(for internal use) cupy counterpart of `qmatmul.had4`: Hadamard transform of a stacked (4 * R, S) device array."""
    R4 = E4.shape[0]
    R2 = R4 >> 1
    R = R2 >> 1
    R3 = R2 + R
    E4_s0 = E4[:R] + E4[R:R2]
    E4_s1 = E4[R2:R3] + E4[R3:]
    E4_d0 = E4[:R] - E4[R:R2]
    E4_d1 = E4[R2:R3] - E4[R3:]
    H4E4 = cp.empty_like(E4)
    H4E4[:R] = E4_s0 + E4_s1
    H4E4[R:R2] = E4_d0 + E4_d1
    H4E4[R2:R3] = E4_s0 - E4_s1
    H4E4[R3:] = E4_d0 - E4_d1
    return H4E4

def matmuldiag_cupy(E4, F4, factor):
    """(for internal use) cupy counterpart of `qmatmul.matmuldiag`: diagonal product of two stacked device arrays,
    computed as four independent real-valued GEMMs (one per diagonal block), each dispatched to cuBLAS via cupy's `.dot`."""
    R4, S = E4.shape
    R2 = R4 >> 1
    R = R2 >> 1
    R3 = R2 + R
    S2 = S << 1
    S3 = S2 + S
    T = F4.shape[1]
    D4 = cp.empty((R4, T), dtype=E4.dtype)
    D4[:R] = factor * (E4[:R].dot(F4[:S]))
    D4[R:R2] = factor * (E4[R:R2].dot(F4[S:S2]))
    D4[R2:R3] = factor * (E4[R2:R3].dot(F4[S2:S3]))
    D4[R3:] = factor * (E4[R3:].dot(F4[S3:]))
    return D4

def permute_cupy(E4, permutation):
    """(for internal use) cupy counterpart of `qmatmul.permute`: block-wise permutation of a stacked (4*R, S) device array.
    `permutation` is a plain length-4 sequence of Python ints (no need to move it to device, since the
    loop below runs on the host and only issues device-to-device slice copies)."""
    R4 = E4.shape[0]
    R = R4 >> 2
    E4p = cp.empty_like(E4)
    for i in range(4):
        p = permutation[i]
        E4p[i * R:(i + 1) * R] = E4[p * R:(p + 1) * R]
    return E4p

def c4_to_c_cupy(C4):
    """(for internal use) cupy counterpart of `qmatmul.c4_to_c`: (4*M, P) device array -> unstacked (M, P, 4) device array."""
    M4, P = C4.shape
    M = M4 >> 2
    M2 = M << 1
    M3 = M2 + M
    C = cp.empty((M, P, 4), dtype=C4.dtype)
    C[:, :, 0] = C4[:M]
    C[:, :, 1] = C4[M:M2]
    C[:, :, 2] = C4[M2:M3]
    C[:, :, 3] = C4[M3:]
    return C

def qmatmul_direct_cupy(A, B, verbose=False):
    """Returns the quaternion matrix product of A and B via the "direct_cupy" approach, for input matrices of type
    either ``float64`` or ``float32``. Implements formula (15) as a single real-valued GEMM of the (4M x 4N) transformation matrix against the (4N x P) stacked
    matrix, dispatched to cuBLAS via cupy's ``.dot``."""
    if verbose:
        print(f"QMATMUL_DIRECT_CUPY...")
        t1 = time.time()
    A_dev = cp.asarray(A)
    B_dev = cp.asarray(B)
    C4_dev = a44_cupy(A_dev).dot(stack_cupy(B_dev))
    C_dev = c4_to_c_cupy(C4_dev)
    cp.cuda.Device().synchronize()
    C = cp.asnumpy(C_dev)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_DIRECT_CUPY DONE. [time: {t2 - t1} s]")
    return C


def qmatmul_algo_cupy(A, B, verbose=False):
    """Returns the quaternion matrix product of A and B via the "algo_cupy" approach, for input matrices of type either ``float64`` or ``float32``."""
    if verbose:
        print(f"QMATMUL_ALGO_CUPY...")
        t1 = time.time()
    M = A.shape[0]
    A_dev = cp.asarray(A)
    B_dev = cp.asarray(B)
    B4 = stack_cupy(B_dev)
    A4 = stack_cupy(A_dev)
    H4A4 = had4_cupy(A4)
    H4B4 = had4_cupy(B4)
    D4u = matmuldiag_cupy(H4A4, H4B4, 0.25)
    H4D4u = had4_cupy(D4u)
    A4p = permute_cupy(A4, (0, 3, 1, 2))
    B4p = permute_cupy(B4, (0, 2, 3, 1))
    D4l = matmuldiag_cupy(A4p, B4p, 2.0)
    C4 = H4D4u - D4l
    C4[:M] = -C4[:M]
    C_dev = c4_to_c_cupy(C4)
    cp.cuda.Device().synchronize()
    C = cp.asnumpy(C_dev)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_ALGO_CUPY DONE. [time: {t2 - t1} s]")
    return C