r"""
This module contains eight computational approaches for multiplication of `quaternion`-valued matrices.
The default approach ``qmatmul_algo_numba_cuda`` is based on the fast algorithm proposed by authors (P. Klęsk, A. Cariow). 
The number of elementary floating-point multiplications involved in the algorithm is reduced `twice` 
with respect to the definition-based formula, regardless of the input matrices. This is owed to a 
suitable representation and decomposition into two products, one of which takes advantage of certain 
diagonal symmetry properties, the other of sparsity.

Altogether, eight approaches (implementation variants) are provided. 
They cover several approaches based on NumPy library, thus supported by the underlying BLAS, but 
also several approaches employing Numba - a just-in-time compiler for Python targeting both CPU and GPU (CUDA). 
The design of CUDA computations for the proposed algorithm involves: six kernel functions with eleven invocations, 
suitable usage of tiling and shared memory, and few host-device memory transfers.

Usage example 1
---------------
Suppose one would like to multiply the following matrices of quaternions:

.. math::
    
    {\scriptsize
    \begin{split}
    \begin{pmatrix}
    2i +j & -1 +i -j + 4k & 5 -i +3j \\
    1 +4i -4j -4k & -5 +3i +3j +4k & 5 +3i +3k \\
    -4 +i -4j + 4k & -i -2j +3k & i -5j +k \\
    1 +i +4j + 2k& -1 -i +2j -4k & 2 +2i -3j -4k \\
    -2 -i +j -k & 5 -4i -3j -3k & 2 -2i -3k
    \end{pmatrix}
    & \cdot
    \begin{pmatrix}
    -3 -4i + 2j -4k & -3 -i +3j -4k \\
    3 - 4i +5j & 5 +i +2j -5k\\
    -2 -4i -2j -4k & -2 -i -4j +2k
    \end{pmatrix} \\    
    &=
    \begin{pmatrix}
    4 -53i -39j +15k & 16 -6i -17j +52k \\
    1 -3i +4j +7k & -30 +3i +30j +56k \\
    44 +23i -16j -38k & 40 -4i -5j +7k \\
    -34 -14i +29j -17k & -40 -62i -10j -21k \\
    -14 -18i +21j -26k & 18 -41j -18k
    \end{pmatrix}.
    \end{split}
    }

With ``qmatmul`` module installed, one can write:

.. code-block:: python

    import qmatmul as qmm
    import numpy as np
    import time

    print(f"QMATMUL EXAMPLE...")
    A = np.array([
        [[ 0,  2,  1,  0], [-1,  1, -1,  4], [ 5, -1,  3,  0]],
        [[ 1,  4, -4, -4], [-5,  3,  3,  4], [ 5,  3,  0,  3]],
        [[-4,  1, -4,  4], [ 0, -1, -2, -3], [ 0,  1, -5,  1]],
        [[ 1,  1,  4,  2], [-1, -1,  2, -4], [ 2,  2, -3, -4]],
        [[-2, -1,  1, -1], [ 5, -4, -3, -3], [ 2, -2,  0, -3]]
        ])
    B = np.array([
        [[-3, -4,  2, -4], [-3, -1,  3, -4]], 
        [[ 3, -4,  5,  0], [ 5,  1,  2, -5]],
        [[-2, -4, -2, -4], [-2, -1, -4,  2]]
        ])
    t1 = time.time()
    C = qmm.dot(A, B)
    t2 = time.time()
    print(f"RESULT -> C:")
    print(C)
    print(f"QMATMUL EXAMPLE DONE. TIME OF qmm.dot: {t2 - t1:.6f} s.")

Running the code above produces the following output:

.. code-block:: console

    QMATMUL EXAMPLE...
    RESULT -> C:
    [[[  4. -53. -39.  15.]
      [ 16.  -6. -17.  52.]]
    
     [[  1.  -3.   4.   7.]
      [-30.   3.  30.  56.]]
    
     [[ 44.  53.   8. -56.]
      [ 10.   8. -11. -23.]]
    
     [[-34. -14.  29. -17.]
      [-40. -62. -10. -21.]]
    
     [[-14. -18.  21. -26.]
      [ 18.   0. -41. -18.]]]
    QMATMUL EXAMPLE DONE. TIME OF qmm.dot: 0.001703 s.
    
Usage example 2 (large arguments)
---------------------------------

In the example below, two large random matrices with quaternions are multiplied.

.. code-block:: python

    print("QMATMUL EXAMPLE (LARGE ARGUMENTS)...")
    M, N, P = 1000, 3000, 2000
    np.random.seed(0)
    A = np.random.rand(M, N, 4) # M x N matrix of quaternions
    B = np.random.rand(N, P, 4) # N x P matrix of quaternions
    t1 = time.time()
    C = qmm.dot(A, B)
    t2 = time.time()
    print(f"RESULT FRAGMENT -> C[:3, :3]:")
    print(C[:3, :3])
    print(f"QMATMUL EXAMPLE (LARGE ARGUMENTS) DONE. TIME OF qmm.dot: {t2 - t1:.6f} s.")

The result is computed fast:

.. code-block:: console

    QMATMUL EXAMPLE (LARGE ARGUMENTS)...
    RESULT FRAGMENT -> C[:3, :3]:
    [[[-1528.6768062   1482.01579334  1482.64352966  1469.29588132]
      [-1474.39555984  1485.26884228  1485.81638515  1486.03433938]
      [-1459.21558118  1487.12054919  1468.20437822  1461.65795584]]
    
     [[-1502.50516591  1465.24326325  1494.5907814   1503.74503685]
      [-1472.32677282  1493.41728185  1487.01751106  1506.41597882]
      [-1460.42240679  1488.1077977   1474.34164258  1504.36179871]]
    
     [[-1558.07311493  1475.34714296  1475.731701    1495.41197899]
      [-1528.83559572  1476.28138065  1474.62854662  1504.35168932]
      [-1508.66904595  1501.06439708  1459.07714887  1471.97545486]]]
    QMATMUL EXAMPLE (LARGE ARGUMENTS) DONE. TIME OF qmm.dot: 0.151431 s.
    

Dependencies
------------
- ``numpy``: required for algebraic numerical computations.

- ``numba`` and ``numba-cuda[cu13]``: required for just-in-time compilation of CUDA kernels (decorated by ``@cuda.jit``) and LLVM-targeted functions (decorated by ``@jit``).

- ``threadpoolctl``: required for managing single- vs multi-threaded modes for CPU-based approaches.   

- For usage of ``qmatmul`` modul, NVIDIA CUDA drivers must be present in the operating system.  

Link to project repository
--------------------------
`https://github.com/pklesk/quaternions <https://github.com/pklesk/quaternions>`_ 
"""

from numba import jit, prange, cuda
from numba import void, float32, float64, int8, int32
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
import os
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
import numpy as np
import time
from threadpoolctl import threadpool_limits

__version__ = "1.0.0"
__author__ = ["Przemysław Klęsk", "Aleksandr Cariow"]
__email__ = ["pklesk@zut.edu.pl", "alexandr.tariov@zut.edu.pl"]

# defaults
DEFAULT_TILE_SIZE = 8

# constants
A_BLOCKS_SIGNS = np.array([
    [1, -1,  -1, -1],
    [1,  1, -1,  1],
    [1,  1, 1, -1],
    [1, -1,  1,  1]
], dtype=np.int8)
A_BLOCKS_PARTS = np.array([
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
], dtype=np.int8)

# functions
def dot(A, B, approach="algo_numba_cuda", extra_args={"verbose": False}):   
    """
    Main function of ``qmatmul`` module allowing to multiply two quaternion-valued matrices.
         
    Args:
        A (numpy.ndarray[:, :, :]):
            first factor (matrix of quaternions).
        B (numpy.ndarray[:, :, :]):
            second factor (matrix of quaternions).
        approach (str):
            choice of computational approach from {``"naive_numba_st"``, ``"naive_numba_parallel"``, ``"direct_numpy_st"``, ``"direct_numpy_parallel"``, ``"direct_numba_cuda"``, ``"algo_numpy_st"``, ``"algo_numpy_parallel"``, ``"algo_numba_cudal"``}, defaults to ``"algo_numba_cuda"``.
        extra_args (dict):
            dictionary of extra arguments to be passed to the invoked function with the chosen computational approach.

    Returns:
        numpy.ndarray[:, :, :]: 
            result of A times B multiplication.
            
    Raises:
        ValueError:
            If input matrices are not 3-dimensional, their last dimension is not 4, or if their inner matrix dimensions do not match for multiplication.            
    """     
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError(f"Input matrices must be 3-dimensional (got A.ndim: {A.ndim}, B.ndim: {B.ndim}).")        
    if A.shape[2] != 4 or B.shape[2] != 4:
        raise ValueError(f"The last dimension of quaternion-valued matrices must be exactly 4 (got A.shape[2]: {A.shape[2]}, B.shape[2]: {B.shape[2]}).")        
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible matrix dimensions for multiplication: A columns ({A.shape[1]}) must match B rows ({B.shape[0]}).")    
    if (A.dtype != np.float32 and A.dtype != np.float64) or (A.dtype != B.dtype):
        A = A.astype(np.float64)
    if B.dtype != A.dtype:
        B = B.astype(np.float64)
    dtype_suffix = ""
    if "numba" in approach: 
        dtype_suffix = "_float32" if (A.dtype == np.float32) and (B.dtype == np.float32) else "_float64"  
    approach_function = globals().get("qmatmul_" + approach + dtype_suffix)
    if not approach_function:        
        dtype_suffix = "_float32" if (A.dtype == np.float32) and (B.dtype == np.float32) else "_float64"
        approach_function = globals().get("qmatmul_algo_numba_cuda" + dtype_suffix)
    return approach_function(A, B, **extra_args)

def qmatrand(M, N, range_min, range_max, dtype=np.float32, rounding=False):
    """Helper function that generates a random three-dimensional ``numpy.ndarray`` of shape ``(M, N, 4)``, meant to represent a matrix of quaternions (the last dimension stores one real and three imaginary parts)."""
    A = (np.random.rand(M, N, 4) * (range_max - range_min) + range_min).astype(dtype)
    if rounding:
        A = np.round(A)
    return A

def stack(E):
    """(`for internal use by` ``qmatmul``) For a three-dimensional ``numpy.ndarray`` of shape ``(R, S, 4)`` (matrix of quaternions) returns its stacked representation - the two-dimensional ``numpy.ndarray`` of shape ``(4 * R, S)`` with slices of imaginary parts stored as blocks of additional rows."""  
    R, S, _ = E.shape
    R2 = R << 1
    R3 = R2 + R
    R4 = R << 2           
    E4 = np.empty((R4, S), dtype=E.dtype)
    E4[:R] = E[:, :, 0]
    E4[R:R2] = E[:, :, 1]
    E4[R2:R3] = E[:, :, 2]
    E4[R3:] = E[:, :, 3]
    return E4  

def a44(A, a_blocks_signs=A_BLOCKS_SIGNS, a_blocks_parts=A_BLOCKS_PARTS):
    """(`for internal use by` ``qmatmul``) For a three-dimensional ``numpy.ndarray`` of shape ``(M, N, 4)`` (matrix of quaternions) returns the two-dimensional transformation matrix ``numpy.ndarray`` of shape ``(4 * M, 4 * N)`` made of suitably permuted and signed blocks of real/imaginary parts."""
    M, N, _ = A.shape
    A44 = np.empty((M << 2, N << 2), dtype=A.dtype)
    for i in range(4):
        iM = i * M
        for j in range(4):
            jN = j * N
            A44[iM:iM + M, jN:jN + N] = a_blocks_signs[i, j] * A[:, :, a_blocks_parts[i, j]] 
    return A44

def a44_ubar(A, a_blocks_parts=A_BLOCKS_PARTS):
    """(`for internal use by` ``qmatmul``) For a three-dimensional ``numpy.ndarray`` of shape ``(M, N, 4)`` (matrix of quaternions) returns the two-dimensional transformation matrix with double diagonal block-symmetry - a ``numpy.ndarray`` of shape ``(4 * M, 4 * N)`` made of suitably permuted blocks of real/imaginary parts."""    
    M, N, _ = A.shape
    A44_ubar = np.empty((M << 2, N << 2), dtype=A.dtype)
    for i in range(4):
        iM = i * M
        for j in range(4):
            jN = j * N
            A44_ubar[iM:iM + M, jN:jN + N] = A[:, :, a_blocks_parts[i, j]] 
    return A44_ubar

def a44_lbar(A):
    """(`for internal use by` ``qmatmul``) For a three-dimensional ``numpy.ndarray`` of shape ``(M, N, 4)`` (matrix of quaternions) returns the sparse transformation matrix ``numpy.ndarray`` of shape ``(4 * M, 4 * N)`` made of suitably placed blocks of real/imaginary parts."""    
    M, N, _ = A.shape
    M2 = M << 1
    M3 = M2 + M
    M4 = M << 2
    N2 = N << 1
    N3 = N2 + N
    N4 = N << 2    
    A44_lbar = np.zeros((M4, N4), dtype=A.dtype)
    A44_lbar[:M, :N] = A[:, :, 0]
    A44_lbar[M:M2, N2:N3] = A[:, :, 3]
    A44_lbar[M2:M3, N3:] = A[:, :, 1]
    A44_lbar[M3:, N:N2] = A[:, :, 2]
    return A44_lbar 

def i4_tilde(M, dtype):
    """(`for internal use by` ``qmatmul``) For a given size ``M`` returns the block-like identity matrix ``numpy.ndarray`` of shape ``(4 * M, 4 * M)`` where the top-left block has negative signs."""
    I4_tilde = np.eye(M << 2, dtype=dtype)
    diag_quarter = I4_tilde[np.arange(M), np.arange(M)]
    I4_tilde[np.arange(M), np.arange(M)] = -diag_quarter
    return I4_tilde


def qmatmul_algolike_numpy(A, B): # only to check correctness (compliance) of computational outcomes
    """(`for internal use by` ``qmatmul``) Function provided only to check the compliance of computational outcomes of the algorithm (main expression with decomposition into two products)."""
    I4_tilde = i4_tilde(A.shape[0], A.dtype)
    A44_ubar = a44_ubar(A)
    A44_lbar = a44_lbar(A)
    B4 = stack(B)        
    C4 = np.dot(I4_tilde, np.dot(A44_ubar, B4) - (2 * A44_lbar).dot(B4))
    C = c4_to_c(C4)
    return C

def c4_to_c(C4):
    """(`for internal use by` ``qmatmul``) For a two-dimensional ``numpy.ndarray`` of shape ``(4 * M, P)`` (a result of matrix-matrix product) returns its unstacked version - the three-dimensional ``numpy.ndarray`` of shape ``(M, P, 4)``, i.e., a matrix of quaternions."""
    M4, P = C4.shape
    M = M4 >> 2
    M2 = M << 1
    M3 = M2 + M    
    C = np.empty((M, P, 4), dtype=C4.dtype)
    C[:, :, 0] = C4[:M]
    C[:, :, 1] = C4[M:M2]
    C[:, :, 2] = C4[M2:M3]
    C[:, :, 3] = C4[M3:]
    return C

@jit(float64[:](float64[:], float64[:]), nopython=True, cache=True)
def qmul_numba_float64(q1, q2):
    """Returns the product of two single quaternions, each stored a 4-element long one-dimensional ``numpy.array`` of type ``float64``."""    
    result = np.zeros(4, dtype=np.float64)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return result  

@jit(float32[:](float32[:], float32[:]), nopython=True, cache=True)
def qmul_numba_float32(q1, q2):
    """Returns the product of two single quaternions, each stored a 4-element long one-dimensional ``numpy.array`` of type ``float32``."""    
    result = np.zeros(4, dtype=np.float32)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return result

@jit(int32[:](int32[:], int32[:]), nopython=True, cache=True)
def qmul_numba_int32(q1, q2):
    """Returns the product of two single quaternions, each stored a 4-element long one-dimensional ``numpy.array`` of type ``int32``."""
    result = np.zeros(4, dtype=np.int32)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return result

def qmatmul_naive_numba_st_float64(A, B, verbose=False):
    """(`to be invoked by the main function` ``qmatmul.dot``) Returns the result of ``dot(A, B, approach="naive_numba_st")`` for inputs of type ``float64``."""
    if verbose:
        print(f"QMATMUL_NAIVE_NUMBA_ST_FLOAT64...")
        t1 = time.time()
    C = qmatmul_naive_numba_st_float64_job(A, B)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_NAIVE_NUMBA_ST_FLOAT64 DONE. [time: {t2 - t1} s]")    
    return C

@jit(float64[:, :, :](float64[:, :, :], float64[:, :, :]), nopython=True, cache=True, parallel=False)
def qmatmul_naive_numba_st_float64_job(A, B):
    """(`for internal use by` ``qmatmul``) Actual computational job function for ``qmatmul_naive_numba_st_float64`` - returns the result of ``dot(A, B, approach="naive_numba_st")``."""
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.float64)
    for m in range(M):
        for p in range(P):
            for n in range(N):
                C[m, p] += qmul_numba_float64(A[m, n], B[n, p])
    return C

def qmatmul_naive_numba_parallel_float64(A, B, verbose=False):
    """(`to be invoked by the main function` ``qmatmul.dot``) Returns the result of ``dot(A, B, approach="naive_numba_parallel")`` for inputs of type ``float64``."""
    if verbose:
        print(f"QMATMUL_NAIVE_NUMBA_PARALLEL_FLOAT64...")
        t1 = time.time()
    C = qmatmul_naive_numba_parallel_float64_job(A, B)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_NAIVE_NUMBA_PARALLEL_FLOAT64 DONE. [time: {t2 - t1} s]")    
    return C

@jit(float64[:, :, :](float64[:, :, :], float64[:, :, :]), nopython=True, cache=True, parallel=True)
def qmatmul_naive_numba_parallel_float64_job(A, B):
    """(`for internal use by` ``qmatmul``) Actual computational job function for ``qmatmul_naive_numba_parallel_float64`` - returns the result of ``dot(A, B, approach="naive_numba_parallel")``."""
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.float64)
    for m in prange(M):
        for p in range(P):
            for n in range(N):
                C[m, p] += qmul_numba_float64(A[m, n], B[n, p])
    return C

def qmatmul_naive_numba_st_float32(A, B, verbose=False):
    if verbose:
        print(f"QMATMUL_NAIVE_NUMBA_ST_FLOAT32...")
        t1 = time.time()
    C = qmatmul_naive_numba_st_float32_job(A, B)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_NAIVE_NUMBA_ST_FLOAT32 DONE. [time: {t2 - t1} s]")    
    return C

@jit(float32[:, :, :](float32[:, :, :], float32[:, :, :]), nopython=True, cache=True, parallel=False)
def qmatmul_naive_numba_st_float32_job(A, B):
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.float32)
    for m in range(M):
        for p in range(P): 
            for n in range(N):
                C[m, p] += qmul_numba_float32(A[m, n], B[n, p])
    return C

def qmatmul_naive_numba_parallel_float32(A, B, verbose=False):
    if verbose:
        print(f"QMATMUL_NAIVE_NUMBA_PARALLEL_FLOAT32...")
        t1 = time.time()
    C = qmatmul_naive_numba_parallel_float32_job(A, B)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_NAIVE_NUMBA_PARALLEL_FLOAT32 DONE. [time: {t2 - t1} s]")    
    return C


@jit(float32[:, :, :](float32[:, :, :], float32[:, :, :]), nopython=True, cache=True, parallel=True)
def qmatmul_naive_numba_parallel_float32_job(A, B):
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.float32)
    for m in prange(M):
        for p in range(P):
            for n in range(N):
                C[m, p] += qmul_numba_float32(A[m, n], B[n, p])
    return C

def qmatmul_direct_numpy_parallel(A, B, verbose=False):
    if verbose:
        print(f"QMATMUL_DIRECT_NUMPY_PARALLEL...")
        t1 = time.time()    
    C4 = a44(A).dot(stack(B))
    C = c4_to_c(C4)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_DIRECT_NUMPY_PARALLEL DONE. [time: {t2 - t1} s]")        
    return C


def qmatmul_direct_numpy_st(A, B, verbose=False):
    if verbose:
        print(f"QMATMUL_DIRECT_NUMPY_ST...")
        t1 = time.time()
    with threadpool_limits(limits=1):    
        C4 = a44(A).dot(stack(B))
        C = c4_to_c(C4)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_DIRECT_NUMPY_ST DONE. [time: {t2 - t1} s]")        
    return C

def had4(E4):
    R4 = E4.shape[0]
    R2 = R4 >> 1
    R = R2 >> 1
    R3 = R2 + R
    E4_s0 = E4[:R] + E4[R:R2]
    E4_s1 = E4[R2:R3] + E4[R3:]
    E4_d0 = E4[:R] - E4[R:R2]
    E4_d1 = E4[R2:R3] - E4[R3:]
    H4E4 = np.empty_like(E4)
    H4E4[:R] = E4_s0 + E4_s1
    H4E4[R:R2] = E4_d0 + E4_d1
    H4E4[R2:R3] = E4_s0 - E4_s1
    H4E4[R3:] = E4_d0 - E4_d1
    return H4E4     

def matmuldiag(E4, F4, factor):
    R4, S = E4.shape
    R2 = R4 >> 1
    R = R2 >> 1
    R3 = R2 + R
    S2 = S << 1
    S3 = S2 + S
    T = F4.shape[1]
    D4 = np.empty((R4, T), dtype=E4.dtype)
    D4[:R] = factor * (E4[:R].dot(F4[:S]))
    D4[R:R2] = factor * (E4[R:R2].dot(F4[S:S2]))
    D4[R2:R3] = factor * (E4[R2:R3].dot(F4[S2:S3]))
    D4[R3:] = factor * (E4[R3:].dot(F4[S3:]))    
    return D4

def permute(E4, permutation):
    R4 = E4.shape[0]
    R = R4 >> 2
    E4p = np.empty_like(E4)
    for i in range(4):
        p = permutation[i]
        E4p[i * R:(i + 1) * R] = E4[p * R:(p + 1) * R]     
    return E4p

def qmatmul_algo_numpy_parallel(A, B, verbose=False):
    if verbose:
        print(f"QMATMUL_ALGO_NUMPY_PARALLEL...")
        t1 = time.time()    
    M = A.shape[0]
    B4 = stack(B)
    A4 = stack(A)        
    H4A4 = had4(A4)                   
    H4B4 = had4(B4)
    D4u = matmuldiag(H4A4, H4B4, 0.25)
    H4D4u = had4(D4u)  
    A4p = permute(A4, np.array([0, 3, 1, 2], dtype=np.int32))
    B4p = permute(B4, np.array([0, 2, 3, 1], dtype=np.int32)) 
    D4l = matmuldiag(A4p, B4p, 2.0)
    C4 = H4D4u - D4l
    C4[:M] = -C4[:M]
    C = c4_to_c(C4)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_ALGO_NUMPY_PARALLEL DONE. [time: {t2 - t1} s]")    
    return C

def qmatmul_algo_numpy_st(A, B, verbose=False):
    if verbose:
        print(f"QMATMUL_ALGO_NUMPY_ST...")
        t1 = time.time()
    with threadpool_limits(limits=1):            
        M = A.shape[0]
        B4 = stack(B)
        A4 = stack(A)        
        H4A4 = had4(A4)                   
        H4B4 = had4(B4)
        D4u = matmuldiag(H4A4, H4B4, 0.25)
        H4D4u = had4(D4u)  
        A4p = permute(A4, np.array([0, 3, 1, 2], dtype=np.int32))
        B4p = permute(B4, np.array([0, 2, 3, 1], dtype=np.int32)) 
        D4l = matmuldiag(A4p, B4p, 2.0)
        C4 = H4D4u - D4l
        C4[:M] = -C4[:M]
        C = c4_to_c(C4)
    if verbose:
        t2 = time.time()
        print(f"QMATMUL_ALGO_NUMPY_ST DONE. [time: {t2 - t1} s]")    
    return C    

def qmatmul_direct_numba_cuda_float64(A, B, tile_size=DEFAULT_TILE_SIZE, verbose=False):
    if verbose:
        print(f"QMATMUL_DIRECT_NUMBA_CUDA_FLOAT64...")
        t1 = time.time()
    M, N, _ = A.shape
    P = B.shape[1]
    N4 = N << 2
    M4 = M << 2
    if verbose:
        t1_b4 = time.time()
    dev_B = cuda.to_device(B)
    dev_B4 = cuda.device_array((N4, P), dtype=np.float64)
    tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
    tpb = tpb_default    
    bpg = (N4 * P + tpb - 1) // tpb
    stack_numba_cuda_job_float64[bpg, tpb](dev_B, dev_B4)
    if verbose:
        cuda.synchronize()    
        t2_b4 = time.time()        
        print(f"[time b4: {t2_b4 - t1_b4} s, bpg: {bpg}, tpb: {tpb}]")
        t1_a44 = time.time()
    dev_A = cuda.to_device(A)
    dev_A44 = cuda.device_array((M4, N4), dtype=np.float64)
    mn_bpg = (M * N + tpb - 1) // tpb 
    bpg = (mn_bpg, 4, 4)
    tpb = tpb_default
    a44_numba_cuda_job_float64[bpg, tpb](dev_A, dev_A44)
    if verbose:
        cuda.synchronize()
        t2_a44 = time.time()        
        print(f"[time a44: {t2_a44 - t1_a44} s, bpg: {bpg}, tpb: {tpb}]")
        t1_c4 = time.time()    
    dev_C4 = cuda.device_array((M4, P), dtype=np.float64)
    bpg_y = (M4 + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size) 
    matmul_numba_cuda_job_float64[bpg, tpb](dev_A44, dev_B4, dev_C4)    
    if verbose:
        cuda.synchronize()
        t2_c4 = time.time()        
        print(f"[time c4: {t2_c4 - t1_c4} s, bpg: {bpg}, tpb: {tpb}]")
        t1_c = time.time()            
    dev_C = cuda.device_array((M, P, 4), dtype=np.float64)
    tpb = tpb_default
    bpg = (M4 * P + tpb - 1) // tpb    
    c4_to_c_numba_cuda_job_float64[bpg, tpb](dev_C4, dev_C)
    cuda.synchronize()
    C = dev_C.copy_to_host()
    if verbose:
        t2_c = time.time()
        print(f"[time c: {t2_c - t1_c} s, bpg: {bpg}, tpb: {tpb}]")    
        t2 = time.time()
        print(f"QMATMUL_DIRECT_NUMBA_CUDA_FLOAT64 DONE. [time: {t2 - t1} s]")
    return C
         
@cuda.jit(void(float64[:, :, :], float64[:, :]))
def stack_numba_cuda_job_float64(E, E4):
    R, S, _ = E.shape
    i_global = cuda.grid(1)
    i_rs, i_im = i_global // 4, i_global % 4    
    if i_rs < R * S:
        r, s = i_rs // S, i_rs % S
        E4[i_im * R + r, s] = E[r, s, i_im]                

@cuda.jit(void(float64[:, :, :], float64[:, :]))
def a44_numba_cuda_job_float64(A, A4):
    const_a_blocks_signs = cuda.const.array_like(A_BLOCKS_SIGNS)
    const_a_blocks_parts = cuda.const.array_like(A_BLOCKS_PARTS)
    M, N, _ = A.shape
    i_mn, block_row, block_col = cuda.grid(3)        
    if i_mn < M * N:
        m, n = i_mn // N, i_mn % N
        A4[block_row * M + m, block_col * N + n] = const_a_blocks_signs[block_row, block_col] * A[m, n, const_a_blocks_parts[block_row, block_col]]
        
@cuda.jit(void(float64[:, :], float64[:, :], float64[:, :]))
def matmul_numba_cuda_job_float64(A44, B4, C4):    
    shared_A = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    shared_B = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    tile_size = cuda.blockDim.x
    M4, P = C4.shape
    N4 = B4.shape[0]
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = by * tile_size + ty
    col = bx * tile_size + tx
    tmp = float64(0.0)
    for k in range(0, N4, tile_size):
        if row < M4 and k + tx < N4:
            shared_A[ty, tx] = A44[row, k + tx]
        else:
            shared_A[ty, tx] = float64(0.0)
        if k + ty < N4 and col < P:
            shared_B[ty, tx] = B4[k + ty, col]
        else:
            shared_B[ty, tx] = float64(0.0)
        cuda.syncthreads()
        for n in range(tile_size):
            tmp += shared_A[ty, n] * shared_B[n, tx]
        cuda.syncthreads()
    if row < M4 and col < P:
        C4[row, col] = tmp
        
@cuda.jit(void(float64[:, :], float64[:, :, :]))
def c4_to_c_numba_cuda_job_float64(C4, C):
    M, P, _ = C.shape
    i_global = cuda.grid(1)
    i_mp, i_im = i_global // 4, i_global % 4    
    if i_mp < M * P:
        m, p = i_mp // P, i_mp % P
        C[m, p, i_im] = C4[i_im * M + m, p] 

def qmatmul_direct_numba_cuda_float32(A, B, tile_size=DEFAULT_TILE_SIZE, verbose=False):
    if verbose:
        print(f"QMATMUL_DIRECT_NUMBA_CUDA_FLOAT32...")
        t1 = time.time()
    M, N, _ = A.shape
    P = B.shape[1]
    N4 = N << 2
    M4 = M << 2
    t1_b4 = time.time()
    dev_B = cuda.to_device(B)
    dev_B4 = cuda.device_array((N4, P), dtype=np.float32)
    tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
    tpb = tpb_default
    bpg = (N4 * P + tpb - 1) // tpb
    stack_numba_cuda_job_float32[bpg, tpb](dev_B, dev_B4)
    if verbose:
        cuda.synchronize()    
        t2_b4 = time.time()        
        print(f"[time b4: {t2_b4 - t1_b4} s, bpg: {bpg}, tpb: {tpb}]")
        t1_a44 = time.time()
    dev_A = cuda.to_device(A)
    dev_A44 = cuda.device_array((M4, N4), dtype=np.float32)
    mn_bpg = (M * N + tpb - 1) // tpb 
    bpg = (mn_bpg, 4, 4)
    tpb = tpb_default
    a44_numba_cuda_job_float32[bpg, tpb](dev_A, dev_A44)
    if verbose:
        cuda.synchronize()
        t2_a44 = time.time()        
        print(f"[time a44: {t2_a44 - t1_a44} s, bpg: {bpg}, tpb: {tpb}]")
        t1_c4 = time.time()    
    dev_C4 = cuda.device_array((M4, P), dtype=np.float64)
    bpg_y = (M4 + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size) 
    matmul_numba_cuda_job_float32[bpg, tpb](dev_A44, dev_B4, dev_C4)    
    if verbose:
        cuda.synchronize()
        t2_c4 = time.time()        
        print(f"[time c4: {t2_c4 - t1_c4} s, bpg: {bpg}, tpb: {tpb}]")
        t1_c = time.time()    
    dev_C = cuda.device_array((M, P, 4), dtype=np.float32)
    tpb = tpb_default
    bpg = (M4 * P + tpb - 1) // tpb    
    c4_to_c_numba_cuda_job_float32[bpg, tpb](dev_C4, dev_C)
    cuda.synchronize()
    C = dev_C.copy_to_host()    
    if verbose:
        t2_c = time.time()
        print(f"[time c: {t2_c - t1_c} s, bpg: {bpg}, tpb: {tpb}]")        
        t2 = time.time()
        print(f"QMATMUL_DIRECT_NUMBA_CUDA_FLOAT32 DONE. [time: {t2 - t1} s]")
    return C
         
@cuda.jit(void(float32[:, :, :], float32[:, :]))
def stack_numba_cuda_job_float32(E, E4):
    R, S, _ = E.shape
    i_global = cuda.grid(1)
    i_rs, i_im = i_global // 4, i_global % 4    
    if i_rs < R * S:
        r, s = i_rs // S, i_rs % S
        E4[i_im * R + r, s] = E[r, s, i_im]                

@cuda.jit(void(float32[:, :, :], float32[:, :]))
def a44_numba_cuda_job_float32(A, A4):
    const_a_blocks_signs = cuda.const.array_like(A_BLOCKS_SIGNS)
    const_a_blocks_parts = cuda.const.array_like(A_BLOCKS_PARTS)
    M, N, _ = A.shape
    i_mn, block_row, block_col = cuda.grid(3)        
    if i_mn < M * N:
        m, n = i_mn // N, i_mn % N
        A4[block_row * M + m, block_col * N + n] = const_a_blocks_signs[block_row, block_col] * A[m, n, const_a_blocks_parts[block_row, block_col]]
        
@cuda.jit(void(float32[:, :], float32[:, :], float32[:, :]))
def matmul_numba_cuda_job_float32(A44, B4, C4):    
    shared_A = cuda.shared.array((16, 16), dtype=float32) # assumed max tile size: 16
    shared_B = cuda.shared.array((16, 16), dtype=float32) # assumed max tile size: 16
    tile_size = cuda.blockDim.x
    M4, P = C4.shape
    N4 = B4.shape[0]
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = by * tile_size + ty
    col = bx * tile_size + tx
    tmp = float32(0.0)
    for k in range(0, N4, tile_size):
        if row < M4 and k + tx < N4:
            shared_A[ty, tx] = A44[row, k + tx]
        else:
            shared_A[ty, tx] = float32(0.0)
        if k + ty < N4 and col < P:
            shared_B[ty, tx] = B4[k + ty, col]
        else:
            shared_B[ty, tx] = float32(0.0)
        cuda.syncthreads()
        for n in range(tile_size):
            tmp += shared_A[ty, n] * shared_B[n, tx]
        cuda.syncthreads()
    if row < M4 and col < P:
        C4[row, col] = tmp
        
@cuda.jit(void(float32[:, :], float32[:, :, :]))
def c4_to_c_numba_cuda_job_float32(C4, C):
    M, P, _ = C.shape
    i_global = cuda.grid(1)
    i_mp, i_im = i_global // 4, i_global % 4    
    if i_mp < M * P:
        m, p = i_mp // P, i_mp % P
        C[m, p, i_im] = C4[i_im * M + m, p]

def qmatmul_algo_numba_cuda_float64(A, B, tile_size=DEFAULT_TILE_SIZE, verbose=False):
    if verbose:
        print(f"QMATMUL_ALGO_NUMBA_CUDA_FLOAT64...")
        t1 = time.time()
    M, N, _ = A.shape
    P = B.shape[1]
    N4 = N << 2
    M4 = M << 2
    tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2    
    if verbose:
        t1_stacks = time.time()
    dev_B = cuda.to_device(B)
    dev_B4 = cuda.device_array((N4, P), dtype=np.float64)
    tpb = tpb_default
    bpg = (N4 * P + tpb - 1) // tpb
    stack_numba_cuda_job_float64[bpg, tpb](dev_B, dev_B4)    
    dev_A = cuda.to_device(A)
    dev_A4 = cuda.device_array((M4, N), dtype=np.float64)
    bpg = (M4 * N + tpb - 1) // tpb 
    stack_numba_cuda_job_float64[bpg, tpb](dev_A, dev_A4)
    if verbose:
        cuda.synchronize()
        t2_stacks = time.time()        
        print(f"[time stacks: {t2_stacks - t1_stacks} s, tpb: {tpb}]")
        t1_h4s = time.time()
    dev_H4B4 = cuda.device_array((N4, P), dtype=np.float64)
    bpg_y = (N + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size)
    had4_numba_cuda_job_float64[bpg, tpb](dev_B4, dev_H4B4)
    dev_H4A4 = cuda.device_array((M4, N), dtype=np.float64)
    bpg_y = (M + tile_size - 1) // tile_size
    bpg_x = (N + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y)
    had4_numba_cuda_job_float64[bpg, tpb](dev_A4, dev_H4A4)
    if verbose:
        cuda.synchronize()
        t2_h4s = time.time()        
        print(f"[time h4s: {t2_h4s - t1_h4s} s, tpb: {tpb}]")        
        t1_d4 = time.time()
    dev_D4 = cuda.device_array((M4, P), dtype=np.float64)
    bpg_y = (M + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y, 4)
    matmuldiag_numba_cuda_job_float64[bpg, tpb](dev_H4A4, dev_H4B4, 0.25, dev_D4)
    if verbose:
        cuda.synchronize()
        t2_d4 = time.time()
        print(f"[time d4: {t2_d4 - t1_d4} s, bpg: {bpg}, tpb: {tpb}]")    
        t1_h4d4 = time.time()
    dev_H4D4 = cuda.device_array((M4, P), dtype=np.float64)
    bpg = (bpg_x, bpg_y)
    had4_numba_cuda_job_float64[bpg, tpb](dev_D4, dev_H4D4)
    if verbose:
        cuda.synchronize()
        t2_h4d4 = time.time()        
        print(f"[time h4d4: {t2_h4d4 - t1_h4d4} s, bpg: {bpg}, tpb: {tpb}]")        
        t1_a4lb4l = time.time()
    dev_A4l = cuda.device_array((M4, N), dtype=np.float64)    
    dev_permutation_a4l = cuda.to_device(np.array([0, 3, 1, 2], dtype=np.int8))
    bpg_x = (M + tile_size - 1) // tile_size
    bpg_y = (N + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    permute_numba_cuda_job_float64[bpg, tpb](dev_A4, dev_permutation_a4l, dev_A4l)    
    dev_B4l = cuda.device_array((N4, P), dtype=np.float64)
    dev_permutation_b4l = cuda.to_device(np.array([0, 2, 3, 1], dtype=np.int8))
    bpg_x = (N + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    permute_numba_cuda_job_float64[bpg, tpb](dev_B4, dev_permutation_b4l, dev_B4l)        
    dev_A4lB4l = cuda.device_array((M4, P), dtype=np.float64)
    bpg_y = (M + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing   
    bpg = (bpg_x, bpg_y, 4)
    matmuldiag_numba_cuda_job_float64[bpg, tpb](dev_A4l, dev_B4l, 2.0, dev_A4lB4l)
    if verbose:
        cuda.synchronize()
        t2_a4lb4l = time.time()        
        print(f"[time a4lb4l: {t2_a4lb4l - t1_a4lb4l} s, tpb: {tpb}]")                
        t1_sub = time.time()
    dev_C4 = cuda.device_array((M4, P), dtype=np.float64)    
    bpg_y = (M4 + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y)    
    matsub_numba_cuda_job_float64[bpg, tpb](dev_H4D4, dev_A4lB4l, dev_C4)    
    if verbose:
        cuda.synchronize()
        t2_sub = time.time()
        print(f"[time sub: {t2_sub - t1_sub} s, bpg: {bpg}, tpb: {tpb}]")
        t1_c = time.time()        
    dev_C = cuda.device_array((M, P, 4), dtype=np.float64)
    tpb = tpb_default
    bpg = (M4 * P + tpb - 1) // tpb
    c4_to_c_numba_cuda_job_float64[bpg, tpb](dev_C4, dev_C)
    cuda.synchronize()    
    C = dev_C.copy_to_host()    
    if verbose:
        t2_c = time.time()
        print(f"[time c: {t2_c - t1_c} s, bpg: {bpg}, tpb: {tpb}]")      
        t2 = time.time()
        print(f"QMATMUL_ALGO_NUMBA_CUDA_FLOAT64 DONE. [time: {t2 - t1} s]")
    return C
        
@cuda.jit(void(float64[:, :], float64[:, :]))
def had4_numba_cuda_job_float64(E4, H4E4):      
    R4, S = E4.shape
    R = R4 >> 2
    R2 = R << 1
    R3 = R2 + R
    tile_size = cuda.blockDim.x    
    row = cuda.blockIdx.y * tile_size + cuda.threadIdx.y
    col = cuda.blockIdx.x * tile_size + cuda.threadIdx.x
    if row < R and col < S:
        e4_0 = E4[row, col]
        e4_1 = E4[row + R, col]
        e4_2 = E4[row + R2, col]
        e4_3 = E4[row + R3, col] 
        s0 = e4_0 + e4_1
        s1 = e4_2 + e4_3
        d0 = e4_0 - e4_1
        d1 = e4_2 - e4_3    
        H4E4[row, col] = s0 + s1
        H4E4[row + R, col] = d0 + d1
        H4E4[row + R2, col] = s0 - s1
        H4E4[row + R3, col] = d0 - d1          
        
@cuda.jit(void(float64[:, :], float64[:, :], float64, float64[:, :]))
def matmuldiag_numba_cuda_job_float64(E4, F4, factor, G4): # E4 shape: (R4 x S), F4 shape: (S4 x T), G4 shape: (R4 x T)     
    shared_E = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    shared_F = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    tile_size = cuda.blockDim.x
    R4, T = G4.shape
    R = R4 >> 2
    S4 = F4.shape[0]
    S = S4 >> 2
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = by * tile_size + ty
    col = bx * tile_size + tx
    tmp = float64(0.0)
    row_bz_R = row + bz * R
    ty_bz_S = ty + bz * S
    for k in range(0, S, tile_size):
        if row < R and k + tx < S:
            shared_E[ty, tx] = E4[row_bz_R, k + tx]
        else:
            shared_E[ty, tx] = float64(0.0)
        if k + ty < S and col < T:
            shared_F[ty, tx] = F4[k + ty_bz_S, col]
        else:
            shared_F[ty, tx] = float64(0.0)
        cuda.syncthreads()
        for s in range(tile_size):
            tmp += shared_E[ty, s] * shared_F[s, tx]
        cuda.syncthreads()
    if row < R and col < T:
        G4[row_bz_R, col] = factor * tmp        

@cuda.jit(void(float64[:, :], int8[:], float64[:, :]))
def permute_numba_cuda_job_float64(E4, permutation, E4_permuted):    
    shared_E4 = cuda.shared.array((16, 16, 4), dtype=float64) # assumed max tile size: 16
    M4, N = E4.shape
    M = M4 >> 2
    M2 = M << 1
    M3 = M2 + M
    tile_size = cuda.blockDim.x
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = bx * tile_size + tx
    col = by * tile_size + ty
    if row < M and col < N:
        shared_E4[tx, ty, 0] = E4[row, col]
        shared_E4[tx, ty, 1] = E4[row + M, col]
        shared_E4[tx, ty, 2] = E4[row + M2, col]
        shared_E4[tx, ty, 3] = E4[row + M3, col]     
        E4_permuted[row, col] = shared_E4[tx, ty, permutation[0]]
        E4_permuted[row + M, col] = shared_E4[tx, ty, permutation[1]]
        E4_permuted[row + M2, col] = shared_E4[tx, ty, permutation[2]]
        E4_permuted[row + M3, col] = shared_E4[tx, ty, permutation[3]]
        
@cuda.jit(void(float64[:, :], float64[:, :], float64[:, :]))
def matsub_numba_cuda_job_float64(C4_left, C4_right, C4):    
    shared_L = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    shared_R = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    tile_size = cuda.blockDim.x
    M4, P = C4.shape
    M = M4 >> 2
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = by * tile_size + ty
    col = bx * tile_size + tx
    if row < M4 and col < P:
        shared_L[tx, ty] = C4_left[row, col]
        shared_R[tx, ty] = C4_right[row, col]
        result = shared_R[tx, ty] - shared_L[tx, ty] if row < M else shared_L[tx, ty] - shared_R[tx, ty]        
        C4[row, col] = result
         
def qmatmul_algo_numba_cuda_float32(A, B, tile_size=DEFAULT_TILE_SIZE, verbose=False):
    if verbose:
        print(f"QMATMUL_ALGO_NUMBA_CUDA_FLOAT32...")
    t1 = time.time()
    M, N, _ = A.shape
    P = B.shape[1]
    N4 = N << 2
    M4 = M << 2
    tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2    
    t1_stacks = time.time()
    dev_B = cuda.to_device(B)
    dev_B4 = cuda.device_array((N4, P), dtype=np.float32)
    tpb = tpb_default
    bpg = (N4 * P + tpb - 1) // tpb
    stack_numba_cuda_job_float32[bpg, tpb](dev_B, dev_B4)
    dev_A = cuda.to_device(A)
    dev_A4 = cuda.device_array((M4, N), dtype=np.float32)
    bpg = (M4 * N + tpb - 1) // tpb 
    stack_numba_cuda_job_float32[bpg, tpb](dev_A, dev_A4)    
    if verbose:
        cuda.synchronize()
        t2_stacks = time.time()
        print(f"[time stacks: {t2_stacks - t1_stacks} s, tpb: {tpb}]")
        t1_h4s = time.time()
    dev_H4B4 = cuda.device_array((N4, P), dtype=np.float32)
    bpg_y = (N + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size)
    had4_numba_cuda_job_float32[bpg, tpb](dev_B4, dev_H4B4)
    dev_H4A4 = cuda.device_array((M4, N), dtype=np.float32)
    bpg_y = (M + tile_size - 1) // tile_size
    bpg_x = (N + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y)
    had4_numba_cuda_job_float32[bpg, tpb](dev_A4, dev_H4A4)        
    if verbose:
        cuda.synchronize()
        t2_h4s = time.time()
        print(f"[time h4s: {t2_h4s - t1_h4s} s, tpb: {tpb}]")        
        t1_d4 = time.time()
    dev_D4 = cuda.device_array((M4, P), dtype=np.float32)
    bpg_y = (M + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y, 4)
    matmuldiag_numba_cuda_job_float32[bpg, tpb](dev_H4A4, dev_H4B4, np.float32(0.25), dev_D4)
    if verbose:
        cuda.synchronize()
        t2_d4 = time.time()        
        print(f"[time d4: {t2_d4 - t1_d4} s, bpg: {bpg}, tpb: {tpb}]")    
        t1_h4d4 = time.time()
    dev_H4D4 = cuda.device_array((M4, P), dtype=np.float32)
    bpg = (bpg_x, bpg_y)
    had4_numba_cuda_job_float32[bpg, tpb](dev_D4, dev_H4D4)
    if verbose:
        cuda.synchronize()
        t2_h4d4 = time.time()        
        print(f"[time h4d4: {t2_h4d4 - t1_h4d4} s, bpg: {bpg}, tpb: {tpb}]")        
        t1_a4lb4l = time.time()
    dev_A4l = cuda.device_array((M4, N), dtype=np.float32)    
    dev_permutation_a4l = cuda.to_device(np.array([0, 3, 1, 2], dtype=np.int8))
    bpg_x = (M + tile_size - 1) // tile_size
    bpg_y = (N + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    permute_numba_cuda_job_float32[bpg, tpb](dev_A4, dev_permutation_a4l, dev_A4l)    
    dev_B4l = cuda.device_array((N4, P), dtype=np.float32)
    dev_permutation_b4l = cuda.to_device(np.array([0, 2, 3, 1], dtype=np.int8))
    bpg_x = (N + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    permute_numba_cuda_job_float32[bpg, tpb](dev_B4, dev_permutation_b4l, dev_B4l)           
    dev_A4lB4l = cuda.device_array((M4, P), dtype=np.float32)
    bpg_y = (M + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing    
    bpg = (bpg_x, bpg_y, 4)
    matmuldiag_numba_cuda_job_float32[bpg, tpb](dev_A4l, dev_B4l, np.float32(2.0), dev_A4lB4l)
    if verbose:
        cuda.synchronize()        
        t2_a4lb4l = time.time()        
        print(f"[time a4lb4l: {t2_a4lb4l - t1_a4lb4l} s, bpg: {bpg}, tpb: {tpb}]")                
        t1_sub = time.time()
    dev_C4 = cuda.device_array((M4, P), dtype=np.float32)
    bpg_y = (M4 + tile_size - 1) // tile_size
    bpg_x = (P + tile_size - 1) // tile_size # associated with columns due to CUDA's column-major indexing
    bpg = (bpg_x, bpg_y)    
    matsub_numba_cuda_job_float32[bpg, tpb](dev_H4D4, dev_A4lB4l, dev_C4)    
    if verbose:
        cuda.synchronize()
        t2_sub = time.time()        
        print(f"[time sub: {t2_sub - t1_sub} s, bpg: {bpg}, tpb: {tpb}]")
        t1_c = time.time()            
    dev_C = cuda.device_array((M, P, 4), dtype=np.float32)
    tpb = tpb_default
    bpg = (M4 * P + tpb - 1) // tpb
    c4_to_c_numba_cuda_job_float32[bpg, tpb](dev_C4, dev_C)
    cuda.synchronize()
    C = dev_C.copy_to_host()
    if verbose:
        t2_c = time.time()
        print(f"[time c: {t2_c - t1_c} s, bpg: {bpg}, tpb: {tpb}]")            
        t2 = time.time()
        print(f"QMATMUL_ALGO_NUMBA_CUDA_FLOAT32 DONE. [time: {t2 - t1} s]")
    return C

@cuda.jit(void(float32[:, :], float32[:, :]))
def had4_numba_cuda_job_float32(E4, H4E4):      
    R4, S = E4.shape
    R = R4 >> 2
    R2 = R << 1
    R3 = R2 + R
    tile_size = cuda.blockDim.x    
    row = cuda.blockIdx.y * tile_size + cuda.threadIdx.y
    col = cuda.blockIdx.x * tile_size + cuda.threadIdx.x
    if row < R and col < S:
        e4_0 = E4[row, col]
        e4_1 = E4[row + R, col]
        e4_2 = E4[row + R2, col]
        e4_3 = E4[row + R3, col] 
        s0 = e4_0 + e4_1
        s1 = e4_2 + e4_3
        d0 = e4_0 - e4_1
        d1 = e4_2 - e4_3    
        H4E4[row, col] = s0 + s1
        H4E4[row + R, col] = d0 + d1
        H4E4[row + R2, col] = s0 - s1
        H4E4[row + R3, col] = d0 - d1        

@cuda.jit(void(float32[:, :], float32[:, :], float32, float32[:, :]))
def matmuldiag_numba_cuda_job_float32(E4, F4, factor, G4): # E4 shape: (R4 x S), F4 shape: (S4 x T), G4 shape: (R4 x T)     
    shared_E = cuda.shared.array((16, 16), dtype=float32) # assumed max tile size: 16
    shared_F = cuda.shared.array((16, 16), dtype=float32) # assumed max tile size: 16
    tile_size = cuda.blockDim.x
    R4, T = G4.shape
    R = R4 >> 2
    S4 = F4.shape[0]
    S = S4 >> 2
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = by * tile_size + ty
    col = bx * tile_size + tx
    tmp = float32(0.0)
    row_bz_R = row + bz * R
    ty_bz_S = ty + bz * S
    for k in range(0, S, tile_size):
        if row < R and k + tx < S:
            shared_E[ty, tx] = E4[row_bz_R, k + tx]
        else:
            shared_E[ty, tx] = float32(0.0)
        if k + ty < S and col < T:
            shared_F[ty, tx] = F4[k + ty_bz_S, col]
        else:
            shared_F[ty, tx] = float32(0.0)
        cuda.syncthreads()
        for s in range(tile_size):
            tmp += shared_E[ty, s] * shared_F[s, tx]
        cuda.syncthreads()
    if row < R and col < T:
        G4[row_bz_R, col] = factor * tmp

@cuda.jit(void(float32[:, :], int8[:], float32[:, :]))
def permute_numba_cuda_job_float32(E4, permutation, E4_permuted):    
    shared_E4 = cuda.shared.array((16, 16, 4), dtype=float32) # assumed max tile size: 16
    M4, N = E4.shape
    M = M4 >> 2
    M2 = M << 1
    M3 = M2 + M
    tile_size = cuda.blockDim.x
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = bx * tile_size + tx
    col = by * tile_size + ty
    if row < M and col < N:
        shared_E4[tx, ty, 0] = E4[row, col]
        shared_E4[tx, ty, 1] = E4[row + M, col]
        shared_E4[tx, ty, 2] = E4[row + M2, col]
        shared_E4[tx, ty, 3] = E4[row + M3, col]     
        E4_permuted[row, col] = shared_E4[tx, ty, permutation[0]]
        E4_permuted[row + M, col] = shared_E4[tx, ty, permutation[1]]
        E4_permuted[row + M2, col] = shared_E4[tx, ty, permutation[2]]
        E4_permuted[row + M3, col] = shared_E4[tx, ty, permutation[3]]
        
@cuda.jit(void(float32[:, :], float32[:, :], float32[:, :]))
def matsub_numba_cuda_job_float32(C4_left, C4_right, C4):    
    shared_L = cuda.shared.array((16, 16), dtype=float32) # assumed max tile size: 16
    shared_R = cuda.shared.array((16, 16), dtype=float32) # assumed max tile size: 16
    tile_size = cuda.blockDim.x
    M4, P = C4.shape
    M = M4 >> 2
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = by * tile_size + ty
    col = bx * tile_size + tx
    if row < M4 and col < P:
        shared_L[tx, ty] = C4_left[row, col]
        shared_R[tx, ty] = C4_right[row, col]
        result = shared_R[tx, ty] - shared_L[tx, ty] if row < M else shared_L[tx, ty] - shared_R[tx, ty]        
        C4[row, col] = result
