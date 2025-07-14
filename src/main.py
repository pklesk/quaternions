import os
NUMPY_SINGLE_THREAD = True
if NUMPY_SINGLE_THREAD:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import time
from numba import jit, prange, cuda
from numba import void, float32, float64, int8, int32
from numba.core.errors import NumbaPerformanceWarning

import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

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

def qmatrand(M, N, range_min, range_max, dtype=np.float32):
    A = (np.random.rand(M, N, 4) * (range_max - range_min) + range_min).astype(dtype)
    return A

def stack(E):
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
    M, N, _ = A.shape
    A44 = np.empty((M << 2, N << 2), dtype=A.dtype)
    for i in range(4):
        iM = i * M
        for j in range(4):
            jN = j * N
            A44[iM:iM + M, jN:jN + N] = a_blocks_signs[i, j] * A[:, :, a_blocks_parts[i, j]] 
    return A44

def a44_ubar(A, a_blocks_parts=A_BLOCKS_PARTS):
    M, N, _ = A.shape
    A44_ubar = np.empty((M << 2, N << 2), dtype=A.dtype)
    for i in range(4):
        iM = i * M
        for j in range(4):
            jN = j * N
            A44_ubar[iM:iM + M, jN:jN + N] = A[:, :, a_blocks_parts[i, j]] 
    return A44_ubar

def a44_lbar(A):
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

def a4_lbar(A):
    M, N, _ = A.shape
    M2 = M << 1
    M3 = M2 + M
    M4 = M << 2
    A4_lbar = np.zeros((M4, N), dtype=A.dtype)
    A4_lbar[:M] = A[:, :, 0]
    A4_lbar[M:M2] = A[:, :, 3]
    A4_lbar[M2:M3] = A[:, :, 1]
    A4_lbar[M3:] = A[:, :, 2]
    return A4_lbar

def i4_tilde(M, dtype):
    I4_tilde = np.eye(M << 2, dtype=dtype)
    diag_quarter = I4_tilde[np.arange(M), np.arange(M)]
    I4_tilde[np.arange(M), np.arange(M)] = -diag_quarter
    return I4_tilde

def c4_to_c(C4):
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
    result = np.zeros(4, dtype=np.float64)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return result  

@jit(float32[:](float32[:], float32[:]), nopython=True, cache=True)
def qmul_numba_float32(q1, q2):
    result = np.zeros(4, dtype=np.float32)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return result

@jit(int32[:](int32[:], int32[:]), nopython=True, cache=True)
def qmul_numba_int32(q1, q2):
    result = np.zeros(4, dtype=np.int32)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return result

@jit(float64[:, :, :](float64[:, :, :], float64[:, :, :]), nopython=True, cache=True, parallel=False)
def qmatmul_naive_numba_st_float64(A, B):
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.float64)
    for m in range(M):
        for p in range(P):
            for n in range(N):
                C[m, p] += qmul_numba_float64(A[m, n], B[n, p])
    return C

@jit(float64[:, :, :](float64[:, :, :], float64[:, :, :]), nopython=True, cache=True, parallel=True)
def qmatmul_naive_numba_parallel_float64(A, B):
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.float64)
    for m in prange(M):
        for p in prange(P):
            for n in range(N):
                C[m, p] += qmul_numba_float64(A[m, n], B[n, p])
    return C

@jit(float32[:, :, :](float32[:, :, :], float32[:, :, :]), nopython=True, cache=True, parallel=False)
def qmatmul_naive_numba_st_float32(A, B):
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.float32)
    for m in range(M):
        for p in range(P):
            for n in range(N):
                C[m, p] += qmul_numba_float32(A[m, n], B[n, p])
    return C

@jit(float32[:, :, :](float32[:, :, :], float32[:, :, :]), nopython=True, cache=True, parallel=True)
def qmatmul_naive_numba_parallel_float32(A, B):
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.float32)
    for m in prange(M):
        for p in range(P):
            for n in range(N):
                C[m, p] += qmul_numba_float32(A[m, n], B[n, p])
    return C

@jit(int32[:, :, :](int32[:, :, :], int32[:, :, :]), nopython=True, cache=True, parallel=False)
def qmatmul_naive_numba_st_int32(A, B):
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.int32)
    for m in range(M):
        for p in range(P):
            for n in range(N):
                C[m, p] += qmul_numba_int32(A[m, n], B[n, p])
    return C

@jit(int32[:, :, :](int32[:, :, :], int32[:, :, :]), nopython=True, cache=True, parallel=True)
def qmatmul_naive_numba_parallel_int32(A, B):
    M, N, _ = A.shape    
    P = B.shape[1]
    C = np.zeros((M, P, 4), dtype=np.int32)
    for m in prange(M):
        for p in range(P):
            for n in range(N):
                C[m, p] += qmul_numba_int32(A[m, n], B[n, p])
    return C

def qmatmul_direct_numpy(A, B):
    C4 = a44(A).dot(stack(B))
    C = c4_to_c(C4)
    return C

def qmatmul_direct_algolike_numpy(A, B):
    I4_tilde = i4_tilde(A.shape[0], A.dtype)
    A44_ubar = a44_ubar(A)
    A44_lbar = a44_lbar(A)
    B4 = stack(B)        
    C4 = np.dot(I4_tilde, np.dot(A44_ubar, B4) - (2 * A44_lbar).dot(B4))
    C = c4_to_c(C4)
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

def qmatmul_algo_numpy(A, B):
    M, N, _ = A.shape
    M2 = M << 1
    M3 = M2 + M
    M4 = M2 << 1
    N2 = N << 1
    N3 = N2 + N     
    B4 = stack(B)
    A4 = stack(A)
    H4A4 = had4(A4)                   
    H4B4 = had4(B4)        
    D4 = np.zeros((M4, P), dtype=np.float64) # type purposely extended due to 0.25 factor 
    D4[:M] = 0.25 * (H4A4[:M].dot(H4B4[:N]))
    D4[M:M2] = 0.25 * (H4A4[M:M2].dot(H4B4[N:N2]))
    D4[M2:M3] = 0.25 * (H4A4[M2:M3].dot(H4B4[N2:N3]))
    D4[M3:] = 0.25 * (H4A4[M3:].dot(H4B4[N3:]))
    H4D4 = had4(D4)  
    A4l = a4_lbar(A) 
    A4lB4l = np.zeros((M4, P), dtype=np.float64) # type purposely extended
    A4lB4l[:M] = (2.0 * A4l[:M]).dot(B4[:N])
    A4lB4l[M:M2] = (2.0 * A4l[M:M2]).dot(B4[N2:N3])
    A4lB4l[M2:M3] = (2.0 * A4l[M2:M3]).dot(B4[N3:])
    A4lB4l[M3:] = (2.0 * A4l[M3:]).dot(B4[N:N2])
    C4 = H4D4 - A4lB4l
    C4[:M] = -C4[:M]
    if C4.dtype != A.dtype:
        C4 = C4.astype(A.dtype)
    C = c4_to_c(C4)
    return C

def qmatmul_direct_numba_cuda_float64(A, B, verbose=False):
    if verbose:
        print(f"QMATMUL_DIRECT_NUMBA_CUDA_FLOAT64...")
    t1 = time.time()
    M, N, _ = A.shape
    P = B.shape[1]
    N4 = N << 2
    M4 = M << 2
    t1_b4 = time.time()
    dev_B = cuda.to_device(B)
    dev_B4 = cuda.device_array((N4, P), dtype=np.float64)
    tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
    tpb = tpb_default
    bpg = (N4 * P + tpb - 1) // tpb
    stack_numba_cuda_job_float64[bpg, tpb](dev_B, dev_B4)
    cuda.synchronize()    
    t2_b4 = time.time()
    if verbose:
        print(f"[time b4: {t2_b4 - t1_b4} s]")
    t1_a44 = time.time()
    dev_A = cuda.to_device(A)
    dev_A44 = cuda.device_array((M4, N4), dtype=np.float64)
    mn_bpg = (M * N + tpb - 1) // tpb 
    bpg = (mn_bpg, 4, 4)
    tpb = tpb_default
    a44_numba_cuda_job_float64[bpg, tpb](dev_A, dev_A44)
    cuda.synchronize()
    t2_a44 = time.time()
    if verbose:
        print(f"[time a44: {t2_a44 - t1_a44} s]")
    t1_c4 = time.time()    
    dev_C4 = cuda.device_array((M4, P), dtype=np.float64)
    tile_size = 8
    bpg_x = (M4 + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size) 
    matmul_numba_cuda_job_float64[bpg, tpb](dev_A44, dev_B4, dev_C4)    
    cuda.synchronize()
    t2_c4 = time.time()
    if verbose:
        print(f"[time c4: {t2_c4 - t1_c4} s, bpg: {bpg}, tpb: {tpb}]")
    t1_c = time.time()    
    dev_C = cuda.device_array((M, P, 4), dtype=np.float64)
    tpb = tpb_default
    bpg = (M4 * P + tpb - 1) // tpb    
    c4_to_c_numba_cuda_job_float64[bpg, tpb](dev_C4, dev_C)
    cuda.synchronize()
    C = dev_C.copy_to_host()
    t2_c = time.time()
    if verbose:
        print(f"[time c: {t2_c - t1_c} s, bpg: {bpg}, tpb: {tpb}]")    
    t2 = time.time()
    if verbose:
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
    row = bx * tile_size + tx
    col = by * tile_size + ty
    tmp = float64(0.0)
    for k in range(0, N4, tile_size):
        if row < M4 and k + ty < N4:
            shared_A[tx, ty] = A44[row, k + ty]
        else:
            shared_A[tx, ty] = float64(0.0)
        if k + tx < N4 and col < P:
            shared_B[tx, ty] = B4[k + tx, col]
        else:
            shared_B[tx, ty] = float64(0.0)
        cuda.syncthreads()
        for n in range(tile_size):
            tmp += shared_A[tx, n] * shared_B[n, ty]
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

def qmatmul_algo_numba_cuda_float64(A, B, verbose=False):
    print(f"QMATMUL_ALGO_NUMBA_CUDA_FLOAT64...")
    t1 = time.time()
    M, N, _ = A.shape
    P = B.shape[1]
    N4 = N << 2
    M4 = M << 2
    tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
    t1_b4 = time.time()
    dev_B = cuda.to_device(B)
    dev_B4 = cuda.device_array((N4, P), dtype=np.float64)
    tpb = tpb_default
    bpg = (N4 * P + tpb - 1) // tpb
    stack_numba_cuda_job_float64[bpg, tpb](dev_B, dev_B4)
    cuda.synchronize()    
    t2_b4 = time.time()
    if verbose:
        print(f"[time b4: {t2_b4 - t1_b4} s]")
    t1_a4 = time.time()
    dev_A = cuda.to_device(A)
    dev_A4 = cuda.device_array((M4, N), dtype=np.float64)
    bpg = (M4 * N + tpb - 1) // tpb 
    stack_numba_cuda_job_float64[bpg, tpb](dev_A, dev_A4)
    cuda.synchronize()
    t2_a4 = time.time()
    if verbose:
        print(f"[time a4: {t2_a4 - t1_a4} s]")
    t1_h4b4 = time.time()
    dev_H4B4 = cuda.device_array((N4, P), dtype=np.float64)
    tile_size = 8
    bpg_x = (N + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size)
    had4_numba_cuda_job_float64[bpg, tpb](dev_B4, dev_H4B4)
    cuda.synchronize()
    t2_h4b4 = time.time()
    if verbose:
        print(f"[time h4b4: {t2_h4b4 - t1_h4b4} s]")
    t1_h4a4 = time.time()
    dev_H4A4 = cuda.device_array((M4, N), dtype=np.float64)
    tile_size = 8
    bpg_x = (M + tile_size - 1) // tile_size
    bpg_y = (N + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size)
    had4_numba_cuda_job_float64[bpg, tpb](dev_A4, dev_H4A4)
    cuda.synchronize()
    t2_h4a4 = time.time()
    if verbose:
        print(f"[time h4a4: {t2_h4a4 - t1_h4a4} s]")        
    t1_d4 = time.time()
    dev_D4 = cuda.device_array((M4, P), dtype=np.float64)
    tile_size = 8
    bpg_x = (M + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size    
    bpg = (bpg_x, bpg_y, 4)
    tpb = (tile_size, tile_size)
    matmuldiag_numba_cuda_job_float64[bpg, tpb](dev_H4A4, dev_H4B4, 0.25, dev_D4)
    cuda.synchronize()
    t2_d4 = time.time()
    if verbose:
        print(f"[time d4: {t2_d4 - t1_d4} s]")    
    t1_h4d4 = time.time()
    dev_H4D4 = cuda.device_array((M4, P), dtype=np.float64)
    tile_size = 8
    bpg_x = (M + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size)
    had4_numba_cuda_job_float64[bpg, tpb](dev_D4, dev_H4D4)
    cuda.synchronize()
    t2_h4d4 = time.time()
    if verbose:
        print(f"[time h4d4: {t2_h4d4 - t1_h4d4} s]")        
    t1_a4lb4l = time.time()
    dev_A4l = cuda.device_array((M4, N), dtype=np.float64)    
    dev_permutation_a4l = cuda.to_device(np.array([0, 3, 1, 2], dtype=np.int8))
    tile_size = 8
    bpg_x = (M + tile_size - 1) // tile_size
    bpg_y = (N + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size)    
    permute_numba_cuda_job_float64[bpg, tpb](dev_A4, dev_permutation_a4l, dev_A4l)    
    dev_B4l = cuda.device_array((N4, P), dtype=np.float64)
    dev_permutation_b4l = cuda.to_device(np.array([0, 2, 3, 1], dtype=np.int8))
    bpg_x = (N + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tile_size = 8
    permute_numba_cuda_job_float64[bpg, tpb](dev_B4, dev_permutation_b4l, dev_B4l)
    cuda.synchronize()        
    dev_A4lB4l = cuda.device_array((M4, P), dtype=np.float64)
    tile_size = 8
    bpg_x = (M + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size    
    bpg = (bpg_x, bpg_y, 4)
    tpb = (tile_size, tile_size)
    matmuldiag_numba_cuda_job_float64[bpg, tpb](dev_A4l, dev_B4l, 2.0, dev_A4lB4l)
    cuda.synchronize()        
    t2_a4lb4l = time.time()
    if verbose:
        print(f"[time a4lb4l: {t2_a4lb4l - t1_a4lb4l} s]")                
    t1_sub = time.time()
    dev_C4 = cuda.device_array((M4, P), dtype=np.float64)    
    bpg_x = (M4 + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size)    
    matsub_numba_cuda_job_float64[bpg, tpb](dev_H4D4, dev_A4lB4l, dev_C4)    
    cuda.synchronize()
    t2_sub = time.time()
    if verbose:
        print(f"[time sub: {t2_sub - t1_sub} s]")        
    dev_C = cuda.device_array((M, P, 4), dtype=np.float64)
    tpb = tpb_default
    bpg = (M4 * P + tpb - 1) // tpb
    c4_to_c_numba_cuda_job_float64[bpg, tpb](dev_C4, dev_C)
    cuda.synchronize()
    C = dev_C.copy_to_host()    
    t2 = time.time()
    print(f"QMATMUL_ALGO_NUMBA_CUDA_FLOAT64 DONE. [time: {t2 - t1} s]")
    return C

@cuda.jit(void(float64[:, :], float64[:, :]))
def had4_numba_cuda_job_float64(E4, H4E4):    
    shared_E4_0 = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    shared_E4_1 = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    shared_E4_2 = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    shared_E4_3 = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    R4, S = E4.shape
    R = R4 >> 2
    R2 = R << 1
    R3 = R2 + R
    tile_size = cuda.blockDim.x
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = bx * tile_size + tx
    col = by * tile_size + ty
    if row < R and col < S:
        shared_E4_0[tx, ty] = E4[row, col]
        shared_E4_1[tx, ty] = E4[row + R, col]
        shared_E4_2[tx, ty] = E4[row + R2, col]
        shared_E4_3[tx, ty] = E4[row + R3, col] 
    else:
        shared_E4_0[tx, ty] = float64(0.0)
        shared_E4_1[tx, ty] = float64(0.0)
        shared_E4_2[tx, ty] = float64(0.0)
        shared_E4_3[tx, ty] = float64(0.0)
    s0 = shared_E4_0[tx, ty] + shared_E4_1[tx, ty]
    s1 = shared_E4_2[tx, ty] + shared_E4_3[tx, ty]
    d0 = shared_E4_0[tx, ty] - shared_E4_1[tx, ty]
    d1 = shared_E4_2[tx, ty] - shared_E4_3[tx, ty]
    if row < R and col < S:    
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
    row = bx * tile_size + tx
    col = by * tile_size + ty
    tmp = float64(0.0)
    for k in range(0, S, tile_size):
        if row < R and k + ty < S:
            shared_E[tx, ty] = E4[row + bz * R, k + ty]
        else:
            shared_E[tx, ty] = float64(0.0)
        if k + tx < S and col < T:
            shared_F[tx, ty] = F4[k + tx + bz * S, col]
        else:
            shared_F[tx, ty] = float64(0.0)
        cuda.syncthreads()
        for s in range(tile_size):
            tmp += shared_E[tx, s] * shared_F[s, ty]
        cuda.syncthreads()
    if row < R and col < T:
        G4[row + bz * R, col] = factor * tmp

@cuda.jit(void(float64[:, :], int8[:], float64[:, :]))
def permute_numba_cuda_job_float64(S4, permutation, S4_permuted):    
    shared_S4 = cuda.shared.array((16, 16, 4), dtype=float64)
    M4, N = S4.shape
    M = M4 >> 2
    M2 = M << 1
    M3 = M2 + M
    tile_size = cuda.blockDim.x
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = bx * tile_size + tx
    col = by * tile_size + ty
    if row < M and col < N:
        shared_S4[tx, ty, 0] = S4[row, col]
        shared_S4[tx, ty, 1] = S4[row + M, col]
        shared_S4[tx, ty, 2] = S4[row + M2, col]
        shared_S4[tx, ty, 3] = S4[row + M3, col]     
        S4_permuted[row, col] = shared_S4[tx, ty, permutation[0]]
        S4_permuted[row + M, col] = shared_S4[tx, ty, permutation[1]]
        S4_permuted[row + M2, col] = shared_S4[tx, ty, permutation[2]]
        S4_permuted[row + M3, col] = shared_S4[tx, ty, permutation[3]]
        
@cuda.jit(void(float64[:, :], float64[:, :], float64[:, :]))
def matsub_numba_cuda_job_float64(C4_left, C4_right, C4):    
    shared_L = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    shared_R = cuda.shared.array((16, 16), dtype=float64) # assumed max tile size: 16
    tile_size = cuda.blockDim.x
    M4, P = C4.shape
    M = M4 >> 2
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = bx * tile_size + tx
    col = by * tile_size + ty
    if row < M4 and col < P:
        shared_L[tx, ty] = C4_left[row, col]
        shared_R[tx, ty] = C4_right[row, col]
        result = shared_R[tx, ty] - shared_L[tx, ty] if row < M else shared_L[tx, ty] - shared_R[tx, ty]        
        C4[row, col] = result
                
if __name__ == "__main__":
    QMATMUL_NAIVE_NUMBA_ST_FUNCTIONS = {
        np.float64: qmatmul_naive_numba_st_float64, 
        np.float32: qmatmul_naive_numba_st_float32,
        np.int32: qmatmul_naive_numba_st_int32
        }
    QMATMUL_NAIVE_NUMBA_PARALLEL_FUNCTIONS = {
        np.float64: qmatmul_naive_numba_parallel_float64, 
        np.float32: qmatmul_naive_numba_parallel_float32,
        np.int32: qmatmul_naive_numba_parallel_int32
        }
    QMATMUL_DIRECT_NUMBA_CUDA_FUNCTIONS = {
        np.float64: qmatmul_direct_numba_cuda_float64, 
        np.float32: None, # TODO
        np.int32: None # TODO
        }
    QMATMUL_ALGO_NUMBA_CUDA_FUNCTIONS = {
        np.float64: qmatmul_algo_numba_cuda_float64, 
        np.float32: None, # TODO
        np.int32: None # TODO
        }    
    
    # experiment settings
    SEED = 0
    M, N, P = 150, 190, 170
    RANGE = 10
    DTYPE = np.float64
    VERBOSE = False         
    APPROACHES = {
        "QMATMUL_NAIVE_NUMBA_ST": (True, QMATMUL_NAIVE_NUMBA_ST_FUNCTIONS[DTYPE]),
        "QMATMUL_NAIVE_NUMBA_PARALLEL": (True, QMATMUL_NAIVE_NUMBA_PARALLEL_FUNCTIONS[DTYPE]),
        "QMATMUL_DIRECT_NUMPY": (True, qmatmul_direct_numpy),
        "QMATMUL_DIRECT_ALGOLIKE_NUMPY": (False, qmatmul_direct_algolike_numpy),
        "QMATMUL_ALGO_NUMPY": (True, qmatmul_algo_numpy),
        "QMATMUL_DIRECT_NUMBA_CUDA": (True, QMATMUL_DIRECT_NUMBA_CUDA_FUNCTIONS[DTYPE]),
        "QMATMUL_ALGO_NUMBA_CUDA": (True, QMATMUL_ALGO_NUMBA_CUDA_FUNCTIONS[DTYPE])        
        }
    
    print(f"QUATERNIONS MAIN... [M: {M}, N: {N}, P: {P}, SEED: {SEED}, RANGE: {RANGE}, DTYPE: {DTYPE}, M x N x P: {M * N * P:.2e}, NUMPY_SINGLE_THREAD: {NUMPY_SINGLE_THREAD}]")            
    np.random.seed(SEED)
    A = qmatrand(M, N, -RANGE, RANGE, DTYPE)
    B = qmatrand(N, P, -RANGE, RANGE, DTYPE)  
    C_ref = None

    # experiment to go  
    for index, (approach_name, (approach_on, approach_function)) in enumerate(APPROACHES.items()):
        if approach_on:
            print(f"APPROACH {index + 1}: {approach_name}...", flush=True) 
            t1 = time.time()
            C = approach_function(A, B)
            t2 = time.time()    
            if C_ref is None:
                C_ref = C
            extra_info = "" if C_ref is None else f", all close: {np.allclose(C, C_ref, atol=1e-2, rtol=1e-3)}, d_inf: {np.max(np.abs(C - C_ref))}"
            if VERBOSE:
                print(f"C:\n {C}")        
            print(f"APPROACH {index + 1}: {approach_name} DONE. [time: {t2 - t1} s{extra_info}]", flush=True)            
        else:
            print(f"APPROACH {index + 1}: {approach_name} OFF.")                    
        
    print("QUATERNIONS MAIN DONE.")