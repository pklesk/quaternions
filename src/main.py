import numpy as np
import quaternion as quat
import time
from numba import jit, prange, cuda
from numba import void, float64
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

q_signs = np.array([
    [1, -1,  -1, -1],
    [1,  1, -1,  1],
    [1,  1, 1, -1],
    [1, -1,  1,  1]
], dtype=np.int8)
q_parts = np.array([
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
], dtype=np.int8)

def qrand(m, n, range_min, range_max, float64_elements=True):
    A = np.random.randint(range_min, range_max + 1, size=(m * n, 4)).astype(np.float64) if not float64_elements else np.random.rand(m * n, 4) * (range_max - range_min) + range_min
    A_quat = quat.as_quat_array(A).reshape(m, n)
    A_numpy = A.reshape(m, n, 4)
    return A_quat, A_numpy


def qmul_mat(A, B):
    m, n = A.shape    
    if B.shape[0] != n:
        raise Exception("Inconsistent shapes of arguments for multiplication")    
    k = B.shape[1]
    C = np.zeros((m, k), dtype=np.quaternion)
    for i in range(m):
        for j in range(k):
            for z in range(n):
                C[i, j] += A[i, z] * B[z, j]
    return C

def qeye(m):
    temp = np.zeros((m * m, 4))
    temp[:, 0] = 1.0
    return quat.as_quat_array(temp).reshape(m, m)

def part(A, i):    
    return quat.as_float_array(A)[:, :, i]

def a4(A):
    m, n = A.shape
    m4 = 4 * m
    n4 = 4 * n
    f = np.empty((m4, n4))
    m2 = 2 * m
    m3 = 3 * m
    n2 = 2 * n
    n3 = 3 * n        
    f[:m, :n] = part(A, 0)
    f[:m, n:n2] = -part(A, 1)
    f[:m, n2:n3] = -part(A, 2)
    f[:m, n3:] = -part(A, 3)
    f[m:m2, :n] = part(A, 1)
    f[m:m2, n:n2] = part(A, 0)
    f[m:m2, n2:n3] = -part(A, 3)
    f[m:m2, n3:] = part(A, 2)    
    f[m2: m3, :n] = part(A, 2)
    f[m2:m3, n:n2] = part(A, 3)
    f[m2:m3, n2:n3] = part(A, 0)
    f[m2:m3, n3:] = -part(A, 1)    
    f[m3:, :n] = part(A, 3)
    f[m3:, n:n2] = -part(A, 2)
    f[m3:, n2:n3] = part(A, 1)
    f[m3:, n3:n4] = part(A, 0)
    return f

def a4_pure(A):
    m, n = A.shape
    m4 = 4 * m
    n4 = 4 * n
    f = np.empty((m4, n4))
    m2 = 2 * m
    m3 = 3 * m
    n2 = 2 * n
    n3 = 3 * n        
    f[:m, :n] = part(A, 0)
    f[:m, n:n2] = part(A, 1)
    f[:m, n2:n3] = part(A, 2)
    f[:m, n3:] = part(A, 3)
    f[m:m2, :n] = part(A, 1)
    f[m:m2, n:n2] = part(A, 0)
    f[m:m2, n2:n3] = part(A, 3)
    f[m:m2, n3:] = part(A, 2)    
    f[m2: m3, :n] = part(A, 2)
    f[m2:m3, n:n2] = part(A, 3)
    f[m2:m3, n2:n3] = part(A, 0)
    f[m2:m3, n3:] = part(A, 1)    
    f[m3:, :n] = part(A, 3)
    f[m3:, n:n2] = part(A, 2)
    f[m3:, n2:n3] = part(A, 1)
    f[m3:, n3:n4] = part(A, 0)
    return f

def a4_pure_n(A):
    m, n = A.shape
    m4 = 4 * m
    f = np.empty((m4, n))
    m2 = 2 * m
    m3 = 3 * m        
    f[:m] = part(A, 0)
    f[m:m2] = part(A, 1)
    f[m2:m3] = part(A, 2)
    f[m3:] = part(A, 3)
    return f

def q4(A):
    m, n = A.shape
    m4 = 4 * m
    n4 = 4 * n
    f = np.zeros((m4, n4))
    f[:m, :n] = part(A, 0)
    f[m:2 * m, 2 * n:3 * n] = part(A, 3)
    f[2 * m: 3 * m, 3 * n:] = part(A, 1)
    f[3 * m:, n:2 * n] = part(A, 2)
    return f    

def a4_hat_from_a(A):
    m, n = A.shape
    m4 = 4 * m
    n4 = 4 * n
    f = np.empty((m4, n4))
    m2 = 2 * m
    m3 = 3 * m
    n2 = 2 * n
    n3 = 3 * n        
    f[:m, :n] = -part(A, 0)
    f[:m, n:n2] = part(A, 1)
    f[:m, n2:n3] = part(A, 2)
    f[:m, n3:] = part(A, 3)
    f[m:m2, :n] = part(A, 1)
    f[m:m2, n:n2] = part(A, 0)
    f[m:m2, n2:n3] = -part(A, 3)
    f[m:m2, n3:] = part(A, 2)    
    f[m2: m3, :n] = part(A, 2)
    f[m2:m3, n:n2] = part(A, 3)
    f[m2:m3, n2:n3] = part(A, 0)
    f[m2:m3, n3:] = -part(A, 1)    
    f[m3:, :n] = part(A, 3)
    f[m3:, n:n2] = -part(A, 2)
    f[m3:, n2:n3] = part(A, 1)
    f[m3:, n3:n4] = part(A, 0)
    return f

def b4(B):
    n, k = B.shape
    n4 = 4 * n
    f = np.empty((n4, k))
    n2 = 2 * n
    n3 = 3 * n        
    f[:n] = part(B, 0)
    f[n:n2] = part(B, 1)
    f[n2:n3] = part(B, 2)
    f[n3:n4] = part(B, 3)
    return f  

def c4_to_c(C4, m, k):
    return quat.as_quat_array(C4.T.reshape(k * 4, m).T.reshape(m * k, 4)).reshape(m, k)   

def c4_to_c_simple(C4, m, k):
    C4_new = np.empty((m, k, 4))
    C4_new[:, :, 0] = C4[:m]
    C4_new[:, :, 1] = C4[m: 2 * m]
    C4_new[:, :, 2] = C4[2 * m:3 * m]
    C4_new[:, :, 3] = C4[3 * m:]
    return numpy_to_quat(C4_new)

def a4_hat_from_a4(A4):
    m4 = A4.shape[0]
    m = m4 // 4
    A4_hat = np.copy(A4)
    A4_hat[:m] = -np.eye(m).dot(A4[:m])
    return A4_hat

def i4_hat(m):
    i4_hat = np.eye(4 * m)
    i4_hat[np.arange(m), np.arange(m)] = -1.0
    return i4_hat


@jit(float64[:](float64[:], float64[:]), nopython=True, cache=True)
def qmul_numba_float64(q1, q2):
    result = np.zeros(4, dtype=np.float64)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return result  

@jit(float64[:, :, :](float64[:, :, :], float64[:, :, :]), nopython=True, cache=True, parallel=True)
def qmul_mat_numba_float64(A, B):
    m, n, _ = A.shape    
    k = B.shape[1]
    C = np.zeros((m, k, 4), dtype=np.float64)
    for i in prange(m):
        for j in range(k):
            for z in range(n):
                C[i, j] += qmul_numba_float64(A[i, z], B[z, j])
    return C

def a_dot_b_full_numba_cuda_float64(A_numpy, B_numpy, result_as_c4=True):
    print(f"A_DOT_B_FULL_NUMBA_CUDA_FLOAT64...")
    t1 = time.time()
    M, N, _ = A_numpy.shape
    P = B_numpy.shape[1]
    N4 = N << 2
    M4 = M << 2
    t1_b4 = time.time()
    dev_B_numpy = cuda.to_device(B_numpy)
    dev_B4 = cuda.device_array((N4, P), dtype=np.float64)
    tpb = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
    bpg = (N4 * P + tpb - 1) // tpb
    b4_numba_cuda_job_float64[bpg, tpb](dev_B_numpy, dev_B4)
    cuda.synchronize()    
    t2_b4 = time.time()
    print(f"[time b4: {t2_b4 - t1_b4} s]")
    t1_a4 = time.time()
    dev_A_numpy = cuda.to_device(A_numpy)
    dev_A4 = cuda.device_array((M4, N4), dtype=np.float64)
    mn_bpg = (M * N + tpb - 1) // tpb 
    bpg = (mn_bpg, 4, 4)
    a4_numba_cuda_job_float64[bpg, tpb](dev_A_numpy, dev_A4)
    cuda.synchronize()
    t2_a4 = time.time()
    print(f"[time a4: {t2_a4 - t1_a4} s]")
    # input("c4")
    t1_c4 = time.time()    
    dev_C4 = cuda.device_array((M4, k), dtype=np.float64)
    # print(f"A4 size [GB]: {dev_A4.nbytes / 1024**3}")
    # print(f"B4 size [GB]: {dev_B4.nbytes / 1024**3}")
    # print(f"C4 size [GB]: {dev_C4.nbytes / 1024**3}")     
    # bpg = (M4, P)
    # a4_dot_b4_numba_cuda_job_float64[bpg, tpb](dev_A4, dev_B4, dev_C4)
    tile_size = 8
    bpg_x = (M4 + tile_size - 1) // tile_size
    bpg_y = (P + tile_size - 1) // tile_size
    bpg = (bpg_x, bpg_y)
    tpb = (tile_size, tile_size) 
    a4_dot_b4_v2_numba_cuda_job_float64[bpg, tpb](dev_A4, dev_B4, dev_C4)    
    cuda.synchronize()
    t2_c4 = time.time()
    print(f"[time c4: {t2_c4 - t1_c4} s, bpg: {bpg}, tpb: {tpb}]")    
    C_result = np.empty((M4, P), dtype=np.float64)
    if result_as_c4:
        dev_C4.copy_to_host(ary=C_result)
    else:
        dev_C_numpy = cuda.device_array((M, P, 4), dtype=np.float64)
        bpg = (M4 * P + tpb - 1) // tpb
        c4_to_c_numpy_numba_cuda_job_float64[bpg, tpb](dev_C4, dev_C_numpy)
        cuda.synchronize()
        C_result = dev_C_numpy.copy_to_host()
    t2 = time.time()
    print(f"A_DOT_B_FULL_NUMBA_CUDA_FLOAT64 DONE. [time: {t2 - t1} s]")
    return C_result
         
@cuda.jit(void(float64[:, :, :], float64[:, :]))
def b4_numba_cuda_job_float64(B_numpy, B4):
    N, P, _ = B_numpy.shape
    i_global = cuda.grid(1)
    i_np, i_im = i_global // 4, i_global % 4    
    if i_np < N * P:
        n, p = i_np // P, i_np % P
        B4[i_im * N + n, p] = B_numpy[n, p, i_im]

@cuda.jit(void(float64[:, :, :], float64[:, :]))
def a4_numba_cuda_job_float64(A_numpy, A4):
    const_q_signs = cuda.const.array_like(q_signs)
    const_q_parts = cuda.const.array_like(q_parts)
    M, N, _ = A_numpy.shape
    i_mn, mat_row, mat_col = cuda.grid(3)        
    if i_mn < M * N:
        m, n = i_mn // N, i_mn % N
        A4[mat_row * M + m, mat_col * N + n] = const_q_signs[mat_row, mat_col] * A_numpy[m, n, const_q_parts[mat_row, mat_col]]

@cuda.jit(void(float64[:, :], float64[:, :], float64[:, :]))
def a4_dot_b4_numba_cuda_job_float64(A4, B4, C4):
    elements = cuda.shared.array(512, dtype=float64) # assumed max corresponding to max tpb
    N4 = B4.shape[0]    
    m4 = cuda.blockIdx.x
    p = cuda.blockIdx.y
    tpb = cuda.blockDim.x
    t = cuda.threadIdx.x
    ept = (N4 + tpb - 1) // tpb
    elements[t] = float64(0.0)        
    cuda.syncthreads()
    e = t
    for _ in range(ept):
        if e < N4:
            elements[t] += A4[m4, e] * B4[e, p] 
        e += tpb
    cuda.syncthreads()
    stride = tpb >> 1
    while stride > 0:
        if t < stride:
            elements[t] += elements[t + stride]
        cuda.syncthreads()
        stride >>= 1
    if t == 0:
        C4[m4, p] = elements[0]
        
@cuda.jit(void(float64[:, :], float64[:, :], float64[:, :]))
def a4_dot_b4_v2_numba_cuda_job_float64(A4, B4, C4):    
    shared_A = cuda.shared.array((16, 16), dtype=float64)
    shared_B = cuda.shared.array((16, 16), dtype=float64)
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
            shared_A[tx, ty] = A4[row, k + ty]
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
def c4_to_c_numpy_numba_cuda_job_float64(C4, C_numpy):
    M, P, _ = C_numpy.shape
    i_global = cuda.grid(1)
    i_mp, i_im = i_global // 4, i_global % 4    
    if i_mp < M * P:
        m, p = i_mp // P, i_mp % P
        C_numpy[m, p, i_im] = C4[i_im * M + m, p] 
            
def numpy_to_quat(A):
    m, n, _ = A.shape
    return quat.as_quat_array(A.reshape(m * n, 4)).reshape(m, n)

if __name__ == "__main__":    
    SEED = 0
    m, n, k = 2000, 8000, 1000
    RANGE = 5        
    VERBOSE = False
    FLOAT64_ELEMENTS = True
    
    print(f"QUATERNIONS MAIN... [m: {m}, n: {n}, k: {k}, seed: {SEED}, range: {RANGE}, mnk: {1.0 * m * n * k:.2e}]")    
            
    np.random.seed(SEED)
    A, A_numpy = qrand(m, n, -RANGE, RANGE, FLOAT64_ELEMENTS)
    B, B_numpy = qrand(n, k, -RANGE, RANGE, FLOAT64_ELEMENTS)  

    # t1 = time.time()
    # C = qmul_mat(A, B)
    # t2 = time.time()
    # print(f"MULTIPLICATION 'NAIVE, PURE PYTHON + QUATERNION MODULE' [time: {t2 - t1} s]")    
    # if VERBOSE:
    #     print(f"A of shape {A.shape}:\n {A}")
    #     print(f"B of shape {B.shape}:\n {B}")
    #     print(f"C of shape {C.shape}:\n {C}")        
        
    # t1 = time.time()
    # C_numba = qmul_mat_numba_float64(A_numpy, B_numpy)
    # t2 = time.time()    
    # C = numpy_to_quat(C_numba)
    # print(f"MULTIPLICATION 'NAIVE, NUMBA' [time: {t2 - t1} s, all close: {np.allclose(C, numpy_to_quat(C_numba))}, d_inf: {np.max(np.abs(C - numpy_to_quat(C_numba)))}]")    
    # if VERBOSE:
    #     print(f"C_numba of shape {C_numba.shape}:\n {C_numba}")        
            
    t1 = time.time()    
    C4_v1 = a4(A).dot(b4(B))
    C_v1 = c4_to_c(C4_v1, m, k)
    t2 = time.time()
    C = C_v1
    print(f"MULTIPLICATION 'VIA FORMULA (2), NUMPY' [time: {t2 - t1} s, all close: {np.allclose(C, C_v1)}, d_inf: {np.max(np.abs(C - C_v1))}]")
    if VERBOSE:
        print(f"C4_v1:\n {C4_v1}")    
        print(f"C_v1:\n {C_v1}")    
    
    # A4 = a4(A)    
    # A4_hat_from_a = a4_hat_from_a(A)
    # A4_hat_from_a4 = a4_hat_from_a4(A4)        
    
    #t1 = time.time()
    # I4_hat = i4_hat(m)
    # Q4 = q4(A)
    # B4 = b4(B)        
    
    ## A4_hat_from_q4 = a4_pure(A) - 2 * Q4             
    ## C4_v2 = np.dot(np.dot(I4_hat, A4_hat_from_q4), B4)
    #A4_pure = a4_pure(A)    
    #C4_v2 = np.dot(I4_hat, np.dot(A4_pure, B4) - (2 * Q4).dot(B4))        
    
    #C_v2 = c4_to_c(C4_v2, m, k)
    #t2 = time.time()
    #print(f"MULTIPLICATION 'VIA FORMULA (3), NUMPY' [time: {t2 - t1} s, all close: {np.allclose(C, C_v2)}, d_inf: {np.max(np.abs(C - C_v2))}]")    
    #if VERBOSE:    
    #    print(f"C4_v2:\n {C4_v2}")        
    #    print(f"C_v2:\n {C_v2}")
    
    t1 = time.time()
    m2 = 2 * m
    m3 = 3 * m
    m4 = 4 * m
    n2 = 2 * n
    n3 = 3 * n
    n4 = 4 * n    
    
    Q4 = q4(A)
    B4 = b4(B)
    A4_pure_n = a4_pure_n(A)                  
    H2 = np.array([[1., 1.], [1., -1.]])
    H4 = np.kron(H2, H2)
    
    # H4En = np.kron(H4, np.eye(n))
    # H4B = H4En.dot(B4)
    H4B2 = np.empty_like(B4)
    B4s1 = B4[:n] + B4[n:n2]
    B4s2 = B4[n2:n3] + B4[n3:n4]
    B4d1 = B4[:n] - B4[n:n2]
    B4d2 = B4[n2:n3] - B4[n3:n4]
    H4B2[:n] = B4s1 + B4s2
    H4B2[n:n2] = B4d1 + B4d2
    H4B2[n2:n3] = B4s1 - B4s2
    H4B2[n3:] = B4d1 - B4d2
    H4B = H4B2
    
    # H4Em = np.kron(H4, np.eye(m))
    # H4A = H4Em.dot(A4_pure_n)
    H4A2 = np.empty_like(A4_pure_n)
    A4s1 = A4_pure_n[:m] + A4_pure_n[m:m2]
    A4s2 = A4_pure_n[m2:m3] + A4_pure_n[m3:]
    A4d1 = A4_pure_n[:m] - A4_pure_n[m:m2]
    A4d2 = A4_pure_n[m2:m3] - A4_pure_n[m3:]
    H4A2[:m] = A4s1 + A4s2
    H4A2[m:m2] = A4d1 + A4d2
    H4A2[m2:m3] = A4s1 - A4s2
    H4A2[m3:] = A4d1 - A4d2
    H4A = H4A2
    
    # D4 = np.zeros((m4, n4), dtype=B4.dtype)
    # D4[:m, :n] = H4A[:m]
    # D4[m:m2, n:n2] = H4A[m:m2]
    # D4[m2:m3, n2:n3] = H4A[m2:m3]
    # D4[m3:, n3:] = H4A[m3:]
    # C4_v3 = np.dot(I4_hat, np.dot(H4Em, np.dot(0.25 * D4, H4B)) - (2.0 * Q4).dot(B4))

    D4H4B = np.zeros((m4, k), dtype=B4.dtype)
    D4H4B[:m] = (0.25 * H4A[:m]).dot(H4B[:n])
    D4H4B[m: m2] = (0.25 * H4A[m:m2]).dot(H4B[n:n2])
    D4H4B[m2: m3] = (0.25 * H4A[m2:m3]).dot(H4B[n2:n3])
    D4H4B[m3:] = (0.25 * H4A[m3:]).dot(H4B[n3:])    
    Q4B = np.zeros((m4, k), dtype=B4.dtype)
    Q4B[:m] = (2.0 * Q4[:m, :n]).dot(B4[:n])
    Q4B[m: m2] = (2.0 * Q4[m:m2, n2:n3]).dot(B4[n2:n3])
    Q4B[m2: m3] = (2.0 * Q4[m2:m3, n3:]).dot(B4[n3:])
    Q4B[m3:] = (2.0 * Q4[m3:, n:n2]).dot(B4[n:n2])    
    
    # H4D4H4B = np.dot(H4Em, D4H4B)    
    H4D4H4B2 = np.empty_like(D4H4B) 
    D4H4Bs1 = D4H4B[:m] + D4H4B[m:m2]
    D4H4Bs2 = D4H4B[m2:m3] + D4H4B[m3:]
    D4H4Bd1 = D4H4B[:m] - D4H4B[m:m2]
    D4H4Bd2 = D4H4B[m2:m3] - D4H4B[m3:]
    H4D4H4B2[:m] = D4H4Bs1 + D4H4Bs2
    H4D4H4B2[m:m2] = D4H4Bd1 + D4H4Bd2
    H4D4H4B2[m2:m3] = D4H4Bs1 - D4H4Bs2
    H4D4H4B2[m3:] = D4H4Bd1 - D4H4Bd2
    H4D4H4B = H4D4H4B2
    
    # I4_hat = i4_hat(m)
    # C4_v3 = np.dot(I4_hat, H4D4H4B - Q4B)
    
    C4_v3 = H4D4H4B - Q4B    
    C4_v3[:m] = -C4_v3[:m]
     
    C_v3 = c4_to_c(C4_v3, m, k)
    t2 = time.time()
    print(f"MULTIPLICATION 'VIA FORMULA (3*), NUMPY' [time: {t2 - t1} s, all close: {np.allclose(C, C_v3)}, d_inf: {np.max(np.abs(C - C_v3))}]")
    if VERBOSE:    
        print(f"C4_v3:\n {C4_v3}")        
        print(f"C_v3:\n {C_v3}")
        
    # input("[press key]")
    t1nc = time.time()
    C4_v4 = a_dot_b_full_numba_cuda_float64(A_numpy, B_numpy)
    C_v4 = c4_to_c(C4_v4, m, k)
    t2nc = time.time()
    if VERBOSE:    
        print(f"C4_v4:\n {C4_v4}")        
        print(f"C_v4:\n {C_v4}")
    print(f"MULTIPLICATION 'VIA FORMULA (2), NUMBA CUDA' [time: {t2nc - t1nc} s, all close: {np.allclose(C, C_v4)}, d_inf: {np.max(np.abs(C - C_v4))}]]")
    
    print("QUATERNIONS MAIN DONE.")
