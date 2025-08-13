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
from utils import cpu_and_system_props, gpu_props, dict_to_str, Logger, experiment_hash_str
import sys
from qmatmul import (
    qmatmul_naive_numba_st_float64, 
    qmatmul_naive_numba_st_float32, 
    qmatmul_naive_numba_parallel_float64,
    qmatmul_naive_numba_parallel_float32,
    qmatmul_direct_numba_cuda_float64,
    qmatmul_direct_numba_cuda_float32,
    qmatmul_algo_numba_cuda_float64,
    qmatmul_algo_numba_cuda_float32,
    qmatmul_direct_numpy,
    qmatmul_algo_numpy)

# global settings                
FOLDER_EXPERIMENTS = "../experiments/"
LINE_SEPARATOR = 208 * "="                
QMATMUL_NAIVE_NUMBA_ST_FUNCTIONS = {
    np.float64: qmatmul_naive_numba_st_float64, 
    np.float32: qmatmul_naive_numba_st_float32
    }
QMATMUL_NAIVE_NUMBA_PARALLEL_FUNCTIONS = {
    np.float64: qmatmul_naive_numba_parallel_float64, 
    np.float32: qmatmul_naive_numba_parallel_float32
    }
QMATMUL_DIRECT_NUMBA_CUDA_FUNCTIONS = {
    np.float64: qmatmul_direct_numba_cuda_float64, 
    np.float32: qmatmul_direct_numba_cuda_float32
    }
QMATMUL_ALGO_NUMBA_CUDA_FUNCTIONS = {
    np.float64: qmatmul_algo_numba_cuda_float64, 
    np.float32: qmatmul_algo_numba_cuda_float32
    }

def qmatrand(M, N, range_min, range_max, dtype=np.float32, rounding=False):
    A = (np.random.rand(M, N, 4) * (range_max - range_min) + range_min).astype(dtype)
    if rounding:
        A = np.round(A)
    return A

# --------------------------------------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    t1_main = time.time()
         
    # experiment settings
    M, N, P = 100, 300, 200
    SEED = 0    
    RANGE = 10
    DTYPE = np.float32 # {np.float64, np.float32} 
    REPETITIONS = 3
    VERBOSE = False     
    APPROACHES = {
        "QMATMUL_NAIVE_NUMBA_ST": (True, QMATMUL_NAIVE_NUMBA_ST_FUNCTIONS[DTYPE]),
        "QMATMUL_NAIVE_NUMBA_PARALLEL": (True, QMATMUL_NAIVE_NUMBA_PARALLEL_FUNCTIONS[DTYPE]),
        "QMATMUL_DIRECT_NUMPY": (True, qmatmul_direct_numpy),
        "QMATMUL_ALGO_NUMPY": (True, qmatmul_algo_numpy),
        "QMATMUL_DIRECT_NUMBA_CUDA": (True, QMATMUL_DIRECT_NUMBA_CUDA_FUNCTIONS[DTYPE]),
        "QMATMUL_ALGO_NUMBA_CUDA": (True, QMATMUL_ALGO_NUMBA_CUDA_FUNCTIONS[DTYPE])        
        }
    APPROACHES_INFO = {key:  (APPROACHES[key][0], APPROACHES[key][1].__name__) for key in APPROACHES.keys()}
    experiment_info = {"M": M, "N": N, "P": P, "SEED": SEED, "RANGE": RANGE, "DTYPE": DTYPE, "REPETITIONS": REPETITIONS, "NUMPY_SINGLE_THREAD": NUMPY_SINGLE_THREAD, **APPROACHES_INFO}    
    c_props = cpu_and_system_props()
    g_props = gpu_props()
    experiment_hs = experiment_hash_str(experiment_info, c_props, g_props)    
    logger = Logger(f"{FOLDER_EXPERIMENTS}{experiment_hs}.log")    
    sys.stdout = logger
    
    # general info
    print(f"QUATERNIONS MAIN...")    
    print(f"HASH STRING: {experiment_hs}")
    print(LINE_SEPARATOR)
    print(f"EXPERIMENT INFO:\n{dict_to_str(experiment_info)}")
    print(LINE_SEPARATOR)
    print(f"CPU AND SYSTEM PROPS:\n{dict_to_str(c_props)}")
    print(f"GPU PROPS:\n{dict_to_str(g_props)}")
    print(LINE_SEPARATOR)         
    np.random.seed(SEED)
    A = qmatrand(M, N, -RANGE, RANGE, DTYPE)
    B = qmatrand(N, P, -RANGE, RANGE, DTYPE)  
    C_ref = None
    time_ref = None

    # memory info
    print("MEMORY INFO:")
    print(f"A: {A.nbytes / 1024**2:.3f} MB") 
    print(f"B: {B.nbytes / 1024**2:.3f} MB")
    print(f"C: {np.empty((M, P, 4), dtype=DTYPE).nbytes / 1024**2:.3f} MB") 
    print(LINE_SEPARATOR)
    
    # experiment to go
    times = {}
    reference_approach_name = None
    for r in range(REPETITIONS):
        print(f"REPETITION: {r + 1}/{REPETITIONS}:")
        for index, (approach_name, (approach_on, approach_function)) in enumerate(APPROACHES.items()):
            reference_info = ""
            if approach_on:
                print(f"APPROACH {index + 1}: {approach_name}...", flush=True) 
                t1 = time.time()
                C = approach_function(A, B)
                t2 = time.time()
                t2_t1 = t2 - t1
                if t2_t1 == 0.0:
                    t2_t1 = 1e-10 # epsilon: 0.1 ns
                if approach_name not in times:
                    times[approach_name] = []
                times[approach_name].append(t2_t1)
                if C_ref is None:
                    C_ref = C
                    time_ref = t2_t1
                    reference_approach_name = approach_name
                    reference_info = ", reference"
                extra_info = "" if C_ref is None else f", all close: {np.allclose(C, C_ref, atol=1e-1, rtol=1e-3)}, d_inf: {np.max(np.abs(C - C_ref))}, speed-up vs reference: {time_ref / t2_t1:.2f}"
                if VERBOSE:
                    print(f"C:\n {C}")        
                print(f"APPROACH {index + 1}: {approach_name} DONE. [time: {t2_t1} s{extra_info}{reference_info}]", flush=True)            
            else:
                print(f"APPROACH {index + 1}: {approach_name} OFF.")    
    print(LINE_SEPARATOR)
    print("SUMMARY:")
    reference_mean_time = np.mean(times[reference_approach_name]) 
    for index, (approach_name, (approach_on, approach_function)) in enumerate(APPROACHES.items()):
        if approach_on:
            reference_info = " (REFERENCE)" if approach_name == reference_approach_name else ""
            time_mean = np.mean(times[approach_name])
            time_std = np.std(times[approach_name])
            speedup = reference_mean_time / time_mean 
            print(f"APPROACH {index + 1}: {approach_name}{reference_info} -> MEAN TIME: {time_mean}, STD: {time_std}, SPEED-UP: {speedup:.2f}", flush=True)
        else:
            print(f"APPROACH {index + 1}: {approach_name} OFF.")            
    print(LINE_SEPARATOR)
    t2_main = time.time()
            
    print(f"QUATERNIONS MAIN DONE. [time: {t2_main - t1_main} s]")    
    sys.stdout = sys.__stdout__
    logger.logfile.close()