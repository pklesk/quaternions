from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
import os
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
import numpy as np
import time
from utils import cpu_and_system_props, gpu_props, dict_to_str, Logger, experiment_hash_str
import sys
import qmatmul as qmm
from qmatmul import (
    qmatrand,
    qmatmul_naive_numba_st_float64, 
    qmatmul_naive_numba_st_float32, 
    qmatmul_naive_numba_parallel_float64,
    qmatmul_naive_numba_parallel_float32,
    qmatmul_direct_numpy_st,    
    qmatmul_direct_numpy_parallel, 
    qmatmul_algo_numpy_st,
    qmatmul_algo_numpy_parallel,   
    qmatmul_direct_numba_cuda_float64,    
    qmatmul_direct_numba_cuda_float32, 
    qmatmul_algo_numba_cuda_float64,
    qmatmul_algo_numba_cuda_float32)

__author__ = ["Przemysław Klęsk", "Aleksandr Cariow"]
__email__ = ["pklesk@zut.edu.pl", "alexandr.tariov@zut.edu.pl"]

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

# --------------------------------------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    t1_main = time.time()
         
    # experiment settings
    M, N, P = 1000, 1000, 1000
    SEED = 0    
    RANGE = 2.0
    DTYPE = np.float32 # {np.float32, np.float64}
    REPETITIONS = 10
    VERBOSE = False
    APPROACHES = {
        "QMATMUL_NAIVE_NUMBA_ST": (False, QMATMUL_NAIVE_NUMBA_ST_FUNCTIONS[DTYPE], {"verbose": False}),
        "QMATMUL_NAIVE_NUMBA_PARALLEL": (True, QMATMUL_NAIVE_NUMBA_PARALLEL_FUNCTIONS[DTYPE], {"verbose": False}),
        "QMATMUL_DIRECT_NUMPY_ST": (True, qmatmul_direct_numpy_st, {"verbose": False}),
        "QMATMUL_DIRECT_NUMPY_PARALLEL": (True, qmatmul_direct_numpy_parallel, {"verbose": False}),
        "QMATMUL_DIRECT_NUMBA_CUDA": (True, QMATMUL_DIRECT_NUMBA_CUDA_FUNCTIONS[DTYPE], {"tile_size": qmm.DEFAULT_TILE_SIZE, "verbose": False}),        
        "QMATMUL_ALGO_NUMPY_ST": (True, qmatmul_algo_numpy_st, {"verbose": False}),
        "QMATMUL_ALGO_NUMPY_PARALLEL": (True, qmatmul_algo_numpy_parallel, {"verbose": False}),        
        "QMATMUL_ALGO_NUMBA_CUDA": (True, QMATMUL_ALGO_NUMBA_CUDA_FUNCTIONS[DTYPE], {"tile_size": qmm.DEFAULT_TILE_SIZE, "verbose": False})        
        }
    APPROACHES_INFO = {key:  (APPROACHES[key][0], APPROACHES[key][1].__name__) for key in APPROACHES.keys()}
    experiment_info = {"M": M, "N": N, "P": P, "SEED": SEED, "RANGE": RANGE, "DTYPE": DTYPE, "REPETITIONS": REPETITIONS, **APPROACHES_INFO}    
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
    print(f"A: {A.nbytes / 1024**2:.3f} MiB") 
    print(f"B: {B.nbytes / 1024**2:.3f} MiB")
    print(f"C: {np.empty((M, P, 4), dtype=DTYPE).nbytes / 1024**2:.2f} MiB") 
    print(LINE_SEPARATOR)
    
    # warm-up
    A_warmup = qmatrand(100, 100, -RANGE, RANGE, DTYPE)
    B_warmup = qmatrand(100, 100, -RANGE, RANGE, DTYPE)      
    print("WARM-UP:")
    for index, (approach_name, (approach_on, approach_function, approach_extra_args)) in enumerate(APPROACHES.items()):
        reference_info = ""
        if approach_on:
            print(f"WARM-UP FOR APPROACH {index + 1}: {approach_name}...", flush=True) 
            t1 = time.time()
            C_warmup = approach_function(A_warmup, B_warmup, **approach_extra_args)                
            t2 = time.time()
            t2_t1 = t2 - t1
            if t2_t1 == 0.0:
                t2_t1 = 1e-10 # epsilon: 0.1 ns
            print(f"WARM-UP FOR APPROACH {index + 1}: {approach_name} DONE. [time: {t2_t1} s]", flush=True)            
        else:
            print(f"WARM-UP APPROACH {index + 1}: {approach_name} OFF.")
    print(LINE_SEPARATOR)
    
    # experiment to be executed
    times = {}
    reference_approach_name = None
    for r in range(REPETITIONS):
        print(f"REPETITION: {r + 1}/{REPETITIONS}:", flush=True)
        for index, (approach_name, (approach_on, approach_function, approach_extra_args)) in enumerate(APPROACHES.items()):
            reference_info = ""
            if approach_on:
                print(f"APPROACH {index + 1}: {approach_name}...", flush=True) 
                t1 = time.time()
                C = approach_function(A, B, **approach_extra_args)                
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
                extra_info = "" if C_ref is None else f", all close: {np.allclose(C, C_ref, rtol=1e-5, atol=1e-2)}, speed-up vs reference: {time_ref / t2_t1:.2f}"
                if VERBOSE:
                    print(f"C:\n {C}", flush=True)        
                print(f"APPROACH {index + 1}: {approach_name} DONE. [time: {t2_t1} s{extra_info}{reference_info}]", flush=True)            
            else:
                print(f"APPROACH {index + 1}: {approach_name} OFF.", flush=True)
    print(LINE_SEPARATOR)
    print("SUMMARY:")
    reference_mean_time = np.mean(times[reference_approach_name]) 
    for index, (approach_name, (approach_on, approach_function, approach_extra_args)) in enumerate(APPROACHES.items()):
        if approach_on:
            reference_info = " (REFERENCE)" if approach_name == reference_approach_name else ""
            time_mean = np.mean(times[approach_name])
            time_std = np.std(times[approach_name])
            speedup = reference_mean_time / time_mean 
            print(f"APPROACH {index + 1}: {approach_name}{reference_info} -> MEAN TIME: {time_mean} s, TIME STD: {time_std} s, STD_%: {(time_std / time_mean) * 100:.1f}%, SPEED-UP: {speedup:.2f}", flush=True)
        else:
            print(f"APPROACH {index + 1}: {approach_name} OFF.")            
    print(LINE_SEPARATOR)
    t2_main = time.time()
                
    print(f"QUATERNIONS MAIN DONE. [hash string: {experiment_hs}, time: {t2_main - t1_main} s]")    
    sys.stdout = sys.__stdout__
    logger.logfile.close()
