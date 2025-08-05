import cpuinfo
import platform
import psutil
from numba import cuda
import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
 
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

def dict_to_str(d, indent=0):
    """Returns a vertically formatted string representation of a dictionary."""
    indent_str = indent * " "
    dict_str = indent_str + "{"
    for i, key in enumerate(d):
        dict_str += "\n" + indent_str + "  "  + str(key) + ": " + str(d[key]) + ("," if i < len(d) - 1 else "")    
    dict_str += "\n" + indent_str + "}"
    return dict_str

def list_to_str(l, indent=0):
    """Returns a vertically formatted string representation of a list."""
    indent_str = indent * " "
    list_str = ""
    for i, elem in enumerate(l):
        list_str += indent_str
        list_str += "[" if i == 0 else " "  
        list_str += str(elem) + (",\n" if i < len(l) - 1 else "]")
    return list_str 

def pickle_objects(fname, some_list):
    """Pickles a list of objects to a binary file."""
    print(f"PICKLE OBJECTS... [to file: {fname}]")
    t1 = time.time()
    try:
        f = open(fname, "wb+")
        pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or pickle the file]")
    t2 = time.time()
    print(f"PICKLE OBJECTS DONE. [time: {t2 - t1} s]")

def unpickle_objects(fname):
    """Returns an a list of objects from a binary file."""
    print(f"UNPICKLE OBJECTS... [from file: {fname}]")
    t1 = time.time()
    try:    
        f = open(fname, "rb")
        some_list = pickle.load(f)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or read the file]")
    t2 = time.time()
    print(f"UNPICKLE OBJECTS DONE. [time: {t2 - t1} s]")
    return some_list

def cpu_and_system_props():
    """Returns a dictionary with properties of CPU and OS."""
    props = {}    
    info = cpuinfo.get_cpu_info()
    un = platform.uname()
    props["cpu_name"] = info["brand_raw"]
    props["ram_size"] = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    props["os_name"] = f"{un.system} {un.release}"
    props["os_version"] = f"{un.version}"
    props["os_machine"] = f"{un.machine}"    
    return props    

def gpu_props():
    """Returns a dictionary with properties of GPU device."""
    gpu = cuda.get_current_device()
    props = {}
    props["name"] = gpu.name.decode("ASCII")
    props["max_threads_per_block"] = gpu.MAX_THREADS_PER_BLOCK
    props["max_block_dim_x"] = gpu.MAX_BLOCK_DIM_X
    props["max_block_dim_y"] = gpu.MAX_BLOCK_DIM_Y
    props["max_block_dim_z"] = gpu.MAX_BLOCK_DIM_Z
    props["max_grid_dim_x"] = gpu.MAX_GRID_DIM_X
    props["max_grid_dim_y"] = gpu.MAX_GRID_DIM_Y
    props["max_grid_dim_z"] = gpu.MAX_GRID_DIM_Z    
    props["max_shared_memory_per_block"] = gpu.MAX_SHARED_MEMORY_PER_BLOCK
    props["async_engine_count"] = gpu.ASYNC_ENGINE_COUNT
    props["can_map_host_memory"] = gpu.CAN_MAP_HOST_MEMORY
    props["multiprocessor_count"] = gpu.MULTIPROCESSOR_COUNT
    props["warp_size"] = gpu.WARP_SIZE
    props["unified_addressing"] = gpu.UNIFIED_ADDRESSING
    props["pci_bus_id"] = gpu.PCI_BUS_ID
    props["pci_device_id"] = gpu.PCI_DEVICE_ID
    props["compute_capability"] = gpu.compute_capability            
    CC_CORES_PER_SM_DICT = {
        (2,0) : 32,
        (2,1) : 48,
        (3,0) : 256,
        (3,5) : 256,
        (3,7) : 256,
        (5,0) : 128,
        (5,2) : 128,
        (6,0) : 64,
        (6,1) : 128,
        (7,0) : 64,
        (7,5) : 64,
        (8,0) : 64,
        (8,6) : 128
        }
    props["cores_per_SM"] = CC_CORES_PER_SM_DICT.get(gpu.compute_capability)
    props["cores_total"] = props["cores_per_SM"] * gpu.MULTIPROCESSOR_COUNT
    return props

def hash_function(s):
    """Returns a hash code (integer) for given string as a base 31 expansion."""
    h = 0
    for c in s:
        h *= 31 
        h += ord(c)
    return h

def hash_str(params, digits):
    return str((hash_function(str(params)) & ((1 << 32) - 1)) % 10**digits).rjust(digits, "0") 

class Logger:
    """Class for simultaneous logging to console and a log file (for purposes of experiments)."""
    def __init__(self, fname):
        """Constructor of ``MCTSNC`` instances."""
        self.logfile = open(fname, "w", encoding="utf-8")  
        
    def write(self, message):
        """Writes a message to console and a log file.""" 
        self.logfile.write(message)
        self.logfile.flush() 
        sys.__stdout__.write(message)

    def flush(self):
        """Empty function required for buffering."""
        pass  
    
def experiment_hash_str(experiment_info, c_props, g_props, all_hs_digits=10, experiment_hs_digits=5, env_hs_digits=3):
    """Returns a hash string for an experiment, based on its settings and properties."""
    experiment_hs =  hash_str(experiment_info, digits=experiment_hs_digits)    
    env_props = {**c_props, **g_props}    
    env_hs =  hash_str(env_props, digits=env_hs_digits)
    all_info = {**experiment_info, **env_props}
    all_hs = hash_str(all_info, digits=all_hs_digits)
    approaches_flags_str = ""
    for key in experiment_info.keys():
        if key.startswith("QMATMUL_"):
            approaches_flags_str += "T" if experiment_info[key][0] else "F" 
    suffix = f"{experiment_info['M']};{experiment_info['N']};{experiment_info['P']};{experiment_info['RANGE']};{np.dtype(experiment_info['DTYPE']).name};{experiment_info['REPETITIONS']};{approaches_flags_str}"
    hs = f"{all_hs}_{experiment_hs}_{env_hs}_[{suffix}]"
    return hs

def speedups_plot():
    args = [1e6, 6.0 * 1e6, 2.7 * 1e7, 1e9, 6.0 * 1e9, 2.7 * 1e10]
    series_float32 = {
        "QMATMUL_NAIVE_NUMBA_ST": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  
        "QMATMUL_NAIVE_NUMBA_PARALLEL": [11.4, 11.5, 12.6, 12.3, 11.5, 11.1],
        "QMATMUL_DIRECT_NUMPY": [135.0, 179.5, 213.5, 282.4, 322.6, 312.7],
        "QMATMUL_ALGO_NUMPY": [134.7, 187.8, 226.6, 404.3, 526.4, 527.0],
        "QMATMUL_DIRECT_NUMBA_CUDA": [67.5, 246.5, 496.8, 1271.4, 1876.7, 1891.7],
        "QMATMUL_ALGO_NUMBA_CUDA": [35.5, 163.3, 433.3, 2771.3, 4142.3, 4316.4]
        }
    series_float64 = {
        "QMATMUL_NAIVE_NUMBA_ST": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  
        "QMATMUL_NAIVE_NUMBA_PARALLEL": [12.0, 12.7, 12.3, 13.0, 10.4, 10.6],
        "QMATMUL_DIRECT_NUMPY": [78.4, 89.4, 116.1, 153.2, 146.4, 153.8],
        "QMATMUL_ALGO_NUMPY": [92.1, 118.6, 146.3, 248.0, 249.5, 273.1],
        "QMATMUL_DIRECT_NUMBA_CUDA": [59.6, 165.2, 299.2, 623.6, 756.9, 854.1],
        "QMATMUL_ALGO_NUMBA_CUDA": [31.9, 127.1, 291.5, 1201.1, 1473.2, 1664.3]
        }
    fontsize_title = 18
    fontsize_labels = 14
    fontsize_legend = 11
    series = series_float32
    title = "SPEED-UPS OF QMATMUL FUNCTIONS (DATA TYPE: FLOAT32)"
    plt.figure(figsize=(12, 6))
    for label, values in series.items():
        plt.plot(args, values, marker="o", markersize=4, label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, color="lightgray", zorder=0)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.3, color="lightgray", zorder=0)
    plt.xlabel(r"$M\cdot N\cdot P$", fontsize=fontsize_labels)
    plt.ylabel("SPEED-UP", fontsize=fontsize_labels)    
    plt.title(title, fontsize=fontsize_title)
    plt.legend(loc="upper left", fontsize=fontsize_legend, labelspacing=0.25)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    speedups_plot()