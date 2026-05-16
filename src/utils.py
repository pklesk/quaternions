import cpuinfo
import platform
import psutil
from numba import cuda
import sys
import numpy as np
from matplotlib import pyplot as plt
 
__author__ = ["Przemysław Klęsk", "Aleksandr Cariow"]
__email__ = ["pklesk@zut.edu.pl", "alexanddr.tariov@zut.edu.pl"]

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
    props["name"] = gpu.name if isinstance(gpu.name, str) else gpu.name.decode("ASCII")
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
        (2, 0): 32,
        (2, 1): 48,
        (3, 0): 256,
        (3, 5): 256,
        (3, 7): 256,
        (5, 0): 128,
        (5, 2): 128,
        (6, 0): 64,
        (6, 1): 128,
        (7, 0): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128,
        (10, 0): 128,
        (10, 1): 128,
        (10, 2): 128,
        (12, 0): 128,
        (12, 1): 128,
        (12, 2): 128
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
    suffix = f"{experiment_info['M']};{experiment_info['N']};{experiment_info['P']};{experiment_info['RANGE']};{np.dtype(experiment_info['DTYPE']).name};" \
        f"{experiment_info['REPETITIONS']};{approaches_flags_str}"
    hs = f"{all_hs}_{experiment_hs}_{env_hs}_[{suffix}]"
    return hs

def speedups_plot(dtype):
    args = [1e6, 6.0 * 1e6, 2.7 * 1e7, 1e9, 6.0 * 1e9, 2.7 * 1e10]
    series_float32 = {
        "QMATMUL_NAIVE_NUMBA_ST": [1.7, 1.5, 1.5, 1.5, 1.6, 1.5],  
        "QMATMUL_NAIVE_NUMBA_PARALLEL": [10.6, 25.5, 48.5, 60.5, 63.1, 59.1],        
        "QMATMUL_DIRECT_NUMPY_ST": [145.5, 290.9, 375.2, 665.4, 765.2, 753.3],        
        "QMATMUL_DIRECT_NUMPY_PARALLEL": [18.6, 92.4, 856.7, 2805.3, 3946.2, 3657.5],        
        "QMATMUL_DIRECT_NUMBA_CUDA": [199.1, 579.2, 1346.2, 3275.4, 6752.9, 8130.6],        
        "QMATMUL_ALGO_NUMPY_ST": [159.4, 345.0, 399.4, 1009.4, 1242.0, 1308.4],        
        "QMATMUL_ALGO_NUMPY_PARALLEL": [490.5, 675.2, 676.1, 2473.1, 2943.8, 4192.3],        
        "QMATMUL_ALGO_NUMBA_CUDA": [107.4, 398.0, 746.2, 6527.9, 12331.5, 15291.1]
    }    
    series_float64 = {
        "QMATMUL_NAIVE_NUMBA_ST": [1.7, 1.5, 1.5, 1.5, 1.5, 1.5],  
        "QMATMUL_NAIVE_NUMBA_PARALLEL": [9.9, 24.5, 47.6, 61.6, 60.8, 59.2],
        "QMATMUL_DIRECT_NUMPY_ST": [111.0, 134.1, 212.1, 302.1, 329.6, 334.6],
        "QMATMUL_DIRECT_NUMPY_PARALLEL": [19.9, 151.9, 587.9, 1179.4, 1494.4, 1641.6],
        "QMATMUL_DIRECT_NUMBA_CUDA": [173.1, 430.5, 587.9, 2105.2, 3299.9, 3799.9],
        "QMATMUL_ALGO_NUMPY_ST": [123.1, 140.5, 212.4, 478.3, 554.3, 600.7],
        "QMATMUL_ALGO_NUMPY_PARALLEL": [288.2, 505.7, 489.4, 1112.7, 1544.9, 2109.0],
        "QMATMUL_ALGO_NUMBA_CUDA": [96.3, 230.6, 499.1, 3672.4, 5544.7, 6563.2]
    }
    styles = {
        "QMATMUL_NAIVE_NUMBA_ST": ("black", "--"),
        "QMATMUL_NAIVE_NUMBA_PARALLEL": ("black", "-"),
        "QMATMUL_DIRECT_NUMPY_ST": ("limegreen", "--"),
        "QMATMUL_DIRECT_NUMPY_PARALLEL": ("limegreen", "-"),
        "QMATMUL_DIRECT_NUMBA_CUDA": ("green", "-"),
        "QMATMUL_ALGO_NUMPY_ST": ("dodgerblue", "--"),
        "QMATMUL_ALGO_NUMPY_PARALLEL": ("dodgerblue", "-"),
        "QMATMUL_ALGO_NUMBA_CUDA": ("blue", "-")
    }
    
    TITLE_FONT_SIZE = 21.0
    LABEL_FONT_SIZE = 10.0    
    AXIS_LABEL_SIZE = 19.0
    LEGEND_FONT_SIZE = 12.0
    COLUMN_SPACING = 1.28
    TICK_LABEL_SIZE = 15.0
    
    if dtype == np.float32:
        series = series_float32
        title = "LOG-LOG PLOT OF SPEED-UPS FOR QMATMUL FUNCTIONS (TYPE: FLOAT32, ENVIRONMENT: 2)"
    else:
        series = series_float64
        title = "LOG-LOG PLOT OF SPEED-UPS FOR QMATMUL FUNCTIONS (TYPE: FLOAT64, ENVIRONMENT: 2)"
    
    plt.figure(figsize=(16, 7))
    linewidth = 2.25
    markersize = 5.5    
    for label, values in series.items():
        color, linestyle = styles.get(label, ("gray", "-"))
        plt.plot(args, values, marker="o", label=label.lower(), color=color, linestyle=linestyle, linewidth=linewidth, markersize=markersize, zorder=10)
    
    last_x = args[-1]
    first_x = args[0]
    
    sorted_series = sorted(series.items(), key=lambda item: item[1][-1], reverse=True)    
    for idx, (label, values) in enumerate(sorted_series):
        last_y = values[-1]
        color, _ = styles.get(label, ("gray", "-"))
        x_pos = last_x * (COLUMN_SPACING ** (idx + 1.8))    
        plt.hlines(y=last_y, xmin=last_x, xmax=x_pos, colors="black", linestyle="-", linewidth=0.8, alpha=0.5, zorder=2)        
        plt.text(x_pos * 1.03, last_y, f"{last_y:.1f}x", color=color, fontsize=LABEL_FONT_SIZE, va="center", ha="left", zorder=11)
        
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", color="#F5F5F5", zorder=1)    
    plt.xlabel(r"$M\cdot N\cdot P$", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("SPEED-UP", fontsize=AXIS_LABEL_SIZE)    
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    
    plt.xlim(first_x * 0.85, last_x * (COLUMN_SPACING ** (len(sorted_series) + 3.2)))
    
    plt.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    plt.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE * 0.8)
    
    plt.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, framealpha=0.9, ncol=1, handlelength=3.0, labelspacing=0.01)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    speedups_plot(dtype=np.float64)
    