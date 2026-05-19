# qmatmul: Fast multiplication of quaternion-valued matrices - algorithm and its implementations for sequential and CUDA computation
We present an algorithm for fast multiplication of matrices whose elements are *quaternions* - hypercomplex numbers consisting of one real and three imaginary parts.
The number of elementary floating-point multiplications involved in the algorithm is reduced *twice* with respect to the definition-based formula, 
regardless of the input matrices. This is owed to a suitable representation and decomposition into two products, one of which takes advantage of certain 
diagonal symmetry properties, the other of sparsity. `qmatmul` contains eight implementation variants, covering: CPU-based single-threaded and parallel
computations, and also GPU/CUDA-based multi-threaded ones. Our design of CUDA computations for the proposed algorithm involves: six kernel functions 
with eleven invocations, suitable usage of tiling and shared memory, and few host-device memory transfers.

## Selected kernels - flows of CUDA computations
<table>
   <tr>
     <td valign="top"><img width="1150" height="1496" alt="had4_flow" src="https://github.com/user-attachments/assets/091bb888-3c58-421a-8165-dff3643dcdbf"/></td>
     <td valign="top"><img width="1492" height="2046" alt="matmuldiag_flow" src="https://github.com/user-attachments/assets/302b37ce-58d9-46d5-9b0f-c9d7bf1ab001"/></td>    
   </tr>
</table>

## Speed-ups
<table>
   <tr>
     <td valign="top"><img src="extras/speedups_float32_5090.png"/></td>
     <td valign="top"><img src="extras/speedups_float64_5090.png"/></td>    
   </tr>
</table>

## Installation
```bash
pip install TODO
```
Note: for further usage, NVIDIA CUDA drivers must be present in the operating system.

## Usage example 1
Suppose one would like to multiply the following matrices of quaternions:

$$
{\tiny
\begin{pmatrix}
2i +j & -1 +i -j + 4k & 5 -i +3j \\
1 +4i -4j -4k & -5 +3i +3j +4k & 5 +3i +3k \\
-4 +i -4j + 4k & -i -2j +3k & i -5j +k \\
1 +i +4j + 2k& -1 -i +2j -4k & 2 +2i -3j -4k \\
-2 -i +j -k & 5 -4i -3j -3k & 2 -2i -3k
\end{pmatrix}
}
{\cdot}
{\tiny
\begin{pmatrix}
-3 -4i + 2j -4k & -3 -i +3j -4k \\
3 - 4i +5j & 5 +i +2j -5k\\
-2 -4i -2j -4k & -2 -i -4j +2k
\end{pmatrix}
}
{\scriptsize =}
{\tiny
\begin{pmatrix}
4 -53i -39j +15k & 16 -6i -17j +52k \\
1 -3i +4j +7k & -30 +3i +30j +56k \\
\end{pmatrix}
}
$$

With `qmatmul` module installed, one can write:
```python
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
```
Running the script above produces the following output:
```bash
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
QMATMUL EXAMPLE DONE. TIME OF qmm.dot: 0.002028 s.
```
## Usage example 2 (large)
With `qmatmul` module installed, one can write e.g.:
```python
import qmatmul as qmm
import numpy as np
import time

print("QMATMUL EXAMPLE...")
M, N, P = 1000, 3000, 2000
np.random.seed(0)
A = np.random.rand(M, N, 4) # M x N matrix of quaternions
B = np.random.rand(N, P, 4) # N x P matrix of quaternions
t1 = time.time()
C = qmm.dot(A, B)
t2 = time.time()
print(f"RESULT FRAGMENT -> C[:3, :3]:")
print(C[:3, :3])
print(f"QMATMUL EXAMPLE DONE. TIME OF qmm.dot: {t2 - t1:.6f} s.")
```
Running the script above produces the following output:
```bash
QMATMUL EXAMPLE...
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
QMATMUL EXAMPLE DONE. TIME OF qmm.dot: 0.209159 s.
```

## Choice of approach

An optional argument `approach` of the function `qmm.dot` allows 
the user to select one of the following eight computational approaches: 
`"naive_st"`, `"naive_parallel"`, `"direct_numpy_st"`, 
`"direct_numpy_parallel"`, `"direct_numpy_st"`, `"direct\_cuda"`,
`"algo_numpy_st"`, `"algo_numpy_parallel"`, `"algo_numba_cuda"`.
The default setting is `"algo_numba_cuda"`.

TODO

## Documentation
TODO

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments and credits
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler.
