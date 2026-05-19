# qmatmul: Fast multiplication of quaternion-valued matrices - algorithm and its implementations for sequential and CUDA computation
We present an algorithm for fast multiplication of matrices whose elements are quaternions - hypercomplex numbers consisting of one real and three imaginary parts.
The number of elementary floating-point multiplications involved in the algorithm is reduced *twice* with respect to the definition-based formula, 
regardless of the input matrices. This is owed to a suitable representation and decomposition into two products, one of which takes advantage of certain 
diagonal symmetry properties, the other of sparsity. `qmatmul` contains eight implementation variants, covering: CPU-based single-threaded and parallel
computations, and also GPU/CUDA-based multi-threaded ones. Our design of CUDA computations for the proposed algorithm involves: six kernel functions 
with eleven invocations, suitable usage of tiling and shared memory, and few host-device memory transfers.

## Selected kernels - flows of CUDA computations
<table>
   <tr>
     <td><img width="1150" height="1496" alt="had4_flow" src="https://github.com/user-attachments/assets/091bb888-3c58-421a-8165-dff3643dcdbf"/></td>
     <td><img width="1492" height="2046" alt="matmuldiag_flow" src="https://github.com/user-attachments/assets/302b37ce-58d9-46d5-9b0f-c9d7bf1ab001"/></td>    
   </tr>
</table>

