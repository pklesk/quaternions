# qmatmul: Fast multiplication of quaternion-valued matrices - algorithm and its implementations for sequential and CUDA computations
We present an algorithm for fast multiplication of matrices whose elements are *quaternions* - hypercomplex numbers consisting of one real and three imaginary parts.
The number of elementary floating-point multiplications involved in the algorithm is reduced *twice* with respect to the definition-based formula, 
regardless of the input matrices. This is owed to a suitable representation and decomposition into two products, one of which takes advantage of certain 
diagonal symmetry properties, the other of sparsity. 

The `qmatmul` package is suitable for Python's ecosystem.
Altogether, we provide 8 implementation variants of matrix-matrix multiplication for quaternion-valued inputs.
The variants cover several approaches based on [NumPy](https://numpy.org), thus supported by BLAS, 
but also several approaches employing [Numba](https://numba.pydata.org) - a just-in-time compiler targeting both CPU and GPU (CUDA). 
Our design of CUDA computations for the proposed algorithm involves: 6 kernel functions with 11 invocations, tiling and shared memory, 
and few host-device memory transfers.

## Installation
```bash
pip install qmatmul
```

For usage examples and more information see the repository at: [https://github.com/pklesk/quaternions](https://github.com/pklesk/quaternions). <br/>

## Documentation
Developer documentation of the project is accessible at: [https://pklesk.github.io/quaternions](https://pklesk.github.io/quaternions). <br/>

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments and credits
- [NumPy](https://numpy.org): the fundamental package for scientific computing with Python.
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler.
