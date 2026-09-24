r"""
Basic correctness tests for the qmatmul package.

Run with: pytest tests/test_qmatmul.py -v
"""
import numpy as np
import pytest

import qmatmul as qmm

# approaches not requiring GPU
CPU_APPROACHES = [
    "naive_numba_st",
    "naive_numba_parallel",
    "direct_numpy_st",
    "direct_numpy_parallel",
    "algo_numpy_st",
    "algo_numpy_parallel",
]

# approaches requiring GPU (run only when CUDA is actually available)
GPU_APPROACHES = [
    "direct_numba_cuda",
    "algo_numba_cuda",
]

DTYPES = [np.float32, np.float64]

def _random_problem(M, N, P, dtype, seed):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-5, 5, size=(M, N, 4)).astype(dtype)
    B = rng.uniform(-5, 5, size=(N, P, 4)).astype(dtype)
    return A, B

def _reference(A, B):
    return qmm.qmatmul_algolike_numpy(A.astype(np.float64), B.astype(np.float64))

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approach", CPU_APPROACHES)
def test_cpu_approaches_match_reference(approach, dtype):
    """Every CPU-based approach must agree with the direct-formula reference
    (qmatmul_algolike_numpy) within a small numerical tolerance."""
    A, B = _random_problem(M=5, N=7, P=3, dtype=dtype, seed=0)
    C = qmm.dot(A, B, approach=approach)
    C_ref = _reference(A, B)
    tol = 1e-3 if dtype == np.float32 else 1e-9
    assert np.allclose(C.astype(np.float64), C_ref, atol=tol, rtol=tol)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approach", GPU_APPROACHES)
def test_gpu_approaches_match_reference(approach, dtype):
    """Same check as above, for GPU-based approaches; skipped when CUDA is
    not available in the current environment."""
    if not qmm.CUDA_AVAILABLE:
        pytest.skip("CUDA is not available in this environment.")
    A, B = _random_problem(M=5, N=7, P=3, dtype=dtype, seed=0)
    C = qmm.dot(A, B, approach=approach)
    C_ref = _reference(A, B)
    tol = 1e-3 if dtype == np.float32 else 1e-9
    assert np.allclose(C.astype(np.float64), C_ref, atol=tol, rtol=tol)

def test_default_dot_falls_back_to_cpu_without_gpu():
    """Without CUDA, the default approach ('algo_numba_cuda') must not raise,
    but transparently fall back to a CPU approach (with a RuntimeWarning),
    still returning a correct result."""
    if qmm.CUDA_AVAILABLE:
        pytest.skip("This test targets environments without a GPU.")
    A, B = _random_problem(M=4, N=5, P=3, dtype=np.float32, seed=1)
    with pytest.warns(RuntimeWarning, match="falling back"):
        C = qmm.dot(A, B)
    C_ref = _reference(A, B)
    assert np.allclose(C.astype(np.float64), C_ref, atol=1e-3)

def test_dot_rejects_wrong_ndim():
    A = np.zeros((2, 2), dtype=np.float32)
    B = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="3-dimensional"):
        qmm.dot(A, B)

def test_dot_rejects_wrong_last_dimension():
    A = np.zeros((2, 3, 5), dtype=np.float32)  # last dim should be 4
    B = np.zeros((3, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="last dimension"):
        qmm.dot(A, B)

def test_dot_rejects_incompatible_inner_dimensions():
    A = np.zeros((2, 3, 4), dtype=np.float32)
    B = np.zeros((5, 2, 4), dtype=np.float32)  # 5 != 3
    with pytest.raises(ValueError, match="Incompatible matrix dimensions"):
        qmm.dot(A, B)