r"""
Standalone script for QCNN forward-pass application example: a small 
quaternion-valued CNN (2 convolutional layers + flatten + dense layer), applied to a
batch of synthetic ("dummy") RGB-D images. Both the direct and proposed-algorithm
CUDA backends are timed per layer.

Scope: forward pass only. SGD (backpropagation training) are explicitly out of 
scope for this illustrative example.

Input encoding: each pixel is represented as a single quaternion, with the three
imaginary parts carrying the R, G, B channels and the real part carrying a fourth,
synthetic channel (e.g. a depth/LiDAR-intensity-style measurement), so that all four
quaternion components are genuinely exercised.

Convolution is implemented as im2col-style unfolding (extracting all k x k receptive
fields into columns) followed by a single qmm.dot call, using "same" zero-padding so
spatial dimensions (H, W) are preserved across both convolutional layers.

Weight initialization: He-uniform for the two convolutional layers (followed by split
QReLU) and Glorot-uniform for the dense layer (no activation on its output), both
adapted for the quaternion Hamilton-product structure.

Link to project repository
--------------------------
`https://github.com/pklesk/quaternions <https://github.com/pklesk/quaternions>`_ 
"""

__author__ = ["Przemysław Klęsk", "Aleksandr Cariow"]
__email__ = ["pklesk@zut.edu.pl", "alexandr.tariov@zut.edu.pl"]

import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import qmatmul as qmm

def unfold_quaternion(X, k, pad=None):
    """X: quaternion feature map, shape (B, C_in, H, W, 4). Applies 'same' zero-padding
    by default (pad=(k-1)//2 on each side of H and W, for odd k), so H_out=H, W_out=W;
    pass pad=0 for a valid (unpadded) convolution. Returns the unfolded matrix, shape
    (C_in*k*k, B*H_out*W_out, 4), ready to be used as the right-hand factor of a
    qmm.dot call (weights as the left factor), plus the output spatial size."""
    if pad is None:
        pad = (k - 1) // 2
    if pad > 0:
        X = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    B, C_in, H, W, _ = X.shape
    patches = sliding_window_view(X, (k, k), axis=(2, 3))   # (B, C_in, H_out, W_out, 4, k, k)
    patches = patches.transpose(1, 5, 6, 0, 2, 3, 4)         # (C_in, k, k, B, H_out, W_out, 4)
    H_out, W_out = H - k + 1, W - k + 1
    return patches.reshape(C_in * k * k, B * H_out * W_out, 4), H_out, W_out

def fold_quaternion(Z, B, H, W):
    """Inverse reshape: conv output Z, shape (C, B*H*W, 4), back to a feature-map
    tensor of shape (B, C, H, W, 4), ready as input to the next convolution."""
    C = Z.shape[0]
    return Z.reshape(C, B, H, W, 4).transpose(1, 0, 2, 3, 4)

def qrelu(Z):
    """Component-wise quaternion ReLU."""
    return np.maximum(Z, 0.0)

def he_range(fan_in, kappa=4):
    """He-uniform range for a quaternion layer followed by (Q)ReLU: R = sqrt(6/(kappa*fan_in)),
    kappa=4 accounts for the four independent real products per summed term in the
    quaternion Hamilton-product structure (kappa=1 recovers the standard real-valued
    He-uniform bound sqrt(6/fan_in))."""
    return np.sqrt(6.0 / (kappa * fan_in))

def glorot_range(fan_in, fan_out, kappa=4):
    """Glorot-uniform range for a quaternion layer with no following activation: R = sqrt(6 / (kappa * (fan_in + fan_out)))."""
    return np.sqrt(6.0 / (kappa * (fan_in + fan_out)))

def timed_dot(W, X, approach, repetitions=10):
    """Runs qmm.dot(W, X, approach) once as warm-up, then `repetitions` times, returning (result, mean_time_ms)."""
    Z = qmm.dot(W, X, approach=approach)  # warm-up (JIT compilation)
    t1 = time.time()
    for _ in range(repetitions):
        Z = qmm.dot(W, X, approach=approach)
    t2 = time.time()
    return Z, (t2 - t1) / repetitions

LINE_SEPARATOR = 256 * "="

# seed / network / data settings
SEED = 2
RANGE = 1.0 # input data range only; layer weight ranges use He/Glorot below
B, H, W = 32, 64, 64 # batch size: 32, input images' shape: 64 x 64
k1, C1 = 5, 128 # conv1: 5x5 kernel, 128 output channels, "same" padding
k2, C2 = 3, 64 # conv2: 3x3 kernel, 64 output channels, "same" padding
M_out = 10 # dense layer: 10 output units

np.random.seed(SEED)

if __name__ == "__main__":
    print(f"QUATERNIONS -> EXAMPLE QCNN FORWARD PASS...")
    print(LINE_SEPARATOR)
    
    # X_batch: an example batch drawn from a dataset - 4 real-valued channels per pixel
    # (e.g. R, G, B, and a depth/LiDAR-intensity channel), standard "channels-first" layout
    X_batch = np.random.uniform(-RANGE, RANGE, size=(B, 4, H, W)).astype(np.float32)
    
    # each pixel's 4 real channels become the 4 parts (real, i, j, k) of a single quaternion:
    # move the channel axis to the end, and insert a size-1 quaternion-channel axis (C_in = 1)
    C_in = 1  # 4 real-valued input channels -> 1 quaternion-valued channel
    X = X_batch.transpose(0, 2, 3, 1)[:, np.newaxis, :, :, :]  # (B, C_in=1, H, W, 4)
    
    results = {}
    
    # conv1 (He initialization, "same" padding)
    U1, H1, W1 = unfold_quaternion(X, k1)
    N1 = U1.shape[0]
    R1 = he_range(N1)
    W1_mat = qmm.qmatrand(C1, N1, -R1, R1, dtype=np.float32)
    print(f"CONV1 -> M: {C1}, N: {N1}, P: {U1.shape[1]} [He init range R: {R1:.5f}, H_out = W_out: {H1}]")
    Z1_algo, t_algo = timed_dot(W1_mat, U1, "algo_numba_cuda")
    Z1_direct, t_direct = timed_dot(W1_mat, U1, "direct_numba_cuda")
    print(f"[qmatmul_algo_numba_cuda: {t_algo * 1000:.3f} ms]")
    print(f"[qmatmul_direct_numba_cuda: {t_direct * 1000:.3f} ms; times ratio direct / algo: {t_direct / t_algo:.2f}]")
    F1 = fold_quaternion(qrelu(Z1_algo), B, H1, W1)
    
    # conv2 (He init, "same" padding)
    U2, H2, W2 = unfold_quaternion(F1, k2)
    N2 = U2.shape[0]
    R2 = he_range(N2)
    W2_mat = qmm.qmatrand(C2, N2, -R2, R2, dtype=np.float32)
    print(f"CONV2 -> M: {C2}, N: {N2}, P: {U2.shape[1]} [He init range R: {R2:.5f}, H_out = W_out: {H2}]")
    Z2_algo, t_algo = timed_dot(W2_mat, U2, "algo_numba_cuda")
    Z2_direct, t_direct = timed_dot(W2_mat, U2, "direct_numba_cuda")
    print(f"[qmatmul_algo_numba_cuda: {t_algo * 1000:.3f} ms]")
    print(f"[qmatmul_direct_numba_cuda: {t_direct * 1000:.3f} ms; times ratio direct / algo: {t_direct / t_algo:.2f}]")
    
    # flatten + dense (Glorot init, no activation on output)
    F2 = qrelu(Z2_algo).reshape(C2, B, H2, W2, 4).transpose(0, 2, 3, 1, 4).reshape(C2 * H2 * W2, B, 4)
    N3 = F2.shape[0]
    R3 = glorot_range(N3, M_out)
    W3_mat = qmm.qmatrand(M_out, N3, -R3, R3, dtype=np.float32)
    print(f"DENSE -> M: {M_out}, N: {N3}, P: {B} [Glorot init range R: {R3:.5f}]")
    Z3_algo, t_algo = timed_dot(W3_mat, F2, "algo_numba_cuda")
    Z3_direct, t_direct = timed_dot(W3_mat, F2, "direct_numba_cuda")
    print(f"[qmatmul_algo_numba_cuda: {t_algo * 1000:.3f} ms]")
    print(f"[qmatmul_direct_numba_cuda: {t_direct * 1000:.3f} ms; times ratio direct / algo: {t_direct / t_algo:.2f}]")
    Z_out = np.transpose(Z3_algo, (1, 0, 2)) # (B, M_out, 4), torch-style batch-first
    
    print("---")    
    print(f"FINAL OUTPUT SHAPE: {Z_out.shape} [(B, M_out, 4)]")
    diff = np.abs(Z3_algo.astype(np.float64) - Z3_direct.astype(np.float64))
    rel_frob = np.linalg.norm(Z3_algo.astype(np.float64) - Z3_direct.astype(np.float64)) / np.linalg.norm(Z3_direct.astype(np.float64))
    print(f"RELATIVE FROBENIUS ERROR [algo vs direct, dense layer]: {rel_frob:.3e}")
    print(f"MAX ABS ERROR: {diff.max():.3e}")
    print(f"TYPICAL |Z3| SCALE (RMS): {np.sqrt(np.mean(Z3_direct.astype(np.float64)**2)):.3e}")
    print(f"SANITY CHECK [algo vs direct -> np.allclose, dense layer]: {np.allclose(Z3_algo, Z3_direct, rtol=1e-5, atol=1e-2)}")

    print(LINE_SEPARATOR)
    print(f"QUATERNIONS -> EXAMPLE QCNN FORWARD PASS DONE.")
    