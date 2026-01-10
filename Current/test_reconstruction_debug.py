#!/usr/bin/env python3
"""
Debug Test: Investigating High Reconstruction Error

The end-to-end test shows reconstruction RMSE of ~245 pixels, which is
unacceptably high for trajectory reconstruction. This test investigates
the root cause.

Possible issues:
1. Basis doesn't capture trajectory shape well
2. Projection is losing information
3. Mean centering issue
4. Numerical issues
"""

import numpy as np
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))

from bspline_basis import (
    BasisAdapter,
    fit_bspline_control_points,
    evaluate_control_point_fit,
    build_interleaved_basis_matrix,
)


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def generate_trajectory(length, distance, angle, jitter=1.0, seed=None):
    """Generate a simple trajectory."""
    if seed:
        np.random.seed(seed)

    t = np.linspace(0, 1, length)
    eased_t = 1 - (1 - t) ** 2

    x = eased_t * distance * np.cos(angle)
    y = eased_t * distance * np.sin(angle)

    if jitter > 0:
        x += gaussian_filter1d(np.random.randn(length) * jitter, sigma=1.0)
        y += gaussian_filter1d(np.random.randn(length) * jitter, sigma=1.0)

    flat = np.empty(2 * length)
    flat[0::2] = x
    flat[1::2] = y
    return flat


def test_control_point_reconstruction():
    """Test reconstruction via control points (not SVD basis)."""
    print_section("TEST 1: Control Point Reconstruction")

    # Generate a simple trajectory
    traj = generate_trajectory(50, 100, np.pi/4, jitter=1.5, seed=42)
    T = len(traj) // 2

    print(f"  Original trajectory: T={T}")
    print(f"    Start: ({traj[0]:.2f}, {traj[1]:.2f})")
    print(f"    End: ({traj[-2]:.2f}, {traj[-1]:.2f})")

    for n_cp in [8, 16, 20, 32]:
        cp = fit_bspline_control_points(traj, n_cp)

        # Reconstruct via B-spline
        B = build_interleaved_basis_matrix(T, n_cp)
        recon = B @ cp

        rmse = np.sqrt(np.mean((traj - recon) ** 2))
        endpoint_error = np.sqrt((traj[-2] - recon[-2])**2 + (traj[-1] - recon[-1])**2)

        print(f"\n  n_control_points={n_cp}:")
        print(f"    Reconstruction RMSE: {rmse:.4f}")
        print(f"    Endpoint error: {endpoint_error:.4f}")
        print(f"    Reconstructed end: ({recon[-2]:.2f}, {recon[-1]:.2f})")


def test_svd_reconstruction():
    """Test reconstruction via SVD basis."""
    print_section("TEST 2: SVD Basis Reconstruction")

    np.random.seed(42)

    # Generate training data with variety
    print("  Generating training data...")
    n_train = 50
    n_cp = 20
    K = 10

    training_data = []
    for i in range(n_train):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i+100)
        training_data.append(traj)

    # Fit control points
    control_points = np.zeros((n_train, 2 * n_cp))
    for i, traj in enumerate(training_data):
        cp = fit_bspline_control_points(traj, n_cp)
        control_points[i] = cp

    # SVD (with centering)
    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    print(f"\n  Basis learned:")
    print(f"    U_ref shape: {U_ref.shape}")
    print(f"    Top singular values: {S[:5]}")
    print(f"    Variance in first K={K}: {np.sum(S[:K]**2)/np.sum(S**2)*100:.2f}%")

    # Create adapter
    adapter = BasisAdapter(U_ref, n_cp)

    # Test reconstruction on a new trajectory
    print("\n  Testing on new trajectories...")

    test_cases = [
        ("Short dist, horizontal", 30, 0),
        ("Medium dist, diagonal", 100, np.pi/4),
        ("Long dist, vertical", 200, np.pi/2),
    ]

    for name, dist, angle in test_cases:
        traj = generate_trajectory(50, dist, angle, jitter=1.0, seed=999)
        T = len(traj) // 2

        # Project and reconstruct
        c = adapter.project(traj)
        recon = adapter.reconstruct(c, T)

        rmse = np.sqrt(np.mean((traj - recon) ** 2))

        print(f"\n  {name}:")
        print(f"    Original end: ({traj[-2]:.2f}, {traj[-1]:.2f})")
        print(f"    Reconstructed end: ({recon[-2]:.2f}, {recon[-1]:.2f})")
        print(f"    RMSE: {rmse:.4f}")

        # Check if coefficients capture the trajectory
        coeff_norm = np.linalg.norm(c)
        print(f"    Coefficient norm: {coeff_norm:.2f}")


def test_projection_matrix_rank():
    """Check if the adapted basis has full rank for projection."""
    print_section("TEST 3: Projection Matrix Analysis")

    np.random.seed(42)

    # Create a simple basis
    n_cp = 20
    K = 8

    # Random orthonormal basis
    A = np.random.randn(2 * n_cp, K)
    U_ref, _, _ = np.linalg.svd(A, full_matrices=False)
    U_ref = U_ref[:, :K]

    adapter = BasisAdapter(U_ref, n_cp)

    for T in [20, 50, 100]:
        U_T = adapter.get_adapted_basis(T)

        # Check rank
        rank = np.linalg.matrix_rank(U_T)

        # Check orthonormality
        ortho = U_T.T @ U_T
        ortho_error = np.max(np.abs(ortho - np.eye(K)))

        print(f"\n  T={T}:")
        print(f"    U_T shape: {U_T.shape}")
        print(f"    Rank: {rank} (should be {K})")
        print(f"    Orthonormality error: {ortho_error:.2e}")

        # Test round-trip
        test_traj = np.random.randn(2 * T)
        c = adapter.project(test_traj)
        recon = adapter.reconstruct(c, T)

        # The projection should capture the component in the basis
        projection_norm = np.linalg.norm(recon)
        original_norm = np.linalg.norm(test_traj)
        ratio = projection_norm / original_norm

        print(f"    Projection/Original norm ratio: {ratio:.4f}")


def test_training_data_impact():
    """Test how training data affects reconstruction of test data."""
    print_section("TEST 4: Training Data Coverage")

    np.random.seed(42)

    n_cp = 20
    K = 10

    # Training data: only short distances in one direction
    print("  Case A: Training on limited data (short, horizontal only)...")
    training_A = []
    for i in range(50):
        length = np.random.randint(30, 60)
        distance = 50 + np.random.rand() * 50  # Only 50-100
        angle = np.random.randn() * 0.2  # Near-horizontal
        traj = generate_trajectory(length, distance, angle, jitter=1.0, seed=i)
        training_A.append(traj)

    # Fit and SVD
    cp_A = np.zeros((50, 2 * n_cp))
    for i, traj in enumerate(training_A):
        cp_A[i] = fit_bspline_control_points(traj, n_cp)

    mean_A = cp_A.mean(axis=0)
    centered_A = cp_A - mean_A
    U_A, S_A, _ = np.linalg.svd(centered_A.T, full_matrices=False)
    U_A = U_A[:, :K]

    adapter_A = BasisAdapter(U_A, n_cp)

    # Training data: diverse
    print("  Case B: Training on diverse data...")
    training_B = []
    for i in range(50):
        length = np.random.randint(30, 80)
        distance = 30 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i+1000)
        training_B.append(traj)

    cp_B = np.zeros((50, 2 * n_cp))
    for i, traj in enumerate(training_B):
        cp_B[i] = fit_bspline_control_points(traj, n_cp)

    mean_B = cp_B.mean(axis=0)
    centered_B = cp_B - mean_B
    U_B, S_B, _ = np.linalg.svd(centered_B.T, full_matrices=False)
    U_B = U_B[:, :K]

    adapter_B = BasisAdapter(U_B, n_cp)

    # Test on out-of-distribution trajectory (long, vertical)
    test_traj = generate_trajectory(50, 200, np.pi/2, jitter=1.0, seed=9999)
    T = 50

    c_A = adapter_A.project(test_traj)
    recon_A = adapter_A.reconstruct(c_A, T)
    rmse_A = np.sqrt(np.mean((test_traj - recon_A) ** 2))

    c_B = adapter_B.project(test_traj)
    recon_B = adapter_B.reconstruct(c_B, T)
    rmse_B = np.sqrt(np.mean((test_traj - recon_B) ** 2))

    print(f"\n  Test trajectory: Long (200px), vertical")
    print(f"    Original end: ({test_traj[-2]:.2f}, {test_traj[-1]:.2f})")
    print(f"\n    Case A (limited training):")
    print(f"      Reconstructed end: ({recon_A[-2]:.2f}, {recon_A[-1]:.2f})")
    print(f"      RMSE: {rmse_A:.4f}")
    print(f"\n    Case B (diverse training):")
    print(f"      Reconstructed end: ({recon_B[-2]:.2f}, {recon_B[-1]:.2f})")
    print(f"      RMSE: {rmse_B:.4f}")


def test_exact_reconstruction():
    """Test if training data can be exactly reconstructed."""
    print_section("TEST 5: Training Data Reconstruction")

    np.random.seed(42)

    n_cp = 20
    K = 10

    # Generate training data
    training_data = []
    for i in range(30):
        length = np.random.randint(40, 60)
        distance = 50 + np.random.rand() * 150
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i)
        training_data.append(traj)

    # Fit control points and SVD
    control_points = np.zeros((30, 2 * n_cp))
    for i, traj in enumerate(training_data):
        control_points[i] = fit_bspline_control_points(traj, n_cp)

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    adapter = BasisAdapter(U_ref, n_cp)

    print("  Reconstructing training data...")
    print(f"  (K={K}, variance explained: {np.sum(S[:K]**2)/np.sum(S**2)*100:.2f}%)")

    rmses = []
    for i, traj in enumerate(training_data[:10]):
        T = len(traj) // 2
        c = adapter.project(traj)
        recon = adapter.reconstruct(c, T)
        rmse = np.sqrt(np.mean((traj - recon) ** 2))
        rmses.append(rmse)

        if i < 3:
            print(f"\n    Trajectory {i}:")
            print(f"      Original: start=({traj[0]:.1f},{traj[1]:.1f}), end=({traj[-2]:.1f},{traj[-1]:.1f})")
            print(f"      Recon:    start=({recon[0]:.1f},{recon[1]:.1f}), end=({recon[-2]:.1f},{recon[-1]:.1f})")
            print(f"      RMSE: {rmse:.4f}")

    print(f"\n  Average RMSE on training data: {np.mean(rmses):.4f}")
    print(f"  Max RMSE on training data: {np.max(rmses):.4f}")


def test_basis_visualization():
    """Visualize what the basis vectors represent."""
    print_section("TEST 6: Basis Vector Analysis")

    np.random.seed(42)

    n_cp = 20
    K = 6

    # Generate diverse training data
    training_data = []
    for i in range(100):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i)
        training_data.append(traj)

    # Fit control points
    control_points = np.zeros((100, 2 * n_cp))
    for i, traj in enumerate(training_data):
        control_points[i] = fit_bspline_control_points(traj, n_cp)

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    print("  Basis vectors at reference length:")
    print(f"  Mean vector (shape {mean.shape}):")
    print(f"    x: {mean[0::2][:5]}...")
    print(f"    y: {mean[1::2][:5]}...")

    for k in range(K):
        u_k = U_ref[:, k]
        x_k = u_k[0::2]
        y_k = u_k[1::2]

        # Interpret: is this mostly x-direction, y-direction, or curved?
        x_var = np.var(x_k)
        y_var = np.var(y_k)

        print(f"\n  Basis vector {k} (singular value {S[k]:.2f}):")
        print(f"    x variance: {x_var:.4f}")
        print(f"    y variance: {y_var:.4f}")
        print(f"    x range: [{x_k.min():.3f}, {x_k.max():.3f}]")
        print(f"    y range: [{y_k.min():.3f}, {y_k.max():.3f}]")


def main():
    print("=" * 70)
    print(" RECONSTRUCTION ERROR DEBUG")
    print("=" * 70)

    test_control_point_reconstruction()
    test_svd_reconstruction()
    test_projection_matrix_rank()
    test_training_data_impact()
    test_exact_reconstruction()
    test_basis_visualization()

    print("\n" + "=" * 70)
    print(" DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
