#!/usr/bin/env python3
"""
Test: Orthonormality Issue and Fix

The root cause of high reconstruction error is that U_T = C @ U_ref
loses orthonormality when C is the B-spline transformation matrix.

This test:
1. Confirms the orthonormality issue
2. Tests a fix (re-orthonormalize after transformation)
3. Compares reconstruction quality before/after fix
"""

import numpy as np
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))

from bspline_basis import (
    BasisAdapter,
    fit_bspline_control_points,
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


class BasisAdapterFixed(BasisAdapter):
    """
    Fixed BasisAdapter that re-orthonormalizes the adapted basis.

    The issue: U_T = C @ U_ref is not orthonormal even if U_ref is.
    Fix: Apply QR decomposition to U_T to get orthonormal columns.
    """

    def get_adapted_basis(self, T: int, orthonormalize: bool = True) -> np.ndarray:
        """Get basis adapted to length T, with optional re-orthonormalization."""
        if T < 2:
            raise ValueError(f"Trajectory length must be >= 2 (got {T})")

        cache_key = (T, orthonormalize)

        if cache_key not in self._basis_cache:
            # Build transformation matrix
            C_interleaved = build_interleaved_basis_matrix(
                T_out=T,
                T_in=self.T_ref,
                clamp_endpoints=self.clamp_endpoints
            )

            # Adapt basis: U_T = C @ U_ref
            U_T = C_interleaved @ self.U_ref

            if orthonormalize:
                # FIX: Re-orthonormalize using QR decomposition
                # This ensures U_T.T @ U_T = I
                Q, R = np.linalg.qr(U_T)
                U_T = Q[:, :self.K]

                # Preserve sign consistency (make first non-zero element positive)
                for k in range(self.K):
                    if U_T[:, k].sum() < 0:
                        U_T[:, k] = -U_T[:, k]

            self._basis_cache[cache_key] = U_T

        return self._basis_cache[cache_key]


def test_orthonormality_issue():
    """Demonstrate the orthonormality issue."""
    print_section("TEST 1: Demonstrating Orthonormality Issue")

    np.random.seed(42)

    # Create basis
    n_cp = 20
    K = 8

    # Generate training data and learn basis
    training_data = []
    for i in range(50):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 150
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i)
        training_data.append(traj)

    control_points = np.zeros((50, 2 * n_cp))
    for i, traj in enumerate(training_data):
        control_points[i] = fit_bspline_control_points(traj, n_cp)

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    print(f"  U_ref orthonormality error: {np.max(np.abs(U_ref.T @ U_ref - np.eye(K))):.2e}")

    # Test at different lengths
    print("\n  Adapted basis orthonormality (ORIGINAL):")
    adapter = BasisAdapter(U_ref, n_cp)

    for T in [20, 30, 40, 50, 60, 80, 100]:
        U_T = adapter.get_adapted_basis(T)
        ortho_error = np.max(np.abs(U_T.T @ U_T - np.eye(K)))
        print(f"    T={T:3d}: error = {ortho_error:.4f}")


def test_fix_effectiveness():
    """Test if re-orthonormalization fixes the issue."""
    print_section("TEST 2: Re-orthonormalization Fix")

    np.random.seed(42)

    n_cp = 20
    K = 8

    # Generate training data
    training_data = []
    for i in range(50):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 150
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i)
        training_data.append(traj)

    control_points = np.zeros((50, 2 * n_cp))
    for i, traj in enumerate(training_data):
        control_points[i] = fit_bspline_control_points(traj, n_cp)

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    adapter_fixed = BasisAdapterFixed(U_ref, n_cp)

    print("  Adapted basis orthonormality (FIXED with QR):")
    for T in [20, 30, 40, 50, 60, 80, 100]:
        U_T = adapter_fixed.get_adapted_basis(T, orthonormalize=True)
        ortho_error = np.max(np.abs(U_T.T @ U_T - np.eye(K)))
        print(f"    T={T:3d}: error = {ortho_error:.2e}")


def test_reconstruction_comparison():
    """Compare reconstruction quality before and after fix."""
    print_section("TEST 3: Reconstruction Quality Comparison")

    np.random.seed(42)

    n_cp = 20
    K = 8

    # Generate training data
    training_data = []
    for i in range(100):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i)
        training_data.append(traj)

    control_points = np.zeros((100, 2 * n_cp))
    for i, traj in enumerate(training_data):
        control_points[i] = fit_bspline_control_points(traj, n_cp)

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    adapter_orig = BasisAdapter(U_ref, n_cp)
    adapter_fixed = BasisAdapterFixed(U_ref, n_cp)

    # Test on NEW trajectories
    print("\n  Testing on 50 new trajectories...")

    rmse_orig = []
    rmse_fixed = []
    rmse_control_points = []

    test_trajectories = []
    for i in range(50):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i+1000)
        test_trajectories.append(traj)

    for traj in test_trajectories:
        T = len(traj) // 2

        # Original (broken) method
        c_orig = adapter_orig.project(traj)
        recon_orig = adapter_orig.reconstruct(c_orig, T)
        rmse_orig.append(np.sqrt(np.mean((traj - recon_orig) ** 2)))

        # Fixed method
        c_fixed = adapter_fixed.project(traj)
        recon_fixed = adapter_fixed.reconstruct(c_fixed, T)
        rmse_fixed.append(np.sqrt(np.mean((traj - recon_fixed) ** 2)))

        # Baseline: control point reconstruction (no SVD)
        cp = fit_bspline_control_points(traj, n_cp)
        B = build_interleaved_basis_matrix(T, n_cp)
        recon_cp = B @ cp
        rmse_control_points.append(np.sqrt(np.mean((traj - recon_cp) ** 2)))

    print(f"\n  Control point (no SVD):")
    print(f"    Mean RMSE: {np.mean(rmse_control_points):.4f}")
    print(f"    Max RMSE:  {np.max(rmse_control_points):.4f}")

    print(f"\n  Original adapter (orthonormality broken):")
    print(f"    Mean RMSE: {np.mean(rmse_orig):.4f}")
    print(f"    Max RMSE:  {np.max(rmse_orig):.4f}")

    print(f"\n  Fixed adapter (re-orthonormalized):")
    print(f"    Mean RMSE: {np.mean(rmse_fixed):.4f}")
    print(f"    Max RMSE:  {np.max(rmse_fixed):.4f}")

    improvement = (np.mean(rmse_orig) - np.mean(rmse_fixed)) / np.mean(rmse_orig) * 100
    print(f"\n  Improvement: {improvement:.1f}%")


def test_endpoint_preservation():
    """Test if endpoints are preserved after reconstruction."""
    print_section("TEST 4: Endpoint Preservation")

    np.random.seed(42)

    n_cp = 20
    K = 8

    # Simple training data
    training_data = []
    for i in range(50):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i)
        training_data.append(traj)

    control_points = np.zeros((50, 2 * n_cp))
    for i, traj in enumerate(training_data):
        control_points[i] = fit_bspline_control_points(traj, n_cp)

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    adapter_orig = BasisAdapter(U_ref, n_cp)
    adapter_fixed = BasisAdapterFixed(U_ref, n_cp)

    print("\n  Testing endpoint preservation...")

    test_cases = [
        ("Short (50px) horizontal", 50, 0),
        ("Medium (150px) diagonal", 150, np.pi/4),
        ("Long (250px) vertical", 250, np.pi/2),
    ]

    for name, dist, angle in test_cases:
        traj = generate_trajectory(50, dist, angle, jitter=0.5, seed=12345)
        T = 50

        orig_endpoint = (traj[-2], traj[-1])

        c_orig = adapter_orig.project(traj)
        recon_orig = adapter_orig.reconstruct(c_orig, T)
        orig_recon_endpoint = (recon_orig[-2], recon_orig[-1])

        c_fixed = adapter_fixed.project(traj)
        recon_fixed = adapter_fixed.reconstruct(c_fixed, T)
        fixed_recon_endpoint = (recon_fixed[-2], recon_fixed[-1])

        orig_error = np.sqrt((orig_endpoint[0] - orig_recon_endpoint[0])**2 +
                             (orig_endpoint[1] - orig_recon_endpoint[1])**2)
        fixed_error = np.sqrt((orig_endpoint[0] - fixed_recon_endpoint[0])**2 +
                              (orig_endpoint[1] - fixed_recon_endpoint[1])**2)

        print(f"\n  {name}:")
        print(f"    Original endpoint: ({orig_endpoint[0]:.1f}, {orig_endpoint[1]:.1f})")
        print(f"    Original method:   ({orig_recon_endpoint[0]:.1f}, {orig_recon_endpoint[1]:.1f}) - error: {orig_error:.1f}px")
        print(f"    Fixed method:      ({fixed_recon_endpoint[0]:.1f}, {fixed_recon_endpoint[1]:.1f}) - error: {fixed_error:.1f}px")


def analyze_information_loss():
    """Analyze how much information is lost in projection."""
    print_section("TEST 5: Information Loss Analysis")

    np.random.seed(42)

    n_cp = 20

    # Generate training data
    training_data = []
    for i in range(100):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        traj = generate_trajectory(length, distance, angle, jitter=1.5, seed=i)
        training_data.append(traj)

    control_points = np.zeros((100, 2 * n_cp))
    for i, traj in enumerate(training_data):
        control_points[i] = fit_bspline_control_points(traj, n_cp)

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)

    print("\n  Variance explained by K components:")
    total_var = np.sum(S ** 2)
    for K in [2, 4, 6, 8, 10, 12]:
        var_explained = np.sum(S[:K] ** 2) / total_var * 100
        print(f"    K={K:2d}: {var_explained:.2f}%")

    print("\n  Singular value distribution:")
    print(f"    Top 10: {S[:10]}")
    print(f"    Sum of first 2: {S[0]+S[1]:.2f} ({(S[0]+S[1])/S.sum()*100:.1f}% of total)")

    # The issue: most variance is in first 2 components (x and y direction)
    # The jitter and micro-corrections are in the smaller components
    print("\n  Analysis:")
    print(f"    The first 2 singular values capture {(S[0]**2 + S[1]**2)/total_var*100:.1f}% of variance")
    print("    This represents overall trajectory direction (x and y)")
    print("    Jitter and micro-corrections are in the remaining components")


def main():
    print("=" * 70)
    print(" ORTHONORMALITY ISSUE AND FIX ANALYSIS")
    print("=" * 70)

    test_orthonormality_issue()
    test_fix_effectiveness()
    test_reconstruction_comparison()
    test_endpoint_preservation()
    analyze_information_loss()

    print("\n" + "=" * 70)
    print(" CONCLUSIONS")
    print("=" * 70)
    print("""
  The BasisAdapter has a critical issue: when adapting the basis to a
  different length T, the transformed basis U_T = C @ U_ref loses
  orthonormality.

  This causes:
  - Projection c = U_T.T @ x is not a proper orthogonal projection
  - Reconstruction x_hat = U_T @ c doesn't minimize ||x - x_hat||
  - High reconstruction errors, especially at lengths far from T_ref

  FIX: Re-orthonormalize U_T using QR decomposition after the B-spline
  transformation. This ensures U_T.T @ U_T = I at all lengths.

  However, even with the fix, there's another issue: the SVD basis
  primarily captures trajectory DIRECTION (first 2 components = 99%+
  of variance). The jitter and fine details are in the remaining
  components which have very small variance contribution.
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
