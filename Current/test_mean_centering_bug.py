#!/usr/bin/env python3
"""
Focused Test: Mean Centering Bug Investigation

The V3_PREPROCESSING_README.md identifies a potential bug (Issue 1, lines 109-157):

- Step 2 (learn_basis_from_trajectories) centers control points before SVD:
    mean = control_points.mean(axis=0)
    centered = control_points - mean

- Step 3 (project_trajectories) projects RAW trajectories WITHOUT subtracting mean:
    c = adapter.project(flat)  # flat is raw trajectory, not centered

This test thoroughly investigates whether this is indeed a bug and quantifies its impact.

Expected behavior (per mathematical derivation in README):
    1. If we learn basis from centered data: U_ref captures directions around the mean
    2. When projecting, we should also center: c = U_T.T @ (trajectory - mean_adapted)
    3. When reconstructing, we should add mean back: traj = U_T @ c + mean_adapted

Current behavior:
    1. Projection: c = U_T.T @ trajectory (no centering)
    2. Reconstruction: traj = U_T @ c (no mean added)

Usage:
    python test_mean_centering_bug.py
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
)


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def generate_realistic_trajectory(
    length: int,
    distance: float,
    angle: float,
    jitter: float = 1.5,
    seed: int = None
) -> np.ndarray:
    """Generate a realistic mouse trajectory."""
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 1, length)
    # Ease-out motion (decelerating)
    eased_t = 1 - (1 - t) ** 2

    # Base trajectory
    x = eased_t * distance * np.cos(angle)
    y = eased_t * distance * np.sin(angle)

    # Add slight curve
    curve = np.random.randn() * distance * 0.05 * np.sin(t * np.pi)
    x += curve * np.sin(angle + np.pi/2)
    y += curve * np.cos(angle + np.pi/2)

    # Add jitter
    jitter_x = gaussian_filter1d(np.random.randn(length) * jitter, sigma=1.0)
    jitter_y = gaussian_filter1d(np.random.randn(length) * jitter, sigma=1.0)
    x += jitter_x
    y += jitter_y

    # Interleave
    flat = np.empty(2 * length)
    flat[0::2] = x
    flat[1::2] = y
    return flat


def test_mean_centering_impact():
    """Test the impact of mean centering on reconstruction quality."""
    print_section("MEAN CENTERING BUG INVESTIGATION")

    np.random.seed(42)

    # Generate training data (variable lengths, like real data)
    print("\n  Generating training trajectories...")
    n_train = 100
    training_trajectories = []

    for i in range(n_train):
        length = np.random.randint(30, 80)
        distance = 50 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        jitter = 1 + np.random.rand() * 2

        traj = generate_realistic_trajectory(length, distance, angle, jitter, seed=i+100)
        training_trajectories.append(traj)

    # Learn basis with centering (like the current implementation)
    print("  Fitting control points and learning basis...")

    n_control_points = 20
    K = 10
    T_ref = n_control_points

    control_points = np.zeros((n_train, 2 * T_ref))
    for i, traj in enumerate(training_trajectories):
        cp = fit_bspline_control_points(traj, n_control_points)
        control_points[i] = cp

    # Compute mean and center (as Step 2 does)
    mean = control_points.mean(axis=0)
    centered = control_points - mean

    # SVD on centered data
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    print(f"\n  Basis learned:")
    print(f"    U_ref shape: {U_ref.shape}")
    print(f"    Mean norm: {np.linalg.norm(mean):.2f}")
    print(f"    Variance explained by K={K}: {(np.sum(S[:K]**2) / np.sum(S**2))*100:.2f}%")

    # Create adapter
    adapter = BasisAdapter(U_ref, T_ref)

    # Test on new trajectories
    print("\n  Testing on 50 new trajectories...")
    n_test = 50

    # Results storage
    results_no_center = []
    results_with_center = []

    for i in range(n_test):
        # Generate test trajectory
        length = np.random.randint(30, 80)
        distance = 50 + np.random.rand() * 200
        angle = np.random.rand() * 2 * np.pi
        jitter = 1 + np.random.rand() * 2

        traj = generate_realistic_trajectory(length, distance, angle, jitter, seed=1000+i)
        T = len(traj) // 2

        # Method 1: Current implementation (NO centering)
        c1 = adapter.project(traj)
        recon1 = adapter.reconstruct(c1, T)
        rmse1 = np.sqrt(np.mean((traj - recon1) ** 2))

        # Method 2: CORRECT implementation (WITH centering)
        mean_T = adapter.reconstruct_mean(mean, T)  # Transform mean to trajectory length
        traj_centered = traj - mean_T
        c2 = adapter.project(traj_centered)
        recon2 = adapter.reconstruct(c2, T) + mean_T  # Add mean back
        rmse2 = np.sqrt(np.mean((traj - recon2) ** 2))

        results_no_center.append(rmse1)
        results_with_center.append(rmse2)

    # Statistics
    no_center = np.array(results_no_center)
    with_center = np.array(results_with_center)

    print("\n  " + "-" * 60)
    print("  RESULTS:")
    print("  " + "-" * 60)

    print(f"\n  Method 1: Current (no centering during projection)")
    print(f"    Mean RMSE:   {no_center.mean():.4f} pixels")
    print(f"    Median RMSE: {np.median(no_center):.4f} pixels")
    print(f"    Max RMSE:    {no_center.max():.4f} pixels")

    print(f"\n  Method 2: Corrected (with centering)")
    print(f"    Mean RMSE:   {with_center.mean():.4f} pixels")
    print(f"    Median RMSE: {np.median(with_center):.4f} pixels")
    print(f"    Max RMSE:    {with_center.max():.4f} pixels")

    improvement = (no_center.mean() - with_center.mean()) / no_center.mean() * 100
    wins = np.sum(with_center < no_center)

    print(f"\n  Comparison:")
    print(f"    Mean RMSE improvement: {improvement:.1f}%")
    print(f"    Method 2 wins: {wins}/{n_test} trajectories ({wins/n_test*100:.0f}%)")

    # Verdict
    print("\n  " + "=" * 60)
    if improvement > 5:  # More than 5% improvement
        print("  VERDICT: BUG CONFIRMED")
        print("  ")
        print("  The current implementation (no centering during projection)")
        print(f"  is {improvement:.1f}% worse than the mathematically correct")
        print("  implementation (with centering).")
        print("  ")
        print("  RECOMMENDED FIX in project_trajectories():")
        print("  ")
        print("    # Transform mean to trajectory length")
        print("    mean_T = adapter.reconstruct_mean(mean, T)")
        print("    # Center trajectory before projection")
        print("    flat_centered = flat - mean_T")
        print("    # Project centered trajectory")
        print("    c = adapter.project(flat_centered)")
        is_bug = True
    elif improvement > 0:
        print("  VERDICT: MINOR ISSUE")
        print(f"  Centering improves RMSE by {improvement:.1f}%")
        print("  Consider adding centering for better accuracy.")
        is_bug = True
    else:
        print("  VERDICT: NOT A BUG")
        print("  Centering does not significantly improve reconstruction.")
        print("  Current implementation is acceptable.")
        is_bug = False

    print("  " + "=" * 60)

    return is_bug, improvement


def test_coefficient_interpretation():
    """Test how coefficients differ with and without centering."""
    print_section("COEFFICIENT INTERPRETATION")

    np.random.seed(42)

    # Generate trajectories with specific patterns
    print("\n  Testing with specific trajectory patterns...")

    patterns = [
        ("Short distance (50px)", 50, 0),
        ("Medium distance (150px)", 150, 0),
        ("Long distance (300px)", 300, 0),
        ("Upward motion", 100, np.pi/2),
        ("Diagonal motion", 100, np.pi/4),
    ]

    # First, learn a basis
    training_trajectories = []
    for i in range(100):
        length = np.random.randint(30, 70)
        distance = 50 + np.random.rand() * 250
        angle = np.random.rand() * 2 * np.pi
        traj = generate_realistic_trajectory(length, distance, angle, jitter=1.5, seed=i)
        training_trajectories.append(traj)

    n_cp = 20
    K = 8

    control_points = np.zeros((100, 2 * n_cp))
    for i, traj in enumerate(training_trajectories):
        cp = fit_bspline_control_points(traj, n_cp)
        control_points[i] = cp

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    adapter = BasisAdapter(U_ref, n_cp)

    print(f"\n  {'Pattern':<25} {'c1[0]':>10} {'c2[0]':>10} {'Diff':>10}")
    print("  " + "-" * 58)

    for name, dist, angle in patterns:
        traj = generate_realistic_trajectory(50, dist, angle, jitter=0.5, seed=999)
        T = 50

        # Without centering
        c1 = adapter.project(traj)

        # With centering
        mean_T = adapter.reconstruct_mean(mean, T)
        c2 = adapter.project(traj - mean_T)

        diff = c1[0] - c2[0]
        print(f"  {name:<25} {c1[0]:>10.2f} {c2[0]:>10.2f} {diff:>10.2f}")

    print("\n  Note: c1[0] (no centering) absorbs the mean component")
    print("        c2[0] (with centering) represents deviation from mean")
    print("        The difference correlates with trajectory's distance from mean")


def test_generation_consistency():
    """Test if generation (adding mean back) is consistent with projection."""
    print_section("GENERATION CONSISTENCY CHECK")

    print("\n  This checks if generate_trajectory_v3.py correctly adds mean back.")
    print("  If preprocessing SUBTRACTS mean before projection,")
    print("  then generation must ADD mean after reconstruction.")
    print("  ")
    print("  Currently:")
    print("    - Preprocessing: does NOT subtract mean during projection")
    print("    - Generation: DOES add mean during reconstruction")
    print("  ")
    print("  This is INCONSISTENT and may cause mean mismatch!")

    # Demonstrate the issue
    np.random.seed(42)

    # Simple test
    training_trajectories = []
    for i in range(50):
        length = np.random.randint(40, 60)
        distance = 100 + np.random.rand() * 100
        angle = np.random.rand() * 2 * np.pi
        traj = generate_realistic_trajectory(length, distance, angle, jitter=1, seed=i)
        training_trajectories.append(traj)

    n_cp = 16
    K = 6

    control_points = np.zeros((50, 2 * n_cp))
    for i, traj in enumerate(training_trajectories):
        cp = fit_bspline_control_points(traj, n_cp)
        control_points[i] = cp

    mean = control_points.mean(axis=0)
    centered = control_points - mean
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    adapter = BasisAdapter(U_ref, n_cp)

    # Test trajectory
    test_traj = training_trajectories[0]
    T = len(test_traj) // 2

    # Current preprocessing behavior (no mean subtraction)
    c_current = adapter.project(test_traj)

    # Current generation behavior (adds mean)
    mean_T = adapter.reconstruct_mean(mean, T)
    recon_current = adapter.reconstruct(c_current, T) + mean_T

    # Error with current inconsistent approach
    error_current = np.sqrt(np.mean((test_traj - recon_current) ** 2))

    # Correct approach 1: Neither subtracts nor adds mean
    recon_neither = adapter.reconstruct(c_current, T)
    error_neither = np.sqrt(np.mean((test_traj - recon_neither) ** 2))

    # Correct approach 2: Both subtracts and adds mean
    c_correct = adapter.project(test_traj - mean_T)
    recon_correct = adapter.reconstruct(c_correct, T) + mean_T
    error_correct = np.sqrt(np.mean((test_traj - recon_correct) ** 2))

    print("\n  Consistency analysis:")
    print(f"    Current (no sub, adds): RMSE = {error_current:.4f}")
    print(f"    Neither (no sub, no add): RMSE = {error_neither:.4f}")
    print(f"    Both (sub & add): RMSE = {error_correct:.4f}")

    print("\n  " + "-" * 60)
    if error_neither < error_current and error_neither < error_correct:
        print("  Best approach: Don't use mean at all")
        print("  Recommendation: Remove mean addition from generate_trajectory_v3.py")
    elif error_correct < error_current:
        print("  Best approach: Use mean consistently (subtract then add)")
        print("  Recommendation: Add mean subtraction to preprocessing")
    else:
        print("  Current approach happens to work reasonably")

    print("  " + "-" * 60)


def main():
    print("=" * 70)
    print(" MEAN CENTERING BUG INVESTIGATION")
    print(" Validating Issue 1 from V3_PREPROCESSING_README.md")
    print("=" * 70)

    is_bug, improvement = test_mean_centering_impact()
    test_coefficient_interpretation()
    test_generation_consistency()

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    if is_bug:
        print(f"\n  BUG CONFIRMED: Mean centering improves reconstruction by {improvement:.1f}%")
        print("\n  REQUIRED FIXES:")
        print("  1. In preprocess_singular_v3_basis_transform.py, project_trajectories():")
        print("     - Before projection: mean_T = adapter.reconstruct_mean(mean, T)")
        print("     - Center trajectory: flat_centered = flat - mean_T")
        print("     - Project centered: c = adapter.project(flat_centered)")
        print("")
        print("  2. Verify generate_trajectory_v3.py adds mean back correctly")
        print("")
    else:
        print("\n  No critical bug found in mean centering.")

    print("=" * 70)

    return 0 if not is_bug else 1


if __name__ == "__main__":
    sys.exit(main())
