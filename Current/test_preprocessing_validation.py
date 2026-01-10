#!/usr/bin/env python3
"""
Comprehensive Validation Test Suite for preprocess_singular_v3_basis_transform.py

This script validates the preprocessing code against V3_PREPROCESSING_README.md
to ensure the implementation matches the documented behavior.

Key validation areas:
1. Step-by-step function correctness
2. Mean centering behavior (potential bug identified in README)
3. Comparison with bspline_basis.py's learn_reference_basis()
4. Data integrity and shape validation
5. Jitter preservation (V3's key innovation)
6. Round-trip reconstruction accuracy

Usage:
    python test_preprocessing_validation.py

Author: Claude Code Validation
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import modules under test
from bspline_basis import (
    BasisAdapter,
    learn_reference_basis,
    fit_bspline_control_points,
    evaluate_control_point_fit,
    build_bspline_basis_matrix,
    build_interleaved_basis_matrix,
)

# Import preprocessing functions
from preprocess_singular_v3_basis_transform import (
    load_trajectories,
    learn_basis_from_trajectories,
    project_trajectories,
    generate_group_anchors,
    compute_residuals,
    split_data,
    get_orientation_id,
    get_distance_group_id,
    trajectory_to_flat,
    flat_to_trajectory,
    DEFAULT_CONFIG,
    ORIENTATIONS,
    DISTANCE_GROUPS,
    NUM_ORIENTATIONS,
    NUM_DISTANCE_GROUPS,
)

# Test configuration
TEST_RESULTS = {
    'passed': 0,
    'failed': 0,
    'warnings': 0,
    'details': []
}


def log_result(test_name: str, passed: bool, message: str = "", warning: bool = False):
    """Log a test result."""
    if warning:
        status = "⚠️ WARNING"
        TEST_RESULTS['warnings'] += 1
    elif passed:
        status = "✓ PASSED"
        TEST_RESULTS['passed'] += 1
    else:
        status = "✗ FAILED"
        TEST_RESULTS['failed'] += 1

    result = f"  {status}: {test_name}"
    if message:
        result += f" - {message}"
    print(result)
    TEST_RESULTS['details'].append((test_name, passed, message, warning))


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


# =============================================================================
# SYNTHETIC TRAJECTORY GENERATION
# =============================================================================

def generate_synthetic_trajectory(
    length: int,
    ideal_distance: float,
    orientation_angle: float,  # radians
    add_jitter: bool = True,
    jitter_magnitude: float = 1.0,
    seed: Optional[int] = None
) -> Dict:
    """
    Generate a synthetic mouse trajectory for testing.

    Args:
        length: Number of points
        ideal_distance: Euclidean distance from start to end
        orientation_angle: Direction in radians (0 = East, pi/2 = North in math coords)
        add_jitter: Whether to add realistic micro-corrections
        jitter_magnitude: Standard deviation of jitter in pixels
        seed: Random seed for reproducibility

    Returns:
        Dictionary matching the JSON trajectory format
    """
    if seed is not None:
        np.random.seed(seed)

    # Base trajectory: eased movement (fast start, slow end - like real mouse)
    t = np.linspace(0, 1, length)
    eased_t = 1 - (1 - t) ** 2  # Ease-out

    # Compute endpoint
    dx = ideal_distance * np.cos(orientation_angle)
    dy = ideal_distance * np.sin(orientation_angle)

    # Base coordinates
    x = eased_t * dx
    y = eased_t * dy

    # Add slight curve (realistic mouse paths aren't perfectly straight)
    curve_magnitude = ideal_distance * 0.05 * np.random.randn()
    curve = curve_magnitude * np.sin(t * np.pi)
    x += curve * np.sin(orientation_angle + np.pi/2)
    y += curve * np.cos(orientation_angle + np.pi/2)

    # Add jitter (micro-corrections)
    if add_jitter:
        from scipy.ndimage import gaussian_filter1d
        # Jitter is slightly correlated (not pure noise)
        jitter_x = gaussian_filter1d(np.random.randn(length) * jitter_magnitude, sigma=1.0)
        jitter_y = gaussian_filter1d(np.random.randn(length) * jitter_magnitude, sigma=1.0)
        x += jitter_x
        y += jitter_y

    # Convert to integers (like real mouse data)
    x = np.round(x).astype(int) + 500  # Offset to positive screen coords
    y = np.round(y).astype(int) + 500

    # Generate timestamps (approximately 8ms between samples, like 120Hz)
    t_stamps = np.cumsum(np.random.randint(6, 12, size=length))

    # Compute actual distance (path length)
    actual_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        't': t_stamps.tolist(),
        'ideal_distance': ideal_distance,
        'actual_distance': float(actual_distance),
        'original_length': length,
        'extracted_length': length,
    }


def generate_test_dataset(n_trajectories: int = 50, seed: int = 42) -> List[Dict]:
    """Generate a diverse test dataset."""
    np.random.seed(seed)
    trajectories = []

    for i in range(n_trajectories):
        # Vary parameters
        length = np.random.randint(25, 100)
        ideal_distance = 30 + np.random.rand() * 300  # 30-330 pixels
        orientation_angle = np.random.rand() * 2 * np.pi
        jitter = np.random.rand() * 2 + 0.5  # 0.5-2.5 pixels

        traj = generate_synthetic_trajectory(
            length=length,
            ideal_distance=ideal_distance,
            orientation_angle=orientation_angle,
            add_jitter=True,
            jitter_magnitude=jitter,
            seed=seed + i
        )
        trajectories.append(traj)

    return trajectories


# =============================================================================
# TEST 1: LOAD TRAJECTORIES
# =============================================================================

def test_load_trajectories():
    """Test Step 1: load_trajectories function."""
    print_section("TEST 1: load_trajectories")

    # Use the example trajectories
    data_dir = Path(__file__).parent / "trajectories_examples"

    if not data_dir.exists():
        log_result("Directory exists", False, f"{data_dir} not found")
        return

    config = DEFAULT_CONFIG.copy()
    config['min_length'] = 15  # Lower threshold for test data

    trajectories = load_trajectories(data_dir, config)

    # Test 1.1: Basic loading
    log_result("Trajectories loaded", len(trajectories) > 0,
               f"Loaded {len(trajectories)} trajectories")

    if len(trajectories) == 0:
        return

    # Test 1.2: Required fields present
    required_fields = ['x', 'y', 'flat', 'length', 'ideal_distance',
                       'orientation_id', 'distance_group_id']
    first_traj = trajectories[0]
    all_fields = all(f in first_traj for f in required_fields)
    log_result("Required fields present", all_fields,
               f"Fields: {list(first_traj.keys())}")

    # Test 1.3: Flat format correct
    traj = trajectories[0]
    flat_correct = len(traj['flat']) == 2 * traj['length']
    log_result("Flat format correct", flat_correct,
               f"flat={len(traj['flat'])}, expected={2*traj['length']}")

    # Test 1.4: Coordinates start at origin (relative)
    x, y = flat_to_trajectory(traj['flat'])
    origin_correct = abs(x[0]) < 1e-10 and abs(y[0]) < 1e-10
    log_result("Relative coords (origin)", origin_correct,
               f"Start point: ({x[0]:.6f}, {y[0]:.6f})")

    # Test 1.5: Orientation ID in valid range
    orient_valid = all(0 <= t['orientation_id'] < NUM_ORIENTATIONS
                       for t in trajectories)
    log_result("Orientation IDs valid", orient_valid,
               f"Range: 0-{NUM_ORIENTATIONS-1}")

    # Test 1.6: Distance group ID in valid range
    dist_valid = all(0 <= t['distance_group_id'] < NUM_DISTANCE_GROUPS
                     for t in trajectories)
    log_result("Distance group IDs valid", dist_valid,
               f"Range: 0-{NUM_DISTANCE_GROUPS-1}")

    return trajectories


# =============================================================================
# TEST 2: LEARN BASIS FROM TRAJECTORIES
# =============================================================================

def test_learn_basis(trajectories: List[Dict]):
    """Test Step 2: learn_basis_from_trajectories function."""
    print_section("TEST 2: learn_basis_from_trajectories")

    if not trajectories or len(trajectories) < 3:
        log_result("Sufficient data", False, "Need at least 3 trajectories")
        return None, None, None

    n_control_points = 16
    K = min(6, len(trajectories))

    U_ref, T_ref, mean, stats = learn_basis_from_trajectories(
        trajectories,
        n_control_points=n_control_points,
        K=K,
        center_data=True
    )

    # Test 2.1: U_ref shape
    expected_shape = (2 * n_control_points, K)
    shape_correct = U_ref.shape == expected_shape
    log_result("U_ref shape", shape_correct,
               f"Got {U_ref.shape}, expected {expected_shape}")

    # Test 2.2: T_ref value
    tref_correct = T_ref == n_control_points
    log_result("T_ref value", tref_correct,
               f"Got {T_ref}, expected {n_control_points}")

    # Test 2.3: Mean shape (if centering enabled)
    if mean is not None:
        mean_shape_correct = len(mean) == 2 * n_control_points
        log_result("Mean shape", mean_shape_correct,
                   f"Got {len(mean)}, expected {2*n_control_points}")
    else:
        log_result("Mean exists", False, "Mean is None but center_data=True")

    # Test 2.4: Orthonormality of U_ref
    orthonormality = U_ref.T @ U_ref
    orthonormality_error = np.max(np.abs(orthonormality - np.eye(K)))
    ortho_ok = orthonormality_error < 1e-10
    log_result("U_ref orthonormal", ortho_ok,
               f"Max deviation from I: {orthonormality_error:.2e}")

    # Test 2.5: Variance explained
    if 'variance_explained' in stats:
        var_ok = 0 < stats['variance_explained'] <= 1
        log_result("Variance explained valid", var_ok,
                   f"{stats['variance_explained']*100:.1f}%")

    # Test 2.6: Compare with bspline_basis.learn_reference_basis()
    print("\n  Comparing with bspline_basis.learn_reference_basis()...")

    # Extract flat trajectories for the reference function
    flat_trajectories = [t['flat'] for t in trajectories]

    U_ref_lib, T_ref_lib, mean_lib = learn_reference_basis(
        flat_trajectories,
        n_control_points=n_control_points,
        K=K,
        center_data=True
    )

    # The bases should span the same space (columns may differ by sign)
    # Check by comparing the projection matrices U @ U.T
    proj_impl = U_ref @ U_ref.T
    proj_lib = U_ref_lib @ U_ref_lib.T
    proj_diff = np.max(np.abs(proj_impl - proj_lib))

    proj_match = proj_diff < 1e-8
    log_result("Matches learn_reference_basis()", proj_match,
               f"Projection matrix diff: {proj_diff:.2e}")

    if not proj_match:
        log_result("Implementation differs from library", True,
                   "Reimplementation may have slight numerical differences",
                   warning=True)

    return U_ref, T_ref, mean


# =============================================================================
# TEST 3: PROJECT TRAJECTORIES
# =============================================================================

def test_project_trajectories(trajectories: List[Dict], U_ref: np.ndarray,
                               T_ref: int, mean: Optional[np.ndarray]):
    """Test Step 3: project_trajectories function."""
    print_section("TEST 3: project_trajectories")

    if U_ref is None:
        log_result("U_ref available", False, "Cannot test without U_ref")
        return None, None

    coefficients, adapter, stats = project_trajectories(
        trajectories, U_ref, T_ref, mean
    )

    K = U_ref.shape[1]
    N = len(trajectories)

    # Test 3.1: Coefficients shape
    shape_correct = coefficients.shape == (N, K)
    log_result("Coefficients shape", shape_correct,
               f"Got {coefficients.shape}, expected ({N}, {K})")

    # Test 3.2: No NaN/Inf values
    no_nan = not np.any(np.isnan(coefficients))
    no_inf = not np.any(np.isinf(coefficients))
    log_result("No NaN/Inf values", no_nan and no_inf,
               f"NaN: {np.sum(np.isnan(coefficients))}, Inf: {np.sum(np.isinf(coefficients))}")

    # Test 3.3: Reconstruction error reasonable
    if 'reconstruction_rmse_mean' in stats:
        rmse_ok = stats['reconstruction_rmse_mean'] < 50  # pixels
        log_result("Reconstruction RMSE", rmse_ok,
                   f"Mean RMSE: {stats['reconstruction_rmse_mean']:.4f} pixels")

    # Test 3.4: Round-trip reconstruction
    print("\n  Testing round-trip reconstruction on sample trajectories...")

    sample_indices = np.random.choice(len(trajectories), min(5, len(trajectories)), replace=False)
    round_trip_errors = []

    for i in sample_indices:
        traj = trajectories[i]
        flat = traj['flat']
        T = traj['length']
        c = coefficients[i]

        # Reconstruct
        recon = adapter.reconstruct(c, T)
        error = np.sqrt(np.mean((flat - recon) ** 2))
        round_trip_errors.append(error)

    mean_rt_error = np.mean(round_trip_errors)
    rt_ok = mean_rt_error < 50
    log_result("Round-trip reconstruction", rt_ok,
               f"Mean RMSE: {mean_rt_error:.4f} pixels")

    # Test 3.5: CRITICAL - Mean centering investigation
    print("\n  CRITICAL: Investigating mean centering behavior...")

    # The README identifies this as a potential bug:
    # Step 2 centers control points before SVD
    # Step 3 projects RAW trajectories without subtracting mean

    # Test if this matters
    test_traj = trajectories[0]
    flat = test_traj['flat']
    T = test_traj['length']

    # Method 1: Current implementation (project raw)
    c1 = adapter.project(flat)

    # Method 2: What the README suggests might be correct
    # Subtract mean (transformed to trajectory length) before projection
    if mean is not None:
        mean_T = adapter.reconstruct_mean(mean, T)
        flat_centered = flat - mean_T
        c2 = adapter.project(flat_centered)

        coeff_diff = np.linalg.norm(c1 - c2)
        coeff_norm = np.linalg.norm(c1)
        relative_diff = coeff_diff / coeff_norm if coeff_norm > 1e-10 else 0

        if relative_diff > 0.1:
            log_result("Mean centering impact", True,
                       f"Centering changes coefficients by {relative_diff*100:.1f}% - INVESTIGATE",
                       warning=True)
        else:
            log_result("Mean centering impact", True,
                       f"Centering has {relative_diff*100:.2f}% effect on coefficients")

        # Check which reconstruction is better
        recon1 = adapter.reconstruct(c1, T)
        recon2 = adapter.reconstruct(c2, T) + mean_T  # Add mean back

        error1 = np.sqrt(np.mean((flat - recon1) ** 2))
        error2 = np.sqrt(np.mean((flat - recon2) ** 2))

        print(f"    Method 1 (no centering) RMSE: {error1:.4f}")
        print(f"    Method 2 (with centering) RMSE: {error2:.4f}")

        if error2 < error1 * 0.9:  # Significantly better
            log_result("Centering improves reconstruction", False,
                       "POTENTIAL BUG: Adding centering reduces error significantly",
                       warning=True)
        elif error1 < error2 * 0.9:
            log_result("Current approach better", True,
                       "Not centering gives better reconstruction")
        else:
            log_result("Centering effect negligible", True,
                       "Both approaches give similar results")

    return coefficients, adapter


# =============================================================================
# TEST 4: GENERATE GROUP ANCHORS
# =============================================================================

def test_generate_anchors(trajectories: List[Dict], coefficients: np.ndarray):
    """Test Step 4: generate_group_anchors function."""
    print_section("TEST 4: generate_group_anchors")

    if coefficients is None:
        log_result("Coefficients available", False, "Cannot test without coefficients")
        return None

    anchors_per_group = 4

    group_anchors, group_stats = generate_group_anchors(
        trajectories,
        coefficients,
        anchors_per_group=anchors_per_group,
        random_seed=42
    )

    K = coefficients.shape[1]

    # Test 4.1: Anchors dictionary exists
    log_result("Anchors dict created", isinstance(group_anchors, dict),
               f"Type: {type(group_anchors)}")

    # Test 4.2: Anchor shapes correct
    all_shapes_ok = True
    total_anchors = 0
    for key, anchors in group_anchors.items():
        if anchors is not None:
            if anchors.ndim != 2 or anchors.shape[1] != K:
                all_shapes_ok = False
            total_anchors += len(anchors)

    log_result("Anchor shapes correct", all_shapes_ok,
               f"Total anchors: {total_anchors}")

    # Test 4.3: Anchors are in coefficient space (reasonable magnitudes)
    anchor_norms = []
    for anchors in group_anchors.values():
        if anchors is not None:
            anchor_norms.extend(np.linalg.norm(anchors, axis=1).tolist())

    if anchor_norms:
        coeff_norms = np.linalg.norm(coefficients, axis=1)
        anchor_range_ok = (min(anchor_norms) > 0 and
                           max(anchor_norms) < np.max(coeff_norms) * 5)
        log_result("Anchor magnitudes reasonable", anchor_range_ok,
                   f"Anchor norm range: [{min(anchor_norms):.2f}, {max(anchor_norms):.2f}]")

    return group_anchors


# =============================================================================
# TEST 5: COMPUTE RESIDUALS
# =============================================================================

def test_compute_residuals(trajectories: List[Dict], coefficients: np.ndarray,
                            group_anchors: Dict):
    """Test Step 5: compute_residuals function."""
    print_section("TEST 5: compute_residuals")

    if coefficients is None or group_anchors is None:
        log_result("Prerequisites available", False, "Missing coefficients or anchors")
        return None, None

    residuals, anchor_indices, stats = compute_residuals(
        trajectories, coefficients, group_anchors
    )

    N = len(trajectories)
    K = coefficients.shape[1]

    # Test 5.1: Residuals shape
    shape_correct = residuals.shape == (N, K)
    log_result("Residuals shape", shape_correct,
               f"Got {residuals.shape}, expected ({N}, {K})")

    # Test 5.2: Anchor indices shape
    idx_shape_correct = anchor_indices.shape == (N,)
    log_result("Anchor indices shape", idx_shape_correct,
               f"Got {anchor_indices.shape}, expected ({N},)")

    # Test 5.3: Residual = coefficient - anchor
    print("\n  Verifying residual computation...")
    verification_errors = []

    for i in range(min(10, N)):
        traj = trajectories[i]
        key = (traj['distance_group_id'], traj['orientation_id'])
        anchors = group_anchors[key]

        if anchors is not None and anchor_indices[i] >= 0:
            expected_residual = coefficients[i] - anchors[anchor_indices[i]]
            actual_residual = residuals[i]
            error = np.max(np.abs(expected_residual - actual_residual))
            verification_errors.append(error)

    if verification_errors:
        max_error = max(verification_errors)
        formula_correct = max_error < 1e-10
        log_result("Residual formula correct", formula_correct,
                   f"Max verification error: {max_error:.2e}")

    # Test 5.4: Residuals smaller than coefficients
    residual_norms = np.linalg.norm(residuals, axis=1)
    coeff_norms = np.linalg.norm(coefficients, axis=1)

    reduction = residual_norms.mean() / coeff_norms.mean()
    reduction_ok = reduction < 1.0  # Should be reduced
    log_result("Residuals reduced from coefficients", reduction_ok,
               f"Mean reduction ratio: {reduction:.2%}")

    return residuals, anchor_indices


# =============================================================================
# TEST 6-8: SPLIT, SAVE, VERIFY
# =============================================================================

def test_split_save_verify(trajectories: List[Dict], coefficients: np.ndarray,
                            residuals: np.ndarray, anchor_indices: np.ndarray):
    """Test Steps 6-8: split_data, save, verify."""
    print_section("TESTS 6-8: Split, Save, Verify")

    if any(x is None for x in [coefficients, residuals, anchor_indices]):
        log_result("Prerequisites available", False, "Missing required data")
        return

    config = DEFAULT_CONFIG.copy()

    # Test 6: Split data
    print("\n  Testing split_data...")
    splits = split_data(trajectories, coefficients, residuals, anchor_indices, config)

    # Test 6.1: All splits present
    splits_present = all(s in splits for s in ['train', 'val', 'test'])
    log_result("All splits present", splits_present,
               f"Splits: {list(splits.keys())}")

    # Test 6.2: Split sizes reasonable
    train_size = len(splits['train']['coefficients'])
    val_size = len(splits['val']['coefficients'])
    test_size = len(splits['test']['coefficients'])
    total = train_size + val_size + test_size

    sizes_ok = total == len(trajectories)
    log_result("Split sizes sum correctly", sizes_ok,
               f"Train: {train_size}, Val: {val_size}, Test: {test_size}, Total: {total}")

    # Test 6.3: original_lengths saved (CRITICAL for V3)
    lengths_saved = 'original_lengths' in splits['train']
    log_result("original_lengths saved (CRITICAL)", lengths_saved,
               "This is required for V3 basis adaptation during generation")

    # Test 6.4: Verify original_lengths match
    if lengths_saved:
        train_idx = splits['train']['indices'] if 'indices' in splits['train'] else None
        if train_idx is not None:
            expected_lengths = np.array([trajectories[i]['length'] for i in train_idx])
            actual_lengths = splits['train']['original_lengths']
            lengths_match = np.array_equal(expected_lengths, actual_lengths)
            log_result("original_lengths values correct", lengths_match,
                       f"Sample: expected={expected_lengths[:3]}, actual={actual_lengths[:3]}")


# =============================================================================
# TEST: JITTER PRESERVATION (V3's KEY INNOVATION)
# =============================================================================

def test_jitter_preservation(trajectories: List[Dict], adapter: BasisAdapter,
                              coefficients: np.ndarray):
    """Test V3's key innovation: jitter/micro-correction preservation."""
    print_section("TEST: Jitter Preservation (V3 Key Innovation)")

    if adapter is None or coefficients is None:
        log_result("Prerequisites available", False, "Missing adapter or coefficients")
        return

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        log_result("scipy available", False, "scipy required for jitter test")
        return

    def compute_jitter_metric(coords):
        """Jitter = RMS of second derivative (acceleration)."""
        if len(coords) < 3:
            return 0
        velocity = np.diff(coords)
        acceleration = np.diff(velocity)
        return np.sqrt(np.mean(acceleration ** 2))

    print("\n  Measuring jitter preservation across trajectories...")

    jitter_ratios_x = []
    jitter_ratios_y = []

    sample_size = min(20, len(trajectories))
    sample_indices = np.random.choice(len(trajectories), sample_size, replace=False)

    for i in sample_indices:
        traj = trajectories[i]
        flat = traj['flat']
        T = traj['length']
        c = coefficients[i]

        x_orig = flat[0::2]
        y_orig = flat[1::2]

        recon = adapter.reconstruct(c, T)
        x_recon = recon[0::2]
        y_recon = recon[1::2]

        orig_jitter_x = compute_jitter_metric(x_orig)
        orig_jitter_y = compute_jitter_metric(y_orig)
        recon_jitter_x = compute_jitter_metric(x_recon)
        recon_jitter_y = compute_jitter_metric(y_recon)

        if orig_jitter_x > 1e-10:
            jitter_ratios_x.append(recon_jitter_x / orig_jitter_x)
        if orig_jitter_y > 1e-10:
            jitter_ratios_y.append(recon_jitter_y / orig_jitter_y)

    if jitter_ratios_x and jitter_ratios_y:
        mean_preservation_x = np.mean(jitter_ratios_x)
        mean_preservation_y = np.mean(jitter_ratios_y)
        mean_preservation = (mean_preservation_x + mean_preservation_y) / 2

        # V3 should preserve more than 30% of jitter
        # (V2 with B-spline smoothing typically preserves <20%)
        jitter_ok = mean_preservation > 0.3
        log_result("Jitter preservation", jitter_ok,
                   f"Mean preservation: {mean_preservation*100:.1f}%")

        print(f"    X-axis: {mean_preservation_x*100:.1f}%")
        print(f"    Y-axis: {mean_preservation_y*100:.1f}%")

        if mean_preservation < 0.3:
            log_result("Low jitter preservation", True,
                       "Consider increasing n_control_points or K",
                       warning=True)
        elif mean_preservation > 0.8:
            log_result("Excellent jitter preservation", True,
                       f"{mean_preservation*100:.1f}% - V3 working as intended!")
    else:
        log_result("Jitter computation", False, "Could not compute jitter ratios")


# =============================================================================
# TEST: COMPARE IMPLEMENTATIONS
# =============================================================================

def test_implementation_comparison():
    """Compare inline implementation vs bspline_basis.py functions."""
    print_section("TEST: Implementation Comparison")

    # Generate synthetic data for comparison
    np.random.seed(42)

    trajectories = []
    for i in range(30):
        T = np.random.randint(30, 70)
        t = np.linspace(0, 1, T)

        # Random trajectory
        x = t * (50 + np.random.rand() * 100) + np.random.randn(T) * 2
        y = (20 + np.random.rand() * 50) * np.sin(t * np.pi) + np.random.randn(T) * 2

        flat = np.empty(2*T)
        flat[0::2] = x - x[0]
        flat[1::2] = y - y[0]
        trajectories.append(flat)

    n_cp = 16
    K = 6

    # Method 1: bspline_basis.py's learn_reference_basis
    print("\n  Testing bspline_basis.learn_reference_basis()...")
    U_lib, T_lib, mean_lib = learn_reference_basis(
        trajectories, n_control_points=n_cp, K=K, center_data=True
    )

    # Method 2: Manual control point fitting + SVD (like preprocessing does)
    print("  Testing inline implementation...")

    control_points = np.zeros((len(trajectories), 2 * n_cp))
    for i, traj in enumerate(trajectories):
        cp = fit_bspline_control_points(traj, n_cp)
        control_points[i] = cp

    mean_inline = control_points.mean(axis=0)
    centered = control_points - mean_inline
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_inline = U[:, :K]

    # Compare projection spaces
    proj_lib = U_lib @ U_lib.T
    proj_inline = U_inline @ U_inline.T
    proj_diff = np.max(np.abs(proj_lib - proj_inline))

    spaces_match = proj_diff < 1e-8
    log_result("Projection spaces match", spaces_match,
               f"Max difference: {proj_diff:.2e}")

    # Compare means
    mean_diff = np.max(np.abs(mean_lib - mean_inline))
    means_match = mean_diff < 1e-10
    log_result("Means match", means_match,
               f"Max difference: {mean_diff:.2e}")

    # Test reconstruction quality with both
    adapter_lib = BasisAdapter(U_lib, T_lib)
    adapter_inline = BasisAdapter(U_inline, n_cp)

    test_traj = trajectories[0]
    T = len(test_traj) // 2

    c_lib = adapter_lib.project(test_traj)
    c_inline = adapter_inline.project(test_traj)

    recon_lib = adapter_lib.reconstruct(c_lib, T)
    recon_inline = adapter_inline.reconstruct(c_inline, T)

    error_lib = np.sqrt(np.mean((test_traj - recon_lib) ** 2))
    error_inline = np.sqrt(np.mean((test_traj - recon_inline) ** 2))

    errors_similar = abs(error_lib - error_inline) < 1
    log_result("Reconstruction errors similar", errors_similar,
               f"Library: {error_lib:.4f}, Inline: {error_inline:.4f}")


# =============================================================================
# TEST: END-TO-END INTEGRATION
# =============================================================================

def test_end_to_end():
    """Run end-to-end test using example trajectories."""
    print_section("TEST: End-to-End Integration")

    data_dir = Path(__file__).parent / "trajectories_examples"

    if not data_dir.exists():
        log_result("Example data available", False, f"{data_dir} not found")
        return

    # Run the full pipeline with example data
    config = DEFAULT_CONFIG.copy()
    config['min_length'] = 15
    config['K'] = 6
    config['n_control_points'] = 16

    print("\n  Step 1: Loading trajectories...")
    trajectories = load_trajectories(data_dir, config)

    if len(trajectories) < 3:
        log_result("Sufficient trajectories", False,
                   f"Only {len(trajectories)} trajectories loaded")
        return

    print("  Step 2: Learning basis...")
    U_ref, T_ref, mean, _ = learn_basis_from_trajectories(
        trajectories, config['n_control_points'], config['K'], True
    )

    print("  Step 3: Projecting trajectories...")
    coefficients, adapter, _ = project_trajectories(
        trajectories, U_ref, T_ref, mean
    )

    print("  Step 4: Generating anchors...")
    group_anchors, _ = generate_group_anchors(
        trajectories, coefficients, config['anchors_per_group'], config['random_seed']
    )

    print("  Step 5: Computing residuals...")
    residuals, anchor_indices, _ = compute_residuals(
        trajectories, coefficients, group_anchors
    )

    print("  Step 6: Splitting data...")
    splits = split_data(
        trajectories, coefficients, residuals, anchor_indices, config
    )

    # Validate end-to-end
    all_valid = True

    # Check data integrity
    train_data = splits['train']
    if len(train_data['coefficients']) > 0:
        # Reconstruct a training sample
        idx = 0
        c = train_data['coefficients'][idx]
        T = train_data['original_lengths'][idx]

        recon = adapter.reconstruct(c, T)

        # Find original trajectory (need to use the index mapping)
        train_indices = train_data.get('indices', None)
        if train_indices is not None:
            orig_flat = trajectories[train_indices[idx]]['flat']
            error = np.sqrt(np.mean((orig_flat - recon) ** 2))

            reconstruction_ok = error < 50
            log_result("End-to-end reconstruction", reconstruction_ok,
                       f"RMSE: {error:.4f} pixels")
            all_valid = all_valid and reconstruction_ok

    log_result("End-to-end pipeline", all_valid,
               "All steps completed successfully")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all validation tests."""
    print("=" * 70)
    print(" PREPROCESSING VALIDATION TEST SUITE")
    print(" Testing preprocess_singular_v3_basis_transform.py")
    print(" Against V3_PREPROCESSING_README.md")
    print("=" * 70)

    np.random.seed(42)

    # Run tests in order
    trajectories = test_load_trajectories()

    U_ref, T_ref, mean = None, None, None
    if trajectories:
        U_ref, T_ref, mean = test_learn_basis(trajectories)

    coefficients, adapter = None, None
    if U_ref is not None:
        coefficients, adapter = test_project_trajectories(
            trajectories, U_ref, T_ref, mean
        )

    group_anchors = None
    if coefficients is not None:
        group_anchors = test_generate_anchors(trajectories, coefficients)

    residuals, anchor_indices = None, None
    if group_anchors is not None:
        residuals, anchor_indices = test_compute_residuals(
            trajectories, coefficients, group_anchors
        )

    if residuals is not None:
        test_split_save_verify(trajectories, coefficients, residuals, anchor_indices)

    if adapter is not None:
        test_jitter_preservation(trajectories, adapter, coefficients)

    test_implementation_comparison()
    test_end_to_end()

    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)

    total = TEST_RESULTS['passed'] + TEST_RESULTS['failed']
    print(f"\n  PASSED:   {TEST_RESULTS['passed']}/{total}")
    print(f"  FAILED:   {TEST_RESULTS['failed']}/{total}")
    print(f"  WARNINGS: {TEST_RESULTS['warnings']}")

    if TEST_RESULTS['failed'] > 0:
        print("\n  FAILED TESTS:")
        for name, passed, msg, warning in TEST_RESULTS['details']:
            if not passed and not warning:
                print(f"    - {name}: {msg}")

    if TEST_RESULTS['warnings'] > 0:
        print("\n  WARNINGS:")
        for name, passed, msg, warning in TEST_RESULTS['details']:
            if warning:
                print(f"    - {name}: {msg}")

    print("\n" + "=" * 70)

    if TEST_RESULTS['failed'] == 0:
        print(" ALL TESTS PASSED")
        if TEST_RESULTS['warnings'] > 0:
            print(f" ({TEST_RESULTS['warnings']} warnings to review)")
    else:
        print(f" {TEST_RESULTS['failed']} TESTS FAILED - REVIEW REQUIRED")

    print("=" * 70)

    return TEST_RESULTS['failed'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
