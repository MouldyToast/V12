#!/usr/bin/env python3
"""
End-to-End Test with Synthetic Trajectories

This test generates a realistic synthetic dataset and runs the full
preprocessing pipeline to validate all steps work correctly together.

This addresses the limitation of only having 6 example trajectories
which is too few for proper testing (especially stratified splits).
"""

import numpy as np
import json
import sys
import tempfile
import shutil
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

# Import preprocessing module
from preprocess_singular_v3_basis_transform import (
    load_trajectories,
    learn_basis_from_trajectories,
    project_trajectories,
    generate_group_anchors,
    compute_residuals,
    split_data,
    save_all_data,
    verify_preprocessing,
    analyze_length_distributions,
    DEFAULT_CONFIG,
    ORIENTATIONS,
    DISTANCE_GROUPS,
    NUM_ORIENTATIONS,
    NUM_DISTANCE_GROUPS,
)

from bspline_basis import BasisAdapter


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def generate_synthetic_trajectory(
    length: int,
    ideal_distance: float,
    orientation_angle: float,
    jitter: float = 1.5,
    seed: int = None
) -> dict:
    """Generate a synthetic mouse trajectory in JSON format."""
    if seed is not None:
        np.random.seed(seed)

    # Time parameterization with ease-out
    t = np.linspace(0, 1, length)
    eased_t = 1 - (1 - t) ** 2

    # Base trajectory (moving toward target)
    dx = ideal_distance * np.cos(orientation_angle)
    dy = ideal_distance * np.sin(orientation_angle)

    x = eased_t * dx
    y = eased_t * dy

    # Add natural curve (mouse paths aren't perfectly straight)
    curve_amount = np.random.randn() * ideal_distance * 0.03
    curve = curve_amount * np.sin(t * np.pi)
    x += curve * np.sin(orientation_angle + np.pi/2)
    y += curve * np.cos(orientation_angle + np.pi/2)

    # Add jitter (micro-corrections)
    if jitter > 0:
        jitter_x = gaussian_filter1d(np.random.randn(length) * jitter, sigma=1.0)
        jitter_y = gaussian_filter1d(np.random.randn(length) * jitter, sigma=1.0)
        x += jitter_x
        y += jitter_y

    # Offset to screen coordinates
    x = x + 500
    y = y + 500

    # Convert to integers (like real data)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    # Generate timestamps (8ms average interval, 120Hz)
    timestamps = np.cumsum(np.random.randint(6, 12, size=length))

    # Compute actual distance (path length)
    actual_distance = np.sum(np.sqrt(np.diff(x.astype(float))**2 +
                                     np.diff(y.astype(float))**2))

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        't': timestamps.tolist(),
        'ideal_distance': float(ideal_distance),
        'actual_distance': float(actual_distance),
        'original_length': length,
        'extracted_length': length,
    }


def generate_diverse_dataset(n_trajectories: int = 200, seed: int = 42) -> list:
    """Generate a diverse dataset covering all orientation/distance groups."""
    np.random.seed(seed)
    trajectories = []

    # Ensure coverage of all groups
    for dist_group in DISTANCE_GROUPS:
        dist_min = dist_group['min']
        dist_max = dist_group['max']
        dist_mid = (dist_min + dist_max) / 2

        for orient_idx, orient_name in enumerate(ORIENTATIONS):
            # Generate multiple trajectories per group
            n_per_group = max(2, n_trajectories // (NUM_DISTANCE_GROUPS * NUM_ORIENTATIONS))

            # Map orientation to angle
            angle_degrees = {
                'E': 0, 'NE': 45, 'N': 90, 'NW': 135,
                'W': 180, 'SW': 225, 'S': 270, 'SE': 315
            }
            angle_rad = np.radians(angle_degrees[orient_name])

            for i in range(n_per_group):
                # Vary within group
                length = np.random.randint(25, 90)
                distance = dist_min + np.random.rand() * (dist_max - dist_min) * 0.9
                distance = max(25, distance)  # Ensure minimum distance

                # Slight angle variation (±10 degrees)
                angle_var = angle_rad + np.radians(np.random.randn() * 5)

                jitter = 0.5 + np.random.rand() * 2.0

                traj = generate_synthetic_trajectory(
                    length=length,
                    ideal_distance=distance,
                    orientation_angle=angle_var,
                    jitter=jitter,
                    seed=seed + len(trajectories)
                )
                trajectories.append(traj)

    # Shuffle
    np.random.shuffle(trajectories)
    return trajectories


def save_trajectories_to_dir(trajectories: list, output_dir: Path):
    """Save trajectories as JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        filepath = output_dir / f"trajectory_{i:04d}.json"
        with open(filepath, 'w') as f:
            json.dump(traj, f)


def run_full_pipeline_test():
    """Run the complete preprocessing pipeline on synthetic data."""
    print_section("GENERATING SYNTHETIC DATASET")

    # Generate dataset
    n_trajectories = 200
    print(f"  Generating {n_trajectories} synthetic trajectories...")
    trajectories_raw = generate_diverse_dataset(n_trajectories, seed=42)

    # Save to temp directory
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"

    print(f"  Saving to {input_dir}...")
    save_trajectories_to_dir(trajectories_raw, input_dir)

    # Configure
    config = DEFAULT_CONFIG.copy()
    config['min_length'] = 20
    config['K'] = 8
    config['n_control_points'] = 20
    config['anchors_per_group'] = 4

    np.random.seed(config['random_seed'])

    # Run pipeline
    print_section("STEP 1: LOADING TRAJECTORIES")
    trajectories = load_trajectories(input_dir, config)
    n_loaded = len(trajectories)
    print(f"\n  Loaded: {n_loaded} trajectories")
    assert n_loaded > 100, f"Too few trajectories loaded: {n_loaded}"

    print_section("STEP 2: LEARNING BASIS")
    U_ref, T_ref, mean, basis_stats = learn_basis_from_trajectories(
        trajectories,
        n_control_points=config['n_control_points'],
        K=config['K'],
        center_data=config['center_data']
    )

    assert U_ref.shape == (2 * config['n_control_points'], config['K']), \
        f"Wrong U_ref shape: {U_ref.shape}"

    # Check orthonormality
    ortho_error = np.max(np.abs(U_ref.T @ U_ref - np.eye(config['K'])))
    assert ortho_error < 1e-10, f"U_ref not orthonormal: {ortho_error}"
    print(f"\n  U_ref orthonormality error: {ortho_error:.2e}")
    print(f"  Variance explained: {basis_stats['variance_explained']*100:.2f}%")

    print_section("STEP 3: PROJECTING TRAJECTORIES")
    coefficients, adapter, proj_stats = project_trajectories(
        trajectories, U_ref, T_ref, mean
    )

    assert coefficients.shape == (n_loaded, config['K']), \
        f"Wrong coefficients shape: {coefficients.shape}"
    assert not np.any(np.isnan(coefficients)), "NaN in coefficients"
    assert not np.any(np.isinf(coefficients)), "Inf in coefficients"

    print(f"\n  Mean reconstruction RMSE: {proj_stats['reconstruction_rmse_mean']:.4f}")

    print_section("STEP 4: GENERATING GROUP ANCHORS")
    group_anchors, group_stats = generate_group_anchors(
        trajectories, coefficients,
        config['anchors_per_group'],
        config['random_seed']
    )

    # Count total anchors
    total_anchors = sum(len(v) if v is not None else 0 for v in group_anchors.values())
    print(f"\n  Total anchors: {total_anchors}")

    print_section("STEP 5: COMPUTING RESIDUALS")
    residuals, anchor_indices, resid_stats = compute_residuals(
        trajectories, coefficients, group_anchors
    )

    assert residuals.shape == coefficients.shape, \
        f"Residuals shape mismatch: {residuals.shape} vs {coefficients.shape}"

    print(f"\n  Residual reduction ratio: {resid_stats['reduction_ratio']*100:.1f}%")

    print_section("STEP 6: SPLITTING DATA")
    splits = split_data(
        trajectories, coefficients, residuals, anchor_indices, config
    )

    # Verify split sizes
    total_split = sum(len(s['coefficients']) for s in splits.values())
    assert total_split == n_loaded, f"Split size mismatch: {total_split} vs {n_loaded}"

    # Verify original_lengths is saved
    assert 'original_lengths' in splits['train'], "original_lengths not in train split!"
    assert 'original_lengths' in splits['val'], "original_lengths not in val split!"
    assert 'original_lengths' in splits['test'], "original_lengths not in test split!"

    print(f"\n  Train: {len(splits['train']['coefficients'])}")
    print(f"  Val: {len(splits['val']['coefficients'])}")
    print(f"  Test: {len(splits['test']['coefficients'])}")

    print_section("STEP 7: SAVING DATA")
    all_stats = {
        'basis': basis_stats,
        'projection': proj_stats,
        'residual': resid_stats,
    }

    output_path = save_all_data(
        output_dir, U_ref, T_ref, mean, group_anchors, splits, config, all_stats
    )

    # Verify files exist
    required_files = ['U_ref.npy', 'config.npy', 'group_anchors.npy']
    if mean is not None:
        required_files.append('mean.npy')

    for f in required_files:
        assert (output_path / f).exists(), f"Missing file: {f}"

    for split_name in ['train', 'val', 'test']:
        split_dir = output_path / split_name
        assert split_dir.exists(), f"Missing split directory: {split_name}"
        assert (split_dir / 'coefficients.npy').exists()
        assert (split_dir / 'original_lengths.npy').exists()

    print("\n  All files saved successfully!")

    print_section("STEP 8: VERIFICATION")
    verification_passed = verify_preprocessing(
        output_path, trajectories, adapter, coefficients, n_samples=20
    )

    print_section("ANALYZING LENGTH DISTRIBUTIONS")
    length_stats = analyze_length_distributions(trajectories, group_anchors, output_path)

    # Additional validation tests
    print_section("ADDITIONAL VALIDATION TESTS")

    # Test 1: Round-trip reconstruction
    print("\n  Test: Round-trip reconstruction accuracy...")
    errors = []
    sample_indices = np.random.choice(n_loaded, min(50, n_loaded), replace=False)
    for i in sample_indices:
        traj = trajectories[i]
        flat = traj['flat']
        T = traj['length']
        c = coefficients[i]

        recon = adapter.reconstruct(c, T)
        rmse = np.sqrt(np.mean((flat - recon) ** 2))
        errors.append(rmse)

    mean_rmse = np.mean(errors)
    max_rmse = np.max(errors)
    print(f"    Mean RMSE: {mean_rmse:.4f}")
    print(f"    Max RMSE: {max_rmse:.4f}")
    assert mean_rmse < 100, f"Mean reconstruction error too high: {mean_rmse}"

    # Test 2: Jitter preservation
    print("\n  Test: Jitter preservation...")

    def compute_jitter(coords):
        if len(coords) < 3:
            return 0
        accel = np.diff(np.diff(coords))
        return np.sqrt(np.mean(accel ** 2))

    jitter_ratios = []
    for i in sample_indices[:20]:
        traj = trajectories[i]
        flat = traj['flat']
        T = traj['length']
        c = coefficients[i]

        x_orig = flat[0::2]
        y_orig = flat[1::2]

        recon = adapter.reconstruct(c, T)
        x_recon = recon[0::2]
        y_recon = recon[1::2]

        orig_jitter = (compute_jitter(x_orig) + compute_jitter(y_orig)) / 2
        recon_jitter = (compute_jitter(x_recon) + compute_jitter(y_recon)) / 2

        if orig_jitter > 1e-10:
            jitter_ratios.append(recon_jitter / orig_jitter)

    if jitter_ratios:
        mean_jitter_preservation = np.mean(jitter_ratios)
        print(f"    Mean jitter preservation: {mean_jitter_preservation*100:.1f}%")
        assert mean_jitter_preservation > 0.3, \
            f"Jitter preservation too low: {mean_jitter_preservation}"

    # Test 3: Coefficient stability across lengths
    print("\n  Test: Coefficient stability...")

    # Reconstruct same trajectory at different lengths
    test_idx = sample_indices[0]
    test_c = coefficients[test_idx]

    recon_30 = adapter.reconstruct(test_c, 30)
    recon_50 = adapter.reconstruct(test_c, 50)
    recon_70 = adapter.reconstruct(test_c, 70)

    # All should have same endpoint (up to interpolation)
    x_30, y_30 = recon_30[0::2], recon_30[1::2]
    x_50, y_50 = recon_50[0::2], recon_50[1::2]
    x_70, y_70 = recon_70[0::2], recon_70[1::2]

    # Endpoints should match within reasonable tolerance
    endpoint_30 = (x_30[-1], y_30[-1])
    endpoint_50 = (x_50[-1], y_50[-1])
    endpoint_70 = (x_70[-1], y_70[-1])

    print(f"    Endpoint at T=30: ({endpoint_30[0]:.1f}, {endpoint_30[1]:.1f})")
    print(f"    Endpoint at T=50: ({endpoint_50[0]:.1f}, {endpoint_50[1]:.1f})")
    print(f"    Endpoint at T=70: ({endpoint_70[0]:.1f}, {endpoint_70[1]:.1f})")

    # Endpoints should be consistent (within a few pixels)
    endpoint_var = np.std([endpoint_30[0], endpoint_50[0], endpoint_70[0]])
    print(f"    Endpoint x variation: {endpoint_var:.2f}")
    assert endpoint_var < 5, f"Endpoint too variable: {endpoint_var}"

    # Cleanup
    print_section("CLEANUP")
    shutil.rmtree(temp_dir)
    print(f"  Removed temp directory: {temp_dir}")

    return True


def main():
    print("=" * 70)
    print(" END-TO-END TEST WITH SYNTHETIC TRAJECTORIES")
    print("=" * 70)

    try:
        success = run_full_pipeline_test()
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 70)
    if success:
        print(" ALL TESTS PASSED")
    else:
        print(" TESTS FAILED")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
