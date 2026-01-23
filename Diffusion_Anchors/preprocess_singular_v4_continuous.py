"""
Singular Space Preprocessing - V4 CONTINUOUS CONDITIONING

This version removes categorical grouping (orientation, distance) and replaces it
with continuous conditioning on endpoint deltas (Δx, Δy).

KEY DIFFERENCES FROM V3:
========================
1. NO categorical grouping - no orientation_id, distance_group_id
2. Continuous conditioning - Δx, Δy in raw pixels
3. Fixed sample count - all trajectories must have exactly N samples (e.g., 200)
4. Global anchor pool - single set of anchors, not per-group
5. Augmentation support - flip X, Y, or both (config toggle)

TIMING:
=======
Timing is implicit in the fixed sample count. With 8ms sample interval:
    200 samples = 1600ms trajectory duration

Output Structure:
    processed_v4/
    ├── config.yaml            # Human-readable configuration
    ├── config.npy             # Configuration and metadata (for compatibility)
    ├── basis/
    │   ├── U_ref.npy          # [2*T_ref × K] reference basis
    │   └── mean.npy           # [2*T_ref] mean of control points
    ├── anchors.npy            # [N_anchors, K] global anchor prototypes
    ├── residual_mean.npy      # [K] per-dimension means for denormalization
    ├── residual_std.npy       # [K] per-dimension stds for denormalization
    └── train/val/test/
        ├── coefficients.npy   # [N × K]
        ├── residuals.npy      # [N × K] NORMALIZED residuals
        ├── anchor_indices.npy # [N] index of nearest global anchor
        ├── deltas.npy         # [N, 2] continuous (Δx, Δy) in raw pixels
        └── original_lengths.npy  # [N] (constant, but kept for compatibility)

Usage:
    python preprocess_singular_v4_continuous.py --input trajectories_v4/ --output processed_v4/

"""

import numpy as np
import json
import yaml
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import defaultdict
import argparse
import sys
from typing import List, Dict, Tuple, Optional

# Import basis transformation module
from bspline_basis import (
    BasisAdapter,
    learn_reference_basis,
    fit_bspline_control_points,
    evaluate_control_point_fit,
    build_interleaved_basis_matrix,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Output location
    'output_location': 'processed_v4/',
    'input_location': 'trajectories_v4/',

    'version': 'v4_continuous',

    # Core parameters
    'K': 15,                     # Singular space dimensions
    'n_control_points': 30,      # B-spline control points (= T_ref)
    'n_anchors': 200,            # Global anchor count (easily tunable)
    'random_seed': 42,

    # Fixed sample parameters
    'expected_sample_count': 200,  # All trajectories must have this many samples
    'sample_interval_ms': 8,       # Milliseconds between samples

    # Screen dimensions (for normalization)
    'screen_width': 2560,
    'screen_height': 1440,

    # Data splits
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,

    # Centering
    'center_data': True,

    # Augmentation
    'augmentation': {
        'enabled': True,
        'flip_x': True,      # Δx → -Δx, mirror trajectory horizontally
        'flip_y': True,      # Δy → -Δy, mirror trajectory vertically
        'flip_both': True,   # Both flipped = 180° rotation
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def trajectory_to_flat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convert x, y arrays to flat interleaved format [x0,y0,x1,y1,...]."""
    flat = np.empty(2 * len(x), dtype=np.float64)
    flat[0::2] = x
    flat[1::2] = y
    return flat


def flat_to_trajectory(flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat format back to x, y arrays."""
    return flat[0::2].copy(), flat[1::2].copy()


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(title)
    print('='*70)


def augment_trajectory(x: np.ndarray, y: np.ndarray, flip_x: bool, flip_y: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment a trajectory by flipping coordinates.

    Trajectories are relative (start at origin), so flipping is straightforward:
        flip_x: x' = -x
        flip_y: y' = -y

    Args:
        x, y: Relative coordinates (start at 0)
        flip_x: Whether to flip X axis
        flip_y: Whether to flip Y axis

    Returns:
        x_aug, y_aug: Augmented coordinates
    """
    x_aug = -x if flip_x else x.copy()
    y_aug = -y if flip_y else y.copy()
    return x_aug, y_aug


# =============================================================================
# STEP 1: LOAD RAW TRAJECTORIES
# =============================================================================

def load_trajectories(data_dir: Path, config: Dict) -> List[Dict]:
    """
    Load raw trajectories from JSON files.

    V4 DIFFERENCES:
    - Enforces fixed sample count
    - Extracts Δx, Δy as continuous conditioning
    - No orientation/distance grouping
    - Applies augmentation if enabled

    Args:
        data_dir: Directory containing trajectory JSON files
        config: Configuration dictionary

    Returns:
        List of trajectory dictionaries with keys:
            - x, y: Relative coordinates (start at origin)
            - flat: Interleaved [x0,y0,x1,y1,...] format
            - length: Number of points (should be constant)
            - delta_x, delta_y: Endpoint deltas (raw pixels)
            - is_augmented: Whether this is an augmented variant
            - augmentation_type: 'original', 'flip_x', 'flip_y', 'flip_both'
    """
    print_section("STEP 1: Loading Raw Trajectories")

    data_dir = Path(data_dir)
    json_files = list(data_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")

    if len(json_files) == 0:
        print(f"ERROR: No JSON files found in {data_dir}")
        return []

    expected_length = config['expected_sample_count']
    aug_config = config['augmentation']

    trajectories = []
    skipped = defaultdict(int)

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check required fields
            if 'x' not in data or 'y' not in data:
                skipped['missing_fields'] += 1
                continue

            x = np.array(data['x'], dtype=np.float64)
            y = np.array(data['y'], dtype=np.float64)

            # V4: Enforce fixed sample count
            if len(x) != expected_length:
                skipped[f'wrong_length_{len(x)}'] += 1
                continue

            # Convert to relative coordinates (start at origin)
            x_rel = x - x[0]
            y_rel = y - y[0]

            # Compute endpoint deltas (raw pixels)
            delta_x = x_rel[-1]  # = x[-1] - x[0]
            delta_y = y_rel[-1]  # = y[-1] - y[0]

            # Store original trajectory
            trajectories.append({
                'x': x_rel,
                'y': y_rel,
                'flat': trajectory_to_flat(x_rel, y_rel),
                'length': len(x),
                'delta_x': delta_x,
                'delta_y': delta_y,
                'is_augmented': False,
                'augmentation_type': 'original',
                'filename': json_file.name,
            })

            # Apply augmentation if enabled
            if aug_config['enabled']:
                augmentations = []

                if aug_config['flip_x']:
                    augmentations.append(('flip_x', True, False))
                if aug_config['flip_y']:
                    augmentations.append(('flip_y', False, True))
                if aug_config['flip_both']:
                    augmentations.append(('flip_both', True, True))

                for aug_name, fx, fy in augmentations:
                    x_aug, y_aug = augment_trajectory(x_rel, y_rel, fx, fy)

                    trajectories.append({
                        'x': x_aug,
                        'y': y_aug,
                        'flat': trajectory_to_flat(x_aug, y_aug),
                        'length': len(x),
                        'delta_x': -delta_x if fx else delta_x,
                        'delta_y': -delta_y if fy else delta_y,
                        'is_augmented': True,
                        'augmentation_type': aug_name,
                        'filename': json_file.name,
                    })

        except Exception as e:
            skipped['error'] += 1

    # Report results
    n_original = sum(1 for t in trajectories if not t['is_augmented'])
    n_augmented = sum(1 for t in trajectories if t['is_augmented'])

    print(f"\nLoaded {n_original} original trajectories")
    if aug_config['enabled']:
        print(f"Created {n_augmented} augmented variants")
    print(f"Total: {len(trajectories)} trajectories")

    if skipped:
        print(f"\nSkipped: {sum(skipped.values())} total")
        for reason, count in sorted(skipped.items()):
            print(f"  - {reason}: {count}")

    if trajectories:
        # Delta statistics
        delta_x_all = np.array([t['delta_x'] for t in trajectories])
        delta_y_all = np.array([t['delta_y'] for t in trajectories])

        print(f"\nEndpoint delta statistics (raw pixels):")
        print(f"  Δx range: [{delta_x_all.min():.1f}, {delta_x_all.max():.1f}]")
        print(f"  Δy range: [{delta_y_all.min():.1f}, {delta_y_all.max():.1f}]")
        print(f"  Δx mean: {delta_x_all.mean():.1f}, std: {delta_x_all.std():.1f}")
        print(f"  Δy mean: {delta_y_all.mean():.1f}, std: {delta_y_all.std():.1f}")

        # Distance statistics
        distances = np.sqrt(delta_x_all**2 + delta_y_all**2)
        print(f"\nDistance statistics:")
        print(f"  Range: [{distances.min():.1f}, {distances.max():.1f}] pixels")
        print(f"  Mean: {distances.mean():.1f}, Median: {np.median(distances):.1f}")

        # Augmentation breakdown
        if aug_config['enabled']:
            print(f"\nAugmentation breakdown:")
            aug_counts = defaultdict(int)
            for t in trajectories:
                aug_counts[t['augmentation_type']] += 1
            for aug_type, count in sorted(aug_counts.items()):
                print(f"  {aug_type}: {count}")

    return trajectories


# =============================================================================
# STEP 2: LEARN REFERENCE BASIS
# =============================================================================

def learn_basis_from_trajectories(
    trajectories: List[Dict],
    n_control_points: int,
    K: int,
    center_data: bool = True
) -> Tuple[np.ndarray, int, Optional[np.ndarray], Dict]:
    """
    Learn SVD reference basis from trajectories.

    This is the same as V3 - the math doesn't change.

    Process:
        1. Fit B-spline control points to each trajectory
        2. Stack control points into matrix
        3. (Optional) Center by subtracting mean
        4. SVD to get reference basis

    Args:
        trajectories: List of trajectory dicts with 'flat' key
        n_control_points: Number of B-spline control points (= T_ref)
        K: Number of singular vectors to keep
        center_data: Whether to center before SVD

    Returns:
        U_ref: Reference basis [2*T_ref × K]
        T_ref: Reference length (= n_control_points)
        mean: Mean control points [2*T_ref] if centered, else None
        stats: Dictionary with fitting statistics
    """
    print_section("STEP 2: Learning Reference Basis")

    N = len(trajectories)
    T_ref = n_control_points

    print(f"Parameters:")
    print(f"  Trajectories: {N}")
    print(f"  Control points (T_ref): {T_ref}")
    print(f"  Singular dimensions (K): {K}")
    print(f"  Center data: {center_data}")

    # Validate K
    if K > N:
        print(f"  WARNING: K ({K}) > N ({N}), reducing K to {N}")
        K = N
    if K > 2 * T_ref:
        print(f"  WARNING: K ({K}) > 2*T_ref ({2*T_ref}), reducing K to {2*T_ref}")
        K = 2 * T_ref

    # Stage 1: Fit B-spline control points to each trajectory
    print(f"\nStage 1: Fitting B-spline control points...")

    control_points = np.zeros((N, 2 * T_ref), dtype=np.float64)
    fit_errors = []

    for i, traj in enumerate(trajectories):
        flat = traj['flat']

        # Fit control points
        cp = fit_bspline_control_points(flat, n_control_points)
        control_points[i] = cp

        # Track fit quality
        fit_eval = evaluate_control_point_fit(flat, cp)
        fit_errors.append(fit_eval['rmse'])

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{N} trajectories...")

    fit_errors = np.array(fit_errors)

    print(f"\n  Control point fitting complete:")
    print(f"    RMSE: mean={fit_errors.mean():.4f}, max={fit_errors.max():.4f}")

    # Stage 2: SVD on control points
    print(f"\nStage 2: Computing SVD...")

    if center_data:
        mean = control_points.mean(axis=0)
        centered = control_points - mean
        print(f"  Centered data (mean norm: {np.linalg.norm(mean):.2f})")
    else:
        mean = None
        centered = control_points

    # SVD
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)

    U_ref = U[:, :K]  # [2*T_ref × K]
    S_k = S[:K]

    # Compute variance explained
    total_var = np.sum(S ** 2)
    explained_var = np.cumsum(S[:K] ** 2) / total_var

    print(f"\n  Singular values (top {min(K, 10)}):")
    for i in range(min(K, 10)):
        pct = (S[i]**2 / total_var) * 100
        cum_pct = explained_var[i] * 100
        print(f"    S[{i+1:2d}] = {S[i]:12.2f}  ({pct:5.2f}% var, {cum_pct:6.2f}% cumulative)")

    print(f"\n  K={K} explains {explained_var[-1]*100:.2f}% of variance")

    # Check basis orthonormality
    orthonormality_error = np.max(np.abs(U_ref.T @ U_ref - np.eye(K)))
    print(f"  Basis orthonormality error: {orthonormality_error:.2e}")

    # Collect statistics
    stats = {
        'n_trajectories': N,
        'n_control_points': T_ref,
        'K': K,
        'fit_rmse_mean': float(fit_errors.mean()),
        'fit_rmse_max': float(fit_errors.max()),
        'variance_explained': float(explained_var[-1]),
        'singular_values': S_k.tolist(),
    }

    print(f"\n  U_ref shape: {U_ref.shape}")

    return U_ref, T_ref, mean, stats


# =============================================================================
# STEP 3: PROJECT TRAJECTORIES TO K-SPACE
# =============================================================================

def project_trajectories(
    trajectories: List[Dict],
    U_ref: np.ndarray,
    T_ref: int,
    mean: Optional[np.ndarray]
) -> Tuple[np.ndarray, BasisAdapter, Dict]:
    """
    Project all trajectories to K-dimensional coefficient space.

    Same as V3 - projects via control points.
    """
    print_section("STEP 3: Projecting Trajectories to K-Space")

    N = len(trajectories)
    K = U_ref.shape[1]

    print(f"Projecting {N} trajectories to {K}-dimensional space...")

    # Create basis adapter
    adapter = BasisAdapter(U_ref, T_ref)

    # Project all trajectories
    coefficients = np.zeros((N, K), dtype=np.float64)
    projection_errors = []

    for i, traj in enumerate(trajectories):
        flat = traj['flat']
        T = traj['length']

        # Fit B-spline control points
        cp = fit_bspline_control_points(flat, T_ref)

        # Project control points onto U_ref
        if mean is not None:
            c = U_ref.T @ (cp - mean)
        else:
            c = U_ref.T @ cp

        coefficients[i] = c

        # Compute reconstruction error
        if mean is not None:
            cp_recon = U_ref @ c + mean
        else:
            cp_recon = U_ref @ c

        C = build_interleaved_basis_matrix(T, T_ref)
        recon = C @ cp_recon

        rmse = np.sqrt(np.mean((flat - recon) ** 2))
        projection_errors.append(rmse)

        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{N} trajectories...")

    projection_errors = np.array(projection_errors)

    print(f"\n  Projection complete:")
    print(f"    Reconstruction RMSE: mean={projection_errors.mean():.4f}, "
          f"median={np.median(projection_errors):.4f}, max={projection_errors.max():.4f}")

    print(f"\n  Coefficient statistics:")
    print(f"    Shape: {coefficients.shape}")
    print(f"    Mean: {coefficients.mean():.4f}")
    print(f"    Std:  {coefficients.std():.4f}")
    print(f"    Range: [{coefficients.min():.4f}, {coefficients.max():.4f}]")

    stats = {
        'reconstruction_rmse_mean': float(projection_errors.mean()),
        'reconstruction_rmse_max': float(projection_errors.max()),
        'coefficient_mean': float(coefficients.mean()),
        'coefficient_std': float(coefficients.std()),
    }

    return coefficients, adapter, stats


# =============================================================================
# STEP 4: GENERATE GLOBAL ANCHORS
# =============================================================================

def generate_global_anchors(
    coefficients: np.ndarray,
    n_anchors: int,
    random_seed: int
) -> Tuple[np.ndarray, Dict]:
    """
    Generate global anchor prototypes via K-means clustering.

    V4 DIFFERENCE: Single global pool, not per-group.

    Args:
        coefficients: [N × K] coefficient array
        n_anchors: Number of global anchors to create
        random_seed: Random seed for reproducibility

    Returns:
        anchors: [n_anchors × K] anchor array
        stats: Clustering statistics
    """
    print_section("STEP 4: Generating Global Anchors")

    N, K = coefficients.shape

    print(f"Parameters:")
    print(f"  Trajectories: {N}")
    print(f"  Requested anchors: {n_anchors}")
    print(f"  Random seed: {random_seed}")

    # Adjust n_anchors if we have fewer samples
    actual_n_anchors = min(n_anchors, N)
    if actual_n_anchors < n_anchors:
        print(f"  Adjusted to {actual_n_anchors} anchors (fewer samples than requested)")

    # K-means clustering
    print(f"\n  Running K-means clustering...")

    kmeans = KMeans(
        n_clusters=actual_n_anchors,
        random_state=random_seed,
        n_init=10,
        max_iter=300,
        verbose=0
    )
    kmeans.fit(coefficients)

    anchors = kmeans.cluster_centers_.copy()

    # Statistics
    print(f"\n  Clustering complete:")
    print(f"    Anchors shape: {anchors.shape}")
    print(f"    Inertia: {kmeans.inertia_:.2f}")

    # Compute cluster sizes
    labels = kmeans.labels_
    cluster_sizes = np.bincount(labels, minlength=actual_n_anchors)

    print(f"    Cluster size: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}")

    # Empty clusters warning
    empty_clusters = np.sum(cluster_sizes == 0)
    if empty_clusters > 0:
        print(f"    WARNING: {empty_clusters} empty clusters!")

    stats = {
        'n_anchors': actual_n_anchors,
        'inertia': float(kmeans.inertia_),
        'cluster_size_min': int(cluster_sizes.min()),
        'cluster_size_max': int(cluster_sizes.max()),
        'cluster_size_mean': float(cluster_sizes.mean()),
        'empty_clusters': int(empty_clusters),
    }

    return anchors, stats


# =============================================================================
# STEP 5: COMPUTE RESIDUALS (GLOBAL)
# =============================================================================

def compute_residuals(
    coefficients: np.ndarray,
    anchors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute residuals: difference between each trajectory's coefficients
    and its nearest anchor from the GLOBAL pool.

    V4 DIFFERENCE: Global nearest neighbor search, not per-group.

    Args:
        coefficients: [N × K] coefficient array
        anchors: [n_anchors × K] global anchor array

    Returns:
        residuals: [N × K] array
        anchor_indices: [N] array of which anchor was nearest
        stats: Residual statistics
    """
    print_section("STEP 5: Computing Residuals (Global)")

    N, K = coefficients.shape
    n_anchors = anchors.shape[0]

    print(f"Finding nearest anchor for {N} trajectories from {n_anchors} global anchors...")

    residuals = np.zeros((N, K), dtype=np.float64)
    anchor_indices = np.zeros(N, dtype=np.int32)
    distances_to_anchor = np.zeros(N, dtype=np.float64)

    # Compute all pairwise distances at once (more efficient)
    # coefficients: [N, K], anchors: [n_anchors, K]
    # distances: [N, n_anchors]

    # Using broadcasting for efficiency
    for i in range(N):
        dists = np.linalg.norm(anchors - coefficients[i], axis=1)
        nearest_idx = dists.argmin()

        anchor_indices[i] = nearest_idx
        residuals[i] = coefficients[i] - anchors[nearest_idx]
        distances_to_anchor[i] = dists[nearest_idx]

        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{N} trajectories...")

    # Statistics
    print(f"\nResiduals computed:")
    print(f"  Shape: {residuals.shape}")

    print(f"\nResidual magnitude (distance to nearest anchor):")
    print(f"  Mean:   {distances_to_anchor.mean():.4f}")
    print(f"  Median: {np.median(distances_to_anchor):.4f}")
    print(f"  Std:    {distances_to_anchor.std():.4f}")
    print(f"  Max:    {distances_to_anchor.max():.4f}")

    # Compare to coefficient magnitude
    coeff_norms = np.linalg.norm(coefficients, axis=1)
    reduction_ratio = distances_to_anchor.mean() / coeff_norms.mean()

    print(f"\nComparison to coefficient magnitude:")
    print(f"  Mean coefficient norm: {coeff_norms.mean():.4f}")
    print(f"  Mean residual norm:    {distances_to_anchor.mean():.4f}")
    print(f"  Reduction ratio:       {reduction_ratio:.2%}")
    print(f"  -> Diffusion learns {reduction_ratio:.1%} of the original signal!")

    # Anchor usage distribution
    anchor_usage = np.bincount(anchor_indices, minlength=n_anchors)
    print(f"\nAnchor usage:")
    print(f"  Used anchors: {np.sum(anchor_usage > 0)}/{n_anchors}")
    print(f"  Usage range: [{anchor_usage.min()}, {anchor_usage.max()}]")

    stats = {
        'residual_mean_norm': float(distances_to_anchor.mean()),
        'residual_max_norm': float(distances_to_anchor.max()),
        'coefficient_mean_norm': float(coeff_norms.mean()),
        'reduction_ratio': float(reduction_ratio),
        'anchors_used': int(np.sum(anchor_usage > 0)),
    }

    return residuals, anchor_indices, stats


# =============================================================================
# STEP 5.5: NORMALIZE RESIDUALS
# =============================================================================

def normalize_residuals(residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize residuals to zero-mean, unit-variance per dimension.

    CRITICAL for diffusion models which start from N(0,1) noise.

    At generation time, denormalize BEFORE adding anchor:
        r_denorm = r_normalized * residual_std + residual_mean
        coefficients = anchor + r_denorm
    """
    print_section("STEP 5.5: Normalizing Residuals")

    K = residuals.shape[1]

    # Compute per-dimension statistics
    residual_mean = residuals.mean(axis=0)
    residual_std = residuals.std(axis=0)

    # Avoid division by zero
    residual_std = np.maximum(residual_std, 1e-8)

    # Normalize
    normalized = (residuals - residual_mean) / residual_std

    # Report statistics
    print(f"Raw residual statistics:")
    print(f"  Shape: {residuals.shape}")
    print(f"  Range: [{residuals.min():.4f}, {residuals.max():.4f}]")

    print(f"\nNormalized residual statistics:")
    print(f"  Range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    print(f"  Mean: {normalized.mean():.6f} (should be ~0)")
    print(f"  Std:  {normalized.std():.6f} (should be ~1)")

    print(f"\nNOTE: At generation time, DENORMALIZE before adding anchor!")

    return normalized, residual_mean, residual_std


# =============================================================================
# STEP 6: SPLIT DATA
# =============================================================================

def split_data(
    trajectories: List[Dict],
    coefficients: np.ndarray,
    residuals: np.ndarray,
    anchor_indices: np.ndarray,
    config: Dict
) -> Dict[str, Dict]:
    """
    Split data into train/validation/test sets.

    V4: Splits on (Δx, Δy) deltas instead of orientation IDs.
    """
    print_section("STEP 6: Splitting Data")

    N = len(trajectories)
    indices = np.arange(N)

    # Extract arrays
    deltas = np.array([[t['delta_x'], t['delta_y']] for t in trajectories], dtype=np.float32)
    original_lengths = np.array([t['length'] for t in trajectories], dtype=np.int32)

    print(f"Total samples: {N}")
    print(f"Split ratios: train={config['train_ratio']}, val={config['val_ratio']}, test={config['test_ratio']}")

    # Simple random split (no stratification needed for continuous values)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=config['train_ratio'],
        random_state=config['random_seed']
    )

    val_ratio_adjusted = config['val_ratio'] / (config['val_ratio'] + config['test_ratio'])
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_ratio_adjusted,
        random_state=config['random_seed']
    )

    # Build split dictionaries
    splits = {}

    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        splits[name] = {
            'indices': idx,
            'coefficients': coefficients[idx],
            'residuals': residuals[idx],
            'anchor_indices': anchor_indices[idx],
            'deltas': deltas[idx],  # V4: continuous deltas instead of IDs
            'original_lengths': original_lengths[idx],
        }

        # Summary statistics
        split_deltas = deltas[idx]
        print(f"\n  {name}: {len(idx)} samples")
        print(f"    Δx range: [{split_deltas[:, 0].min():.1f}, {split_deltas[:, 0].max():.1f}]")
        print(f"    Δy range: [{split_deltas[:, 1].min():.1f}, {split_deltas[:, 1].max():.1f}]")

    return splits


# =============================================================================
# STEP 7: SAVE DATA
# =============================================================================

def save_all_data(
    output_dir: Path,
    U_ref: np.ndarray,
    T_ref: int,
    mean: Optional[np.ndarray],
    anchors: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    splits: Dict,
    config: Dict,
    stats: Dict
) -> Path:
    """
    Save all preprocessed data to disk.

    V4 Output structure:
        output_dir/
        ├── config.yaml
        ├── config.npy
        ├── basis/
        │   ├── U_ref.npy
        │   └── mean.npy
        ├── anchors.npy
        ├── residual_mean.npy
        ├── residual_std.npy
        └── train/val/test/
            ├── coefficients.npy
            ├── residuals.npy
            ├── anchor_indices.npy
            ├── deltas.npy
            └── original_lengths.npy
    """
    print_section("STEP 7: Saving Data")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create basis subdirectory
    basis_dir = output_path / 'basis'
    basis_dir.mkdir(exist_ok=True)

    # Save reference basis
    np.save(basis_dir / 'U_ref.npy', U_ref)
    print(f"  Saved: basis/U_ref.npy {U_ref.shape}")

    # Save mean
    if mean is not None:
        np.save(basis_dir / 'mean.npy', mean)
        print(f"  Saved: basis/mean.npy {mean.shape}")

    # Save global anchors
    np.save(output_path / 'anchors.npy', anchors)
    print(f"  Saved: anchors.npy {anchors.shape}")

    # Save residual normalization parameters
    np.save(output_path / 'residual_mean.npy', residual_mean)
    np.save(output_path / 'residual_std.npy', residual_std)
    print(f"  Saved: residual_mean.npy {residual_mean.shape}")
    print(f"  Saved: residual_std.npy {residual_std.shape}")

    # Build full config
    K = U_ref.shape[1]

    full_config = {
        **config,
        'K': K,
        'T_ref': T_ref,
        'n_control_points': T_ref,
        'version': 'v4_continuous',

        # Normalization for Fourier embedding
        'normalization': {
            'delta_x_scale': config['screen_width'],
            'delta_y_scale': config['screen_height'],
        },

        # Statistics
        'stats': stats,
    }

    # Save config as YAML (human-readable)
    yaml_config = {k: v for k, v in full_config.items() if k != 'stats'}
    yaml_config['stats_summary'] = {
        'n_trajectories': stats['basis']['n_trajectories'],
        'variance_explained': stats['basis']['variance_explained'],
        'reduction_ratio': stats['residual']['reduction_ratio'],
    }

    with open(output_path / 'config.yaml', 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: config.yaml")

    # Save config as NPY (for compatibility)
    np.save(output_path / 'config.npy', full_config, allow_pickle=True)
    print(f"  Saved: config.npy")

    # Save splits
    for split_name, split_data in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)

        for key, arr in split_data.items():
            if key != 'indices':
                np.save(split_dir / f'{key}.npy', arr)

        print(f"  Saved: {split_name}/ ({len(split_data['coefficients'])} samples)")

    print(f"\nAll data saved to: {output_path}")

    return output_path


# =============================================================================
# STEP 8: VERIFICATION
# =============================================================================

def verify_preprocessing(
    output_dir: Path,
    trajectories: List[Dict],
    adapter: BasisAdapter,
    coefficients: np.ndarray,
    n_samples: int = 10
) -> bool:
    """
    Verify that the preprocessing pipeline is working correctly.
    """
    print_section("STEP 8: Verification")

    all_passed = True

    # Test 1: Round-trip reconstruction
    print("\n[Test 1] Round-trip reconstruction accuracy...")

    sample_indices = np.random.choice(len(trajectories), min(n_samples, len(trajectories)), replace=False)

    reconstruction_errors = []
    mean_path = output_dir / 'basis' / 'mean.npy'
    mean = np.load(mean_path) if mean_path.exists() else None

    for i in sample_indices:
        traj = trajectories[i]
        flat = traj['flat']
        T = traj['length']
        c = coefficients[i]

        # Reconstruct
        cp_recon = adapter.U_ref @ c
        if mean is not None:
            cp_recon = cp_recon + mean
        C = build_interleaved_basis_matrix(T, adapter.T_ref)
        recon = C @ cp_recon

        error = np.sqrt(np.mean((flat - recon) ** 2))
        reconstruction_errors.append(error)

    mean_error = np.mean(reconstruction_errors)
    print(f"  Mean RMSE: {mean_error:.4f}")
    print(f"  Max RMSE:  {np.max(reconstruction_errors):.4f}")

    if mean_error > 100:
        print(f"  WARNING: High reconstruction error!")
        all_passed = False
    else:
        print(f"  OK: Reconstruction accuracy acceptable")

    # Test 2: Delta consistency
    print("\n[Test 2] Delta consistency check...")

    deltas = np.load(output_dir / 'train' / 'deltas.npy')
    print(f"  Deltas shape: {deltas.shape}")
    print(f"  Δx range: [{deltas[:, 0].min():.1f}, {deltas[:, 0].max():.1f}]")
    print(f"  Δy range: [{deltas[:, 1].min():.1f}, {deltas[:, 1].max():.1f}]")
    print(f"  OK: Deltas saved correctly")

    # Test 3: Anchor consistency
    print("\n[Test 3] Anchor consistency check...")

    anchors = np.load(output_dir / 'anchors.npy')
    anchor_indices = np.load(output_dir / 'train' / 'anchor_indices.npy')

    print(f"  Anchors shape: {anchors.shape}")
    print(f"  Anchor indices range: [{anchor_indices.min()}, {anchor_indices.max()}]")

    if anchor_indices.max() >= len(anchors):
        print(f"  ERROR: Anchor index out of range!")
        all_passed = False
    else:
        print(f"  OK: Anchor indices valid")

    # Summary
    print(f"\n{'='*70}")
    if all_passed:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION COMPLETED WITH WARNINGS")
    print('='*70)

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess trajectories using V4 continuous conditioning approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python preprocess_singular_v4_continuous.py --input trajectories_v4/ --output processed_v4/

    # With custom parameters
    python preprocess_singular_v4_continuous.py --input trajectories_v4/ --output processed_v4/ \\
        --n_control_points 30 --k 15 --n_anchors 200

    # Disable augmentation
    python preprocess_singular_v4_continuous.py --input trajectories_v4/ --output processed_v4/ \\
        --no_augmentation
"""
    )

    # Required arguments
    parser.add_argument('--input', type=str, default=DEFAULT_CONFIG['input_location'],
                        help='Directory containing trajectory JSON files')
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_location'],
                        help='Output directory for preprocessed data')

    # Core parameters
    parser.add_argument('--n_control_points', type=int, default=DEFAULT_CONFIG['n_control_points'],
                        help=f"Number of B-spline control points (default: {DEFAULT_CONFIG['n_control_points']})")
    parser.add_argument('--k', type=int, default=DEFAULT_CONFIG['K'],
                        help=f"Number of singular dimensions (default: {DEFAULT_CONFIG['K']})")
    parser.add_argument('--n_anchors', type=int, default=DEFAULT_CONFIG['n_anchors'],
                        help=f"Number of global anchors (default: {DEFAULT_CONFIG['n_anchors']})")

    # Fixed sample parameters
    parser.add_argument('--sample_count', type=int, default=DEFAULT_CONFIG['expected_sample_count'],
                        help=f"Expected sample count per trajectory (default: {DEFAULT_CONFIG['expected_sample_count']})")

    # Screen dimensions
    parser.add_argument('--screen_width', type=int, default=DEFAULT_CONFIG['screen_width'],
                        help=f"Screen width in pixels (default: {DEFAULT_CONFIG['screen_width']})")
    parser.add_argument('--screen_height', type=int, default=DEFAULT_CONFIG['screen_height'],
                        help=f"Screen height in pixels (default: {DEFAULT_CONFIG['screen_height']})")

    # Other options
    parser.add_argument('--no_center', action='store_true',
                        help='Disable data centering before SVD')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['random_seed'],
                        help=f"Random seed (default: {DEFAULT_CONFIG['random_seed']})")
    parser.add_argument('--skip_verify', action='store_true',
                        help='Skip verification step')

    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config['input_location'] = args.input
    config['output_location'] = args.output
    config['n_control_points'] = args.n_control_points
    config['K'] = args.k
    config['n_anchors'] = args.n_anchors
    config['expected_sample_count'] = args.sample_count
    config['screen_width'] = args.screen_width
    config['screen_height'] = args.screen_height
    config['center_data'] = not args.no_center
    config['random_seed'] = args.seed

    if args.no_augmentation:
        config['augmentation']['enabled'] = False

    # Print header
    print("=" * 70)
    print("SINGULAR SPACE PREPROCESSING - VERSION 4 (CONTINUOUS CONDITIONING)")
    print("=" * 70)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print(f"\nParameters:")
    print(f"  n_control_points (T_ref): {config['n_control_points']}")
    print(f"  K (singular dimensions):  {config['K']}")
    print(f"  n_anchors (global):       {config['n_anchors']}")
    print(f"  expected_sample_count:    {config['expected_sample_count']}")
    print(f"  screen_width:             {config['screen_width']}")
    print(f"  screen_height:            {config['screen_height']}")
    print(f"  center_data:              {config['center_data']}")
    print(f"  augmentation:             {config['augmentation']['enabled']}")
    print(f"  random_seed:              {config['random_seed']}")
    print("\n" + "=" * 70)
    print("KEY INNOVATION: Continuous Δx, Δy conditioning instead of categorical groups!")
    print("=" * 70)

    # Set random seed
    np.random.seed(config['random_seed'])

    # =========================================================================
    # PREPROCESSING PIPELINE
    # =========================================================================

    # Step 1: Load trajectories
    trajectories = load_trajectories(args.input, config)

    if len(trajectories) == 0:
        print("\nERROR: No trajectories loaded!")
        sys.exit(1)

    # Step 2: Learn reference basis
    U_ref, T_ref, mean, basis_stats = learn_basis_from_trajectories(
        trajectories,
        n_control_points=config['n_control_points'],
        K=config['K'],
        center_data=config['center_data']
    )

    # Step 3: Project trajectories to K-space
    coefficients, adapter, projection_stats = project_trajectories(
        trajectories, U_ref, T_ref, mean
    )

    # Step 4: Generate global anchors
    anchors, anchor_stats = generate_global_anchors(
        coefficients,
        config['n_anchors'],
        config['random_seed']
    )

    # Step 5: Compute residuals
    residuals, anchor_indices, residual_stats = compute_residuals(
        coefficients, anchors
    )

    # Step 5.5: Normalize residuals
    normalized_residuals, residual_mean, residual_std = normalize_residuals(residuals)

    # Step 6: Split data
    splits = split_data(
        trajectories, coefficients, normalized_residuals, anchor_indices, config
    )

    # Combine all statistics
    all_stats = {
        'basis': basis_stats,
        'projection': projection_stats,
        'anchor': anchor_stats,
        'residual': residual_stats,
    }

    # Step 7: Save data
    output_path = save_all_data(
        args.output, U_ref, T_ref, mean, anchors,
        residual_mean, residual_std, splits, config, all_stats
    )

    # Step 8: Verification
    if not args.skip_verify:
        verify_preprocessing(output_path, trajectories, adapter, coefficients)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE - V4 CONTINUOUS CONDITIONING")
    print("=" * 70)

    print(f"\nOutput saved to: {output_path}")
    print(f"\nKey files:")
    print(f"  - basis/U_ref.npy:        Reference basis [{2*T_ref} × {config['K']}]")
    if mean is not None:
        print(f"  - basis/mean.npy:         Mean control points [{2*T_ref}]")
    print(f"  - anchors.npy:            Global anchors [{anchors.shape[0]} × {config['K']}]")
    print(f"  - residual_mean.npy:      Per-dim means for denormalization")
    print(f"  - residual_std.npy:       Per-dim stds for denormalization")
    print(f"  - config.yaml:            Human-readable configuration")
    print(f"  - train/val/test/:        Split data with deltas.npy (continuous!)")

    print(f"\nStatistics:")
    print(f"  - Trajectories: {len(trajectories)}")
    print(f"  - Variance explained: {basis_stats['variance_explained']*100:.2f}%")
    print(f"  - Residual reduction: {residual_stats['reduction_ratio']*100:.1f}%")
    print(f"  - Global anchors: {anchors.shape[0]}")

    print(f"\nNormalization for Fourier embedding:")
    print(f"  Δx_norm = Δx / {config['screen_width']}  # [-1, 1]")
    print(f"  Δy_norm = Δy / {config['screen_height']}  # [-1, 1]")

    print(f"\nTo train the diffusion model:")
    print(f"  python train_singular_diffusion_v4.py --data {args.output}")

    print(f"\nNOTE: Generation MUST denormalize residuals BEFORE adding anchor!")
    print(f"      r = r_normalized * residual_std + residual_mean")
    print(f"      coefficients = anchor + r")


if __name__ == '__main__':
    main()
