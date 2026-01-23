"""
Singular Space Preprocessing - V5 ANTICIPATORY (THREE-DOT FLOW)

This version processes trajectories from the three-dot recorder which captures
ANTICIPATORY MOVEMENT - how the A→B trajectory is shaped by knowing C exists.

KEY INNOVATION:
===============
Each trajectory has DUAL conditioning:
    - AB: Current segment (orientation, distance) - where we're going
    - BC: Next segment (orientation, distance) - where we're going AFTER
    - Turn angle: How much direction changes from AB to BC

The model learns: "Given current direction AND next direction,
                   here's how the trajectory should curve"

ANCHOR STRATEGY: Per Turn Category
==================================
Instead of 40 groups (orientation × distance) or 1600 groups (AB × BC),
we group by TURN CATEGORY:
    - straight: -22.5° to 22.5°
    - slight_right: 22.5° to 67.5°
    - hard_right: 67.5° to 135°
    - reverse: 135° to 180° (and -135° to -180°)
    - slight_left: -67.5° to -22.5°
    - hard_left: -135° to -67.5°

This captures the KEY insight: trajectory SHAPE depends primarily on how
much you're turning, not the absolute directions.

Output Structure:
    processed_v5/
    ├── config.npy
    ├── basis/
    │   ├── U_ref.npy
    │   └── mean.npy
    ├── turn_category_anchors.npy   # Per turn-category anchors
    ├── residual_mean.npy
    ├── residual_std.npy
    └── train/val/test/
        ├── coefficients.npy
        ├── residuals.npy
        ├── anchor_indices.npy
        ├── orientation_ids_AB.npy   # Current segment orientation (0-7)
        ├── distance_ids_AB.npy      # Current segment distance group (0-4)
        ├── orientation_ids_BC.npy   # Next segment orientation (0-7)
        ├── distance_ids_BC.npy      # Next segment distance group (0-4)
        ├── turn_angles.npy          # Raw turn angles in degrees
        ├── turn_category_ids.npy    # Turn category (0-6)
        └── original_lengths.npy

Usage:
    python preprocess_singular_v5_anticipatory.py --input segments/ --output processed_v5/
"""

import numpy as np
import json
import math
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import defaultdict
import argparse
import sys
from typing import List, Dict, Tuple, Optional

from bspline_basis import (
    BasisAdapter,
    fit_bspline_control_points,
    evaluate_control_point_fit,
    build_interleaved_basis_matrix,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'output_location': 'processed_v5/',
    'input_location': 'segments/',

    'version': 'v5_anticipatory',

    # Core parameters
    'K': 15,
    'n_control_points': 30,
    'anchors_per_turn_category': 25,
    'random_seed': 42,

    # Length constraints
    'min_length': 20,
    'max_length_filter': 300,

    # Data splits
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,

    # Centering
    'center_data': True,
}

# =============================================================================
# ORIENTATION AND DISTANCE DEFINITIONS (same as V3)
# =============================================================================

ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NUM_ORIENTATIONS = 8

DISTANCE_GROUPS = [
    {"name": "XS", "id": 0, "min": 0,   "max": 159},
    {"name": "S",  "id": 1, "min": 159, "max": 291},
    {"name": "M",  "id": 2, "min": 291, "max": 423},
    {"name": "L",  "id": 3, "min": 423, "max": 555},
    {"name": "XL", "id": 4, "min": 555, "max": float('inf')},
]
NUM_DISTANCE_GROUPS = 5

# =============================================================================
# TURN CATEGORY DEFINITIONS - THE KEY FOR V5
# =============================================================================

TURN_CATEGORIES = [
    {"name": "straight",     "id": 0, "min": -22.5,  "max": 22.5},
    {"name": "slight_right", "id": 1, "min": 22.5,   "max": 67.5},
    {"name": "hard_right",   "id": 2, "min": 67.5,   "max": 135.0},
    {"name": "reverse_right","id": 3, "min": 135.0,  "max": 180.0},
    {"name": "slight_left",  "id": 4, "min": -67.5,  "max": -22.5},
    {"name": "hard_left",    "id": 5, "min": -135.0, "max": -67.5},
    {"name": "reverse_left", "id": 6, "min": -180.0, "max": -135.0},
]
NUM_TURN_CATEGORIES = 7
TURN_CATEGORY_NAMES = [tc["name"] for tc in TURN_CATEGORIES]


def get_turn_category_id(turn_angle: float) -> int:
    """
    Get turn category ID from turn angle in degrees.

    Turn angle is positive for right turns, negative for left turns.
    Range: -180 to 180 degrees.
    """
    # Normalize to -180 to 180
    while turn_angle > 180:
        turn_angle -= 360
    while turn_angle < -180:
        turn_angle += 360

    for cat in TURN_CATEGORIES:
        if cat["min"] <= turn_angle < cat["max"]:
            return cat["id"]

    # Edge case: exactly 180 or -180
    if abs(turn_angle) >= 179.9:
        return 3 if turn_angle > 0 else 6  # reverse_right or reverse_left

    return 0  # Default to straight


def get_distance_group_id(distance: float) -> int:
    """Get distance group ID (0-4) from distance in pixels."""
    for group in DISTANCE_GROUPS:
        if group["min"] <= distance < group["max"]:
            return group["id"]
    return NUM_DISTANCE_GROUPS - 1


def get_orientation_id(orient_str: str) -> int:
    """Convert orientation string to ID."""
    if orient_str in ORIENTATIONS:
        return ORIENTATIONS.index(orient_str)
    return 2  # Default to E


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def trajectory_to_flat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convert x, y arrays to flat interleaved format [x0,y0,x1,y1,...]."""
    flat = np.empty(2 * len(x), dtype=np.float64)
    flat[0::2] = x
    flat[1::2] = y
    return flat


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(title)
    print('='*70)


# =============================================================================
# STEP 1: LOAD SEGMENT TRAJECTORIES (THREE-DOT FORMAT)
# =============================================================================

def load_trajectories(data_dir: Path, config: Dict) -> List[Dict]:
    """
    Load trajectories from three-dot recorder segment JSON files.

    Expected JSON structure (from SapiRecorderThreeDot):
    {
        "segment_id": 1,
        "A": {"x": 1250, "y": 710},
        "B": {"x": 1500, "y": 600},
        "C": {"x": 1700, "y": 600},
        "AB": {
            "distance": 280,
            "orientation": "NE",
            "orientation_id": 1,
            "distance_group": 1,
            ...
        },
        "BC": {
            "distance": 200,
            "orientation": "E",
            "orientation_id": 2,
            "distance_group": 1,
            ...
        },
        "turn": {
            "angle_deg": -45.0,
            "category": "slight_left"
        },
        "trajectory": {
            "x": [...],
            "y": [...],
            "timestamps": [...]
        }
    }
    """
    print_section("STEP 1: Loading Segment Trajectories (Three-Dot Format)")

    data_dir = Path(data_dir)

    # Look for segment JSON files
    json_files = list(data_dir.glob('**/segment_*.json'))

    if len(json_files) == 0:
        # Try looking in segments subdirectory
        json_files = list(data_dir.glob('**/segments/segment_*.json'))

    print(f"Found {len(json_files)} segment JSON files")

    if len(json_files) == 0:
        print(f"ERROR: No segment_*.json files found in {data_dir}")
        return []

    trajectories = []
    skipped = defaultdict(int)

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check required fields
            required = ['trajectory', 'AB', 'BC', 'turn']
            if not all(k in data for k in required):
                skipped['missing_fields'] += 1
                continue

            # Extract trajectory
            traj_data = data['trajectory']
            if 'x' not in traj_data or 'y' not in traj_data:
                skipped['missing_trajectory'] += 1
                continue

            x = np.array(traj_data['x'], dtype=np.float64)
            y = np.array(traj_data['y'], dtype=np.float64)

            # Length filtering
            if len(x) < config['min_length']:
                skipped['too_short'] += 1
                continue
            if len(x) > config['max_length_filter']:
                skipped['too_long'] += 1
                continue

            # Convert to relative coordinates (start at origin)
            x_rel = x - x[0]
            y_rel = y - y[0]

            # Extract AB conditioning
            ab = data['AB']
            orient_AB = ab.get('orientation', 'E')
            orient_id_AB = ab.get('orientation_id', get_orientation_id(orient_AB))
            dist_AB = ab.get('distance', 100)
            dist_id_AB = ab.get('distance_group', get_distance_group_id(dist_AB))

            # Extract BC conditioning (THE KEY FOR V5!)
            bc = data['BC']
            orient_BC = bc.get('orientation', 'E')
            orient_id_BC = bc.get('orientation_id', get_orientation_id(orient_BC))
            dist_BC = bc.get('distance', 100)
            dist_id_BC = bc.get('distance_group', get_distance_group_id(dist_BC))

            # Extract turn info
            turn = data['turn']
            turn_angle = turn.get('angle_deg', 0.0)
            turn_category_id = get_turn_category_id(turn_angle)

            # Store trajectory with all conditioning
            trajectories.append({
                'x': x_rel,
                'y': y_rel,
                'flat': trajectory_to_flat(x_rel, y_rel),
                'length': len(x),

                # AB conditioning (current segment)
                'orientation_id_AB': orient_id_AB,
                'distance_id_AB': dist_id_AB,
                'distance_AB': dist_AB,

                # BC conditioning (next segment - anticipation!)
                'orientation_id_BC': orient_id_BC,
                'distance_id_BC': dist_id_BC,
                'distance_BC': dist_BC,

                # Turn info
                'turn_angle': turn_angle,
                'turn_category_id': turn_category_id,

                'filename': json_file.name,
            })

        except Exception as e:
            skipped['error'] += 1
            if skipped['error'] <= 3:
                print(f"  Error loading {json_file.name}: {e}")

    # Report results
    print(f"\nLoaded {len(trajectories)} trajectories")

    if skipped:
        print(f"Skipped: {sum(skipped.values())} total")
        for reason, count in sorted(skipped.items()):
            print(f"  - {reason}: {count}")

    if trajectories:
        # Length statistics
        lengths = [t['length'] for t in trajectories]
        print(f"\nTrajectory lengths:")
        print(f"  Range: [{min(lengths)}, {max(lengths)}]")
        print(f"  Median: {np.median(lengths):.0f}")
        print(f"  Mean: {np.mean(lengths):.1f}")

        # Turn category distribution
        print(f"\nTurn category distribution:")
        turn_counts = defaultdict(int)
        for t in trajectories:
            turn_counts[t['turn_category_id']] += 1

        for cat_id in range(NUM_TURN_CATEGORIES):
            name = TURN_CATEGORY_NAMES[cat_id]
            count = turn_counts[cat_id]
            pct = count / len(trajectories) * 100
            print(f"  {name:<15} {count:>6} ({pct:>5.1f}%)")

        # Turn angle statistics
        turn_angles = [t['turn_angle'] for t in trajectories]
        print(f"\nTurn angle statistics:")
        print(f"  Range: [{min(turn_angles):.1f}°, {max(turn_angles):.1f}°]")
        print(f"  Mean: {np.mean(turn_angles):.1f}°")
        print(f"  Std: {np.std(turn_angles):.1f}°")

        # AB orientation distribution
        print(f"\nAB orientation distribution:")
        orient_counts = defaultdict(int)
        for t in trajectories:
            orient_counts[t['orientation_id_AB']] += 1
        for oid in range(NUM_ORIENTATIONS):
            print(f"  {ORIENTATIONS[oid]}: {orient_counts[oid]}")

    return trajectories


# =============================================================================
# STEP 2: LEARN REFERENCE BASIS (UNCHANGED FROM V3)
# =============================================================================

def learn_basis_from_trajectories(
    trajectories: List[Dict],
    n_control_points: int,
    K: int,
    center_data: bool = True
) -> Tuple[np.ndarray, int, Optional[np.ndarray], Dict]:
    """
    Learn SVD reference basis from trajectories.
    Same as V3 - math doesn't change.
    """
    print_section("STEP 2: Learning Reference Basis")

    N = len(trajectories)
    T_ref = n_control_points

    print(f"Parameters:")
    print(f"  Trajectories: {N}")
    print(f"  Control points (T_ref): {T_ref}")
    print(f"  Singular dimensions (K): {K}")
    print(f"  Center data: {center_data}")

    if K > N:
        print(f"  WARNING: K ({K}) > N ({N}), reducing K to {N}")
        K = N
    if K > 2 * T_ref:
        print(f"  WARNING: K ({K}) > 2*T_ref ({2*T_ref}), reducing K to {2*T_ref}")
        K = 2 * T_ref

    # Fit B-spline control points
    print(f"\nStage 1: Fitting B-spline control points...")

    control_points = np.zeros((N, 2 * T_ref), dtype=np.float64)
    fit_errors = []

    for i, traj in enumerate(trajectories):
        flat = traj['flat']
        cp = fit_bspline_control_points(flat, n_control_points)
        control_points[i] = cp

        fit_eval = evaluate_control_point_fit(flat, cp)
        fit_errors.append(fit_eval['rmse'])

        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{N}...")

    fit_errors = np.array(fit_errors)
    print(f"\n  Fitting complete: RMSE mean={fit_errors.mean():.4f}, max={fit_errors.max():.4f}")

    # SVD
    print(f"\nStage 2: Computing SVD...")

    if center_data:
        mean = control_points.mean(axis=0)
        centered = control_points - mean
        print(f"  Centered data (mean norm: {np.linalg.norm(mean):.2f})")
    else:
        mean = None
        centered = control_points

    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]
    S_k = S[:K]

    total_var = np.sum(S ** 2)
    explained_var = np.cumsum(S[:K] ** 2) / total_var

    print(f"\n  Singular values (top {min(K, 8)}):")
    for i in range(min(K, 8)):
        pct = (S[i]**2 / total_var) * 100
        cum_pct = explained_var[i] * 100
        print(f"    S[{i+1:2d}] = {S[i]:10.2f}  ({pct:5.2f}% var, {cum_pct:6.2f}% cumulative)")

    print(f"\n  K={K} explains {explained_var[-1]*100:.2f}% of variance")

    stats = {
        'n_trajectories': N,
        'n_control_points': T_ref,
        'K': K,
        'fit_rmse_mean': float(fit_errors.mean()),
        'fit_rmse_max': float(fit_errors.max()),
        'variance_explained': float(explained_var[-1]),
        'singular_values': S_k.tolist(),
    }

    return U_ref, T_ref, mean, stats


# =============================================================================
# STEP 3: PROJECT TRAJECTORIES TO K-SPACE (UNCHANGED FROM V3)
# =============================================================================

def project_trajectories(
    trajectories: List[Dict],
    U_ref: np.ndarray,
    T_ref: int,
    mean: Optional[np.ndarray]
) -> Tuple[np.ndarray, BasisAdapter, Dict]:
    """Project trajectories to K-space via control points."""
    print_section("STEP 3: Projecting Trajectories to K-Space")

    N = len(trajectories)
    K = U_ref.shape[1]

    print(f"Projecting {N} trajectories to {K}-dimensional space...")

    adapter = BasisAdapter(U_ref, T_ref)
    coefficients = np.zeros((N, K), dtype=np.float64)
    projection_errors = []

    for i, traj in enumerate(trajectories):
        flat = traj['flat']
        T = traj['length']

        cp = fit_bspline_control_points(flat, T_ref)

        if mean is not None:
            c = U_ref.T @ (cp - mean)
        else:
            c = U_ref.T @ cp

        coefficients[i] = c

        # Reconstruction error
        if mean is not None:
            cp_recon = U_ref @ c + mean
        else:
            cp_recon = U_ref @ c

        C = build_interleaved_basis_matrix(T, T_ref)
        recon = C @ cp_recon
        rmse = np.sqrt(np.mean((flat - recon) ** 2))
        projection_errors.append(rmse)

        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{N}...")

    projection_errors = np.array(projection_errors)

    print(f"\n  Projection complete:")
    print(f"    RMSE: mean={projection_errors.mean():.4f}, max={projection_errors.max():.4f}")
    print(f"    Coefficient std: {coefficients.std():.4f}")

    stats = {
        'reconstruction_rmse_mean': float(projection_errors.mean()),
        'reconstruction_rmse_max': float(projection_errors.max()),
        'coefficient_std': float(coefficients.std()),
    }

    return coefficients, adapter, stats


# =============================================================================
# STEP 4: GENERATE TURN-CATEGORY ANCHORS (V5 SPECIFIC)
# =============================================================================

def generate_turn_category_anchors(
    trajectories: List[Dict],
    coefficients: np.ndarray,
    anchors_per_category: int,
    random_seed: int
) -> Tuple[Dict, Dict]:
    """
    Generate anchor prototypes for each TURN CATEGORY.

    This is the V5 innovation: group by how much the direction changes,
    not by absolute direction. Trajectories with similar turns have
    similar shapes regardless of compass direction.
    """
    print_section("STEP 4: Generating Turn-Category Anchors")

    np.random.seed(random_seed)
    K = coefficients.shape[1]

    print(f"Parameters:")
    print(f"  Anchors per turn category: {anchors_per_category}")
    print(f"  Turn categories: {NUM_TURN_CATEGORIES}")
    print(f"  Random seed: {random_seed}")

    # Group by turn category
    groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        cat_id = traj['turn_category_id']
        groups[cat_id].append(i)

    turn_anchors = {}
    turn_stats = {}

    print(f"\n  {'Category':<15} {'Count':>8} {'Anchors':>8} {'Inertia':>12}")
    print(f"  {'-'*45}")

    total_anchors = 0

    for cat_id in range(NUM_TURN_CATEGORIES):
        indices = groups[cat_id]
        cat_name = TURN_CATEGORY_NAMES[cat_id]
        n_samples = len(indices)

        if n_samples == 0:
            turn_anchors[cat_id] = None
            n_anchors = 0
            inertia = 0.0

        elif n_samples <= anchors_per_category:
            turn_anchors[cat_id] = coefficients[indices].copy()
            n_anchors = n_samples
            inertia = 0.0

        else:
            cat_coeffs = coefficients[indices]
            n_clusters = min(anchors_per_category, n_samples)

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_seed,
                n_init=10,
                max_iter=300
            )
            kmeans.fit(cat_coeffs)

            turn_anchors[cat_id] = kmeans.cluster_centers_.copy()
            n_anchors = n_clusters
            inertia = kmeans.inertia_

        total_anchors += n_anchors

        turn_stats[cat_id] = {
            'name': cat_name,
            'count': n_samples,
            'n_anchors': n_anchors,
            'inertia': float(inertia),
        }

        print(f"  {cat_name:<15} {n_samples:>8} {n_anchors:>8} {inertia:>12.2f}")

    print(f"\n  Total anchors: {total_anchors}")

    return turn_anchors, turn_stats


# =============================================================================
# STEP 5: COMPUTE RESIDUALS (PER TURN CATEGORY)
# =============================================================================

def compute_residuals(
    trajectories: List[Dict],
    coefficients: np.ndarray,
    turn_anchors: Dict
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute residuals using nearest anchor within same TURN CATEGORY.
    """
    print_section("STEP 5: Computing Residuals (Per Turn Category)")

    N = len(trajectories)
    K = coefficients.shape[1]

    residuals = np.zeros((N, K), dtype=np.float64)
    anchor_indices = np.zeros(N, dtype=np.int32)
    distances_to_anchor = []

    no_anchor_count = 0

    for i, traj in enumerate(trajectories):
        cat_id = traj['turn_category_id']
        anchors = turn_anchors[cat_id]

        if anchors is None:
            residuals[i] = coefficients[i]
            anchor_indices[i] = -1
            distances_to_anchor.append(np.linalg.norm(coefficients[i]))
            no_anchor_count += 1
        else:
            dists = np.linalg.norm(anchors - coefficients[i], axis=1)
            nearest_idx = dists.argmin()

            residuals[i] = coefficients[i] - anchors[nearest_idx]
            anchor_indices[i] = nearest_idx
            distances_to_anchor.append(dists[nearest_idx])

    distances_to_anchor = np.array(distances_to_anchor)

    print(f"Residuals computed:")
    print(f"  Shape: {residuals.shape}")
    print(f"  Without anchors: {no_anchor_count}")

    print(f"\nResidual magnitude:")
    print(f"  Mean: {distances_to_anchor.mean():.4f}")
    print(f"  Max:  {distances_to_anchor.max():.4f}")

    coeff_norms = np.linalg.norm(coefficients, axis=1)
    reduction_ratio = distances_to_anchor.mean() / coeff_norms.mean()

    print(f"\nReduction ratio: {reduction_ratio:.2%}")
    print(f"  → Diffusion learns {reduction_ratio:.1%} of original signal")

    stats = {
        'residual_mean_norm': float(distances_to_anchor.mean()),
        'residual_max_norm': float(distances_to_anchor.max()),
        'coefficient_mean_norm': float(coeff_norms.mean()),
        'reduction_ratio': float(reduction_ratio),
        'no_anchor_count': no_anchor_count,
    }

    return residuals, anchor_indices, stats


# =============================================================================
# STEP 5.5: NORMALIZE RESIDUALS
# =============================================================================

def normalize_residuals(residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize residuals to zero-mean, unit-variance per dimension."""
    print_section("STEP 5.5: Normalizing Residuals")

    residual_mean = residuals.mean(axis=0)
    residual_std = residuals.std(axis=0)
    residual_std = np.maximum(residual_std, 1e-8)

    normalized = (residuals - residual_mean) / residual_std

    print(f"Raw residuals: range=[{residuals.min():.4f}, {residuals.max():.4f}]")
    print(f"Normalized:    range=[{normalized.min():.4f}, {normalized.max():.4f}]")
    print(f"               mean={normalized.mean():.6f}, std={normalized.std():.6f}")

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
    Split data with V5 conditioning arrays.
    """
    print_section("STEP 6: Splitting Data")

    N = len(trajectories)
    indices = np.arange(N)

    # Extract all conditioning arrays
    orient_ids_AB = np.array([t['orientation_id_AB'] for t in trajectories], dtype=np.int32)
    dist_ids_AB = np.array([t['distance_id_AB'] for t in trajectories], dtype=np.int32)
    orient_ids_BC = np.array([t['orientation_id_BC'] for t in trajectories], dtype=np.int32)
    dist_ids_BC = np.array([t['distance_id_BC'] for t in trajectories], dtype=np.int32)
    turn_angles = np.array([t['turn_angle'] for t in trajectories], dtype=np.float32)
    turn_category_ids = np.array([t['turn_category_id'] for t in trajectories], dtype=np.int32)
    original_lengths = np.array([t['length'] for t in trajectories], dtype=np.int32)

    print(f"Total samples: {N}")
    print(f"Split ratios: train={config['train_ratio']}, val={config['val_ratio']}, test={config['test_ratio']}")

    # Stratify by turn category for balanced splits
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=config['train_ratio'],
        random_state=config['random_seed'],
        stratify=turn_category_ids
    )

    val_ratio_adjusted = config['val_ratio'] / (config['val_ratio'] + config['test_ratio'])
    temp_turn_ids = turn_category_ids[temp_idx]

    unique, counts = np.unique(temp_turn_ids, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0

    if min_count >= 2:
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            random_state=config['random_seed'],
            stratify=temp_turn_ids
        )
    else:
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            random_state=config['random_seed']
        )

    splits = {}

    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        splits[name] = {
            'indices': idx,
            'coefficients': coefficients[idx],
            'residuals': residuals[idx],
            'anchor_indices': anchor_indices[idx],

            # AB conditioning (current segment)
            'orientation_ids_AB': orient_ids_AB[idx],
            'distance_ids_AB': dist_ids_AB[idx],

            # BC conditioning (next segment - anticipation!)
            'orientation_ids_BC': orient_ids_BC[idx],
            'distance_ids_BC': dist_ids_BC[idx],

            # Turn info
            'turn_angles': turn_angles[idx],
            'turn_category_ids': turn_category_ids[idx],

            'original_lengths': original_lengths[idx],
        }

        # Summary
        print(f"\n  {name}: {len(idx)} samples")
        turn_dist = np.bincount(turn_category_ids[idx], minlength=NUM_TURN_CATEGORIES)
        print(f"    Turn categories: {dict(zip(TURN_CATEGORY_NAMES, turn_dist))}")

    return splits


# =============================================================================
# STEP 7: SAVE DATA
# =============================================================================

def save_all_data(
    output_dir: Path,
    U_ref: np.ndarray,
    T_ref: int,
    mean: Optional[np.ndarray],
    turn_anchors: Dict,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    splits: Dict,
    config: Dict,
    stats: Dict
) -> Path:
    """Save all preprocessed data."""
    print_section("STEP 7: Saving Data")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create basis subdirectory
    basis_dir = output_path / 'basis'
    basis_dir.mkdir(exist_ok=True)

    # Save basis
    np.save(basis_dir / 'U_ref.npy', U_ref)
    print(f"  Saved: basis/U_ref.npy {U_ref.shape}")

    if mean is not None:
        np.save(basis_dir / 'mean.npy', mean)
        print(f"  Saved: basis/mean.npy {mean.shape}")

    # Save turn-category anchors
    np.save(output_path / 'turn_category_anchors.npy', turn_anchors, allow_pickle=True)
    n_anchors = sum(len(v) if v is not None else 0 for v in turn_anchors.values())
    print(f"  Saved: turn_category_anchors.npy ({n_anchors} total anchors)")

    # Save residual normalization
    np.save(output_path / 'residual_mean.npy', residual_mean)
    np.save(output_path / 'residual_std.npy', residual_std)
    print(f"  Saved: residual_mean.npy, residual_std.npy")

    # Build config
    K = U_ref.shape[1]

    full_config = {
        **config,
        'K': K,
        'T_ref': T_ref,
        'n_control_points': T_ref,
        'version': 'v5_anticipatory',

        # V5 specific
        'num_orientations': NUM_ORIENTATIONS,
        'num_distance_groups': NUM_DISTANCE_GROUPS,
        'num_turn_categories': NUM_TURN_CATEGORIES,
        'orientations': ORIENTATIONS,
        'distance_groups': DISTANCE_GROUPS,
        'turn_categories': TURN_CATEGORIES,
        'turn_category_names': TURN_CATEGORY_NAMES,

        'stats': stats,
    }

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
    """Verify preprocessing correctness."""
    print_section("STEP 8: Verification")

    all_passed = True

    # Test reconstruction
    print("\n[Test 1] Reconstruction accuracy...")

    sample_indices = np.random.choice(len(trajectories), min(n_samples, len(trajectories)), replace=False)

    mean_path = output_dir / 'basis' / 'mean.npy'
    mean = np.load(mean_path) if mean_path.exists() else None

    errors = []
    for i in sample_indices:
        traj = trajectories[i]
        flat = traj['flat']
        T = traj['length']
        c = coefficients[i]

        cp_recon = adapter.U_ref @ c
        if mean is not None:
            cp_recon = cp_recon + mean
        C = build_interleaved_basis_matrix(T, adapter.T_ref)
        recon = C @ cp_recon

        error = np.sqrt(np.mean((flat - recon) ** 2))
        errors.append(error)

    print(f"  Mean RMSE: {np.mean(errors):.4f}")
    print(f"  Max RMSE:  {np.max(errors):.4f}")

    if np.mean(errors) > 100:
        print(f"  WARNING: High reconstruction error!")
        all_passed = False
    else:
        print(f"  OK")

    # Test conditioning arrays
    print("\n[Test 2] Conditioning arrays...")

    train_dir = output_dir / 'train'
    required_files = [
        'orientation_ids_AB.npy',
        'distance_ids_AB.npy',
        'orientation_ids_BC.npy',
        'distance_ids_BC.npy',
        'turn_angles.npy',
        'turn_category_ids.npy',
    ]

    for fname in required_files:
        fpath = train_dir / fname
        if fpath.exists():
            arr = np.load(fpath)
            print(f"  OK: {fname} {arr.shape}")
        else:
            print(f"  MISSING: {fname}")
            all_passed = False

    print(f"\n{'='*70}")
    print("VERIFICATION PASSED" if all_passed else "VERIFICATION FAILED")
    print('='*70)

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess three-dot trajectories with anticipatory conditioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python preprocess_singular_v5_anticipatory.py --input segments/ --output processed_v5/

    python preprocess_singular_v5_anticipatory.py --input three_dot_flow/session_*/segments/ \\
        --output processed_v5/ --anchors_per_category 30
"""
    )

    parser.add_argument('--input', type=str, default=DEFAULT_CONFIG['input_location'])
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_location'])
    parser.add_argument('--n_control_points', type=int, default=DEFAULT_CONFIG['n_control_points'])
    parser.add_argument('--k', type=int, default=DEFAULT_CONFIG['K'])
    parser.add_argument('--anchors_per_category', type=int, default=DEFAULT_CONFIG['anchors_per_turn_category'])
    parser.add_argument('--min_length', type=int, default=DEFAULT_CONFIG['min_length'])
    parser.add_argument('--max_length', type=int, default=DEFAULT_CONFIG['max_length_filter'])
    parser.add_argument('--no_center', action='store_true')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['random_seed'])
    parser.add_argument('--skip_verify', action='store_true')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['input_location'] = args.input
    config['output_location'] = args.output
    config['n_control_points'] = args.n_control_points
    config['K'] = args.k
    config['anchors_per_turn_category'] = args.anchors_per_category
    config['min_length'] = args.min_length
    config['max_length_filter'] = args.max_length
    config['center_data'] = not args.no_center
    config['random_seed'] = args.seed

    # Header
    print("=" * 70)
    print("SINGULAR SPACE PREPROCESSING - V5 ANTICIPATORY (THREE-DOT)")
    print("=" * 70)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print(f"\nParameters:")
    print(f"  n_control_points: {config['n_control_points']}")
    print(f"  K:                {config['K']}")
    print(f"  anchors_per_turn_category: {config['anchors_per_turn_category']}")
    print(f"  center_data:      {config['center_data']}")
    print("\n" + "=" * 70)
    print("KEY INNOVATION: Dual conditioning (AB + BC) with turn-category anchors")
    print("=" * 70)

    np.random.seed(config['random_seed'])

    # Pipeline
    trajectories = load_trajectories(args.input, config)
    if len(trajectories) == 0:
        print("\nERROR: No trajectories loaded!")
        sys.exit(1)

    U_ref, T_ref, mean, basis_stats = learn_basis_from_trajectories(
        trajectories, config['n_control_points'], config['K'], config['center_data']
    )

    coefficients, adapter, projection_stats = project_trajectories(
        trajectories, U_ref, T_ref, mean
    )

    turn_anchors, turn_stats = generate_turn_category_anchors(
        trajectories, coefficients, config['anchors_per_turn_category'], config['random_seed']
    )

    residuals, anchor_indices, residual_stats = compute_residuals(
        trajectories, coefficients, turn_anchors
    )

    normalized_residuals, residual_mean, residual_std = normalize_residuals(residuals)

    splits = split_data(
        trajectories, coefficients, normalized_residuals, anchor_indices, config
    )

    all_stats = {
        'basis': basis_stats,
        'projection': projection_stats,
        'residual': residual_stats,
        'turn_categories': turn_stats,
    }

    output_path = save_all_data(
        args.output, U_ref, T_ref, mean, turn_anchors,
        residual_mean, residual_std, splits, config, all_stats
    )

    if not args.skip_verify:
        verify_preprocessing(output_path, trajectories, adapter, coefficients)

    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE - V5 ANTICIPATORY")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print(f"\nKey files:")
    print(f"  - basis/U_ref.npy, mean.npy")
    print(f"  - turn_category_anchors.npy ({NUM_TURN_CATEGORIES} categories)")
    print(f"  - train/: orientation_ids_AB, orientation_ids_BC, turn_angles, etc.")
    print(f"\nStatistics:")
    print(f"  - Trajectories: {len(trajectories)}")
    print(f"  - Variance explained: {basis_stats['variance_explained']*100:.2f}%")
    print(f"  - Residual reduction: {residual_stats['reduction_ratio']*100:.1f}%")
    print(f"\nConditioning for training:")
    print(f"  - AB: orientation (0-7), distance (0-4)")
    print(f"  - BC: orientation (0-7), distance (0-4)")
    print(f"  - Turn category (0-6) for anchor selection")


if __name__ == '__main__':
    main()
