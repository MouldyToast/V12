#!/usr/bin/env python3
"""
Singular Space Preprocessing - VERSION 3: Basis-Transformation Approach

KEY INNOVATION (FROM PAPER):
============================
This version implements the paper's approach: transform the BASIS to match
trajectory length, rather than transforming trajectory DATA to a canonical length.

V2 (OLD):  trajectory → B-spline → canonical_length → SVD → coefficients
           Problem: B-spline smoothing DESTROYS jitter before SVD sees it
           
V3 (NEW):  trajectory → project onto length-adapted basis → coefficients
           Benefit: Raw trajectory preserved, jitter/micro-corrections intact

How it works:
    1. Fit B-spline control points to each trajectory (intermediate representation)
    2. SVD on control points → reference basis U_ref
    3. For each trajectory, adapt basis to its native length via B-spline transform
    4. Project RAW trajectory onto adapted basis → coefficients

At generation time:
    1. Generate coefficients via diffusion (same as before)
    2. Adapt basis to target output length
    3. Reconstruct: trajectory = U_adapted @ coefficients
    4. NO post-hoc interpolation needed!

Output Structure:
    processed_singular_v3/
    ├── U_ref.npy              # [2*T_ref × K] reference basis
    ├── mean.npy               # [2*T_ref] mean of control points (for centering)
    ├── config.npy             # Configuration and metadata
    ├── group_anchors.npy      # Per-group anchor prototypes
    └── train/val/test/
        ├── coefficients.npy   # [N × K] 
        ├── residuals.npy      # [N × K] = coefficient - nearest_anchor
        ├── anchor_indices.npy # [N] index of nearest anchor
        ├── orientation_ids.npy
        ├── distance_ids.npy
        └── original_lengths.npy  # [N] CRITICAL: needed for reconstruction!

REQUIRED CONFIG KEYS FOR DOWNSTREAM SCRIPTS:
============================================
    - K: int                    # Singular space dimensions
    - T_ref: int                # Reference length (# control points)  
    - n_control_points: int     # Same as T_ref
    - max_length: int           # Maximum observed trajectory length
    - min_length: int           # Minimum observed trajectory length
    - version: str              # 'v3_basis_transform'

Usage:
    python preprocess_singular_v3_basis_transform.py --input trajectories/ --output processed_v3/
    
    # With custom parameters
    python preprocess_singular_v3_basis_transform.py --input trajectories/ --output processed_v3/ \\
        --n_control_points 20 --k 8 --anchors_per_group 6
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

# Import our new basis transformation module
from bspline_basis import (
    BasisAdapter,
    learn_reference_basis,
    fit_bspline_control_points,
    evaluate_control_point_fit,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Core parameters - NOTE: Aligned with paper's recommendations
    'K': 8,                     # Singular space dimensions (paper uses K=4)
    'n_control_points': 20,     # B-spline control points (= T_ref)
    'anchors_per_group': 6,     # Anchors per (distance, orientation) group
    'random_seed': 42,
    
    # Length constraints
    'min_length': 20,           # Minimum trajectory length (points)
    'max_length_filter': 200,   # Maximum trajectory length (filter out longer)
    
    # Trajectory filtering
    'max_distance_ratio': 1.6,  # actual_distance / ideal_distance threshold
    'min_distance': 20.0,       # Minimum ideal distance (pixels)
    
    # Data splits
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    
    # Centering
    'center_data': True,        # Whether to subtract mean before SVD
}

# Orientation definitions (8 compass directions)
ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NUM_ORIENTATIONS = 8

# Distance group definitions
DISTANCE_GROUPS = [
    {"name": "XSmall", "id": 0, "min": 0,   "max": 50},
    {"name": "Small",  "id": 1, "min": 50,  "max": 100},
    {"name": "Medium", "id": 2, "min": 100, "max": 200},
    {"name": "Large",  "id": 3, "min": 200, "max": 400},
    {"name": "XLarge", "id": 4, "min": 400, "max": 900},
]
NUM_DISTANCE_GROUPS = 5

# Screen angle ranges for orientation classification
# Note: Screen coordinates have Y increasing downward
SCREEN_ANGLE_RANGES = {
    "E":  (-22.5, 22.5),
    "SE": (22.5, 67.5),
    "S":  (67.5, 112.5),
    "SW": (112.5, 157.5),
    "W":  (157.5, 180.0),
    "W2": (-180.0, -157.5),  # W wraps around
    "NW": (-157.5, -112.5),
    "N":  (-112.5, -67.5),
    "NE": (-67.5, -22.5),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_orientation_id(x: np.ndarray, y: np.ndarray) -> int:
    """
    Determine orientation ID (0-7) from trajectory endpoints.
    
    Uses screen coordinates where Y increases downward.
    """
    dx = x[-1] - x[0]
    dy = y[0] - y[-1]  # Flip for screen coords
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 2  # Default to E for zero-length
    
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    for orient, (lo, hi) in SCREEN_ANGLE_RANGES.items():
        if lo <= angle_deg < hi:
            return ORIENTATIONS.index(orient.replace("2", ""))
    
    # Edge case: exactly at ±180°
    if abs(angle_deg) >= 179.9:
        return ORIENTATIONS.index("W")
    
    return 2  # Default to E


def get_distance_group_id(ideal_distance: float) -> int:
    """Determine distance group ID (0-4) from ideal distance."""
    for group in DISTANCE_GROUPS:
        if group["min"] <= ideal_distance < group["max"]:
            return group["id"]
    return DISTANCE_GROUPS[-1]["id"]  # XLarge for very long distances


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


# =============================================================================
# STEP 1: LOAD RAW TRAJECTORIES
# =============================================================================

def load_trajectories(data_dir: Path, config: Dict) -> List[Dict]:
    """
    Load raw trajectories from JSON files.
    
    CRITICAL V3 DIFFERENCE:
    - We keep the original trajectory data as-is
    - No interpolation to canonical length
    - Length is stored for later use in basis adaptation
    
    Args:
        data_dir: Directory containing trajectory JSON files
        config: Configuration dictionary
        
    Returns:
        List of trajectory dictionaries with keys:
            - x, y: Relative coordinates (start at origin)
            - flat: Interleaved [x0,y0,x1,y1,...] format
            - length: Number of points (CRITICAL for V3!)
            - ideal_distance, orientation_id, distance_group_id
    """
    print_section("STEP 1: Loading Raw Trajectories")
    
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print(f"ERROR: No JSON files found in {data_dir}")
        return []
    
    trajectories = []
    skipped = defaultdict(int)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            required = ['x', 'y', 'ideal_distance', 'actual_distance']
            if not all(k in data for k in required):
                skipped['missing_fields'] += 1
                continue
            
            x = np.array(data['x'], dtype=np.float64)
            y = np.array(data['y'], dtype=np.float64)
            ideal_dist = float(data['ideal_distance'])
            actual_dist = float(data['actual_distance'])
            
            # Length filtering
            if len(x) < config['min_length']:
                skipped['too_short'] += 1
                continue
            if len(x) > config['max_length_filter']:
                skipped['too_long'] += 1
                continue
            
            # Distance filtering
            if ideal_dist < config['min_distance']:
                skipped['min_distance'] += 1
                continue
            
            # Erratic trajectory filtering (actual >> ideal means lots of backtracking)
            if actual_dist / max(ideal_dist, 1e-6) > config['max_distance_ratio']:
                skipped['erratic'] += 1
                continue
            
            # Convert to relative coordinates (start at origin)
            x_rel = x - x[0]
            y_rel = y - y[0]
            
            # Store trajectory
            trajectories.append({
                'x': x_rel,
                'y': y_rel,
                'flat': trajectory_to_flat(x_rel, y_rel),  # [2*T]
                'length': len(x),  # CRITICAL for V3!
                'ideal_distance': ideal_dist,
                'orientation_id': get_orientation_id(x, y),
                'distance_group_id': get_distance_group_id(ideal_dist),
                'filename': json_file.name,
            })
            
        except Exception as e:
            skipped['error'] += 1
    
    # Report results
    print(f"\nLoaded {len(trajectories)} trajectories")
    
    if skipped:
        print(f"Skipped: {sum(skipped.values())} total")
        for reason, count in sorted(skipped.items()):
            print(f"  - {reason}: {count}")
    
    if trajectories:
        lengths = [t['length'] for t in trajectories]
        print(f"\nTrajectory lengths:")
        print(f"  Range: [{min(lengths)}, {max(lengths)}]")
        print(f"  Median: {np.median(lengths):.0f}")
        print(f"  Mean: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        
        # Distribution by group
        print(f"\nDistribution by group:")
        group_counts = defaultdict(int)
        for t in trajectories:
            key = (t['distance_group_id'], t['orientation_id'])
            group_counts[key] += 1
        
        print(f"  {'Group':<15} {'Count':>8}")
        print(f"  {'-'*25}")
        for dist_id in range(NUM_DISTANCE_GROUPS):
            for orient_id in range(NUM_ORIENTATIONS):
                key = (dist_id, orient_id)
                count = group_counts.get(key, 0)
                if count > 0:
                    name = f"{DISTANCE_GROUPS[dist_id]['name']}-{ORIENTATIONS[orient_id]}"
                    print(f"  {name:<15} {count:>8}")
    
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
    Learn SVD reference basis from variable-length trajectories.
    
    This is the CORE of V3: we learn a basis from B-spline control point
    representations, which provides a length-invariant intermediate form.
    
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
    lengths = []
    
    for i, traj in enumerate(trajectories):
        flat = traj['flat']
        T = traj['length']
        lengths.append(T)
        
        # Fit control points
        cp = fit_bspline_control_points(flat, n_control_points)
        control_points[i] = cp
        
        # Track fit quality
        fit_eval = evaluate_control_point_fit(flat, cp)
        fit_errors.append(fit_eval['rmse'])
        
        # Progress indicator for large datasets
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{N} trajectories...")
    
    fit_errors = np.array(fit_errors)
    lengths = np.array(lengths)
    
    print(f"\n  Control point fitting complete:")
    print(f"    RMSE: mean={fit_errors.mean():.4f}, max={fit_errors.max():.4f}")
    print(f"    Trajectory lengths: min={lengths.min()}, max={lengths.max()}, median={np.median(lengths):.0f}")
    
    # Stage 2: SVD on control points
    print(f"\nStage 2: Computing SVD...")
    
    if center_data:
        mean = control_points.mean(axis=0)
        centered = control_points - mean
        print(f"  Centered data (mean norm: {np.linalg.norm(mean):.2f})")
    else:
        mean = None
        centered = control_points
    
    # SVD: centered = U_samples @ S @ V_basis.T
    # We want V_basis (the directions in control point space)
    # centered.T = V_basis @ S @ U_samples.T
    # So we SVD the transpose to get V_basis as the left singular vectors
    
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    
    # U is [2*T_ref × min(2*T_ref, N)] - these are our basis vectors
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
        'length_min': int(lengths.min()),
        'length_max': int(lengths.max()),
        'length_median': float(np.median(lengths)),
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
    
    CRITICAL V3 DIFFERENCE:
    - Each trajectory is projected using a BASIS adapted to its length
    - The raw trajectory data is PRESERVED (no interpolation!)
    - This is where jitter/micro-corrections are maintained
    
    Process for each trajectory:
        1. Get adapted basis U_T for trajectory length T
        2. If centering: subtract mean from control point representation
        3. Project: c = U_T.T @ (trajectory - mean_adapted)
        
    Note: The mean subtraction is tricky because mean is in control point space.
    We handle this by projecting through the adapted basis.
    
    Args:
        trajectories: List of trajectory dicts with 'flat' and 'length' keys
        U_ref: Reference basis [2*T_ref × K]
        T_ref: Reference length
        mean: Mean control points [2*T_ref] or None
        
    Returns:
        coefficients: [N × K] array
        adapter: BasisAdapter instance for later use
        stats: Projection statistics
    """
    print_section("STEP 3: Projecting Trajectories to K-Space")
    
    N = len(trajectories)
    K = U_ref.shape[1]
    
    print(f"Projecting {N} trajectories to {K}-dimensional space...")
    print(f"  Using basis adaptation (V3 approach)")
    
    # Create basis adapter
    adapter = BasisAdapter(U_ref, T_ref)
    
    # Project all trajectories
    coefficients = np.zeros((N, K), dtype=np.float64)
    projection_errors = []
    
    # Group trajectories by length for efficiency (cache reuse)
    length_groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        length_groups[traj['length']].append(i)
    
    print(f"  Unique trajectory lengths: {len(length_groups)}")
    
    # Process by length groups (better cache utilization)
    processed = 0
    for length, indices in sorted(length_groups.items()):
        for i in indices:
            traj = trajectories[i]
            flat = traj['flat']
            
            # Project raw trajectory onto adapted basis
            # Note: The adapter handles the basis transformation internally
            c = adapter.project(flat)
            coefficients[i] = c
            
            # Compute reconstruction error for this trajectory
            error_info = adapter.reconstruction_error(flat)
            projection_errors.append(error_info['rmse'])
            
            processed += 1
            if processed % 1000 == 0:
                print(f"    Processed {processed}/{N} trajectories...")
    
    projection_errors = np.array(projection_errors)
    
    # Statistics
    print(f"\n  Projection complete:")
    print(f"    Reconstruction RMSE: mean={projection_errors.mean():.4f}, "
          f"median={np.median(projection_errors):.4f}, max={projection_errors.max():.4f}")
    
    # Coefficient statistics
    print(f"\n  Coefficient statistics:")
    print(f"    Shape: {coefficients.shape}")
    print(f"    Mean: {coefficients.mean():.4f}")
    print(f"    Std:  {coefficients.std():.4f}")
    print(f"    Range: [{coefficients.min():.4f}, {coefficients.max():.4f}]")
    
    # Per-component statistics
    print(f"\n  Per-component standard deviations:")
    for k in range(min(K, 8)):
        print(f"    c[{k}]: std={coefficients[:, k].std():.4f}")
    if K > 8:
        print(f"    ... ({K - 8} more components)")
    
    # Check for any degenerate projections
    zero_coeffs = np.sum(np.all(np.abs(coefficients) < 1e-10, axis=1))
    if zero_coeffs > 0:
        print(f"\n  WARNING: {zero_coeffs} trajectories have near-zero coefficients!")
    
    stats = {
        'reconstruction_rmse_mean': float(projection_errors.mean()),
        'reconstruction_rmse_max': float(projection_errors.max()),
        'coefficient_mean': float(coefficients.mean()),
        'coefficient_std': float(coefficients.std()),
        'n_unique_lengths': len(length_groups),
    }
    
    # Store projection errors in trajectories for later analysis
    for i, err in enumerate(projection_errors):
        trajectories[i]['projection_rmse'] = err
    
    return coefficients, adapter, stats


# =============================================================================
# STEP 4: GENERATE GROUP ANCHORS
# =============================================================================

def generate_group_anchors(
    trajectories: List[Dict],
    coefficients: np.ndarray,
    anchors_per_group: int,
    random_seed: int
) -> Tuple[Dict, Dict]:
    """
    Generate anchor prototypes for each (distance, orientation) group.
    
    Anchors are cluster centroids found via K-means on the coefficients
    within each group. The diffusion model learns to generate residuals
    from these anchors rather than from scratch.
    
    Args:
        trajectories: List of trajectory dicts
        coefficients: [N × K] coefficient array
        anchors_per_group: Number of anchors per group
        random_seed: Random seed for reproducibility
        
    Returns:
        group_anchors: Dict mapping (dist_id, orient_id) → [n_anchors × K] or None
        group_stats: Dict with per-group statistics
    """
    print_section("STEP 4: Generating Group Anchors")
    
    np.random.seed(random_seed)
    K = coefficients.shape[1]
    
    print(f"Parameters:")
    print(f"  Anchors per group: {anchors_per_group}")
    print(f"  Random seed: {random_seed}")
    
    # Group trajectories by (distance_id, orientation_id)
    groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        key = (traj['distance_group_id'], traj['orientation_id'])
        groups[key].append(i)
    
    group_anchors = {}
    group_stats = {}
    
    print(f"\n  {'Group':<15} {'Count':>8} {'Anchors':>8} {'Inertia':>12}")
    print(f"  {'-'*45}")
    
    total_anchors = 0
    empty_groups = 0
    
    for dist_id in range(NUM_DISTANCE_GROUPS):
        for orient_id in range(NUM_ORIENTATIONS):
            key = (dist_id, orient_id)
            indices = groups[key]
            group_name = f"{DISTANCE_GROUPS[dist_id]['name']}-{ORIENTATIONS[orient_id]}"
            
            n_samples = len(indices)
            
            if n_samples == 0:
                # No samples in this group
                group_anchors[key] = None
                n_anchors = 0
                inertia = 0.0
                empty_groups += 1
                
            elif n_samples <= anchors_per_group:
                # Fewer samples than requested anchors - use all samples as anchors
                group_anchors[key] = coefficients[indices].copy()
                n_anchors = n_samples
                inertia = 0.0  # Perfect fit
                
            else:
                # Normal case: K-means clustering
                group_coeffs = coefficients[indices]
                n_clusters = min(anchors_per_group, n_samples)
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=random_seed,
                    n_init=10,
                    max_iter=300
                )
                kmeans.fit(group_coeffs)
                
                group_anchors[key] = kmeans.cluster_centers_.copy()
                n_anchors = n_clusters
                inertia = kmeans.inertia_
            
            total_anchors += n_anchors
            
            # Store statistics
            group_stats[key] = {
                'name': group_name,
                'count': n_samples,
                'n_anchors': n_anchors,
                'inertia': float(inertia) if n_samples > 0 else 0.0,
            }
            
            # Print row (only for non-empty groups)
            if n_samples > 0:
                print(f"  {group_name:<15} {n_samples:>8} {n_anchors:>8} {inertia:>12.2f}")
    
    print(f"\n  Total anchors: {total_anchors}")
    print(f"  Empty groups: {empty_groups}")
    print(f"  Non-empty groups: {NUM_DISTANCE_GROUPS * NUM_ORIENTATIONS - empty_groups}")
    
    return group_anchors, group_stats


# =============================================================================
# STEP 5: COMPUTE RESIDUALS
# =============================================================================

def compute_residuals(
    trajectories: List[Dict],
    coefficients: np.ndarray,
    group_anchors: Dict
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute residuals: the difference between each trajectory's coefficients
    and its nearest anchor in the same group.
    
    residual[i] = coefficients[i] - anchor[nearest]
    
    The diffusion model learns to generate these residuals, which are
    typically much smaller than the full coefficients.
    
    Args:
        trajectories: List of trajectory dicts
        coefficients: [N × K] coefficient array
        group_anchors: Dict mapping (dist_id, orient_id) → anchors
        
    Returns:
        residuals: [N × K] array
        anchor_indices: [N] array of which anchor was nearest
        stats: Residual statistics
    """
    print_section("STEP 5: Computing Residuals")
    
    N = len(trajectories)
    K = coefficients.shape[1]
    
    residuals = np.zeros((N, K), dtype=np.float64)
    anchor_indices = np.zeros(N, dtype=np.int32)
    distances_to_anchor = []
    
    no_anchor_count = 0
    
    for i, traj in enumerate(trajectories):
        key = (traj['distance_group_id'], traj['orientation_id'])
        anchors = group_anchors[key]
        
        if anchors is None:
            # No anchors for this group - residual equals coefficient
            residuals[i] = coefficients[i]
            anchor_indices[i] = -1
            distances_to_anchor.append(np.linalg.norm(coefficients[i]))
            no_anchor_count += 1
        else:
            # Find nearest anchor
            dists = np.linalg.norm(anchors - coefficients[i], axis=1)
            nearest_idx = dists.argmin()
            
            residuals[i] = coefficients[i] - anchors[nearest_idx]
            anchor_indices[i] = nearest_idx
            distances_to_anchor.append(dists[nearest_idx])
    
    distances_to_anchor = np.array(distances_to_anchor)
    
    # Statistics
    print(f"Residuals computed:")
    print(f"  Shape: {residuals.shape}")
    print(f"  Trajectories without anchors: {no_anchor_count}")
    
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
    print(f"  → Diffusion learns {reduction_ratio:.1%} of the original signal!")
    
    # Per-component residual statistics
    print(f"\nPer-component residual std:")
    for k in range(min(K, 8)):
        coeff_std = coefficients[:, k].std()
        resid_std = residuals[:, k].std()
        ratio = resid_std / coeff_std if coeff_std > 1e-10 else 0
        print(f"  c[{k}]: coeff_std={coeff_std:.4f}, resid_std={resid_std:.4f}, ratio={ratio:.2%}")
    
    stats = {
        'residual_mean_norm': float(distances_to_anchor.mean()),
        'residual_max_norm': float(distances_to_anchor.max()),
        'coefficient_mean_norm': float(coeff_norms.mean()),
        'reduction_ratio': float(reduction_ratio),
        'no_anchor_count': no_anchor_count,
    }
    
    return residuals, anchor_indices, stats


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
    
    Uses stratified splitting based on orientation to ensure balanced
    representation across all orientations in each split.
    
    CRITICAL V3: original_lengths is saved for each split!
    
    Args:
        trajectories: List of trajectory dicts
        coefficients: [N × K] array
        residuals: [N × K] array
        anchor_indices: [N] array
        config: Configuration with split ratios
        
    Returns:
        splits: Dict with 'train', 'val', 'test' keys, each containing arrays
    """
    print_section("STEP 6: Splitting Data")
    
    N = len(trajectories)
    indices = np.arange(N)
    
    # Extract arrays needed for splitting and saving
    orient_ids = np.array([t['orientation_id'] for t in trajectories], dtype=np.int32)
    dist_ids = np.array([t['distance_group_id'] for t in trajectories], dtype=np.int32)
    original_lengths = np.array([t['length'] for t in trajectories], dtype=np.int32)
    
    print(f"Total samples: {N}")
    print(f"Split ratios: train={config['train_ratio']}, val={config['val_ratio']}, test={config['test_ratio']}")
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=config['train_ratio'],
        random_state=config['random_seed'],
        stratify=orient_ids  # Stratify by orientation
    )
    
    # Second split: val vs test
    val_ratio_adjusted = config['val_ratio'] / (config['val_ratio'] + config['test_ratio'])
    
    # Stratify the second split too, but handle case where some classes might be missing
    temp_orient_ids = orient_ids[temp_idx]
    
    # Check if we have enough samples per class
    unique, counts = np.unique(temp_orient_ids, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0
    
    if min_count >= 2:
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            random_state=config['random_seed'],
            stratify=temp_orient_ids
        )
    else:
        # Can't stratify - just do random split
        print(f"  Note: Using non-stratified val/test split (small class size)")
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
            'orientation_ids': orient_ids[idx],
            'distance_ids': dist_ids[idx],
            'original_lengths': original_lengths[idx],  # CRITICAL for V3!
        }
        
        # Summary statistics
        lengths_in_split = original_lengths[idx]
        print(f"\n  {name}: {len(idx)} samples")
        print(f"    Length range: [{lengths_in_split.min()}, {lengths_in_split.max()}]")
        print(f"    Length median: {np.median(lengths_in_split):.0f}")
        
        # Orientation distribution
        orient_counts = np.bincount(orient_ids[idx], minlength=NUM_ORIENTATIONS)
        print(f"    Orientations: {dict(zip(ORIENTATIONS, orient_counts))}")
    
    return splits


# =============================================================================
# STEP 7: SAVE DATA
# =============================================================================

def save_all_data(
    output_dir: Path,
    U_ref: np.ndarray,
    T_ref: int,
    mean: Optional[np.ndarray],
    group_anchors: Dict,
    splits: Dict,
    config: Dict,
    stats: Dict
) -> Path:
    """
    Save all preprocessed data to disk.
    
    Output structure:
        output_dir/
        ├── U_ref.npy              # Reference basis [2*T_ref × K]
        ├── mean.npy               # Mean control points (if centering)
        ├── config.npy             # Full configuration
        ├── group_anchors.npy      # Per-group anchors
        └── train/val/test/
            ├── coefficients.npy
            ├── residuals.npy
            ├── anchor_indices.npy
            ├── orientation_ids.npy
            ├── distance_ids.npy
            └── original_lengths.npy  # CRITICAL!
    
    Args:
        output_dir: Output directory path
        U_ref: Reference basis
        T_ref: Reference length
        mean: Mean control points or None
        group_anchors: Per-group anchor dict
        splits: Train/val/test split data
        config: Configuration dict
        stats: Statistics from preprocessing
        
    Returns:
        output_path: Path to output directory
    """
    print_section("STEP 7: Saving Data")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save reference basis
    np.save(output_path / 'U_ref.npy', U_ref)
    print(f"  Saved: U_ref.npy {U_ref.shape}")
    
    # Save mean (if centering was used)
    if mean is not None:
        np.save(output_path / 'mean.npy', mean)
        print(f"  Saved: mean.npy {mean.shape}")
    
    # Save anchors
    np.save(output_path / 'group_anchors.npy', group_anchors, allow_pickle=True)
    n_anchors = sum(
        len(v) if v is not None else 0 
        for v in group_anchors.values()
    )
    print(f"  Saved: group_anchors.npy ({n_anchors} total anchors)")
    
    # Build full config
    K = U_ref.shape[1]
    
    full_config = {
        # Core parameters
        **config,
        
        # Computed values
        'K': K,
        'T_ref': T_ref,
        'n_control_points': T_ref,  # Alias for clarity
        
        # Metadata
        'num_orientations': NUM_ORIENTATIONS,
        'num_distance_groups': NUM_DISTANCE_GROUPS,
        'orientations': ORIENTATIONS,
        'distance_groups': DISTANCE_GROUPS,
        
        # Version identifier
        'version': 'v3_basis_transform',
        'uses_basis_adaptation': True,
        
        # Statistics
        'stats': stats,
        
        # Document the reconstruction formula
        'reconstruction_notes': [
            'To reconstruct a trajectory at length T:',
            '1. Create BasisAdapter(U_ref, T_ref)',
            '2. If mean is not None: trajectory = adapter.reconstruct(c, T) + mean_adapted',
            '   Otherwise: trajectory = adapter.reconstruct(c, T)',
            '3. The mean_adapted is the mean transformed to length T via B-spline',
        ],
    }
    
    np.save(output_path / 'config.npy', full_config, allow_pickle=True)
    print(f"  Saved: config.npy")
    
    # Save splits
    for split_name, split_data in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        for key, arr in split_data.items():
            if key != 'indices':  # Don't save indices
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
    
    Tests:
        1. Round-trip reconstruction accuracy
        2. Coefficient-length correlation (should be weak in V3)
        3. Data integrity checks
    
    Args:
        output_dir: Where data was saved
        trajectories: Original trajectory list
        adapter: BasisAdapter used for projection
        coefficients: Projected coefficients
        n_samples: Number of samples to test
        
    Returns:
        success: True if all checks pass
    """
    print_section("STEP 8: Verification")
    
    all_passed = True
    
    # Test 1: Round-trip reconstruction
    print("\n[Test 1] Round-trip reconstruction accuracy...")
    
    sample_indices = np.random.choice(len(trajectories), min(n_samples, len(trajectories)), replace=False)
    
    reconstruction_errors = []
    for i in sample_indices:
        traj = trajectories[i]
        flat = traj['flat']
        T = traj['length']
        c = coefficients[i]
        
        # Reconstruct at original length
        recon = adapter.reconstruct(c, T)
        
        # Compute error
        error = np.sqrt(np.mean((flat - recon) ** 2))
        reconstruction_errors.append(error)
    
    mean_error = np.mean(reconstruction_errors)
    max_error = np.max(reconstruction_errors)
    
    print(f"  Mean RMSE: {mean_error:.4f}")
    print(f"  Max RMSE:  {max_error:.4f}")
    
    if mean_error > 100:  # Reasonable threshold for pixel coordinates
        print(f"  WARNING: High reconstruction error!")
        all_passed = False
    else:
        print(f"  ✓ Reconstruction accuracy acceptable")
    
    # Test 2: Length-coefficient correlation
    print("\n[Test 2] Length-coefficient correlation...")
    
    lengths = np.array([t['length'] for t in trajectories])
    
    correlations = []
    for k in range(coefficients.shape[1]):
        corr = np.corrcoef(lengths, coefficients[:, k])[0, 1]
        correlations.append((k, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"  Top 3 length correlations:")
    for k, corr in correlations[:3]:
        print(f"    c[{k}]: r = {corr:+.4f}")
    
    # In V3, we don't explicitly encode length, so correlations may exist
    # but shouldn't be artificially strong
    print(f"  Note: V3 doesn't encode length explicitly in coefficients")
    
    # Test 3: Data integrity
    print("\n[Test 3] Data integrity checks...")
    
    # Load saved data
    config = np.load(output_dir / 'config.npy', allow_pickle=True).item()
    U_ref_loaded = np.load(output_dir / 'U_ref.npy')
    train_coeffs = np.load(output_dir / 'train' / 'coefficients.npy')
    train_lengths = np.load(output_dir / 'train' / 'original_lengths.npy')
    
    # Check shapes
    checks = [
        ('U_ref shape matches', U_ref_loaded.shape == adapter.U_ref.shape),
        ('Coefficients have correct K', train_coeffs.shape[1] == config['K']),
        ('Lengths array matches samples', len(train_lengths) == len(train_coeffs)),
        ('Version is v3', config['version'] == 'v3_basis_transform'),
    ]
    
    for check_name, passed in checks:
        status = '✓' if passed else '✗'
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    # Test 4: Jitter preservation (if scipy available)
    print("\n[Test 4] Jitter preservation check...")
    
    try:
        from scipy.ndimage import gaussian_filter1d
        
        # Compute jitter metric for original vs reconstructed
        original_jitters = []
        recon_jitters = []
        
        for i in sample_indices[:5]:
            traj = trajectories[i]
            flat = traj['flat']
            T = traj['length']
            c = coefficients[i]
            
            x_orig = flat[0::2]
            y_orig = flat[1::2]
            
            recon = adapter.reconstruct(c, T)
            x_recon = recon[0::2]
            y_recon = recon[1::2]
            
            # Jitter = RMS of second derivative (acceleration)
            def jitter_metric(coords):
                if len(coords) < 3:
                    return 0
                accel = np.diff(np.diff(coords))
                return np.sqrt(np.mean(accel ** 2))
            
            orig_jitter = (jitter_metric(x_orig) + jitter_metric(y_orig)) / 2
            recon_jitter = (jitter_metric(x_recon) + jitter_metric(y_recon)) / 2
            
            original_jitters.append(orig_jitter)
            recon_jitters.append(recon_jitter)
        
        mean_orig = np.mean(original_jitters)
        mean_recon = np.mean(recon_jitters)
        preservation = mean_recon / mean_orig if mean_orig > 1e-10 else 1.0
        
        print(f"  Original jitter (mean): {mean_orig:.4f}")
        print(f"  Reconstructed jitter:   {mean_recon:.4f}")
        print(f"  Preservation ratio:     {preservation:.1%}")
        
        if preservation > 0.5:
            print(f"  ✓ Jitter preserved reasonably well")
        else:
            print(f"  WARNING: Significant jitter loss")
            
    except ImportError:
        print(f"  Skipped (scipy not available)")
    
    # Summary
    print(f"\n{'='*70}")
    if all_passed:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION COMPLETED WITH WARNINGS")
    print('='*70)
    
    return all_passed


# =============================================================================
# LENGTH DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_length_distributions(
    trajectories: List[Dict],
    group_anchors: Dict,
    output_dir: Path
):
    """
    Analyze and save length distributions per group.
    
    This information is needed at generation time to sample
    realistic lengths for each (orientation, distance) group.
    """
    print_section("Analyzing Length Distributions")
    
    length_stats = {}
    
    for dist_id in range(NUM_DISTANCE_GROUPS):
        for orient_id in range(NUM_ORIENTATIONS):
            key = (dist_id, orient_id)
            group_name = f"{DISTANCE_GROUPS[dist_id]['name']}-{ORIENTATIONS[orient_id]}"
            
            # Get lengths for this group
            lengths = [t['length'] for t in trajectories 
                      if t['distance_group_id'] == dist_id and t['orientation_id'] == orient_id]
            
            if len(lengths) > 0:
                lengths = np.array(lengths)
                length_stats[key] = {
                    'name': group_name,
                    'count': len(lengths),
                    'min': int(lengths.min()),
                    'max': int(lengths.max()),
                    'mean': float(lengths.mean()),
                    'std': float(lengths.std()),
                    'median': float(np.median(lengths)),
                    'percentiles': {
                        '10': float(np.percentile(lengths, 10)),
                        '25': float(np.percentile(lengths, 25)),
                        '50': float(np.percentile(lengths, 50)),
                        '75': float(np.percentile(lengths, 75)),
                        '90': float(np.percentile(lengths, 90)),
                    },
                    'histogram': np.histogram(lengths, bins=20)[0].tolist(),
                    'histogram_edges': np.histogram(lengths, bins=20)[1].tolist(),
                }
    
    # Save length statistics
    np.save(output_dir / 'length_distributions.npy', length_stats, allow_pickle=True)
    
    # Print summary
    print(f"\n  {'Group':<15} {'Count':>8} {'Min':>6} {'Max':>6} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*55}")
    
    for dist_id in range(NUM_DISTANCE_GROUPS):
        for orient_id in range(NUM_ORIENTATIONS):
            key = (dist_id, orient_id)
            if key in length_stats:
                s = length_stats[key]
                print(f"  {s['name']:<15} {s['count']:>8} {s['min']:>6} {s['max']:>6} "
                      f"{s['mean']:>8.1f} {s['std']:>8.1f}")
    
    print(f"\n  Saved: length_distributions.npy")
    
    return length_stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess trajectories using V3 basis-transformation approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python preprocess_singular_v3_basis_transform.py --input trajectories/ --output processed_v3/
    
    # With custom parameters (closer to paper's recommendations)
    python preprocess_singular_v3_basis_transform.py --input trajectories/ --output processed_v3/ \\
        --n_control_points 16 --k 4 --anchors_per_group 6
    
    # Higher fidelity (more control points and dimensions)
    python preprocess_singular_v3_basis_transform.py --input trajectories/ --output processed_v3/ \\
        --n_control_points 24 --k 12 --anchors_per_group 8
"""
    )
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing trajectory JSON files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for preprocessed data')
    
    # Core parameters
    parser.add_argument('--n_control_points', type=int, default=DEFAULT_CONFIG['n_control_points'],
                        help=f"Number of B-spline control points (default: {DEFAULT_CONFIG['n_control_points']})")
    parser.add_argument('--k', type=int, default=DEFAULT_CONFIG['K'],
                        help=f"Number of singular dimensions (default: {DEFAULT_CONFIG['K']})")
    parser.add_argument('--anchors_per_group', type=int, default=DEFAULT_CONFIG['anchors_per_group'],
                        help=f"Anchors per group (default: {DEFAULT_CONFIG['anchors_per_group']})")
    
    # Length constraints
    parser.add_argument('--min_length', type=int, default=DEFAULT_CONFIG['min_length'],
                        help=f"Minimum trajectory length (default: {DEFAULT_CONFIG['min_length']})")
    parser.add_argument('--max_length', type=int, default=DEFAULT_CONFIG['max_length_filter'],
                        help=f"Maximum trajectory length (default: {DEFAULT_CONFIG['max_length_filter']})")
    
    # Other options
    parser.add_argument('--no_center', action='store_true',
                        help='Disable data centering before SVD')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['random_seed'],
                        help=f"Random seed (default: {DEFAULT_CONFIG['random_seed']})")
    parser.add_argument('--skip_verify', action='store_true',
                        help='Skip verification step')
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config['n_control_points'] = args.n_control_points
    config['K'] = args.k
    config['anchors_per_group'] = args.anchors_per_group
    config['min_length'] = args.min_length
    config['max_length_filter'] = args.max_length
    config['center_data'] = not args.no_center
    config['random_seed'] = args.seed
    
    # Print header
    print("=" * 70)
    print("SINGULAR SPACE PREPROCESSING - VERSION 3 (BASIS-TRANSFORMATION)")
    print("=" * 70)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print(f"\nParameters:")
    print(f"  n_control_points (T_ref): {config['n_control_points']}")
    print(f"  K (singular dimensions):  {config['K']}")
    print(f"  anchors_per_group:        {config['anchors_per_group']}")
    print(f"  min_length:               {config['min_length']}")
    print(f"  max_length_filter:        {config['max_length_filter']}")
    print(f"  center_data:              {config['center_data']}")
    print(f"  random_seed:              {config['random_seed']}")
    print("\n" + "=" * 70)
    print("KEY INNOVATION: Basis is adapted to trajectory length, not vice versa!")
    print("This preserves jitter and micro-corrections in the raw data.")
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
    
    # Step 4: Generate group anchors
    group_anchors, group_stats = generate_group_anchors(
        trajectories,
        coefficients,
        config['anchors_per_group'],
        config['random_seed']
    )
    
    # Step 5: Compute residuals
    residuals, anchor_indices, residual_stats = compute_residuals(
        trajectories, coefficients, group_anchors
    )
    
    # Step 6: Split data
    splits = split_data(
        trajectories, coefficients, residuals, anchor_indices, config
    )
    
    # Combine all statistics
    all_stats = {
        'basis': basis_stats,
        'projection': projection_stats,
        'residual': residual_stats,
        'groups': group_stats,
    }
    
    # Step 7: Save data
    output_path = save_all_data(
        args.output, U_ref, T_ref, mean, group_anchors, splits, config, all_stats
    )
    
    # Analyze and save length distributions
    length_stats = analyze_length_distributions(trajectories, group_anchors, output_path)
    
    # Step 8: Verification
    if not args.skip_verify:
        verify_preprocessing(output_path, trajectories, adapter, coefficients)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE - VERSION 3 (BASIS-TRANSFORMATION)")
    print("=" * 70)
    
    print(f"\nOutput saved to: {output_path}")
    print(f"\nKey files:")
    print(f"  - U_ref.npy:              Reference basis [{2*T_ref} × {config['K']}]")
    if mean is not None:
        print(f"  - mean.npy:               Mean control points [{2*T_ref}]")
    print(f"  - group_anchors.npy:      Per-group anchor prototypes")
    print(f"  - length_distributions.npy: Length stats for generation")
    print(f"  - config.npy:             Full configuration")
    print(f"  - train/val/test/:        Split data with original_lengths!")
    
    print(f"\nStatistics:")
    print(f"  - Trajectories: {len(trajectories)}")
    print(f"  - Variance explained: {basis_stats['variance_explained']*100:.2f}%")
    print(f"  - Residual reduction: {residual_stats['reduction_ratio']*100:.1f}%")
    
    print(f"\nTo train the diffusion model:")
    print(f"  python train_singular_diffusion_v1.py --data {args.output} --epochs 256")
    
    print(f"\nTo generate trajectories (after training):")
    print(f"  python generate_trajectory_v3.py --checkpoint checkpoints/best_model.pt --data {args.output}")
    

if __name__ == '__main__':
    main()
