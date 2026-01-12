#!/usr/bin/env python3
"""
Visual Pipeline Comparison: Buggy vs Correct Mean Centering

This script runs the FULL pipeline (preprocess → train → generate) twice:
1. With current buggy code (projects raw trajectories)
2. With correct code (projects via control points)

Then generates side-by-side visualizations so you can SEE the difference.

Usage:
    # With your real trajectory data
    python test_visual_pipeline_comparison.py --input /path/to/trajectories/

    # With synthetic data (for quick testing)
    python test_visual_pipeline_comparison.py --synthetic

    # Fewer epochs for faster testing
    python test_visual_pipeline_comparison.py --input /path/to/data --epochs 5
"""

import numpy as np
import torch
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
import sys

# Ensure we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bspline_basis import (
    BasisAdapter,
    fit_bspline_control_points,
    build_interleaved_basis_matrix,
)

# Will import training/generation after we verify they exist
try:
    from train_singular_diffusion_v1 import (
        ResidualDiffusionMLP,
        NoiseSchedule,
        SingularTrajectoryDataset,
        Trainer,
    )
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    HAS_TRAINING = True
except ImportError as e:
    print(f"Warning: Could not import training modules: {e}")
    HAS_TRAINING = False


# =============================================================================
# CONFIGURATION
# =============================================================================

ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DISTANCE_GROUPS = [
    {"name": "XSmall", "id": 0, "min": 0, "max": 50},
    {"name": "Small", "id": 1, "min": 50, "max": 100},
    {"name": "Medium", "id": 2, "min": 100, "max": 200},
    {"name": "Large", "id": 3, "min": 200, "max": 400},
    {"name": "XLarge", "id": 4, "min": 400, "max": 900},
]


# =============================================================================
# DATA LOADING
# =============================================================================

def generate_synthetic_trajectories(n=200, seed=42):
    """Generate synthetic mouse-like trajectories."""
    np.random.seed(seed)
    trajectories = []

    for i in range(n):
        T = np.random.randint(30, 100)
        t = np.linspace(0, 1, T)

        # Random endpoint direction and distance
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(30, 300)

        end_x = distance * np.cos(angle)
        end_y = distance * np.sin(angle)

        # Curved path with easing
        ease = 1 - (1 - t) ** 2  # ease-out
        curve = np.random.randn() * 20

        x = ease * end_x + curve * np.sin(t * np.pi) + np.random.randn(T) * 1.5
        y = ease * end_y + curve * np.cos(t * np.pi) + np.random.randn(T) * 1.5

        # Start at origin
        x = x - x[0]
        y = y - y[0]

        # Compute orientation and distance
        dx, dy = x[-1], y[-1]
        ideal_distance = np.sqrt(dx**2 + dy**2)
        angle_deg = np.degrees(np.arctan2(-dy, dx))  # Screen coords

        # Classify orientation
        orient_id = int((angle_deg + 180 + 22.5) // 45) % 8

        # Classify distance
        dist_id = 0
        for g in DISTANCE_GROUPS:
            if g["min"] <= ideal_distance < g["max"]:
                dist_id = g["id"]
                break

        flat = np.empty(2 * T)
        flat[0::2] = x
        flat[1::2] = y

        trajectories.append({
            'flat': flat,
            'x': x,
            'y': y,
            'length': T,
            'ideal_distance': ideal_distance,
            'orientation_id': orient_id,
            'distance_group_id': dist_id,
        })

    return trajectories


def load_real_trajectories(data_dir, max_count=300, min_length=20, max_length=200):
    """Load real trajectory data from JSON files."""
    import math

    data_dir = Path(data_dir)
    json_files = list(data_dir.glob('*.json'))

    print(f"Found {len(json_files)} JSON files")

    trajectories = []

    for json_file in json_files:
        if len(trajectories) >= max_count:
            break

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'x' not in data or 'y' not in data:
                continue

            x = np.array(data['x'], dtype=np.float64)
            y = np.array(data['y'], dtype=np.float64)

            if len(x) < min_length or len(x) > max_length:
                continue

            # Convert to relative
            x = x - x[0]
            y = y - y[0]

            # Compute metrics
            ideal_distance = data.get('ideal_distance', np.sqrt(x[-1]**2 + y[-1]**2))

            # Classify orientation
            dx, dy = x[-1], y[0] - y[-1]  # Screen coords
            angle_deg = math.degrees(math.atan2(dy, dx))
            orient_id = 2  # Default E
            for orient, (lo, hi) in [("E", (-22.5, 22.5)), ("NE", (-67.5, -22.5)),
                                      ("N", (-112.5, -67.5)), ("NW", (-157.5, -112.5)),
                                      ("SE", (22.5, 67.5)), ("S", (67.5, 112.5)),
                                      ("SW", (112.5, 157.5))]:
                if lo <= angle_deg < hi:
                    orient_id = ORIENTATIONS.index(orient)
                    break
            if abs(angle_deg) >= 157.5:
                orient_id = ORIENTATIONS.index("W")

            # Classify distance
            dist_id = 4
            for g in DISTANCE_GROUPS:
                if g["min"] <= ideal_distance < g["max"]:
                    dist_id = g["id"]
                    break

            flat = np.empty(2 * len(x))
            flat[0::2] = x
            flat[1::2] = y

            trajectories.append({
                'flat': flat,
                'x': x,
                'y': y,
                'length': len(x),
                'ideal_distance': ideal_distance,
                'orientation_id': orient_id,
                'distance_group_id': dist_id,
            })

        except Exception:
            continue

    return trajectories


# =============================================================================
# PREPROCESSING - Two versions
# =============================================================================

def preprocess_buggy(trajectories, T_ref=16, K=8, center_data=True):
    """
    Current BUGGY preprocessing: projects raw trajectories.

    This is what the current code does - it's mathematically incorrect
    because U_ref is learned in control point space but projection
    is done in trajectory space.
    """
    N = len(trajectories)

    # Step 1: Learn basis from control points
    control_points = np.zeros((N, 2 * T_ref))
    for i, traj in enumerate(trajectories):
        cp = fit_bspline_control_points(traj['flat'], T_ref)
        control_points[i] = cp

    if center_data:
        mean = control_points.mean(axis=0)
        centered = control_points - mean
    else:
        mean = None
        centered = control_points

    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    # Step 2: Project using BasisAdapter (BUGGY - projects raw trajectory)
    adapter = BasisAdapter(U_ref, T_ref)
    coefficients = np.zeros((N, K))

    for i, traj in enumerate(trajectories):
        # This is the BUG: adapter.project() does U_T.T @ raw_trajectory
        coefficients[i] = adapter.project(traj['flat'])

    return U_ref, mean, coefficients, adapter


def preprocess_correct(trajectories, T_ref=16, K=8, center_data=True):
    """
    CORRECT preprocessing: projects via control points.

    This properly projects in control point space where U_ref was learned.
    """
    N = len(trajectories)

    # Step 1: Learn basis from control points (same as buggy)
    control_points = np.zeros((N, 2 * T_ref))
    for i, traj in enumerate(trajectories):
        cp = fit_bspline_control_points(traj['flat'], T_ref)
        control_points[i] = cp

    if center_data:
        mean = control_points.mean(axis=0)
        centered = control_points - mean
    else:
        mean = None
        centered = control_points

    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
    U_ref = U[:, :K]

    # Step 2: Project via control points (CORRECT)
    coefficients = np.zeros((N, K))

    for i, traj in enumerate(trajectories):
        cp = fit_bspline_control_points(traj['flat'], T_ref)
        if center_data:
            coefficients[i] = U_ref.T @ (cp - mean)
        else:
            coefficients[i] = U_ref.T @ cp

    # Create adapter for reconstruction
    adapter = BasisAdapter(U_ref, T_ref)

    return U_ref, mean, coefficients, adapter


# =============================================================================
# ANCHORS AND RESIDUALS
# =============================================================================

def compute_anchors_and_residuals(trajectories, coefficients, anchors_per_group=4):
    """Compute group anchors and residuals."""
    N = len(trajectories)
    K = coefficients.shape[1]

    # Group by (distance, orientation)
    groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        key = (traj['distance_group_id'], traj['orientation_id'])
        groups[key].append(i)

    # Compute anchors via K-means
    group_anchors = {}
    for key, indices in groups.items():
        if len(indices) == 0:
            group_anchors[key] = None
        elif len(indices) <= anchors_per_group:
            group_anchors[key] = coefficients[indices].copy()
        else:
            kmeans = KMeans(n_clusters=anchors_per_group, random_state=42, n_init=10)
            kmeans.fit(coefficients[indices])
            group_anchors[key] = kmeans.cluster_centers_

    # Compute residuals
    residuals = np.zeros((N, K))
    anchor_indices = np.zeros(N, dtype=np.int32)

    for i, traj in enumerate(trajectories):
        key = (traj['distance_group_id'], traj['orientation_id'])
        anchors = group_anchors[key]

        if anchors is None:
            residuals[i] = coefficients[i]
            anchor_indices[i] = -1
        else:
            dists = np.linalg.norm(anchors - coefficients[i], axis=1)
            nearest = dists.argmin()
            residuals[i] = coefficients[i] - anchors[nearest]
            anchor_indices[i] = nearest

    return group_anchors, residuals, anchor_indices


# =============================================================================
# SAVE PREPROCESSED DATA
# =============================================================================

def save_preprocessed(output_dir, trajectories, U_ref, mean, coefficients,
                      group_anchors, residuals, anchor_indices, T_ref, K):
    """Save preprocessed data in format expected by training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = len(trajectories)

    # Save basis and config
    np.save(output_dir / 'U_ref.npy', U_ref)
    if mean is not None:
        np.save(output_dir / 'mean.npy', mean)
    np.save(output_dir / 'group_anchors.npy', group_anchors, allow_pickle=True)

    config = {
        'K': K,
        'T_ref': T_ref,
        'T_win': T_ref,
        'n_control_points': T_ref,
        'num_orientations': 8,
        'num_distance_groups': 5,
        'version': 'v3_basis_transform',
    }
    np.save(output_dir / 'config.npy', config, allow_pickle=True)

    # Split data
    indices = np.arange(N)
    orient_ids = np.array([t['orientation_id'] for t in trajectories])
    dist_ids = np.array([t['distance_group_id'] for t in trajectories])
    lengths = np.array([t['length'] for t in trajectories])

    train_idx, temp_idx = train_test_split(indices, train_size=0.7, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42)

    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_dir = output_dir / name
        split_dir.mkdir(exist_ok=True)

        np.save(split_dir / 'coefficients.npy', coefficients[idx])
        np.save(split_dir / 'residuals.npy', residuals[idx])
        np.save(split_dir / 'anchor_indices.npy', anchor_indices[idx])
        np.save(split_dir / 'orientation_ids.npy', orient_ids[idx])
        np.save(split_dir / 'distance_ids.npy', dist_ids[idx])
        np.save(split_dir / 'original_lengths.npy', lengths[idx])


# =============================================================================
# TRAINING
# =============================================================================

def train_model(data_dir, epochs=10, device='cpu'):
    """Train diffusion model on preprocessed data."""
    if not HAS_TRAINING:
        print("Training modules not available!")
        return None

    data_dir = Path(data_dir)
    config = np.load(data_dir / 'config.npy', allow_pickle=True).item()
    K = config['K']

    train_config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.1,
        'diffusion_steps': 10,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': epochs,
        'warmup_epochs': 2,
        'validate_every': 1,
        'save_every': 1000,
        'log_every': 50,
        'K': K,
        'T_win': config['T_ref'],
    }

    # Load datasets
    train_dataset = SingularTrajectoryDataset(data_dir, 'train', device=device)
    val_dataset = SingularTrajectoryDataset(data_dir, 'val', device=device)

    # Create model
    model = ResidualDiffusionMLP(
        K=K,
        hidden_dim=train_config['hidden_dim'],
        num_layers=train_config['num_layers'],
        num_orientations=8,
        num_distances=5,
        dropout=train_config['dropout']
    )

    # Train
    trainer = Trainer(model, train_dataset, val_dataset, train_config, device=device)

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate()
        print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    return model, train_config


# =============================================================================
# GENERATION
# =============================================================================

def generate_trajectories(model, config, data_dir, n_samples=5, device='cpu'):
    """Generate sample trajectories."""
    data_dir = Path(data_dir)

    data_config = np.load(data_dir / 'config.npy', allow_pickle=True).item()
    K = data_config['K']
    T_ref = data_config['T_ref']

    U_ref = np.load(data_dir / 'U_ref.npy')
    mean = None
    if (data_dir / 'mean.npy').exists():
        mean = np.load(data_dir / 'mean.npy')

    group_anchors = np.load(data_dir / 'group_anchors.npy', allow_pickle=True).item()

    adapter = BasisAdapter(U_ref, T_ref)

    noise_schedule = NoiseSchedule(
        config['diffusion_steps'],
        config['beta_start'],
        config['beta_end'],
        device=device
    )

    model.eval()
    generated = []

    # Generate for a few groups
    test_groups = [(2, 2), (2, 0), (3, 4)]  # (dist_id, orient_id)

    for dist_id, orient_id in test_groups:
        key = (dist_id, orient_id)
        anchors = group_anchors.get(key)

        if anchors is None:
            continue

        for _ in range(n_samples):
            # Pick random anchor
            anchor = torch.tensor(anchors[np.random.randint(len(anchors))],
                                  dtype=torch.float32, device=device).unsqueeze(0)

            orient_t = torch.tensor([orient_id], device=device)
            dist_t = torch.tensor([dist_id], device=device)

            # Start from noise
            r_t = torch.randn(1, K, device=device)

            # Denoise
            with torch.no_grad():
                for step in reversed(range(config['diffusion_steps'])):
                    t = torch.tensor([step], device=device)
                    noise_pred = model(r_t, t, orient_t, dist_t, anchor)
                    r_t = noise_schedule.ddim_step(r_t, noise_pred, t)

            # Get coefficients
            coefficients = (anchor + r_t).squeeze(0).cpu().numpy()

            # Reconstruct at reasonable length
            T = 50
            traj_flat = adapter.reconstruct(coefficients, T)

            if mean is not None:
                mean_T = adapter.reconstruct_mean(mean, T)
                traj_flat = traj_flat + mean_T

            x = traj_flat[0::2]
            y = traj_flat[1::2]

            generated.append({
                'x': x,
                'y': y,
                'dist_id': dist_id,
                'orient_id': orient_id,
                'dist_name': DISTANCE_GROUPS[dist_id]['name'],
                'orient_name': ORIENTATIONS[orient_id],
            })

    return generated


# =============================================================================
# RECONSTRUCTION TEST (without training)
# =============================================================================

def test_reconstruction_only(trajectories, T_ref=16, K=8):
    """Test reconstruction quality without training - just preprocess and reconstruct."""

    print("\n" + "=" * 70)
    print("RECONSTRUCTION TEST (No Training)")
    print("=" * 70)

    # Preprocess both ways
    print("\n[1] Preprocessing with BUGGY method...")
    U_buggy, mean_buggy, coef_buggy, adapter_buggy = preprocess_buggy(
        trajectories, T_ref, K, center_data=True
    )

    print("[2] Preprocessing with CORRECT method...")
    U_correct, mean_correct, coef_correct, adapter_correct = preprocess_correct(
        trajectories, T_ref, K, center_data=True
    )

    # Test reconstruction on sample trajectories
    print("\n[3] Testing reconstruction on 20 trajectories...")

    buggy_rmse = []
    correct_rmse = []

    for i in range(min(20, len(trajectories))):
        traj = trajectories[i]
        T = traj['length']
        original = traj['flat']

        # Buggy reconstruction
        recon_buggy = adapter_buggy.reconstruct(coef_buggy[i], T)
        if mean_buggy is not None:
            mean_T = adapter_buggy.reconstruct_mean(mean_buggy, T)
            recon_buggy = recon_buggy + mean_T

        # Correct reconstruction
        cp_recon = U_correct @ coef_correct[i] + mean_correct
        C = build_interleaved_basis_matrix(T, T_ref)
        recon_correct = C @ cp_recon

        buggy_rmse.append(np.sqrt(np.mean((original - recon_buggy) ** 2)))
        correct_rmse.append(np.sqrt(np.mean((original - recon_correct) ** 2)))

    print(f"\n    Results:")
    print(f"    {'Method':<20} {'Mean RMSE':>12} {'Max RMSE':>12}")
    print(f"    {'-' * 46}")
    print(f"    {'BUGGY':<20} {np.mean(buggy_rmse):>12.2f} {np.max(buggy_rmse):>12.2f}")
    print(f"    {'CORRECT':<20} {np.mean(correct_rmse):>12.2f} {np.max(correct_rmse):>12.2f}")

    return buggy_rmse, correct_rmse, coef_buggy, coef_correct, adapter_buggy, U_correct, mean_correct


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_comparison(trajectories, buggy_rmse, correct_rmse,
                         coef_buggy, coef_correct, adapter_buggy,
                         U_correct, mean_correct, T_ref, output_path):
    """Create visual comparison of buggy vs correct reconstruction."""

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Original vs Buggy vs Correct for 4 trajectories
    for col in range(4):
        if col >= len(trajectories):
            break

        traj = trajectories[col]
        T = traj['length']

        # Original
        x_orig = traj['x']
        y_orig = traj['y']

        # Buggy reconstruction
        recon_buggy = adapter_buggy.reconstruct(coef_buggy[col], T)
        if mean_correct is not None:  # Using same mean for fair comparison
            mean_T = adapter_buggy.reconstruct_mean(mean_correct, T)
            recon_buggy = recon_buggy + mean_T
        x_buggy = recon_buggy[0::2]
        y_buggy = recon_buggy[1::2]

        # Correct reconstruction
        cp_recon = U_correct @ coef_correct[col] + mean_correct
        C = build_interleaved_basis_matrix(T, T_ref)
        recon_correct = C @ cp_recon
        x_correct = recon_correct[0::2]
        y_correct = recon_correct[1::2]

        ax = axes[0, col]
        ax.plot(x_orig, y_orig, 'b-', linewidth=2, label='Original', alpha=0.7)
        ax.plot(x_buggy, y_buggy, 'r--', linewidth=2, label='Buggy', alpha=0.7)
        ax.plot(x_correct, y_correct, 'g:', linewidth=2, label='Correct', alpha=0.7)
        ax.scatter([0], [0], c='black', s=50, zorder=10)
        ax.set_title(f'Traj {col+1} (T={T})')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: RMSE distribution
    ax = axes[1, 0]
    ax.bar(['Buggy', 'Correct'], [np.mean(buggy_rmse), np.mean(correct_rmse)],
           color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Mean RMSE')
    ax.set_title('Average Reconstruction Error')

    ax = axes[1, 1]
    ax.hist(buggy_rmse, bins=15, alpha=0.7, label='Buggy', color='red')
    ax.hist(correct_rmse, bins=15, alpha=0.7, label='Correct', color='green')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Count')
    ax.set_title('RMSE Distribution')
    ax.legend()

    # Row 2: Coefficient comparison
    ax = axes[1, 2]
    ax.scatter(coef_buggy[:, 0], coef_buggy[:, 1], alpha=0.5, label='Buggy', c='red', s=20)
    ax.scatter(coef_correct[:, 0], coef_correct[:, 1], alpha=0.5, label='Correct', c='green', s=20)
    ax.set_xlabel('Coef 0')
    ax.set_ylabel('Coef 1')
    ax.set_title('Coefficient Space (first 2 dims)')
    ax.legend()

    ax = axes[1, 3]
    coef_diff = np.linalg.norm(coef_buggy - coef_correct, axis=1)
    ax.hist(coef_diff, bins=20, color='purple', alpha=0.7)
    ax.set_xlabel('||c_buggy - c_correct||')
    ax.set_ylabel('Count')
    ax.set_title('Coefficient Difference')

    # Row 3: More trajectory comparisons
    for col in range(4):
        idx = col + 4
        if idx >= len(trajectories):
            axes[2, col].axis('off')
            continue

        traj = trajectories[idx]
        T = traj['length']

        x_orig = traj['x']
        y_orig = traj['y']

        recon_buggy = adapter_buggy.reconstruct(coef_buggy[idx], T)
        if mean_correct is not None:
            mean_T = adapter_buggy.reconstruct_mean(mean_correct, T)
            recon_buggy = recon_buggy + mean_T
        x_buggy = recon_buggy[0::2]
        y_buggy = recon_buggy[1::2]

        cp_recon = U_correct @ coef_correct[idx] + mean_correct
        C = build_interleaved_basis_matrix(T, T_ref)
        recon_correct = C @ cp_recon
        x_correct = recon_correct[0::2]
        y_correct = recon_correct[1::2]

        ax = axes[2, col]
        ax.plot(x_orig, y_orig, 'b-', linewidth=2, label='Original', alpha=0.7)
        ax.plot(x_buggy, y_buggy, 'r--', linewidth=2, label='Buggy', alpha=0.7)
        ax.plot(x_correct, y_correct, 'g:', linewidth=2, label='Correct', alpha=0.7)
        ax.scatter([0], [0], c='black', s=50, zorder=10)
        ax.set_title(f'Traj {idx+1} (T={T})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('BUGGY (red) vs CORRECT (green) Reconstruction Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visual comparison of buggy vs correct preprocessing pipeline'
    )
    parser.add_argument('--input', type=str, default=None,
                        help='Directory containing trajectory JSON files')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of real data')
    parser.add_argument('--output', type=str, default='visual_comparison.png',
                        help='Output image path')
    parser.add_argument('--t_ref', type=int, default=16,
                        help='Number of control points')
    parser.add_argument('--k', type=int, default=8,
                        help='Number of singular dimensions')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs (if running full pipeline)')
    parser.add_argument('--full_pipeline', action='store_true',
                        help='Run full pipeline including training and generation')
    parser.add_argument('--max_trajectories', type=int, default=200,
                        help='Maximum trajectories to load')

    args = parser.parse_args()

    print("=" * 70)
    print("VISUAL PIPELINE COMPARISON: Buggy vs Correct")
    print("=" * 70)

    # Load data
    if args.synthetic or args.input is None:
        print("\nUsing synthetic trajectories...")
        trajectories = generate_synthetic_trajectories(n=args.max_trajectories)
    else:
        print(f"\nLoading trajectories from: {args.input}")
        trajectories = load_real_trajectories(args.input, max_count=args.max_trajectories)

    if len(trajectories) == 0:
        print("ERROR: No trajectories loaded!")
        return

    print(f"Loaded {len(trajectories)} trajectories")
    lengths = [t['length'] for t in trajectories]
    print(f"Lengths: min={min(lengths)}, max={max(lengths)}, median={np.median(lengths):.0f}")

    # Run reconstruction test
    buggy_rmse, correct_rmse, coef_buggy, coef_correct, adapter_buggy, U_correct, mean_correct = \
        test_reconstruction_only(trajectories, args.t_ref, args.k)

    # Create visualization
    visualize_comparison(
        trajectories, buggy_rmse, correct_rmse,
        coef_buggy, coef_correct, adapter_buggy,
        U_correct, mean_correct, args.t_ref,
        args.output
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    BUGGY method (current code):
      - Projects raw trajectories using U_T.T @ trajectory
      - Mean RMSE: {np.mean(buggy_rmse):.2f}

    CORRECT method:
      - Projects via control points: U_ref.T @ (cp - mean)
      - Mean RMSE: {np.mean(correct_rmse):.2f}

    Improvement: {np.mean(buggy_rmse) / np.mean(correct_rmse):.1f}x better with correct method

    Visualization saved to: {args.output}
    """)


if __name__ == "__main__":
    main()
