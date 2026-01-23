#!/usr/bin/env python3
"""
Trajectory Generation - VERSION 5 (Anticipatory)

V5 INNOVATION: Dual Conditioning (AB + BC)
==========================================
Generates trajectories that capture ANTICIPATORY MOVEMENT - how the path
from A→B is shaped by knowing where C (the next target) is located.

Generation Process:
    1. Given (orient_AB, dist_AB, orient_BC, dist_BC), compute turn_category
    2. Select anchor from turn_category_anchors[turn_cat]
    3. Sample target length from AB segment distribution
    4. Initialize residual from Gaussian noise
    5. Denoise through M diffusion steps (DDIM) with dual conditioning
    6. DENORMALIZE residual: r = r_normalized * std + mean
    7. Add denormalized residual to anchor: c = anchor + r
    8. Reconstruct: trajectory = U_ref @ c

Turn Categories (7):
    0: straight      (-22.5° to 22.5°)
    1: slight_right  (22.5° to 67.5°)
    2: hard_right    (67.5° to 135°)
    3: reverse_right (135° to 180°)
    4: slight_left   (-67.5° to -22.5°)
    5: hard_left     (-135° to -67.5°)
    6: reverse_left  (-180° to -135°)

Usage:
    # Generate for all turn categories
    python generate_trajectory_v5.py --checkpoint checkpoints_v5/best_model.pt \\
        --data processed_v5/ --visualize

    # Generate specific segment pair
    python generate_trajectory_v5.py --checkpoint checkpoints_v5/best_model.pt \\
        --data processed_v5/ --orient_AB E --dist_AB Medium \\
        --orient_BC N --dist_BC Small --visualize
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from typing import Dict, Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_singular_diffusion_v5 import ResidualDiffusionMLP_V5, NoiseSchedule
from bspline_basis import BasisAdapter, build_interleaved_basis_matrix


# =============================================================================
# CONFIGURATION
# =============================================================================

ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DISTANCE_GROUPS = ["XSmall", "Small", "Medium", "Large", "XLarge"]

TURN_CATEGORIES = [
    {"name": "straight",      "id": 0, "min": -22.5,  "max": 22.5,   "color": "#2ca02c"},
    {"name": "slight_right",  "id": 1, "min": 22.5,   "max": 67.5,   "color": "#ff7f0e"},
    {"name": "hard_right",    "id": 2, "min": 67.5,   "max": 135.0,  "color": "#d62728"},
    {"name": "reverse_right", "id": 3, "min": 135.0,  "max": 180.0,  "color": "#9467bd"},
    {"name": "slight_left",   "id": 4, "min": -67.5,  "max": -22.5,  "color": "#1f77b4"},
    {"name": "hard_left",     "id": 5, "min": -135.0, "max": -67.5,  "color": "#17becf"},
    {"name": "reverse_left",  "id": 6, "min": -180.0, "max": -135.0, "color": "#e377c2"},
]

TURN_CATEGORY_NAMES = [tc["name"] for tc in TURN_CATEGORIES]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def orientation_to_angle(orient_id: int) -> float:
    """Convert orientation ID (0-7) to angle in degrees."""
    # N=0°, NE=45°, E=90°, SE=135°, S=180°, SW=225°, W=270°, NW=315°
    return orient_id * 45.0


def compute_turn_angle(orient_AB: int, orient_BC: int) -> float:
    """
    Compute turn angle from AB orientation to BC orientation.

    Positive = right turn, Negative = left turn.
    Returns angle in range [-180, 180].
    """
    angle_AB = orientation_to_angle(orient_AB)
    angle_BC = orientation_to_angle(orient_BC)

    # Turn angle: how much we rotate from AB direction to BC direction
    turn = angle_BC - angle_AB

    # Normalize to [-180, 180]
    while turn > 180:
        turn -= 360
    while turn <= -180:
        turn += 360

    return turn


def turn_angle_to_category(turn_angle: float) -> int:
    """Map turn angle to category ID (0-6)."""
    for tc in TURN_CATEGORIES:
        if tc["min"] <= turn_angle < tc["max"]:
            return tc["id"]
        # Handle edge case for exactly 180°
        if tc["name"] == "reverse_right" and turn_angle == 180.0:
            return tc["id"]

    # Fallback (shouldn't happen)
    return 0


def compute_turn_category(orient_AB: int, orient_BC: int) -> int:
    """Compute turn category from AB and BC orientations."""
    turn_angle = compute_turn_angle(orient_AB, orient_BC)
    return turn_angle_to_category(turn_angle)


def get_representative_orientations(turn_cat: int) -> Tuple[int, int]:
    """Get representative (orient_AB, orient_BC) pair for a turn category."""
    # Start with East (orient_AB = 2)
    orient_AB = 2

    # Map turn category to BC orientation
    # Turn category angle ranges determine the orient_BC
    if turn_cat == 0:    # straight: continue East
        orient_BC = 2
    elif turn_cat == 1:  # slight_right: turn to SE
        orient_BC = 3
    elif turn_cat == 2:  # hard_right: turn to S
        orient_BC = 4
    elif turn_cat == 3:  # reverse_right: turn to SW
        orient_BC = 5
    elif turn_cat == 4:  # slight_left: turn to NE
        orient_BC = 1
    elif turn_cat == 5:  # hard_left: turn to N
        orient_BC = 0
    elif turn_cat == 6:  # reverse_left: turn to NW
        orient_BC = 7
    else:
        orient_BC = 2

    return orient_AB, orient_BC


# =============================================================================
# V5 GENERATOR CLASS
# =============================================================================

class TrajectoryGeneratorV5:
    """
    Generate trajectories using V5 anticipatory diffusion model.

    Key innovation: Dual conditioning on AB (current) and BC (next) segments,
    capturing anticipatory movement patterns.
    """

    def __init__(self, checkpoint_path: str, data_dir: str, device: str = 'cuda'):
        self.device = device
        self.data_dir = Path(data_dir)

        # Load data config
        self.config = np.load(self.data_dir / 'config.npy', allow_pickle=True).item()
        self.K = self.config['K']
        self.T_ref = self.config.get('T_ref', self.config.get('n_control_points', 64))
        self.version = self.config.get('version', 'unknown')

        # Verify V5 format
        if 'v5' not in self.version.lower():
            print(f"  WARNING: Data appears to be {self.version}, not V5")
            print(f"           V5 generation may not work correctly")

        # Load reference basis
        if (self.data_dir / 'U_ref.npy').exists():
            self.U_ref = np.load(self.data_dir / 'U_ref.npy')
            print(f"  Loaded U_ref.npy: {self.U_ref.shape}")
        elif (self.data_dir / 'U_k.npy').exists():
            self.U_ref = np.load(self.data_dir / 'U_k.npy')
            print(f"  Loaded U_k.npy (fallback): {self.U_ref.shape}")
        else:
            raise FileNotFoundError("No basis file found (U_ref.npy or U_k.npy)")

        # Load mean if available
        self.mean = None
        if (self.data_dir / 'mean.npy').exists():
            self.mean = np.load(self.data_dir / 'mean.npy')
            print(f"  Loaded mean.npy: {self.mean.shape}")

        # Load residual normalization parameters
        self.residual_mean = None
        self.residual_std = None
        if (self.data_dir / 'residual_mean.npy').exists():
            self.residual_mean = np.load(self.data_dir / 'residual_mean.npy')
            self.residual_std = np.load(self.data_dir / 'residual_std.npy')
            print(f"  Loaded residual normalization: mean {self.residual_mean.shape}, std {self.residual_std.shape}")

        # Create basis adapter
        self.adapter = BasisAdapter(self.U_ref, self.T_ref)
        print(f"  Created BasisAdapter (T_ref={self.T_ref}, K={self.K})")

        # Load length distributions (keyed by AB segment)
        self.length_distributions = None
        if (self.data_dir / 'length_distributions.npy').exists():
            self.length_distributions = np.load(
                self.data_dir / 'length_distributions.npy',
                allow_pickle=True
            ).item()
            print(f"  Loaded length distributions for {len(self.length_distributions)} groups")

        # V5: Load turn category anchors (not group anchors)
        self.turn_anchors = np.load(
            self.data_dir / 'turn_category_anchors.npy',
            allow_pickle=True
        ).item()
        print(f"  Loaded turn_category_anchors for {len(self.turn_anchors)} categories")

        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint['config']

        self.model = ResidualDiffusionMLP_V5(
            K=self.K,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_orientations=self.config.get('num_orientations', 8),
            num_distances=self.config.get('num_distance_groups', 5),
            num_turn_categories=self.config.get('num_turn_categories', 7),
            dropout=0.0  # No dropout at inference
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Noise schedule for DDIM sampling
        self.noise_schedule = NoiseSchedule(
            model_config['diffusion_steps'],
            model_config['beta_start'],
            model_config['beta_end'],
            device=device
        )
        self.num_steps = model_config['diffusion_steps']

        print(f"\nLoaded V5 model from {checkpoint_path}")
        print(f"  K={self.K}, T_ref={self.T_ref}, diffusion_steps={self.num_steps}")
        print(f"  Version: {self.version}")
        print(f"  Mode: ANTICIPATORY (dual AB+BC conditioning)")

    def get_anchor(self, turn_cat: int, anchor_idx: Optional[int] = None) -> torch.Tensor:
        """Get anchor prototype for a turn category."""
        anchors = self.turn_anchors.get(turn_cat)

        if anchors is None or len(anchors) == 0:
            print(f"  WARNING: No anchors for turn_cat={turn_cat}, using zeros")
            return torch.zeros(self.K, device=self.device)

        if anchor_idx is None:
            anchor_idx = np.random.randint(len(anchors))
        else:
            anchor_idx = min(anchor_idx, len(anchors) - 1)

        return torch.tensor(anchors[anchor_idx], dtype=torch.float32, device=self.device)

    def sample_length(self, orient_AB: int, dist_AB: int) -> int:
        """Sample a realistic length from AB segment's distribution."""
        key = (dist_AB, orient_AB)

        if self.length_distributions is not None and key in self.length_distributions:
            stats = self.length_distributions[key]
            mean = stats['mean']
            std = stats['std']
            min_len = stats['min']
            max_len = stats['max']

            length = int(np.random.normal(mean, std))
            length = max(min_len, min(max_len, length))
            return length
        else:
            # Fallback defaults
            defaults = {0: 30, 1: 45, 2: 60, 3: 80, 4: 100}
            return defaults.get(dist_AB, 60)

    @torch.no_grad()
    def generate_single(
        self,
        orient_AB: int,
        dist_AB: int,
        orient_BC: int,
        dist_BC: int,
        turn_cat: Optional[int] = None,
        anchor_idx: Optional[int] = None,
        seed: Optional[int] = None,
        target_length: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a single trajectory with V5 dual conditioning.

        Args:
            orient_AB: Current segment orientation (0-7 or string)
            dist_AB: Current segment distance group (0-4 or string)
            orient_BC: Next segment orientation (0-7 or string)
            dist_BC: Next segment distance group (0-4 or string)
            turn_cat: Turn category (computed if None)
            anchor_idx: Which anchor to use (None = random)
            seed: Random seed for reproducibility
            target_length: Specific length (None = sample from distribution)

        Returns:
            trajectory: [T, 2] numpy array
            info: dict with generation metadata
        """
        # Convert strings to IDs
        if isinstance(orient_AB, str):
            orient_AB = ORIENTATIONS.index(orient_AB)
        if isinstance(dist_AB, str):
            dist_AB = DISTANCE_GROUPS.index(dist_AB)
        if isinstance(orient_BC, str):
            orient_BC = ORIENTATIONS.index(orient_BC)
        if isinstance(dist_BC, str):
            dist_BC = DISTANCE_GROUPS.index(dist_BC)

        # Compute turn category if not provided
        if turn_cat is None:
            turn_cat = compute_turn_category(orient_AB, orient_BC)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Determine target length from AB segment
        if target_length is None:
            target_length = self.sample_length(orient_AB, dist_AB)

        # Get anchor based on turn category
        anchor = self.get_anchor(turn_cat, anchor_idx)
        anchor = anchor.unsqueeze(0)  # [1, K]

        # Prepare condition tensors
        orient_AB_t = torch.tensor([orient_AB], device=self.device)
        dist_AB_t = torch.tensor([dist_AB], device=self.device)
        orient_BC_t = torch.tensor([orient_BC], device=self.device)
        dist_BC_t = torch.tensor([dist_BC], device=self.device)
        turn_cat_t = torch.tensor([turn_cat], device=self.device)

        # Initialize residual from noise
        r_t = torch.randn(1, self.K, device=self.device)

        # DDIM denoising loop with V5 conditioning
        for step in reversed(range(self.num_steps)):
            t = torch.tensor([step], device=self.device)
            noise_pred = self.model(
                r_t, t,
                orient_AB_t, dist_AB_t,
                orient_BC_t, dist_BC_t,
                turn_cat_t,
                anchor
            )
            r_t = self.noise_schedule.ddim_step(r_t, noise_pred, t)

        # Final residual (normalized)
        r_0 = r_t.squeeze(0).cpu().numpy()

        # DENORMALIZE residuals
        if self.residual_mean is not None and self.residual_std is not None:
            r_0 = r_0 * self.residual_std + self.residual_mean

        # Add anchor to get coefficients
        coefficients = anchor.squeeze(0).cpu().numpy() + r_0

        # Reconstruct trajectory
        cp_recon = self.U_ref @ coefficients
        if self.mean is not None:
            cp_recon = cp_recon + self.mean

        C = build_interleaved_basis_matrix(target_length, self.T_ref)
        trajectory_flat = C @ cp_recon

        # Reshape to [T, 2]
        trajectory = np.column_stack([
            trajectory_flat[0::2],
            trajectory_flat[1::2]
        ])

        # Force start at origin
        trajectory = trajectory - trajectory[0]

        # Compute turn angle for info
        turn_angle = compute_turn_angle(orient_AB, orient_BC)

        # Build info dict
        info = {
            'coefficients': coefficients,
            'target_length': target_length,
            'orient_AB': orient_AB,
            'orient_AB_name': ORIENTATIONS[orient_AB],
            'dist_AB': dist_AB,
            'dist_AB_name': DISTANCE_GROUPS[dist_AB],
            'orient_BC': orient_BC,
            'orient_BC_name': ORIENTATIONS[orient_BC],
            'dist_BC': dist_BC,
            'dist_BC_name': DISTANCE_GROUPS[dist_BC],
            'turn_cat': turn_cat,
            'turn_cat_name': TURN_CATEGORY_NAMES[turn_cat],
            'turn_angle': turn_angle,
            'seed': seed,
        }

        return trajectory, info

    @torch.no_grad()
    def generate_batch(
        self,
        orient_ABs: List[int],
        dist_ABs: List[int],
        orient_BCs: List[int],
        dist_BCs: List[int],
        turn_cats: Optional[List[int]] = None,
        target_lengths: Optional[List[int]] = None
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Generate multiple trajectories with V5 conditioning."""
        B = len(orient_ABs)

        # Compute turn categories if not provided
        if turn_cats is None:
            turn_cats = [
                compute_turn_category(o_ab, o_bc)
                for o_ab, o_bc in zip(orient_ABs, orient_BCs)
            ]

        # Determine lengths from AB segments
        if target_lengths is None:
            target_lengths = [
                self.sample_length(o, d)
                for o, d in zip(orient_ABs, dist_ABs)
            ]

        # Get anchors by turn category
        anchors = torch.stack([
            self.get_anchor(tc)
            for tc in turn_cats
        ])  # [B, K]

        # Prepare condition tensors
        orient_AB_t = torch.tensor(orient_ABs, device=self.device)
        dist_AB_t = torch.tensor(dist_ABs, device=self.device)
        orient_BC_t = torch.tensor(orient_BCs, device=self.device)
        dist_BC_t = torch.tensor(dist_BCs, device=self.device)
        turn_cat_t = torch.tensor(turn_cats, device=self.device)

        # Initialize from noise
        r_t = torch.randn(B, self.K, device=self.device)

        # DDIM denoising loop
        for step in reversed(range(self.num_steps)):
            t = torch.full((B,), step, device=self.device)
            noise_pred = self.model(
                r_t, t,
                orient_AB_t, dist_AB_t,
                orient_BC_t, dist_BC_t,
                turn_cat_t,
                anchors
            )
            r_t = self.noise_schedule.ddim_step(r_t, noise_pred, t)

        # Residuals (normalized)
        residuals = r_t.cpu().numpy()

        # DENORMALIZE
        if self.residual_mean is not None and self.residual_std is not None:
            residuals = residuals * self.residual_std + self.residual_mean

        # Coefficients
        coefficients = anchors.cpu().numpy() + residuals

        # Reconstruct each trajectory
        trajectories = []
        for i in range(B):
            T = target_lengths[i]

            cp_recon = self.U_ref @ coefficients[i]
            if self.mean is not None:
                cp_recon = cp_recon + self.mean

            C = build_interleaved_basis_matrix(T, self.T_ref)
            traj_flat = C @ cp_recon

            traj = np.column_stack([traj_flat[0::2], traj_flat[1::2]])
            traj = traj - traj[0]
            trajectories.append(traj)

        return trajectories, target_lengths

    def generate_for_turn_category(
        self,
        turn_cat: int,
        samples: int = 10,
        dist_AB: int = 2,  # Medium
        dist_BC: int = 2   # Medium
    ) -> List[Tuple[np.ndarray, Dict]]:
        """Generate samples for a specific turn category."""
        results = []

        # Get representative orientations for this turn category
        orient_AB, orient_BC = get_representative_orientations(turn_cat)

        for i in range(samples):
            traj, info = self.generate_single(
                orient_AB, dist_AB,
                orient_BC, dist_BC,
                turn_cat=turn_cat,
                seed=turn_cat * 1000 + i
            )
            results.append((traj, info))

        return results

    def generate_for_all_turn_categories(
        self,
        samples_per_category: int = 10,
        dist_AB: int = 2,
        dist_BC: int = 2
    ) -> Dict[str, List[Tuple[np.ndarray, Dict]]]:
        """Generate samples for all 7 turn categories."""
        results = {}

        print(f"\nGenerating {samples_per_category} samples per turn category...")

        for tc in TURN_CATEGORIES:
            turn_cat = tc["id"]
            name = tc["name"]

            samples = self.generate_for_turn_category(
                turn_cat, samples_per_category, dist_AB, dist_BC
            )
            results[name] = samples

            lengths = [info['target_length'] for _, info in samples]
            print(f"  {name}: {len(samples)} trajectories, mean length={np.mean(lengths):.0f}")

        return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_turn_categories(
    generator: TrajectoryGeneratorV5,
    samples_per_category: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize generated trajectories for all 7 turn categories."""
    fig, axes = plt.subplots(1, 7, figsize=(21, 4))

    for tc in TURN_CATEGORIES:
        turn_cat = tc["id"]
        name = tc["name"]
        color = tc["color"]
        ax = axes[turn_cat]

        samples = generator.generate_for_turn_category(
            turn_cat, samples_per_category
        )

        lengths = []
        for traj, info in samples:
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.5, linewidth=1.5)
            lengths.append(info['target_length'])

        # Mark origin
        ax.scatter(0, 0, c='black', s=50, marker='o', zorder=10)

        # Add arrow showing turn direction
        orient_AB, orient_BC = get_representative_orientations(turn_cat)
        turn_angle = compute_turn_angle(orient_AB, orient_BC)

        ax.set_title(f'{name}\n({turn_angle:+.0f}°)', fontsize=10)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)

    plt.suptitle('V5 Anticipatory Trajectories by Turn Category', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()
    return fig


def visualize_single_trajectory(
    trajectory: np.ndarray,
    title: str = "Generated Trajectory",
    info: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize a single V5 trajectory."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.8)

    # Mark start and end
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100,
               marker='o', zorder=5, label='Start (A)', edgecolors='black')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100,
               marker='X', zorder=5, label='End (B)', edgecolors='black')

    # Add info to title
    if info is not None:
        title += f"\nAB: {info['dist_AB_name']}-{info['orient_AB_name']} → "
        title += f"BC: {info['dist_BC_name']}-{info['orient_BC_name']}"
        title += f"\nTurn: {info['turn_cat_name']} ({info['turn_angle']:+.0f}°), Length={info['target_length']}"

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()
    return fig


def visualize_anticipatory_comparison(
    generator: TrajectoryGeneratorV5,
    orient_AB: int = 2,  # E
    dist_AB: int = 2,    # Medium
    dist_BC: int = 2,    # Medium
    samples: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare how the same A→B segment changes based on where C is.

    Shows the same AB segment with different BC directions,
    demonstrating anticipatory movement.
    """
    fig, axes = plt.subplots(1, 7, figsize=(21, 4))

    for tc in TURN_CATEGORIES:
        turn_cat = tc["id"]
        name = tc["name"]
        color = tc["color"]
        ax = axes[turn_cat]

        _, orient_BC = get_representative_orientations(turn_cat)

        for i in range(samples):
            traj, info = generator.generate_single(
                orient_AB, dist_AB,
                orient_BC, dist_BC,
                turn_cat=turn_cat,
                seed=turn_cat * 1000 + i
            )
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.5, linewidth=1.5)

        ax.scatter(0, 0, c='black', s=50, marker='o', zorder=10)

        turn_angle = compute_turn_angle(orient_AB, orient_BC)
        ax.set_title(f'Next: {ORIENTATIONS[orient_BC]}\n({turn_angle:+.0f}°)', fontsize=10)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Same A→B ({DISTANCE_GROUPS[dist_AB]}-{ORIENTATIONS[orient_AB]}), Different C Directions',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate trajectories using V5 (Anticipatory) model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate for all turn categories (7-panel view)
    python generate_trajectory_v5.py --checkpoint checkpoints_v5/best_model.pt \\
        --data processed_v5/ --visualize

    # Generate specific segment pair
    python generate_trajectory_v5.py --checkpoint checkpoints_v5/best_model.pt \\
        --data processed_v5/ --orient_AB E --dist_AB Medium \\
        --orient_BC N --dist_BC Small --visualize

    # Show anticipatory comparison (same AB, different BC)
    python generate_trajectory_v5.py --checkpoint checkpoints_v5/best_model.pt \\
        --data processed_v5/ --anticipatory_demo --visualize
"""
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to V5 model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to preprocessed V5 data directory')
    parser.add_argument('--output', type=str, default='generated_v5',
                        help='Output directory')

    # AB segment options
    parser.add_argument('--orient_AB', type=str, default=None,
                        help='AB orientation (N, NE, E, SE, S, SW, W, NW)')
    parser.add_argument('--dist_AB', type=str, default='Medium',
                        help='AB distance group')

    # BC segment options
    parser.add_argument('--orient_BC', type=str, default=None,
                        help='BC orientation (N, NE, E, SE, S, SW, W, NW)')
    parser.add_argument('--dist_BC', type=str, default='Medium',
                        help='BC distance group')

    parser.add_argument('--length', type=int, default=None,
                        help='Specific trajectory length')
    parser.add_argument('--samples', type=int, default=10,
                        help='Samples per category')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--anticipatory_demo', action='store_true',
                        help='Show same AB with different BC directions')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    print("=" * 70)
    print("TRAJECTORY GENERATION - VERSION 5 (ANTICIPATORY)")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create generator
    print("\nInitializing V5 generator...")
    generator = TrajectoryGeneratorV5(args.checkpoint, args.data, device=args.device)

    # Generate based on arguments
    if args.orient_AB and args.orient_BC:
        # Specific segment pair
        print(f"\nGenerating: AB={args.dist_AB}-{args.orient_AB} → BC={args.dist_BC}-{args.orient_BC}")

        trajectory, info = generator.generate_single(
            args.orient_AB, args.dist_AB,
            args.orient_BC, args.dist_BC,
            seed=args.seed,
            target_length=args.length
        )

        print(f"\n  Generated trajectory:")
        print(f"    Shape: {trajectory.shape}")
        print(f"    Turn: {info['turn_cat_name']} ({info['turn_angle']:+.0f}°)")
        print(f"    Length: {info['target_length']} points")
        print(f"    End: [{trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f}]")

        if args.visualize:
            vis_path = output_dir / f"trajectory_{args.orient_AB}_{args.orient_BC}.png"
            visualize_single_trajectory(
                trajectory,
                f'Generated: {args.orient_AB}→{args.orient_BC}',
                info=info,
                save_path=str(vis_path)
            )

        np.save(output_dir / f"trajectory_{args.orient_AB}_{args.orient_BC}.npy", trajectory)

    else:
        # Generate for all turn categories
        print(f"\nGenerating for all 7 turn categories...")

        if args.visualize:
            vis_path = output_dir / 'turn_categories_v5.png'
            visualize_turn_categories(
                generator,
                samples_per_category=args.samples,
                save_path=str(vis_path)
            )

        results = generator.generate_for_all_turn_categories(
            samples_per_category=args.samples
        )

        # Save trajectories
        save_results = {
            name: [traj for traj, _ in samples]
            for name, samples in results.items()
        }
        np.save(output_dir / 'generated_trajectories.npy', save_results, allow_pickle=True)
        print(f"\nSaved trajectories to {output_dir / 'generated_trajectories.npy'}")

    # Optional: Anticipatory demo
    if args.anticipatory_demo and args.visualize:
        print("\nGenerating anticipatory comparison...")
        vis_path = output_dir / 'anticipatory_comparison.png'
        visualize_anticipatory_comparison(
            generator,
            samples=args.samples,
            save_path=str(vis_path)
        )

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE (V5 ANTICIPATORY)")
    print("=" * 70)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == '__main__':
    main()
