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
# CHAIN GENERATOR - Continuous Multi-Waypoint Paths
# =============================================================================

class ChainGenerator:
    """
    Generate continuous trajectories through multiple waypoints.

    Key features:
    - Subdivides long segments into trainable distances
    - Each segment is generated with anticipation of the NEXT segment
    - Transforms (scale/rotate/translate) generated segments to actual coordinates
    - Smooth concatenation at waypoints

    Usage:
        chain_gen = ChainGenerator(generator)

        # From explicit waypoints
        waypoints = [(100, 100), (400, 300), (200, 500), (600, 400)]
        trajectory = chain_gen.generate_chain(waypoints)

        # Random path
        waypoints, trajectory = chain_gen.generate_random_path(
            start=(500, 500), num_waypoints=6
        )
    """

    # Distance group boundaries (pixels) - must match preprocessing
    DISTANCE_BOUNDARIES = [170, 340, 510, 687]  # XSmall/Small/Medium/Large/XLarge max

    # Target segment length for subdivision (middle of Medium range)
    TARGET_SEGMENT_LENGTH = 350

    # Screen dimensions for random path generation
    DEFAULT_SCREEN_WIDTH = 2560
    DEFAULT_SCREEN_HEIGHT = 1440

    def __init__(
        self,
        generator: TrajectoryGeneratorV5,
        screen_width: int = None,
        screen_height: int = None
    ):
        self.generator = generator
        self.screen_width = screen_width or self.DEFAULT_SCREEN_WIDTH
        self.screen_height = screen_height or self.DEFAULT_SCREEN_HEIGHT

    def classify_orientation(self, dx: float, dy: float) -> int:
        """
        Classify movement direction into orientation ID (0-7).

        Orientation mapping (screen coordinates, Y increases downward):
            0: N  (up)        - angle ~= -90°
            1: NE (up-right)  - angle ~= -45°
            2: E  (right)     - angle ~= 0°
            3: SE (down-right)- angle ~= 45°
            4: S  (down)      - angle ~= 90°
            5: SW (down-left) - angle ~= 135°
            6: W  (left)      - angle ~= 180°
            7: NW (up-left)   - angle ~= -135°
        """
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 2  # Default to East for zero movement

        # atan2 returns angle in radians, convert to degrees
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Map angle to orientation
        # E=0°, SE=45°, S=90°, SW=135°, W=180°/-180°, NW=-135°, N=-90°, NE=-45°
        # We need to map to: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7

        # Shift so N (up, -90°) becomes 0
        shifted = angle_deg + 90  # Now N=0, E=90, S=180, W=270

        # Normalize to [0, 360)
        shifted = shifted % 360

        # Quantize to 8 bins (each 45°)
        orient_id = int((shifted + 22.5) / 45) % 8

        return orient_id

    def classify_distance(self, dx: float, dy: float) -> int:
        """
        Classify movement distance into distance group (0-4).

        Groups: XSmall(0), Small(1), Medium(2), Large(3), XLarge(4)
        """
        dist = np.sqrt(dx**2 + dy**2)

        for i, boundary in enumerate(self.DISTANCE_BOUNDARIES):
            if dist <= boundary:
                return i

        return 4  # XLarge

    def subdivide_segment(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        target_length: float = None
    ) -> List[Tuple[float, float]]:
        """
        Subdivide a long segment into natural-length pieces.

        If the segment is within the max trainable distance, returns [start, end].
        Otherwise, inserts intermediate waypoints along the line.

        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            target_length: Target length for each sub-segment

        Returns:
            List of waypoints including start and end
        """
        if target_length is None:
            target_length = self.TARGET_SEGMENT_LENGTH

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        total_dist = np.sqrt(dx**2 + dy**2)

        # If within max trainable distance, no subdivision needed
        max_trainable = self.DISTANCE_BOUNDARIES[-1]
        if total_dist <= max_trainable:
            return [start, end]

        # Calculate number of subdivisions
        n_segments = int(np.ceil(total_dist / target_length))
        n_segments = max(2, n_segments)  # At least 2 segments

        # Create intermediate waypoints
        waypoints = [start]
        for i in range(1, n_segments):
            t = i / n_segments
            wp = (start[0] + t * dx, start[1] + t * dy)
            waypoints.append(wp)
        waypoints.append(end)

        return waypoints

    def transform_trajectory(
        self,
        trajectory: np.ndarray,
        start: Tuple[float, float],
        target_direction: Tuple[float, float]
    ) -> np.ndarray:
        """
        Transform a generated trajectory: ROTATE + TRANSLATE only. NO SCALING.

        The generated trajectory starts at origin. We rotate it to point toward
        the target direction, then translate to start position.

        CRITICAL: We do NOT scale the trajectory. The model generates natural
        lengths for the distance group. Scaling would destroy the natural shape.

        Args:
            trajectory: [T, 2] array starting at origin
            start: Start position (x, y)
            target_direction: Direction to point toward (dx, dy from start)

        Returns:
            Transformed trajectory [T, 2]
        """
        # Target direction angle
        target_angle = np.arctan2(target_direction[1], target_direction[0])

        # Generated trajectory direction (from endpoint)
        gen_endpoint = trajectory[-1]
        gen_angle = np.arctan2(gen_endpoint[1], gen_endpoint[0])

        # Rotation needed to align generated direction with target
        rotation = target_angle - gen_angle

        # Apply rotation (NO SCALING!)
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotated = np.column_stack([
            trajectory[:, 0] * cos_r - trajectory[:, 1] * sin_r,
            trajectory[:, 0] * sin_r + trajectory[:, 1] * cos_r
        ])

        # Translate to start position
        translated = rotated + np.array(start)

        return translated

    def generate_chain(
        self,
        waypoints: List[Tuple[float, float]],
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a continuous trajectory through waypoints.

        Algorithm:
        1. Subdivide any segments that exceed max trainable distance
        2. For each segment, generate with anticipation of next segment
        3. Transform each segment to actual coordinates
        4. Concatenate smoothly (remove duplicate points)

        Args:
            waypoints: List of (x, y) waypoints
            seed: Random seed for reproducibility
            verbose: Print progress information

        Returns:
            trajectory: [N, 2] full trajectory array
            info: Dict with generation metadata
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Step 1: Subdivide long segments
        expanded_waypoints = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            sub_wps = self.subdivide_segment(waypoints[i], waypoints[i + 1])
            expanded_waypoints.extend(sub_wps[1:])  # Skip first (already added)

        if verbose and len(expanded_waypoints) > len(waypoints):
            print(f"  Subdivided {len(waypoints)} waypoints → {len(expanded_waypoints)} waypoints")

        waypoints = expanded_waypoints
        n_segments = len(waypoints) - 1

        # Step 2: Generate each segment
        # IMPORTANT: We use actual endpoints, not waypoints, for seamless concatenation
        # Waypoints are DIRECTION GUIDES, not exact targets
        segments = []
        segment_infos = []
        current_pos = waypoints[0]  # Start at first waypoint

        for i in range(n_segments):
            # Current position is either first waypoint or actual endpoint of previous segment
            A = current_pos
            B = waypoints[i + 1]  # Target waypoint (direction guide)
            C = waypoints[i + 2] if i + 2 < len(waypoints) else None

            # Compute AB conditioning (direction from A toward B)
            dx_AB = B[0] - A[0]
            dy_AB = B[1] - A[1]
            orient_AB = self.classify_orientation(dx_AB, dy_AB)
            dist_AB = self.classify_distance(dx_AB, dy_AB)

            # Compute BC conditioning (anticipation - direction from B toward C)
            if C is not None:
                dx_BC = C[0] - B[0]
                dy_BC = C[1] - B[1]
                orient_BC = self.classify_orientation(dx_BC, dy_BC)
                dist_BC = self.classify_distance(dx_BC, dy_BC)
            else:
                # Final segment - no anticipation, continue straight
                orient_BC = orient_AB
                dist_BC = dist_AB

            # Generate segment (model outputs natural length for distance group)
            traj, info = self.generator.generate_single(
                orient_AB, dist_AB,
                orient_BC, dist_BC,
                seed=None  # Already seeded at start
            )

            # Transform: rotate to target direction, translate to start
            # NO SCALING - we trust the model's natural output length
            direction = (dx_AB, dy_AB)
            transformed = self.transform_trajectory(traj, A, direction)

            segments.append(transformed)

            # CRITICAL: Next segment starts at actual endpoint, not waypoint B
            current_pos = (transformed[-1, 0], transformed[-1, 1])

            segment_infos.append({
                'segment_idx': i,
                'start': A,
                'target_waypoint': B,
                'actual_endpoint': current_pos,
                'next_waypoint': C,
                'orient_AB': orient_AB,
                'dist_AB': dist_AB,
                'orient_BC': orient_BC,
                'dist_BC': dist_BC,
                'turn_cat': info['turn_cat'],
                'turn_cat_name': info['turn_cat_name'],
                'turn_angle': info['turn_angle'],
                'generated_length': len(traj),
            })

        # Step 3: Concatenate segments (already seamless - each starts at previous endpoint)
        full_trajectory = [segments[0]]
        for seg in segments[1:]:
            full_trajectory.append(seg[1:])  # Skip first point (duplicate of previous endpoint)

        full_trajectory = np.vstack(full_trajectory)

        # Build info
        chain_info = {
            'original_waypoints': waypoints[:len(waypoints)],
            'expanded_waypoints': waypoints,
            'n_segments': n_segments,
            'total_points': len(full_trajectory),
            'segment_infos': segment_infos,
            'seed': seed,
        }

        if verbose:
            print(f"  Generated chain: {n_segments} segments, {len(full_trajectory)} points")
            turn_summary = {}
            for si in segment_infos:
                tc = si['turn_cat_name']
                turn_summary[tc] = turn_summary.get(tc, 0) + 1
            print(f"  Turn categories: {turn_summary}")

        return full_trajectory, chain_info

    def generate_random_path(
        self,
        start: Tuple[float, float] = None,
        num_waypoints: int = 5,
        min_distance: float = 200,
        max_distance: float = 500,
        margin: float = 50,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[List[Tuple[float, float]], np.ndarray, Dict]:
        """
        Generate random waypoints and trajectory.

        Args:
            start: Starting position (default: center of screen)
            num_waypoints: Number of waypoints to generate
            min_distance: Minimum distance between consecutive waypoints
            max_distance: Maximum distance between consecutive waypoints
            margin: Minimum distance from screen edges
            seed: Random seed
            verbose: Print progress

        Returns:
            waypoints: List of (x, y) waypoints
            trajectory: [N, 2] full trajectory array
            info: Dict with generation metadata
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Default start at center
        if start is None:
            start = (self.screen_width / 2, self.screen_height / 2)

        waypoints = [start]

        for _ in range(num_waypoints - 1):
            last = waypoints[-1]

            # Try to find a valid next waypoint
            for attempt in range(100):
                # Random direction
                angle = np.random.uniform(0, 2 * np.pi)
                # Random distance
                dist = np.random.uniform(min_distance, max_distance)

                dx = dist * np.cos(angle)
                dy = dist * np.sin(angle)

                new_x = last[0] + dx
                new_y = last[1] + dy

                # Check bounds
                if (margin <= new_x <= self.screen_width - margin and
                    margin <= new_y <= self.screen_height - margin):
                    waypoints.append((new_x, new_y))
                    break
            else:
                # Couldn't find valid point, clamp to bounds
                new_x = np.clip(last[0] + dx, margin, self.screen_width - margin)
                new_y = np.clip(last[1] + dy, margin, self.screen_height - margin)
                waypoints.append((new_x, new_y))

        if verbose:
            print(f"\nGenerated {len(waypoints)} random waypoints")

        # Generate chain through waypoints
        trajectory, chain_info = self.generate_chain(waypoints, seed=None, verbose=verbose)

        chain_info['random_seed'] = seed
        chain_info['min_distance'] = min_distance
        chain_info['max_distance'] = max_distance

        return waypoints, trajectory, chain_info


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


def visualize_chain(
    trajectory: np.ndarray,
    chain_info: Dict,
    title: str = "Chain Trajectory",
    show_waypoints: bool = True,
    show_segments: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a chain trajectory with waypoints and segment info.

    Shows:
    - Actual trajectory (colored by turn category)
    - Target waypoints (direction guides) in orange
    - Actual start (green) and end (red)

    Args:
        trajectory: [N, 2] full trajectory array
        chain_info: Info dict from generate_chain()
        title: Plot title
        show_waypoints: Draw waypoint markers
        show_segments: Color-code segments by turn category
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    waypoints = chain_info['expanded_waypoints']
    segment_infos = chain_info['segment_infos']

    if show_segments and len(segment_infos) > 0:
        # Color-code by segment
        point_idx = 0
        for i, seg_info in enumerate(segment_infos):
            seg_len = seg_info['generated_length']
            if i > 0:
                seg_len -= 1  # Account for removed duplicate

            end_idx = min(point_idx + seg_len, len(trajectory))

            # Get color from turn category
            turn_cat = seg_info['turn_cat']
            color = TURN_CATEGORIES[turn_cat]['color']

            seg_traj = trajectory[point_idx:end_idx + 1]
            ax.plot(seg_traj[:, 0], seg_traj[:, 1],
                   color=color, linewidth=2, alpha=0.8)

            point_idx = end_idx
    else:
        # Single color
        ax.plot(trajectory[:, 0], trajectory[:, 1],
               'b-', linewidth=2, alpha=0.8)

    # Draw waypoints (these are DIRECTION GUIDES, path may not hit them exactly)
    if show_waypoints and len(waypoints) > 0:
        wp_array = np.array(waypoints)

        # Intermediate waypoints (direction guides)
        if len(wp_array) > 2:
            ax.scatter(wp_array[1:-1, 0], wp_array[1:-1, 1],
                      c='orange', s=80, marker='D', zorder=10,
                      edgecolors='black', linewidths=1, alpha=0.6,
                      label='Target waypoints (guides)')

        # First waypoint (actual start)
        ax.scatter(wp_array[0, 0], wp_array[0, 1],
                  c='green', s=150, marker='o', zorder=11,
                  edgecolors='black', linewidths=2,
                  label='Start')

        # Last waypoint (target end - may not be exact)
        ax.scatter(wp_array[-1, 0], wp_array[-1, 1],
                  c='orange', s=100, marker='D', zorder=10,
                  edgecolors='black', linewidths=1, alpha=0.6)

        # Actual trajectory endpoint
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                  c='red', s=150, marker='X', zorder=11,
                  edgecolors='black', linewidths=2,
                  label='Actual end')

        # Add waypoint labels
        for i, (x, y) in enumerate(waypoints):
            ax.annotate(f'W{i}', (x, y), textcoords='offset points',
                       xytext=(5, 5), fontsize=8, alpha=0.7)

    # Add info to title
    n_seg = chain_info['n_segments']
    n_pts = chain_info['total_points']
    title += f"\n{n_seg} segments, {n_pts} points"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()
    return fig


def visualize_chain_segments(
    chain_info: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize chain segment details - turn categories and angles.
    """
    segment_infos = chain_info['segment_infos']
    n_segments = len(segment_infos)

    if n_segments == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Turn category distribution
    ax1 = axes[0]
    turn_cats = [si['turn_cat'] for si in segment_infos]
    turn_names = [TURN_CATEGORY_NAMES[tc] for tc in turn_cats]

    # Count occurrences
    from collections import Counter
    counts = Counter(turn_names)

    names = [tc['name'] for tc in TURN_CATEGORIES]
    values = [counts.get(n, 0) for n in names]
    colors = [tc['color'] for tc in TURN_CATEGORIES]

    ax1.bar(range(7), values, color=colors)
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Count')
    ax1.set_title('Turn Category Distribution')

    # Right: Turn angles over path
    ax2 = axes[1]
    turn_angles = [si['turn_angle'] for si in segment_infos]
    segment_indices = range(len(turn_angles))

    colors_per_seg = [TURN_CATEGORIES[tc]['color'] for tc in turn_cats]
    ax2.bar(segment_indices, turn_angles, color=colors_per_seg)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('Turn Angle (°)')
    ax2.set_title('Turn Angles Along Path')
    ax2.set_ylim(-180, 180)

    plt.suptitle('Chain Segment Analysis', fontsize=14, fontweight='bold')
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

    # CHAIN GENERATION - through explicit waypoints
    python generate_trajectory_v5.py --checkpoint checkpoints_v5/best_model.pt \\
        --data processed_v5/ --chain "100,100;400,300;200,500;600,400" --visualize

    # RANDOM PATH - generate random waypoints and chain
    python generate_trajectory_v5.py --checkpoint checkpoints_v5/best_model.pt \\
        --data processed_v5/ --random_path 6 --visualize
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

    # Chain generation options
    parser.add_argument('--chain', type=str, default=None,
                        help='Generate chain through waypoints: "x1,y1;x2,y2;x3,y3;..."')
    parser.add_argument('--random_path', type=int, default=None,
                        help='Generate random path with N waypoints')
    parser.add_argument('--start', type=str, default=None,
                        help='Start position for random path: "x,y" (default: screen center)')
    parser.add_argument('--screen_width', type=int, default=2560,
                        help='Screen width for random path generation')
    parser.add_argument('--screen_height', type=int, default=1440,
                        help='Screen height for random path generation')

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

    # =========================================================================
    # CHAIN GENERATION MODE
    # =========================================================================
    if args.chain:
        print("\n" + "=" * 70)
        print("CHAIN GENERATION MODE")
        print("=" * 70)

        # Parse waypoints from string "x1,y1;x2,y2;x3,y3"
        try:
            waypoints = []
            for wp_str in args.chain.split(';'):
                x, y = map(float, wp_str.strip().split(','))
                waypoints.append((x, y))
            print(f"\nParsed {len(waypoints)} waypoints:")
            for i, (x, y) in enumerate(waypoints):
                print(f"  W{i}: ({x:.1f}, {y:.1f})")
        except ValueError as e:
            print(f"\nERROR: Invalid waypoint format: {e}")
            print("Expected format: --chain \"x1,y1;x2,y2;x3,y3\"")
            return

        # Create chain generator
        chain_gen = ChainGenerator(
            generator,
            screen_width=args.screen_width,
            screen_height=args.screen_height
        )

        # Generate chain
        trajectory, chain_info = chain_gen.generate_chain(
            waypoints, seed=args.seed, verbose=True
        )

        # Save trajectory
        np.save(output_dir / 'chain_trajectory.npy', trajectory)
        print(f"\nSaved trajectory to {output_dir / 'chain_trajectory.npy'}")

        # Visualize
        if args.visualize:
            vis_path = output_dir / 'chain_trajectory.png'
            visualize_chain(
                trajectory, chain_info,
                title='Chain Trajectory',
                save_path=str(vis_path)
            )

            seg_vis_path = output_dir / 'chain_segments.png'
            visualize_chain_segments(chain_info, save_path=str(seg_vis_path))

        print("\n" + "=" * 70)
        print("CHAIN GENERATION COMPLETE")
        print("=" * 70)
        return

    # =========================================================================
    # RANDOM PATH GENERATION MODE
    # =========================================================================
    if args.random_path:
        print("\n" + "=" * 70)
        print("RANDOM PATH GENERATION MODE")
        print("=" * 70)

        # Parse start position
        start = None
        if args.start:
            try:
                x, y = map(float, args.start.split(','))
                start = (x, y)
            except ValueError:
                print(f"WARNING: Invalid start format '{args.start}', using center")

        # Create chain generator
        chain_gen = ChainGenerator(
            generator,
            screen_width=args.screen_width,
            screen_height=args.screen_height
        )

        # Generate random path
        waypoints, trajectory, chain_info = chain_gen.generate_random_path(
            start=start,
            num_waypoints=args.random_path,
            seed=args.seed,
            verbose=True
        )

        # Save
        np.save(output_dir / 'random_path_trajectory.npy', trajectory)
        np.save(output_dir / 'random_path_waypoints.npy', np.array(waypoints))
        print(f"\nSaved trajectory to {output_dir / 'random_path_trajectory.npy'}")

        # Visualize
        if args.visualize:
            vis_path = output_dir / 'random_path_trajectory.png'
            visualize_chain(
                trajectory, chain_info,
                title=f'Random Path ({args.random_path} waypoints)',
                save_path=str(vis_path)
            )

            seg_vis_path = output_dir / 'random_path_segments.png'
            visualize_chain_segments(chain_info, save_path=str(seg_vis_path))

        print("\n" + "=" * 70)
        print("RANDOM PATH GENERATION COMPLETE")
        print("=" * 70)
        return

    # =========================================================================
    # SINGLE SEGMENT / CATEGORY GENERATION MODE
    # =========================================================================

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
