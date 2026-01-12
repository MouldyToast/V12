#!/usr/bin/env python3
"""
Trajectory Generation from Trained Diffusion Model - VERSION 3 (Basis-Transformation)

This script generates mouse trajectories using the trained residual diffusion model.
VERSION 3 uses the basis-transformation approach where the BASIS is adapted to
the target length, rather than interpolating trajectory data.

KEY DIFFERENCE FROM V2:
=======================
V2: Generate coefficients → reconstruct at T_win → B-spline interpolate to target
    Problem: Post-hoc interpolation can smooth away generated details
    
V3: Generate coefficients → adapt basis to target length → reconstruct directly
    Benefit: No post-processing, generated characteristics preserved exactly

Generation Process:
    1. Given (orientation_id, distance_id), select a group anchor
    2. Sample target length from group's length distribution
    3. Initialize residual from Gaussian noise
    4. Denoise through M=10 DDIM steps
    5. Add denoised residual to anchor: c = anchor + r_0
    6. Adapt basis to target length: U_T = adapter.get_adapted_basis(T)
    7. Reconstruct: trajectory = U_T @ c (directly at target length!)

Usage:
    # Generate for specific group
    python generate_trajectory_v3.py --checkpoint checkpoints/best_model.pt \\
        --data processed_v3/ --orient N --dist Medium --visualize
    
    # Generate for all groups
    python generate_trajectory_v3.py --checkpoint checkpoints/best_model.pt --data processed_v3/ --all_groups --visualize --samples 20
    
    # Generate with specific length
    python generate_trajectory_v3.py --checkpoint checkpoints/best_model.pt \\
        --data processed_v3/ --orient E --dist Large --length 80
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

from train_singular_diffusion_v1 import ResidualDiffusionMLP, NoiseSchedule
from bspline_basis import BasisAdapter, build_interleaved_basis_matrix


# =============================================================================
# CONFIGURATION
# =============================================================================

ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DISTANCE_GROUPS = ["XSmall", "Small", "Medium", "Large", "XLarge"]

ORIENTATION_COLORS = {
    "N": "#1f77b4", "NE": "#ff7f0e", "E": "#2ca02c", "SE": "#d62728",
    "S": "#9467bd", "SW": "#8c564b", "W": "#e377c2", "NW": "#7f7f7f"
}


# =============================================================================
# V3 GENERATOR CLASS
# =============================================================================

class TrajectoryGeneratorV3:
    """
    Generate trajectories using trained diffusion model - VERSION 3 (Basis-Transformation).
    
    Key innovation: Uses BasisAdapter to adapt the SVD basis to the target length,
    rather than reconstructing at a canonical length and then interpolating.
    """
    
    def __init__(self, checkpoint_path: str, data_dir: str, device: str = 'cuda'):
        self.device = device
        self.data_dir = Path(data_dir)
        
        # Load data config
        self.config = np.load(self.data_dir / 'config.npy', allow_pickle=True).item()
        self.K = self.config['K']
        self.T_ref = self.config.get('T_ref', self.config.get('n_control_points', 64))
        self.version = self.config.get('version', 'unknown')



        
        # Verify V3 format
        if 'v3' not in self.version.lower() and not self.config.get('uses_basis_adaptation', False):
            print(f"  WARNING: Data appears to be {self.version}, not V3")
            print(f"           V3 generation may not work correctly")
        
        # Load reference basis
        # V3 uses U_ref.npy, V1/V2 use U_k.npy
        if (self.data_dir / 'U_ref.npy').exists():
            self.U_ref = np.load(self.data_dir / 'U_ref.npy')
            print(f"  Loaded U_ref.npy: {self.U_ref.shape}")
        elif (self.data_dir / 'U_k.npy').exists():
            self.U_ref = np.load(self.data_dir / 'U_k.npy')
            print(f"  Loaded U_k.npy (V1/V2 fallback): {self.U_ref.shape}")
        else:
            raise FileNotFoundError("No basis file found (U_ref.npy or U_k.npy)")
        
        # Load mean if available (for centered data)
        self.mean = None
        if (self.data_dir / 'mean.npy').exists():
            self.mean = np.load(self.data_dir / 'mean.npy')
            print(f"  Loaded mean.npy: {self.mean.shape}")
        
        # Create basis adapter
        self.adapter = BasisAdapter(self.U_ref, self.T_ref)
        print(f"  Created BasisAdapter (T_ref={self.T_ref}, K={self.K})")
        
        # Load length distributions for sampling
        self.length_distributions = None
        if (self.data_dir / 'length_distributions.npy').exists():
            self.length_distributions = np.load(
                self.data_dir / 'length_distributions.npy', 
                allow_pickle=True
            ).item()
            print(f"  Loaded length distributions for {len(self.length_distributions)} groups")
        
        # Load group anchors
        self.group_anchors = np.load(
            self.data_dir / 'group_anchors.npy', 
            allow_pickle=True
        ).item()
        
        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint['config']
        
        self.model = ResidualDiffusionMLP(
            K=self.K,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_orientations=self.config.get('num_orientations', 8),
            num_distances=self.config.get('num_distance_groups', 5),
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
        
        print(f"\nLoaded model from {checkpoint_path}")
        print(f"  K={self.K}, T_ref={self.T_ref}, diffusion_steps={self.num_steps}")
        print(f"  Version: {self.version}")



        print(f"DEBUG anchors sample: {list(self.group_anchors.values())[0][0][:3]}")
    
    def get_anchor(self, orient_id: int, dist_id: int, anchor_idx: Optional[int] = None) -> torch.Tensor:
        """Get anchor prototype for a group."""
        key = (dist_id, orient_id)
        anchors = self.group_anchors.get(key)
        
        if anchors is None:
            return torch.zeros(self.K, device=self.device)
        
        if anchor_idx is None:
            anchor_idx = np.random.randint(len(anchors))
        else:
            anchor_idx = min(anchor_idx, len(anchors) - 1)
        
        return torch.tensor(anchors[anchor_idx], dtype=torch.float32, device=self.device)
    
    def sample_length(self, orient_id: int, dist_id: int) -> int:
        """
        Sample a realistic length from the group's length distribution.
        
        Falls back to median length if no distribution available.
        """
        key = (dist_id, orient_id)
        
        if self.length_distributions is not None and key in self.length_distributions:
            stats = self.length_distributions[key]
            
            # Sample from normal distribution clipped to observed range
            mean = stats['mean']
            std = stats['std']
            min_len = stats['min']
            max_len = stats['max']
            
            # Sample and clip
            length = int(np.random.normal(mean, std))
            length = max(min_len, min(max_len, length))
            
            return length
        else:
            # Fallback: use reasonable defaults based on distance group
            defaults = {0: 30, 1: 45, 2: 60, 3: 80, 4: 100}  # XSmall to XLarge
            return defaults.get(dist_id, 60)
    
    @torch.no_grad()
    def generate_single(
        self, 
        orient_id: int, 
        dist_id: int, 
        anchor_idx: Optional[int] = None,
        seed: Optional[int] = None,
        target_length: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a single trajectory.
        
        Args:
            orient_id: int 0-7 (or string like "N", "NE", etc.)
            dist_id: int 0-4 (or string like "XSmall", "Small", etc.)
            anchor_idx: which anchor to use (None = random)
            seed: random seed for reproducibility
            target_length: specific length to generate (None = sample from distribution)
        
        Returns:
            trajectory: [T, 2] numpy array at target length
            info: dict with generation metadata
        """
        # Convert string to id if needed
        if isinstance(orient_id, str):
            orient_id = ORIENTATIONS.index(orient_id)
        if isinstance(dist_id, str):
            dist_id = DISTANCE_GROUPS.index(dist_id)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Determine target length
        if target_length is None:
            target_length = self.sample_length(orient_id, dist_id)
        
        # Get anchor
        anchor = self.get_anchor(orient_id, dist_id, anchor_idx)
        anchor = anchor.unsqueeze(0)  # [1, K]
        
        # Prepare condition tensors
        orient_tensor = torch.tensor([orient_id], device=self.device)
        dist_tensor = torch.tensor([dist_id], device=self.device)
        
        # Initialize residual from Gaussian noise
        r_t = torch.randn(1, self.K, device=self.device)
        
        # DDIM denoising loop
        for step in reversed(range(self.num_steps)):
            t = torch.tensor([step], device=self.device)
            noise_pred = self.model(r_t, t, orient_tensor, dist_tensor, anchor)
            r_t = self.noise_schedule.ddim_step(r_t, noise_pred, t)
        
        # Final residual
        r_0 = r_t.squeeze(0)  # [K]
        
        # Add anchor to get coefficients
        coefficients = (anchor.squeeze(0) + r_0).cpu().numpy()  # [K]
        # === FIXED RECONSTRUCTION ===
        # Reconstruct via control points (consistent with fixed preprocessing)
        # Step 1: Reconstruct control points in U_ref space
        cp_recon = self.U_ref @ coefficients
        if self.mean is not None:
            cp_recon = cp_recon + self.mean

        # Step 2: Interpolate control points to target trajectory length
        C = build_interleaved_basis_matrix(target_length, self.T_ref)
        trajectory_flat = C @ cp_recon

        
        # Reshape to [T, 2]
        trajectory = np.column_stack([
            trajectory_flat[0::2],
            trajectory_flat[1::2]
        ])
        
        # Build info dict
        info = {
            'coefficients': coefficients,
            'target_length': target_length,
            'orient_id': orient_id,
            'orient_name': ORIENTATIONS[orient_id],
            'dist_id': dist_id,
            'dist_name': DISTANCE_GROUPS[dist_id],
            'seed': seed,
        }
        
        return trajectory, info
    
    @torch.no_grad()
    def generate_batch(
        self, 
        orient_ids: List[int], 
        dist_ids: List[int],
        target_lengths: Optional[List[int]] = None
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate multiple trajectories.
        
        Note: Due to variable lengths, returns a list of trajectories
        rather than a single array.
        
        Args:
            orient_ids: List of orientation IDs
            dist_ids: List of distance group IDs
            target_lengths: Optional list of target lengths
            
        Returns:
            trajectories: List of [T_i, 2] arrays
            lengths: List of trajectory lengths
        """
        B = len(orient_ids)
        
        # Determine lengths
        if target_lengths is None:
            target_lengths = [
                self.sample_length(o, d) 
                for o, d in zip(orient_ids, dist_ids)
            ]
        
        # Get anchors
        anchors = torch.stack([
            self.get_anchor(o, d) 
            for o, d in zip(orient_ids, dist_ids)
        ])  # [B, K]
        
        # Prepare condition tensors
        orient_tensor = torch.tensor(orient_ids, device=self.device)
        dist_tensor = torch.tensor(dist_ids, device=self.device)
        
        # Initialize from noise
        r_t = torch.randn(B, self.K, device=self.device)
        
        # DDIM denoising loop
        for step in reversed(range(self.num_steps)):
            t = torch.full((B,), step, device=self.device)
            noise_pred = self.model(r_t, t, orient_tensor, dist_tensor, anchors)
            r_t = self.noise_schedule.ddim_step(r_t, noise_pred, t)
        
        # Coefficients
        coefficients = (anchors + r_t).cpu().numpy()  # [B, K]
        
        # FIXED: Reconstruct each trajectory via control points
        trajectories = []
        for i in range(B):
            T = target_lengths[i]

            # Step 1: Reconstruct control points
            cp_recon = self.U_ref @ coefficients[i]
            if self.mean is not None:
                cp_recon = cp_recon + self.mean

            # Step 2: Interpolate to target length
            C = build_interleaved_basis_matrix(T, self.T_ref)
            traj_flat = C @ cp_recon

            traj = np.column_stack([traj_flat[0::2], traj_flat[1::2]])
            trajectories.append(traj)

        return trajectories, target_lengths
    
    def generate_for_all_groups(self, samples_per_group: int = 10) -> Dict:
        """
        Generate samples for all 40 (distance × orientation) groups.
        
        Returns dict mapping group_name → list of trajectories
        """
        results = {}
        
        print(f"\nGenerating {samples_per_group} samples per group...")
        
        for dist_id, dist_name in enumerate(DISTANCE_GROUPS):
            for orient_id, orient_name in enumerate(ORIENTATIONS):
                key = f"{dist_name}-{orient_name}"
                trajectories = []
                lengths = []
                
                for i in range(samples_per_group):
                    traj, info = self.generate_single(
                        orient_id, dist_id, 
                        seed=i * 1000 + dist_id * 100 + orient_id
                    )
                    trajectories.append(traj)
                    lengths.append(info['target_length'])
                
                results[key] = trajectories
                
                mean_len = np.mean(lengths)
                print(f"  {key}: {samples_per_group} trajectories, mean length={mean_len:.0f}")
        
        return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_single_trajectory(
    trajectory: np.ndarray, 
    title: str = "Generated Trajectory",
    info: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize a single trajectory."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.8)
    
    # Mark start and end
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, 
              marker='o', zorder=5, label='Start', edgecolors='black')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, 
              marker='X', zorder=5, label='End', edgecolors='black')
    
    # Add info to title
    if info is not None:
        title += f"\n{info['dist_name']}-{info['orient_name']}, Length={info['target_length']}"
    
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


def visualize_all_groups(
    generator: TrajectoryGeneratorV3, 
    samples_per_group: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize generated trajectories for all 40 groups."""
    fig, axes = plt.subplots(5, 8, figsize=(24, 15))
    
    for dist_id, dist_name in enumerate(DISTANCE_GROUPS):
        for orient_id, orient_name in enumerate(ORIENTATIONS):
            ax = axes[dist_id, orient_id]
            
            lengths = []
            for i in range(samples_per_group):
                traj, info = generator.generate_single(
                    orient_id, dist_id,
                    seed=i * 1000 + dist_id * 100 + orient_id
                )
                color = ORIENTATION_COLORS[orient_name]
                ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, linewidth=1.5)
                lengths.append(info['target_length'])
            
            # Mark origin
            ax.scatter(0, 0, c='black', s=50, marker='o', zorder=10)
            
            # Title with mean length
            mean_len = np.mean(lengths)
            ax.set_title(f'{dist_name}-{orient_name}\n(L≈{mean_len:.0f})', fontsize=9)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
    
    plt.suptitle('Generated Trajectories - V3 (Basis-Transformation)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()
    return fig


def visualize_length_comparison(
    generator: TrajectoryGeneratorV3,
    orient_id: int = 2,  # E
    dist_id: int = 2,    # Medium
    lengths: List[int] = [30, 50, 80, 120],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the same trajectory at different lengths.
    
    Demonstrates V3's ability to generate at arbitrary lengths
    using the same coefficients.
    """
    fig, axes = plt.subplots(1, len(lengths), figsize=(5 * len(lengths), 5))
    
    # Generate one set of coefficients
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get anchor and generate residual once
    anchor = generator.get_anchor(orient_id, dist_id, anchor_idx=0)
    anchor = anchor.unsqueeze(0)
    
    orient_tensor = torch.tensor([orient_id], device=generator.device)
    dist_tensor = torch.tensor([dist_id], device=generator.device)
    
    r_t = torch.randn(1, generator.K, device=generator.device)
    
    for step in reversed(range(generator.num_steps)):
        t = torch.tensor([step], device=generator.device)
        noise_pred = generator.model(r_t, t, orient_tensor, dist_tensor, anchor)
        r_t = generator.noise_schedule.ddim_step(r_t, noise_pred, t)
    
    coefficients = (anchor.squeeze(0) + r_t.squeeze(0)).cpu().numpy()
    
    # Reconstruct at different lengths
    for ax, T in zip(axes, lengths):
        traj_flat = generator.adapter.reconstruct(coefficients, T)
        if generator.mean is not None:
            mean_adapted = generator.adapter.reconstruct_mean(generator.mean, T)
            traj_flat = traj_flat + mean_adapted
        
        traj = np.column_stack([traj_flat[0::2], traj_flat[1::2]])
        
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=80, marker='o', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=80, marker='X', zorder=5)
        ax.scatter(0, 0, c='black', s=50, marker='+', zorder=10)
        
        ax.set_title(f'Length = {T}', fontsize=12)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    group_name = f"{DISTANCE_GROUPS[dist_id]}-{ORIENTATIONS[orient_id]}"
    plt.suptitle(f'Same Coefficients at Different Lengths ({group_name})', fontsize=14)
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
        description='Generate trajectories using V3 (Basis-Transformation) approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate for specific group
    python generate_trajectory_v3.py --checkpoint checkpoints/best_model.pt \\
        --data processed_v3/ --orient E --dist Medium --visualize
    
    # Generate for all groups
    python generate_trajectory_v3.py --checkpoint checkpoints/best_model.pt \\
        --data processed_v3/ --all_groups --samples 20 --visualize
    
    # Generate with specific length
    python generate_trajectory_v3.py --checkpoint checkpoints/best_model.pt \\
        --data processed_v3/ --orient N --dist Large --length 100
"""
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to preprocessed V3 data directory')
    parser.add_argument('--output', type=str, default='generated_v3',
                        help='Output directory')
    
    # Generation options
    parser.add_argument('--orient', type=str, default=None,
                        help='Orientation (N, NE, E, SE, S, SW, W, NW)')
    parser.add_argument('--dist', type=str, default=None,
                        help='Distance group (XSmall, Small, Medium, Large, XLarge)')
    parser.add_argument('--length', type=int, default=None,
                        help='Specific trajectory length (default: sample from distribution)')
    parser.add_argument('--all_groups', action='store_true',
                        help='Generate for all 40 groups')
    parser.add_argument('--samples', type=int, default=10,
                        help='Samples per group')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--length_demo', action='store_true',
                        help='Show same coefficients at different lengths')
    
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRAJECTORY GENERATION - VERSION 3 (BASIS-TRANSFORMATION)")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create generator
    print("\nInitializing generator...")
    generator = TrajectoryGeneratorV3(args.checkpoint, args.data, device=args.device)
    
    # Generate trajectories
    if args.all_groups or (args.orient is None and args.dist is None):
        print(f"\nGenerating trajectories for all 40 groups...")
        
        if args.visualize:
            vis_path = output_dir / 'all_groups_v3.png'
            visualize_all_groups(generator, samples_per_group=args.samples, save_path=str(vis_path))
        
        results = generator.generate_for_all_groups(samples_per_group=args.samples)
        np.save(output_dir / 'generated_trajectories.npy', results, allow_pickle=True)
        print(f"\nSaved trajectories to {output_dir / 'generated_trajectories.npy'}")
        
    elif args.orient and args.dist:
        print(f"\nGenerating trajectory: {args.dist}-{args.orient}")
        
        trajectory, info = generator.generate_single(
            args.orient, args.dist,
            seed=args.seed,
            target_length=args.length
        )
        
        print(f"\n  Generated trajectory:")
        print(f"    Shape: {trajectory.shape}")
        print(f"    Length: {info['target_length']} points")
        print(f"    Start: [{trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f}]")
        print(f"    End: [{trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f}]")
        
        # Compute path length
        diffs = np.diff(trajectory, axis=0)
        path_length = np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2))
        endpoint_dist = np.sqrt(trajectory[-1, 0]**2 + trajectory[-1, 1]**2)
        print(f"    Path length: {path_length:.2f} pixels")
        print(f"    Endpoint distance: {endpoint_dist:.2f} pixels")
        
        if args.visualize:
            vis_path = output_dir / f'trajectory_{args.dist}_{args.orient}.png'
            visualize_single_trajectory(
                trajectory, 
                f'Generated: {args.dist}-{args.orient}',
                info=info, 
                save_path=str(vis_path)
            )
        
        np.save(output_dir / f'trajectory_{args.dist}_{args.orient}.npy', trajectory)
        
    else:
        print("\nERROR: Please specify --orient and --dist, or use --all_groups")
        return
    
    # Optional: Length demo visualization
    if args.length_demo:
        print("\nGenerating length comparison visualization...")
        vis_path = output_dir / 'length_comparison.png'
        visualize_length_comparison(generator, save_path=str(vis_path))
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == '__main__':
    main()
