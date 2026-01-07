#!/usr/bin/env python3
"""
Validate recorded trajectories for training data quality.

Checks:
- Start velocity below threshold (started from rest)
- End velocity below threshold (ended at rest)
- Plots trajectories with velocity overlay

Usage:
    python validate_trajectories.py recorded_trajectories/
    python validate_trajectories.py recorded_trajectories/ --threshold 50
"""

import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def load_trajectory(filepath: Path) -> dict:
    """Load a trajectory JSON file."""
    with open(filepath) as f:
        return json.load(f)


def compute_velocities(x: List[float], y: List[float],
                       timestamps: List[float]) -> List[float]:
    """Compute velocity at each point."""
    velocities = [0.0]  # First point has no velocity
    for i in range(1, len(x)):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0:
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            v = math.sqrt(dx*dx + dy*dy) / dt
            velocities.append(v)
        else:
            velocities.append(velocities[-1])
    return velocities


def validate_trajectory(traj: dict, threshold: float) -> Tuple[bool, float, float, str]:
    """
    Validate a single trajectory.
    Returns: (is_valid, start_velocity, end_velocity, reason)
    """
    x, y = traj['x'], traj['y']
    timestamps = traj['timestamps']

    if len(x) < 3:
        return False, 0, 0, "Too few points"

    velocities = compute_velocities(x, y, timestamps)

    # Use average of first/last 3 points for stability
    start_vel = np.mean(velocities[:3]) if len(velocities) >= 3 else velocities[0]
    end_vel = np.mean(velocities[-3:]) if len(velocities) >= 3 else velocities[-1]

    if start_vel > threshold:
        return False, start_vel, end_vel, f"Start velocity too high ({start_vel:.1f} px/s)"
    if end_vel > threshold:
        return False, start_vel, end_vel, f"End velocity too high ({end_vel:.1f} px/s)"

    return True, start_vel, end_vel, "OK"


def plot_trajectory(traj: dict, ax, title: str = ""):
    """Plot a single trajectory with velocity coloring."""
    x, y = traj['x'], traj['y']
    timestamps = traj['timestamps']
    velocities = compute_velocities(x, y, timestamps)

    # Normalize velocities for coloring
    v_arr = np.array(velocities)
    v_norm = v_arr / (v_arr.max() + 1e-6)

    # Plot segments colored by velocity
    for i in range(len(x) - 1):
        color = plt.cm.coolwarm(v_norm[i])
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, linewidth=2)

    # Mark start and end
    ax.scatter([x[0]], [y[0]], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([x[-1]], [y[-1]], c='red', s=100, marker='x', label='End', zorder=5)

    # Mark target if available
    if 'target' in traj:
        ax.scatter([traj['target'][0]], [traj['target'][1]],
                   c='blue', s=150, marker='*', label='Target', zorder=5)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Screen coordinates
    ax.legend(loc='upper right', fontsize=8)


def main():
    parser = argparse.ArgumentParser(description='Validate trajectory recordings')
    parser.add_argument('directory', type=Path, help='Directory with trajectory JSON files')
    parser.add_argument('--threshold', '-t', type=float, default=100.0,
                        help='Max velocity threshold for start/end (px/s, default: 100)')
    parser.add_argument('--plot', '-p', type=int, default=0,
                        help='Number of trajectories to plot (0=none, -1=all failed)')
    parser.add_argument('--plot-all', action='store_true', help='Plot all trajectories')

    args = parser.parse_args()

    # Find trajectory files
    files = sorted(args.directory.glob('traj_*.json'))
    if not files:
        print(f"No trajectory files found in {args.directory}")
        return

    print(f"\n{'='*60}")
    print(f"TRAJECTORY VALIDATION")
    print(f"{'='*60}")
    print(f"Directory: {args.directory}")
    print(f"Files: {len(files)}")
    print(f"Velocity threshold: {args.threshold} px/s")
    print("="*60 + "\n")

    # Validate all
    valid_count = 0
    invalid_count = 0
    failed_trajectories = []
    all_start_vels = []
    all_end_vels = []

    for f in files:
        traj = load_trajectory(f)
        is_valid, start_v, end_v, reason = validate_trajectory(traj, args.threshold)
        all_start_vels.append(start_v)
        all_end_vels.append(end_v)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            failed_trajectories.append((f, traj, reason))
            print(f"  FAIL: {f.name} - {reason}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Valid:   {valid_count}/{len(files)} ({100*valid_count/len(files):.1f}%)")
    print(f"Invalid: {invalid_count}/{len(files)}")
    print(f"\nStart velocity - mean: {np.mean(all_start_vels):.1f}, max: {np.max(all_start_vels):.1f} px/s")
    print(f"End velocity   - mean: {np.mean(all_end_vels):.1f}, max: {np.max(all_end_vels):.1f} px/s")
    print("="*60 + "\n")

    # Plotting
    to_plot = []
    if args.plot_all:
        to_plot = [(f, load_trajectory(f), "OK") for f in files]
    elif args.plot == -1:
        to_plot = failed_trajectories
    elif args.plot > 0:
        # Plot first N trajectories
        to_plot = [(f, load_trajectory(f), "Sample") for f in files[:args.plot]]

    if to_plot:
        n = len(to_plot)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (filepath, traj, reason) in enumerate(to_plot):
            plot_trajectory(traj, axes[i], f"{filepath.name}\n{reason}")

        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f"Trajectory Validation (threshold: {args.threshold} px/s)", y=1.02)
        plt.show()


if __name__ == '__main__':
    main()
