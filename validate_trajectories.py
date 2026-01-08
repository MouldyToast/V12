#!/usr/bin/env python3
"""
Trajectory Validation - Individual analysis with issue detection.

Creates one PNG per trajectory showing:
- Path visualization with start/end/target markers
- Speed over time graph
- Acceleration over time graph

Checks for issues:
- Start/end velocity (should be near zero if settled)
- Sampling consistency (should be ~8ms intervals)
- Jumps/teleports (large position gaps)
- Target alignment (did path end near target?)
- Jitter (high-frequency noise)
- Duration and point count

Usage:
    python validate_trajectories.py recorded_trajectories/
    python validate_trajectories.py recorded_trajectories/ --output validation_output/
"""

import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


@dataclass
class ValidationConfig:
    # Velocity thresholds (px/s)
    max_start_velocity: float = 50.0
    max_end_velocity: float = 50.0

    # Sampling (seconds)
    expected_interval: float = 0.008  # 8ms
    interval_tolerance: float = 0.004  # +/- 4ms

    # Jump detection
    max_jump_pixels: float = 100.0  # Single frame jump threshold

    # Target alignment
    max_target_distance: float = 50.0  # End point should be within this of target

    # Jitter detection
    jitter_threshold: float = 5.0  # Consecutive direction changes < this = jitter
    max_jitter_ratio: float = 0.3  # More than 30% jitter points = issue

    # Minimum requirements
    min_points: int = 10
    min_duration_ms: float = 50.0


@dataclass
class ValidationResult:
    filename: str
    is_valid: bool
    issues: List[str]

    # Metrics
    point_count: int
    duration_ms: float
    start_velocity: float
    end_velocity: float
    max_velocity: float
    avg_velocity: float
    max_acceleration: float

    # Sampling
    avg_interval_ms: float
    min_interval_ms: float
    max_interval_ms: float

    # Path quality
    target_distance: float
    jitter_ratio: float
    jump_count: int


def load_trajectory(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def get_timestamps_sec(traj: dict) -> List[float]:
    """Get timestamps in seconds. Handles both old and new format."""
    if 't' in traj:
        # New format: 't' is integer ms (0, 8, 16, 24...)
        return [t / 1000.0 for t in traj['t']]
    elif 'timestamps' in traj:
        # Old format: 'timestamps' is absolute seconds, convert to relative
        t0 = traj['timestamps'][0]
        return [t - t0 for t in traj['timestamps']]
    else:
        # Fallback: assume 8ms intervals
        return [i * 0.008 for i in range(len(traj['x']))]


def compute_velocities(x: List[int], y: List[int],
                       t_sec: List[float]) -> np.ndarray:
    """Compute velocity at each point (px/s)."""
    velocities = [0.0]
    for i in range(1, len(x)):
        dt = t_sec[i] - t_sec[i-1]
        if dt > 0:
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            velocities.append(math.sqrt(dx*dx + dy*dy) / dt)
        else:
            velocities.append(0.0)
    return np.array(velocities)


def compute_accelerations(velocities: np.ndarray,
                          t_sec: List[float]) -> np.ndarray:
    """Compute acceleration at each point (px/s^2)."""
    accelerations = [0.0]
    for i in range(1, len(velocities)):
        dt = t_sec[i] - t_sec[i-1]
        if dt > 0:
            accelerations.append((velocities[i] - velocities[i-1]) / dt)
        else:
            accelerations.append(0.0)
    return np.array(accelerations)


def detect_jitter(x: List[float], y: List[float], threshold: float) -> float:
    """Detect jitter (rapid direction changes). Returns ratio of jittery segments."""
    if len(x) < 4:
        return 0.0

    jitter_count = 0
    for i in range(2, len(x)):
        # Vector from i-2 to i-1
        dx1 = x[i-1] - x[i-2]
        dy1 = y[i-1] - y[i-2]
        # Vector from i-1 to i
        dx2 = x[i] - x[i-1]
        dy2 = y[i] - y[i-1]

        # Check if direction reversed (dot product negative) with small movement
        dot = dx1*dx2 + dy1*dy2
        mag1 = math.sqrt(dx1*dx1 + dy1*dy1)
        mag2 = math.sqrt(dx2*dx2 + dy2*dy2)

        if mag1 < threshold and mag2 < threshold and dot < 0:
            jitter_count += 1

    return jitter_count / (len(x) - 2)


def detect_jumps(x: List[float], y: List[float], threshold: float) -> int:
    """Count number of large position jumps."""
    jump_count = 0
    for i in range(1, len(x)):
        dist = math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        if dist > threshold:
            jump_count += 1
    return jump_count


def validate_trajectory(traj: dict, config: ValidationConfig,
                        filename: str) -> ValidationResult:
    """Validate a single trajectory and return detailed results."""
    x, y = traj['x'], traj['y']
    t_sec = get_timestamps_sec(traj)  # Handles both old and new format
    target = traj.get('target', [x[-1], y[-1]])

    issues = []

    # Basic checks
    point_count = len(x)
    if point_count < config.min_points:
        issues.append(f"Too few points: {point_count} < {config.min_points}")

    duration_ms = t_sec[-1] * 1000  # Already relative (starts at 0)
    if duration_ms < config.min_duration_ms:
        issues.append(f"Too short: {duration_ms:.0f}ms < {config.min_duration_ms}ms")

    # Compute velocities and accelerations
    velocities = compute_velocities(x, y, t_sec)
    accelerations = compute_accelerations(velocities, t_sec)

    # Velocity checks (average of first/last 3 points)
    start_vel = np.mean(velocities[:3]) if len(velocities) >= 3 else velocities[0]
    end_vel = np.mean(velocities[-3:]) if len(velocities) >= 3 else velocities[-1]

    if start_vel > config.max_start_velocity:
        issues.append(f"High start velocity: {start_vel:.0f} px/s > {config.max_start_velocity}")
    if end_vel > config.max_end_velocity:
        issues.append(f"High end velocity: {end_vel:.0f} px/s > {config.max_end_velocity}")

    # Sampling interval checks (convert to ms)
    intervals = np.diff(t_sec) * 1000
    avg_interval = np.mean(intervals)
    min_interval = np.min(intervals)
    max_interval = np.max(intervals)

    expected_ms = config.expected_interval * 1000
    tolerance_ms = config.interval_tolerance * 1000

    if avg_interval < expected_ms - tolerance_ms or avg_interval > expected_ms + tolerance_ms:
        issues.append(f"Inconsistent sampling: avg {avg_interval:.1f}ms (expected {expected_ms}ms)")

    if max_interval > expected_ms * 3:
        issues.append(f"Large sample gap: {max_interval:.1f}ms")

    # Target alignment
    target_dist = math.sqrt((x[-1] - target[0])**2 + (y[-1] - target[1])**2)
    if target_dist > config.max_target_distance:
        issues.append(f"Missed target: {target_dist:.0f}px away")

    # Jitter detection
    jitter_ratio = detect_jitter(x, y, config.jitter_threshold)
    if jitter_ratio > config.max_jitter_ratio:
        issues.append(f"High jitter: {jitter_ratio*100:.0f}% of path")

    # Jump detection
    jump_count = detect_jumps(x, y, config.max_jump_pixels)
    if jump_count > 0:
        issues.append(f"Position jumps detected: {jump_count}")

    return ValidationResult(
        filename=filename,
        is_valid=len(issues) == 0,
        issues=issues,
        point_count=point_count,
        duration_ms=duration_ms,
        start_velocity=start_vel,
        end_velocity=end_vel,
        max_velocity=np.max(velocities),
        avg_velocity=np.mean(velocities),
        max_acceleration=np.max(np.abs(accelerations)),
        avg_interval_ms=avg_interval,
        min_interval_ms=min_interval,
        max_interval_ms=max_interval,
        target_distance=target_dist,
        jitter_ratio=jitter_ratio,
        jump_count=jump_count
    )


def create_trajectory_plot(traj: dict, result: ValidationResult,
                           output_path: Path, config: ValidationConfig) -> None:
    """Create a detailed PNG for a single trajectory."""
    x, y = np.array(traj['x']), np.array(traj['y'])
    t_sec = get_timestamps_sec(traj)
    target = traj.get('target', [x[-1], y[-1]])

    # Convert to ms for plotting
    t_ms = np.array(t_sec) * 1000

    velocities = compute_velocities(traj['x'], traj['y'], t_sec)
    accelerations = compute_accelerations(velocities, t_sec)

    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.5, 1, 1], height_ratios=[1, 1])

    # === Path Plot (left, spans both rows) ===
    ax_path = fig.add_subplot(gs[:, 0])

    # Color path by velocity
    v_arr = np.array(velocities)
    v_norm = v_arr / (v_arr.max() + 1e-6)

    for i in range(len(x) - 1):
        color = plt.cm.coolwarm(v_norm[i])
        ax_path.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, linewidth=2)

    # Markers
    ax_path.scatter([x[0]], [y[0]], c='green', s=150, marker='o',
                    label=f'Start (v={result.start_velocity:.0f})', zorder=5, edgecolors='white')
    ax_path.scatter([x[-1]], [y[-1]], c='red', s=150, marker='s',
                    label=f'End (v={result.end_velocity:.0f})', zorder=5, edgecolors='white')
    ax_path.scatter([target[0]], [target[1]], c='blue', s=200, marker='*',
                    label=f'Target ({result.target_distance:.0f}px away)', zorder=5)

    ax_path.set_xlabel('X (pixels)')
    ax_path.set_ylabel('Y (pixels)')
    ax_path.set_aspect('equal')
    ax_path.invert_yaxis()
    ax_path.legend(loc='upper right', fontsize=9)
    ax_path.set_title('Trajectory Path\n(color = velocity)', fontsize=11)
    ax_path.grid(True, alpha=0.3)

    # === Speed Plot (top right) ===
    ax_speed = fig.add_subplot(gs[0, 1:])
    ax_speed.plot(t_ms, velocities, 'b-', linewidth=1.5, label='Speed')
    ax_speed.axhline(y=config.max_start_velocity, color='g', linestyle='--',
                     alpha=0.7, label=f'Start threshold ({config.max_start_velocity})')
    ax_speed.axhline(y=config.max_end_velocity, color='r', linestyle='--',
                     alpha=0.7, label=f'End threshold ({config.max_end_velocity})')

    # Mark start/end regions
    ax_speed.axvspan(0, t_ms[min(3, len(t_ms)-1)], alpha=0.2, color='green', label='Start region')
    ax_speed.axvspan(t_ms[max(0, len(t_ms)-3)], t_ms[-1], alpha=0.2, color='red', label='End region')

    ax_speed.set_xlabel('Time (ms)')
    ax_speed.set_ylabel('Speed (px/s)')
    ax_speed.set_title('Speed Over Time', fontsize=11)
    ax_speed.legend(loc='upper right', fontsize=8)
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_xlim(0, t_ms[-1])

    # === Acceleration Plot (bottom right) ===
    ax_accel = fig.add_subplot(gs[1, 1:])
    ax_accel.plot(t_ms, accelerations, 'r-', linewidth=1.5, alpha=0.8)
    ax_accel.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax_accel.set_xlabel('Time (ms)')
    ax_accel.set_ylabel('Acceleration (px/s²)')
    ax_accel.set_title('Acceleration Over Time', fontsize=11)
    ax_accel.grid(True, alpha=0.3)
    ax_accel.set_xlim(0, t_ms[-1])

    # === Title with status ===
    status = "VALID" if result.is_valid else "ISSUES FOUND"
    status_color = 'green' if result.is_valid else 'red'

    title_lines = [f"{result.filename} - {status}"]
    title_lines.append(f"Points: {result.point_count} | Duration: {result.duration_ms:.0f}ms | "
                       f"Avg interval: {result.avg_interval_ms:.1f}ms")
    if result.issues:
        title_lines.append("Issues: " + "; ".join(result.issues[:3]))

    fig.suptitle('\n'.join(title_lines), fontsize=12, color=status_color,
                 fontweight='bold' if not result.is_valid else 'normal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_summary_plot(results: List[ValidationResult], output_path: Path) -> None:
    """Create a summary plot showing distribution of key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    valid = [r for r in results if r.is_valid]
    invalid = [r for r in results if not r.is_valid]

    # 1. Start velocities
    ax = axes[0, 0]
    start_vels = [r.start_velocity for r in results]
    colors = ['green' if r.is_valid else 'red' for r in results]
    ax.bar(range(len(start_vels)), start_vels, color=colors, alpha=0.7)
    ax.axhline(y=50, color='orange', linestyle='--', label='Threshold')
    ax.set_xlabel('Trajectory')
    ax.set_ylabel('Start Velocity (px/s)')
    ax.set_title('Start Velocities')

    # 2. End velocities
    ax = axes[0, 1]
    end_vels = [r.end_velocity for r in results]
    ax.bar(range(len(end_vels)), end_vels, color=colors, alpha=0.7)
    ax.axhline(y=50, color='orange', linestyle='--', label='Threshold')
    ax.set_xlabel('Trajectory')
    ax.set_ylabel('End Velocity (px/s)')
    ax.set_title('End Velocities')

    # 3. Duration distribution
    ax = axes[0, 2]
    durations = [r.duration_ms for r in results]
    ax.hist(durations, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Duration Distribution')

    # 4. Sample interval distribution
    ax = axes[1, 0]
    intervals = [r.avg_interval_ms for r in results]
    ax.hist(intervals, bins=20, color='teal', alpha=0.7, edgecolor='black')
    ax.axvline(x=8, color='orange', linestyle='--', label='Expected (8ms)')
    ax.set_xlabel('Avg Sample Interval (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Sampling Consistency')
    ax.legend()

    # 5. Target distance
    ax = axes[1, 1]
    target_dists = [r.target_distance for r in results]
    ax.bar(range(len(target_dists)), target_dists, color=colors, alpha=0.7)
    ax.axhline(y=50, color='orange', linestyle='--', label='Threshold')
    ax.set_xlabel('Trajectory')
    ax.set_ylabel('Target Distance (px)')
    ax.set_title('End Position to Target')

    # 6. Valid vs Invalid pie
    ax = axes[1, 2]
    sizes = [len(valid), len(invalid)]
    labels = [f'Valid ({len(valid)})', f'Invalid ({len(invalid)})']
    colors_pie = ['#4CAF50', '#f44336']
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title('Validation Summary')

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Validate trajectory recordings')
    parser.add_argument('directory', type=Path, help='Directory with trajectory JSON files')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output directory for PNGs (default: <directory>/validation/)')
    parser.add_argument('--threshold', '-t', type=float, default=50.0,
                        help='Max start/end velocity threshold (px/s, default: 50)')
    parser.add_argument('--only-invalid', action='store_true',
                        help='Only generate PNGs for invalid trajectories')

    args = parser.parse_args()

    # Setup output directory
    output_dir = args.output or (args.directory / 'validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find trajectory files
    files = sorted(args.directory.glob('traj_*.json'))
    if not files:
        print(f"No trajectory files found in {args.directory}")
        return

    # Configure validation
    config = ValidationConfig(
        max_start_velocity=args.threshold,
        max_end_velocity=args.threshold
    )

    print(f"\n{'='*60}")
    print("TRAJECTORY VALIDATION")
    print(f"{'='*60}")
    print(f"Input:  {args.directory}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(files)}")
    print(f"Velocity threshold: {args.threshold} px/s")
    print("="*60 + "\n")

    # Validate all trajectories
    results = []
    for f in files:
        traj = load_trajectory(f)
        result = validate_trajectory(traj, config, f.name)
        results.append(result)

        # Generate individual PNG
        if not args.only_invalid or not result.is_valid:
            png_path = output_dir / f"{f.stem}.png"
            create_trajectory_plot(traj, result, png_path, config)

        # Print status
        status = "OK" if result.is_valid else "FAIL"
        symbol = "+" if result.is_valid else "X"
        print(f"  [{symbol}] {f.name}: {status}")
        if result.issues:
            for issue in result.issues:
                print(f"      - {issue}")

    # Summary
    valid = [r for r in results if r.is_valid]
    invalid = [r for r in results if not r.is_valid]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Valid:   {len(valid)}/{len(results)} ({100*len(valid)/len(results):.1f}%)")
    print(f"Invalid: {len(invalid)}/{len(results)}")

    if results:
        print(f"\nStart velocity - mean: {np.mean([r.start_velocity for r in results]):.1f}, "
              f"max: {np.max([r.start_velocity for r in results]):.1f} px/s")
        print(f"End velocity   - mean: {np.mean([r.end_velocity for r in results]):.1f}, "
              f"max: {np.max([r.end_velocity for r in results]):.1f} px/s")
        print(f"Avg interval   - mean: {np.mean([r.avg_interval_ms for r in results]):.2f}ms")

    # Generate summary plot
    summary_path = output_dir / "_summary.png"
    create_summary_plot(results, summary_path)
    print(f"\nSummary plot: {summary_path}")
    print(f"Individual PNGs: {output_dir}/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
