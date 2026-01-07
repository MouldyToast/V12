#!/usr/bin/env python3
"""
Trajectory Splitter - FIXED VERSION

Fixes the critical timing bug in trajectory_splitter_adaptive.py.

THE BUG:
  Old logic kept points based on distance from last KEPT point.
  This created fake timing gaps (e.g., 358ms instead of 2ms).

THE FIX:
  1. Find where movement STARTS (first position change)
  2. Find where movement ENDS (last position change)
  3. Keep the point just before first movement (starting position)
  4. Keep ALL points from start to end (including zero-movement intervals)
  5. Remove only leading and trailing stationary periods

This preserves:
  - True 500Hz timing (dt always ~2ms)
  - Zero-movement intervals during movement (real pixel quantization)
  - Mid-trajectory pauses (real human hesitation)

  ****CURRENT**** 
  STEP 1 - Good data start_idx = first_move_idx - 1
  STEP 2 - Good data start_idx = first_move_idx - 2
  STEP 3 - python reconstruct_trajectories.py trajectories/ trajectories_reconstructed/ --verbose

Usage:
    python trajectory_splitter_adaptive.py combined_all_sessions.csv trajectories/
"""

import csv
import json
import os
import math
import numpy as np
from pathlib import Path
import argparse


def extract_movement_period(x_list, y_list, t_list):
    """
    Extract the movement period, removing only leading and trailing stationary.
    
    Keeps ALL points during movement, including zero-movement intervals.
    This preserves true 500Hz timing throughout.
    
    Args:
        x_list: List of x coordinates
        y_list: List of y coordinates  
        t_list: List of timestamps in ms
        
    Returns:
        extracted_x, extracted_y, extracted_t
    """
    if len(x_list) < 2:
        return x_list, y_list, t_list
    
    # Find first movement (first position change)
    first_move_idx = None
    for i in range(1, len(x_list)):
        if x_list[i] != x_list[i-1] or y_list[i] != y_list[i-1]:
            first_move_idx = i
            break
    
    # If no movement found, return empty
    if first_move_idx is None:
        return [], [], []
    
    # Find last movement
    last_move_idx = first_move_idx
    for i in range(first_move_idx, len(x_list)):
        if x_list[i] != x_list[i-1] or y_list[i] != y_list[i-1]:
            last_move_idx = i
    
    # Start from point BEFORE first movement (starting position)
    # End at last movement (inclusive)
    start_idx = first_move_idx - 2
    end_idx = last_move_idx + 2  # +1 because slice is exclusive
    
    return x_list[start_idx:end_idx], y_list[start_idx:end_idx], t_list[start_idx:end_idx]


def split_and_extract_trajectories(csv_path, output_dir="trajectories",
                                   min_moves=1):
    """
    Split CSV into trajectories and extract movement periods.
    
    Args:
        csv_path: Path to SapiRecorder CSV
        output_dir: Output directory for trajectory JSON files
        min_moves: Minimum movement points to consider valid trajectory
        
    Returns:
        Number of trajectories extracted, stats dict
    """
    
    print(f"Reading CSV: {csv_path}")
    
    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        events = list(reader)
    
    print(f"Total events: {len(events):,}")
    
    # Split on click events (Released)
    trajectories_raw = []
    current_trajectory = []
    
    for event in events:
        if event['state'] == 'Move':
            current_trajectory.append({
                'x': int(event['x']),
                'y': int(event['y']),
                't': int(event['client timestamp'])
            })
        
        elif event['state'] == 'Released':
            # End current trajectory
            if len(current_trajectory) >= min_moves:
                trajectories_raw.append(current_trajectory)
            current_trajectory = []
    
    # Add final trajectory if exists
    if len(current_trajectory) >= min_moves:
        trajectories_raw.append(current_trajectory)
    
    print(f"Extracted {len(trajectories_raw)} raw trajectories")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each trajectory
    stats = {
        'original_points_total': 0,
        'extracted_points_total': 0,
        'leading_removed_total': 0,
        'trailing_removed_total': 0,
        'trajectories_saved': 0,
        'trajectories_skipped': 0,
    }
    
    saved_count = 0

    skipped_details = []
    
    for i, traj_raw in enumerate(trajectories_raw, 1):
        # Extract coordinates and timestamps
        x_list = [point['x'] for point in traj_raw]
        y_list = [point['y'] for point in traj_raw]
        t_list = [point['t'] for point in traj_raw]
        
        original_len = len(x_list)
        stats['original_points_total'] += original_len

        start_x, start_y = x_list[0], y_list[0]
        end_x, end_y = x_list[-1], y_list[-1]
        total_distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Extract movement period (THE FIX)
        x_ext, y_ext, t_ext = extract_movement_period(x_list, y_list, t_list)
        
        # Skip if too short after extraction
        if len(x_ext) < min_moves:
            stats['trajectories_skipped'] += 1

            skipped_details.append({
                'trajectory_id': i,
                'original_length': original_len,
                'extracted_length': len(x_ext),
                'total_distance': total_distance,
                'duration_ms': t_list[-1] - t_list[0] if len(t_list) > 1 else 0
            })

            continue
        
        extracted_len = len(x_ext)
        stats['extracted_points_total'] += extracted_len
        
        # Count what was removed
        # Find first/last movement indices to calculate removed points
        first_move_idx = None
        last_move_idx = None
        for j in range(1, len(x_list)):
            if x_list[j] != x_list[j-1] or y_list[j] != y_list[j-1]:
                if first_move_idx is None:
                    first_move_idx = j
                last_move_idx = j
        
        leading_removed = first_move_idx - 1 if first_move_idx else 0
        trailing_removed = original_len - last_move_idx - 1 if last_move_idx else 0
        stats['leading_removed_total'] += leading_removed
        stats['trailing_removed_total'] += trailing_removed
        
        # Calculate metadata
        start_x, start_y = x_ext[0], y_ext[0]
        end_x, end_y = x_ext[-1], y_ext[-1]
        
        # Ideal distance (straight line)
        ideal_distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Actual distance (sum of all movements)
        actual_distance = 0.0
        for j in range(1, len(x_ext)):
            dx = x_ext[j] - x_ext[j-1]
            dy = y_ext[j] - y_ext[j-1]
            actual_distance += math.sqrt(dx**2 + dy**2)
        
        # Create trajectory JSON
        trajectory = {
            'x': x_ext,
            'y': y_ext,
            't': t_ext,
            'ideal_distance': ideal_distance,
            'actual_distance': actual_distance,
            'original_length': original_len,
            'extracted_length': extracted_len,
            'leading_removed': leading_removed,
            'trailing_removed': trailing_removed,
            'processing': 'Movement period extraction (leading/trailing stationary removed, all else preserved)'
        }
        
        # Save
        output_path = os.path.join(output_dir, f'trajectory_{i:04d}.json')
        with open(output_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        saved_count += 1
        
        # Progress
        if saved_count % 100 == 0:
            print(f"  Processed {saved_count} trajectories...")
    
    stats['trajectories_saved'] = saved_count
    
    if skipped_details:
        print("\n" + "="*70)
        print(f"SKIPPED TRAJECTORIES ANALYSIS ({len(skipped_details)} total)")
        print("="*70)
        print(f"{'ID':<6} {'Original':<10} {'Extracted':<10} {'Distance':<12} {'Duration':<10}")
        print("-"*70)
        
        for detail in skipped_details[:20]:  # Show first 20
            print(f"{detail['trajectory_id']:<6} "
                  f"{detail['original_length']:<10} "
                  f"{detail['extracted_length']:<10} "
                  f"{detail['total_distance']:<12.1f} "
                  f"{detail['duration_ms']:<10}")
        
        if len(skipped_details) > 20:
            print(f"... and {len(skipped_details) - 20} more")
        
        # Statistics
        distances = [d['total_distance'] for d in skipped_details]
        durations = [d['duration_ms'] for d in skipped_details]
        extracted = [d['extracted_length'] for d in skipped_details]
        
        print(f"\nSkipped trajectories statistics:")
        print(f"  Distance: min={min(distances):.1f}px, max={max(distances):.1f}px, mean={np.mean(distances):.1f}px")
        print(f"  Duration: min={min(durations)}ms, max={max(durations)}ms, mean={np.mean(durations):.0f}ms")
        print(f"  Extracted points: min={min(extracted)}, max={max(extracted)}, mean={np.mean(extracted):.1f}")
        print("="*70)

    return saved_count, stats


def verify_timing(output_dir, num_samples=5):
    """Verify that extracted trajectories have correct timing."""
    print(f"\n{'='*70}")
    print("VERIFICATION: Checking timing consistency")
    print(f"{'='*70}")
    
    json_files = sorted(Path(output_dir).glob('*.json'))[:num_samples]
    
    all_good = True
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        t = np.array(data['t'])
        x = np.array(data['x'])
        y = np.array(data['y'])
        
        dt = np.diff(t)
        dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        
        # Check timing consistency (should be ~2ms throughout)
        max_dt = dt.max()
        min_dt = dt.min()
        
        # Count zero-movement intervals
        zero_movement = (dist == 0).sum()
        
        status = "✓" if max_dt <= 5 else "✗"  # Allow up to 5ms for timing jitter
        if max_dt > 5:
            all_good = False
        
        print(f"\n{json_file.name}:")
        print(f"  Points: {len(t)} (original: {data['original_length']})")
        print(f"  Removed: {data['leading_removed']} leading, {data['trailing_removed']} trailing")
        print(f"  dt range: {min_dt:.0f}ms to {max_dt:.0f}ms {status}")
        print(f"  Zero-movement intervals: {zero_movement} (pixel quantization + pauses)")
    
    if all_good:
        print(f"\n✓ All trajectories have consistent ~2ms timing!")
    else:
        print(f"\n✗ Some trajectories have timing issues - please check!")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(
        description='Split CSV and extract movement periods (FIXED VERSION)'
    )
    parser.add_argument(
        'csv_path',
        help='Path to SapiRecorder CSV file'
    )
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='trajectories',
        help='Output directory (default: raw_trajectories_adaptive)'
    )
    parser.add_argument(
        '--min-moves',
        type=int,
        default=5,
        help='Minimum movement points per trajectory (default: 10)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify timing after extraction'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRAJECTORY SPLITTER - FIXED VERSION")
    print("=" * 70)
    print("""
This version FIXES the timing bug in trajectory_splitter_adaptive.py.

THE BUG (old version):
  - Kept points based on distance from last KEPT point
  - Created fake timing gaps (e.g., 358ms instead of 2ms)
  - Corrupted speed/acceleration calculations

THE FIX (this version):
  - Only removes leading stationary (before movement starts)
  - Only removes trailing stationary (after movement ends)
  - Keeps ALL points during movement (including zero-movement)
  - Preserves true 500Hz timing (dt always ~2ms)
""")
    print("=" * 70 + "\n")
    
    count, stats = split_and_extract_trajectories(
        args.csv_path,
        args.output_dir,
        min_moves=args.min_moves
    )
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nTrajectories saved: {count}")
    print(f"Trajectories skipped (too short): {stats['trajectories_skipped']}")
    print(f"Output directory: {args.output_dir}/")
    
    print(f"\nPoints summary:")
    print(f"  Original total: {stats['original_points_total']:,}")
    print(f"  After extraction: {stats['extracted_points_total']:,}")
    print(f"  Leading stationary removed: {stats['leading_removed_total']:,}")
    print(f"  Trailing stationary removed: {stats['trailing_removed_total']:,}")
    
    if args.verify or True:  # Always verify
        verify_timing(args.output_dir)
    
    print(f"\nNext steps:")
    print(f"  1. Run preprocessing:")
    print(f"     python preprocess_v4.py --input_dir {args.output_dir}")
    print(f"  2. Train model:")
    print(f"     python train_v4.py")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
