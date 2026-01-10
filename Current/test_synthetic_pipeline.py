#!/usr/bin/env python3
"""
Synthetic End-to-End Pipeline Test

This test runs the complete pipeline:
1. Generate synthetic trajectories
2. Preprocess with preprocess_singular_v3_basis_transform.py
3. Train with train_singular_diffusion_v1.py for a few epochs
4. Generate with generate_trajectory_v3.py

This catches:
- Config mismatches between modules
- Runtime errors in training
- Generation issues
"""

import numpy as np
import json
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# Settings
N_TRAJECTORIES = 300
N_EPOCHS = 3
BATCH_SIZE = 64


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def generate_synthetic_trajectory(
    length: int,
    ideal_distance: float,
    orientation_angle: float,
    jitter: float = 1.5,
    seed: int = None
) -> dict:
    """Generate a synthetic mouse trajectory in JSON format."""
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 1, length)
    eased_t = 1 - (1 - t) ** 2

    dx = ideal_distance * np.cos(orientation_angle)
    dy = ideal_distance * np.sin(orientation_angle)

    x = eased_t * dx
    y = eased_t * dy

    curve_amount = np.random.randn() * ideal_distance * 0.03
    curve = curve_amount * np.sin(t * np.pi)
    x += curve * np.sin(orientation_angle + np.pi/2)
    y += curve * np.cos(orientation_angle + np.pi/2)

    if jitter > 0:
        jitter_x = gaussian_filter1d(np.random.randn(length) * jitter, sigma=1.0)
        jitter_y = gaussian_filter1d(np.random.randn(length) * jitter, sigma=1.0)
        x += jitter_x
        y += jitter_y

    x = x + 500
    y = y + 500
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    timestamps = np.cumsum(np.random.randint(6, 12, size=length))
    actual_distance = np.sum(np.sqrt(np.diff(x.astype(float))**2 +
                                     np.diff(y.astype(float))**2))

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        't': timestamps.tolist(),
        'ideal_distance': float(ideal_distance),
        'actual_distance': float(actual_distance),
        'original_length': length,
        'extracted_length': length,
    }


def generate_diverse_dataset(n_trajectories: int, seed: int = 42) -> list:
    """Generate a diverse dataset covering all groups."""
    np.random.seed(seed)
    trajectories = []

    ORIENTATIONS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    DISTANCE_GROUPS = [
        {'name': 'XSmall', 'min': 25, 'max': 60},
        {'name': 'Small', 'min': 60, 'max': 120},
        {'name': 'Medium', 'min': 120, 'max': 200},
        {'name': 'Large', 'min': 200, 'max': 320},
        {'name': 'XLarge', 'min': 320, 'max': 500},
    ]

    angle_map = {
        'E': 0, 'NE': 45, 'N': 90, 'NW': 135,
        'W': 180, 'SW': 225, 'S': 270, 'SE': 315
    }

    # Ensure coverage of all groups
    for dist_group in DISTANCE_GROUPS:
        for orient_name in ORIENTATIONS:
            n_per_group = max(2, n_trajectories // (5 * 8))

            angle_rad = np.radians(angle_map[orient_name])

            for i in range(n_per_group):
                length = np.random.randint(25, 90)
                dist_min = dist_group['min']
                dist_max = dist_group['max']
                distance = dist_min + np.random.rand() * (dist_max - dist_min) * 0.9
                distance = max(25, distance)

                angle_var = angle_rad + np.radians(np.random.randn() * 5)
                jitter = 0.5 + np.random.rand() * 2.0

                traj = generate_synthetic_trajectory(
                    length=length,
                    ideal_distance=distance,
                    orientation_angle=angle_var,
                    jitter=jitter,
                    seed=seed + len(trajectories)
                )
                trajectories.append(traj)

    np.random.shuffle(trajectories)
    return trajectories


def save_trajectories(trajectories: list, output_dir: Path):
    """Save trajectories as JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        filepath = output_dir / f"trajectory_{i:04d}.json"
        with open(filepath, 'w') as f:
            json.dump(traj, f)


def run_command(cmd: list, description: str, cwd: Path = None) -> bool:
    """Run a command and return success status."""
    print(f"\n  Running: {' '.join(cmd[:4])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )

        if result.returncode != 0:
            print(f"\n  FAILED: {description}")
            print(f"  Return code: {result.returncode}")
            print(f"\n  STDOUT:\n{result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
            print(f"\n  STDERR:\n{result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr}")
            return False

        # Show last few lines of output
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) > 10:
            print("  ...")
        for line in output_lines[-10:]:
            print(f"  {line}")

        return True

    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"\n  ERROR: {description}")
        print(f"  {e}")
        return False


def test_pipeline():
    """Run the full pipeline test."""
    print_section("SYNTHETIC PIPELINE TEST")

    # Create temp directories
    temp_dir = Path(tempfile.mkdtemp())
    raw_data_dir = temp_dir / "raw_trajectories"
    processed_dir = temp_dir / "processed_v3"
    checkpoint_dir = temp_dir / "checkpoints"
    generated_dir = temp_dir / "generated"

    current_dir = Path(__file__).parent

    print(f"\n  Temp directory: {temp_dir}")
    print(f"  Current directory: {current_dir}")

    success = True

    try:
        # Step 1: Generate synthetic data
        print_section("STEP 1: Generate Synthetic Data")
        print(f"  Generating {N_TRAJECTORIES} trajectories...")
        trajectories = generate_diverse_dataset(N_TRAJECTORIES, seed=42)
        save_trajectories(trajectories, raw_data_dir)
        print(f"  Saved to {raw_data_dir}")

        # Step 2: Run preprocessing
        print_section("STEP 2: Preprocess Data")
        preprocess_cmd = [
            sys.executable,
            str(current_dir / "preprocess_singular_v3_basis_transform.py"),
            "--input", str(raw_data_dir),
            "--output", str(processed_dir),
            "--n_control_points", "16",
            "--k", "8",
            "--min_length", "20",
            "--anchors_per_group", "4",
            "--seed", "42"
        ]

        if not run_command(preprocess_cmd, "Preprocessing"):
            success = False
            return success

        # Verify preprocessing output
        print("\n  Verifying preprocessing output...")
        required_files = ['config.npy', 'U_ref.npy', 'group_anchors.npy']
        for f in required_files:
            if not (processed_dir / f).exists():
                print(f"  MISSING: {f}")
                success = False
            else:
                print(f"  OK: {f}")

        # Check config has required keys
        config = np.load(processed_dir / 'config.npy', allow_pickle=True).item()
        required_keys = ['K', 'T_win', 'T_ref', 'num_orientations', 'num_distance_groups']
        for key in required_keys:
            if key not in config:
                print(f"  MISSING CONFIG KEY: {key}")
                success = False
            else:
                print(f"  CONFIG: {key} = {config[key]}")

        if not success:
            return success

        # Step 3: Run training (few epochs)
        print_section("STEP 3: Train Model")
        train_cmd = [
            sys.executable,
            str(current_dir / "train_singular_diffusion_v1.py"),
            "--data", str(processed_dir),
            "--output", str(checkpoint_dir),
            "--epochs", str(N_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--device", "cpu"  # Use CPU for testing
        ]

        if not run_command(train_cmd, "Training"):
            success = False
            return success

        # Verify training output
        print("\n  Verifying training output...")
        if not (checkpoint_dir / "latest.pt").exists():
            print("  MISSING: latest.pt")
            success = False
        else:
            print("  OK: latest.pt")

            # Check checkpoint contents
            import torch
            checkpoint = torch.load(checkpoint_dir / "latest.pt", map_location='cpu', weights_only=False)
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
            print(f"  Checkpoint K: {checkpoint['config']['K']}")

        if not success:
            return success

        # Step 4: Test generation
        print_section("STEP 4: Generate Trajectories")
        generate_cmd = [
            sys.executable,
            str(current_dir / "generate_trajectory_v3.py"),
            "--checkpoint", str(checkpoint_dir / "latest.pt"),
            "--data", str(processed_dir),
            "--output", str(generated_dir),
            "--orient", "E",
            "--dist", "Medium",
            "--samples", "5",
            "--device", "cpu"
        ]

        if not run_command(generate_cmd, "Generation"):
            success = False
            return success

        # Verify generation output
        print("\n  Verifying generation output...")
        npy_files = list(generated_dir.glob("*.npy"))
        if len(npy_files) == 0:
            print("  MISSING: No trajectory files generated")
            success = False
        else:
            for f in npy_files:
                traj = np.load(f)
                print(f"  OK: {f.name} - shape {traj.shape}")

        print_section("PIPELINE TEST COMPLETE")
        if success:
            print("  ALL STEPS PASSED!")
        else:
            print("  SOME STEPS FAILED!")

    finally:
        # Cleanup
        print(f"\n  Cleaning up {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    return success


def main():
    print("=" * 70)
    print(" SYNTHETIC END-TO-END PIPELINE TEST")
    print("=" * 70)
    print(f"\n  Settings:")
    print(f"    N_TRAJECTORIES: {N_TRAJECTORIES}")
    print(f"    N_EPOCHS: {N_EPOCHS}")
    print(f"    BATCH_SIZE: {BATCH_SIZE}")

    success = test_pipeline()

    print("\n" + "=" * 70)
    if success:
        print(" ALL TESTS PASSED")
    else:
        print(" TESTS FAILED - SEE ABOVE FOR DETAILS")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
