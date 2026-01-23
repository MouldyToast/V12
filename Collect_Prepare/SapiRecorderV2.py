#!/usr/bin/env python3
"""
SapiRecorderV2 - Fixed Sample Count Trajectory Recorder
Version: 2.0.0

Records mouse trajectories with fixed sample count for continuous conditioning.
Designed for V4 preprocessing pipeline with (Δx, Δy) conditioning.

Features:
- Fixed 200 samples per trajectory (1600ms at 8ms interval)
- Visual waypoint grid as movement guides
- Press ENTER to start recording (from any position)
- Auto-stops when sample count reached
- Saves JSON with relative coordinates and deltas
- Thread-safe: samples to memory, file I/O on main thread

Usage:
    python SapiRecorderV2.py
    python SapiRecorderV2.py --output trajectories_v4/ --samples 200
"""

import tkinter as tk
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from pynput import mouse
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Recording parameters
    'sample_count': 200,            # Fixed samples per trajectory
    'sample_interval_ms': 8,        # 125Hz sampling rate

    # Screen dimensions
    'screen_width': 2560,
    'screen_height': 1440,

    # Window dimensions (slightly smaller than screen for window borders)
    'window_width': 2500,
    'window_height': 1400,

    # Waypoint grid
    'grid_cols': 8,
    'grid_rows': 5,
    'waypoint_radius': 15,
    'waypoint_color': '#404040',    # Dark gray - visible but not distracting

    # Output
    'output_dir': 'trajectories_v4/',
}


# =============================================================================
# RECORDER CLASS
# =============================================================================

class SapiRecorderV2:
    def __init__(self, master, config):
        self.master = master
        self.config = config
        self.master.title("SapiRecorderV2 - Fixed Sample Recorder")

        # Window setup
        self.width = config['window_width']
        self.height = config['window_height']
        self.master.geometry(f"{self.width}x{self.height}")

        # Canvas
        self.canvas = tk.Canvas(master, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Recording state
        self.state = 'WAITING'  # WAITING, RECORDING, SAVING
        self.samples_x = []
        self.samples_y = []
        self.start_x = 0
        self.start_y = 0
        self.recording_start_time = None

        # Statistics
        self.trajectories_recorded = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Thread safety
        self.canvas_lock = threading.Lock()
        self.canvas_x = 0
        self.canvas_y = 0

        # Sampling thread
        self.mouse_controller = mouse.Controller()
        self.sampling_active = True
        self.sampling_thread = threading.Thread(target=self._sample_loop, daemon=True)

        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Bindings
        self.master.bind("<Return>", self.on_enter_key)
        self.master.bind("<Escape>", lambda e: self.quit_app())
        self.master.protocol("WM_DELETE_WINDOW", self.quit_app)

        # Start canvas position updater
        self.update_canvas_position()

        # Draw initial UI
        self.draw_waypoints()
        self.draw_ui()

        # Start sampling thread
        self.sampling_thread.start()

        # Print startup info
        print("\n" + "=" * 60)
        print("SapiRecorderV2 - Fixed Sample Trajectory Recorder")
        print("=" * 60)
        print(f"Samples per trajectory: {config['sample_count']}")
        print(f"Sample interval: {config['sample_interval_ms']}ms (125Hz)")
        print(f"Duration per trajectory: {config['sample_count'] * config['sample_interval_ms']}ms")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        print("\nPress ENTER to start recording")
        print("Press ESCAPE to quit")
        print("=" * 60)

    def update_canvas_position(self):
        """Update canvas position for coordinate conversion (thread-safe)."""
        try:
            new_x = self.canvas.winfo_rootx()
            new_y = self.canvas.winfo_rooty()
            with self.canvas_lock:
                self.canvas_x = new_x
                self.canvas_y = new_y
        except Exception:
            pass
        self.master.after(100, self.update_canvas_position)

    def draw_waypoints(self):
        """Draw the waypoint grid as visual guides."""
        self.canvas.delete("waypoints")

        cols = self.config['grid_cols']
        rows = self.config['grid_rows']
        radius = self.config['waypoint_radius']
        color = self.config['waypoint_color']

        # Calculate spacing with margins
        margin_x = self.width / (cols + 1)
        margin_y = self.height / (rows + 1)

        for row in range(rows):
            for col in range(cols):
                x = margin_x * (col + 1)
                y = margin_y * (row + 1)

                self.canvas.create_oval(
                    x - radius, y - radius,
                    x + radius, y + radius,
                    fill=color, outline=color,
                    tags="waypoints"
                )

    def draw_ui(self):
        """Draw status text."""
        self.canvas.delete("ui")

        if self.state == 'WAITING':
            # Main instruction
            self.canvas.create_text(
                self.width // 2, 40,
                text="Press ENTER to start recording",
                font=("Arial", 20, "bold"),
                fill="green",
                tags="ui"
            )

            # Counter
            self.canvas.create_text(
                self.width // 2, 80,
                text=f"Trajectories recorded: {self.trajectories_recorded}",
                font=("Arial", 14),
                fill="gray",
                tags="ui"
            )

        elif self.state == 'RECORDING':
            sample_count = len(self.samples_x)
            target = self.config['sample_count']

            # Progress
            self.canvas.create_text(
                self.width // 2, 40,
                text=f"Recording... {sample_count}/{target}",
                font=("Arial", 24, "bold"),
                fill="red",
                tags="ui"
            )

            # Progress bar
            bar_width = 400
            bar_height = 20
            bar_x = (self.width - bar_width) // 2
            bar_y = 70

            # Background
            self.canvas.create_rectangle(
                bar_x, bar_y,
                bar_x + bar_width, bar_y + bar_height,
                fill="#333333", outline="#555555",
                tags="ui"
            )

            # Fill
            fill_width = int(bar_width * sample_count / target)
            if fill_width > 0:
                self.canvas.create_rectangle(
                    bar_x, bar_y,
                    bar_x + fill_width, bar_y + bar_height,
                    fill="red", outline="",
                    tags="ui"
                )

        elif self.state == 'SAVING':
            self.canvas.create_text(
                self.width // 2, 50,
                text=f"Saved! #{self.trajectories_recorded}",
                font=("Arial", 24, "bold"),
                fill="cyan",
                tags="ui"
            )

    def _sample_loop(self):
        """Poll mouse position at 125Hz (thread-safe)."""
        interval = self.config['sample_interval_ms'] / 1000.0
        next_time = time.perf_counter()

        while self.sampling_active:
            if self.state == 'RECORDING':
                try:
                    pos = self.mouse_controller.position

                    with self.canvas_lock:
                        offset_x = self.canvas_x
                        offset_y = self.canvas_y

                    canvas_x = pos[0] - offset_x
                    canvas_y = pos[1] - offset_y

                    # Store sample (relative to start position)
                    rel_x = canvas_x - self.start_x
                    rel_y = canvas_y - self.start_y

                    self.samples_x.append(rel_x)
                    self.samples_y.append(rel_y)

                    # Check if we've reached target sample count
                    if len(self.samples_x) >= self.config['sample_count']:
                        self.state = 'SAVING'
                        # Trigger save on main thread
                        self.master.after(0, self.save_trajectory)

                    # Update UI periodically (every 10 samples to reduce overhead)
                    if len(self.samples_x) % 10 == 0:
                        self.master.after(0, self.draw_ui)

                except Exception as e:
                    print(f"Sampling error: {e}")

            # Maintain 125Hz timing
            next_time += interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're behind schedule, reset timing
                next_time = time.perf_counter()

    def on_enter_key(self, event):
        """Handle ENTER key press to start recording."""
        if self.state != 'WAITING':
            return

        # Get current mouse position
        pos = self.mouse_controller.position

        with self.canvas_lock:
            offset_x = self.canvas_x
            offset_y = self.canvas_y

        canvas_x = pos[0] - offset_x
        canvas_y = pos[1] - offset_y

        # Check if mouse is within canvas bounds
        if not (0 <= canvas_x <= self.width and 0 <= canvas_y <= self.height):
            print("Mouse outside recording area - move cursor to window first")
            return

        # Start recording
        self.start_x = canvas_x
        self.start_y = canvas_y
        self.samples_x = [0]  # First sample is always (0, 0) relative
        self.samples_y = [0]
        self.recording_start_time = time.time()
        self.state = 'RECORDING'

        print(f"Recording started at ({canvas_x:.0f}, {canvas_y:.0f})")
        self.draw_ui()

    def save_trajectory(self):
        """Save the recorded trajectory to JSON file."""
        if self.state != 'SAVING':
            return

        # Compute deltas
        delta_x = self.samples_x[-1]  # Last position relative to start
        delta_y = self.samples_y[-1]

        # Build trajectory data
        trajectory_data = {
            'x': [int(x) for x in self.samples_x],
            'y': [int(y) for y in self.samples_y],
            'delta_x': int(delta_x),
            'delta_y': int(delta_y),
            'sample_count': len(self.samples_x),
            'sample_interval_ms': self.config['sample_interval_ms'],
            'start_position': {
                'x': int(self.start_x),
                'y': int(self.start_y)
            },
            'timestamp': datetime.now().isoformat(),
        }

        # Generate filename
        self.trajectories_recorded += 1
        filename = f"traj_{self.session_id}_{self.trajectories_recorded:04d}.json"
        filepath = self.output_dir / filename

        # Write JSON
        with open(filepath, 'w') as f:
            json.dump(trajectory_data, f, indent=2)

        # Calculate stats
        duration_ms = len(self.samples_x) * self.config['sample_interval_ms']
        distance = (delta_x**2 + delta_y**2) ** 0.5

        print(f"Saved: {filename}")
        print(f"  Samples: {len(self.samples_x)} | Duration: {duration_ms}ms")
        print(f"  Delta: ({delta_x}, {delta_y}) | Distance: {distance:.1f}px")

        # Update UI briefly
        self.draw_ui()

        # Return to waiting state after brief delay
        self.master.after(500, self.return_to_waiting)

    def return_to_waiting(self):
        """Return to waiting state."""
        self.state = 'WAITING'
        self.samples_x = []
        self.samples_y = []
        self.draw_ui()

    def quit_app(self):
        """Exit cleanly."""
        print("\nShutting down...")

        self.sampling_active = False

        # Wait for sampling thread to finish
        if self.sampling_thread.is_alive():
            self.sampling_thread.join(timeout=0.5)

        print(f"\nSession complete!")
        print(f"  Trajectories recorded: {self.trajectories_recorded}")
        print(f"  Output directory: {self.output_dir}")

        self.master.destroy()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SapiRecorderV2 - Fixed Sample Count Trajectory Recorder'
    )

    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_dir'],
                        help=f"Output directory (default: {DEFAULT_CONFIG['output_dir']})")
    parser.add_argument('--samples', type=int, default=DEFAULT_CONFIG['sample_count'],
                        help=f"Samples per trajectory (default: {DEFAULT_CONFIG['sample_count']})")
    parser.add_argument('--interval', type=int, default=DEFAULT_CONFIG['sample_interval_ms'],
                        help=f"Sample interval in ms (default: {DEFAULT_CONFIG['sample_interval_ms']})")
    parser.add_argument('--width', type=int, default=DEFAULT_CONFIG['window_width'],
                        help=f"Window width (default: {DEFAULT_CONFIG['window_width']})")
    parser.add_argument('--height', type=int, default=DEFAULT_CONFIG['window_height'],
                        help=f"Window height (default: {DEFAULT_CONFIG['window_height']})")

    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config['output_dir'] = args.output
    config['sample_count'] = args.samples
    config['sample_interval_ms'] = args.interval
    config['window_width'] = args.width
    config['window_height'] = args.height

    # Create and run
    root = tk.Tk()
    app = SapiRecorderV2(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()
