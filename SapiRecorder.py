#!/usr/bin/env python3
"""
SapiAgent Two-Dot User-Controlled Recorder
Version: 6.2.0 - WITH TRAJECTORY VALIDATION

Changes from v6.1.2:
- Added real-time trajectory validation (start/end velocity checks)
- Automatic rejection of bad trajectories with visual feedback
- Thread-safe CSV truncation for invalid recordings
- Validation statistics tracking
- Respawn same dots on rejection (don't advance combination)
"""

import tkinter as tk
import random
import math
import csv
import os
import time
import threading
from datetime import datetime
from pynput import mouse

# Configuration
DOT_RADIUS = 30
SESSION_TYPE = '3min'
SESSION_DURATION_MS = 900000 if SESSION_TYPE == '1min' else 900000
BREAK_DURATION_SEC = 10
OUTPUT_DIR = r'D:\V12\V12_bspline_basis-Original\user0001'
MAX_DISTANCE_FROM_CENTER = 690
TARGET_TRAJECTORIES = 120  # One complete cycle through all combinations

# Validation Configuration
VALIDATION_ENABLED = True  # Set to False to disable validation
START_VELOCITY_THRESHOLD = 4  # px/s - max velocity for valid start dwell
END_VELOCITY_THRESHOLD = 4    # px/s - max velocity for valid end dwell
MIN_TRAJECTORY_LENGTH = 12     # Minimum Move events required
DWELL_CHECK_POINTS = 4         # Number of points to check at start/end

# Task Configuration
DISTANCE_THRESHOLDS = [93, 225, 357, 489, 621]
#5 thresholds (×8 = 40 recordings)
#DISTANCE_THRESHOLDS = [27, 34, 43, 54, 68, 86, 108, 136, 171, 215, 271, 341, 429, 540, 667]
#15 thresholds (×8 = 120 recordings)
#DISTANCE_THRESHOLDS = [27, 31, 36, 41, 47, 54, 62, 71, 82, 94, 108, 124, 143, 164, 189, 217, 250, 288, 331, 381, 438, 504, 580, 667]
#184 23 x 8
ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

SCREEN_ANGLE_RANGES = {
    "E":  (-22.5, 22.5), "SE": (22.5, 67.5), "S":  (67.5, 112.5),
    "SW": (112.5, 157.5), "W":  (157.5, 202.5), "NW": (-157.5, -112.5),
    "N":  (-112.5, -67.5), "NE": (-67.5, -22.5),
}

def generate_systematic_combinations():
    """Generate all 120 combinations (15 distances × 8 orientations) systematically."""
    combinations = []
    for distance in DISTANCE_THRESHOLDS:
        for orientation in ORIENTATIONS:
            combinations.append((distance, orientation))
    
    #random.shuffle(combinations)
    return combinations

SYSTEMATIC_COMBINATIONS = generate_systematic_combinations()
print(f"✓ Generated {len(SYSTEMATIC_COMBINATIONS)} systematic combinations")

class TwoDotRecorder:
    def __init__(self, master):
        self.master = master
        self.master.title("SapiAgent Two-Dot Recorder v6.2.0 + VALIDATION")
        
        # Window setup
        self.width, self.height = 2500, 1420
        self.master.geometry(f"{self.width}x{self.height}")
        
        # Canvas
        self.canvas = tk.Canvas(master, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Session state
        self.base_session_id = datetime.now().strftime("session_%Y_%m_%d")
        self.session_number = self.get_next_session_number()
        self.session_start_time = None
        self.is_active = False
        self.is_on_break = False
        self.total_sessions_completed = 0
        self.canvas_lock = threading.Lock()
        self.trajectories_recorded = 0
        self.stop_recording_movements = False
        
        # File writing
        self.csv_file = None
        self.csv_writer = None
        self.event_count = 0
        self.flush_counter = 0
        
        # ⭐ NEW: CSV write lock for thread safety
        self.csv_write_lock = threading.Lock()
        
        # ⭐ NEW: Validation tracking
        self.trajectory_start_position = 0  # Byte position where current trajectory starts
        self.validation_enabled = VALIDATION_ENABLED
        self.validation_stats = {
            'total_attempts': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'rejection_reasons': []
        }
        
        # Two-dot system
        self.dot_A = None
        self.dot_B = None
        self.recording_enabled = False
        
        # UI update guard
        self._updating_ui = False
        
        # Systematic combinations
        self.combinations = SYSTEMATIC_COMBINATIONS
        self.combination_index = 0
        self.total_combinations = len(self.combinations)
        
        # Start position
        self.start_x = self.width // 2
        self.start_y = self.height // 2
        
        # Bindings
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.master.bind("<Return>", self.on_return_key)
        self.master.bind("<Escape>", lambda e: self.quit_app())
        self.master.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # H125z position sampler
        self.mouse_controller = mouse.Controller()
        self.sampling_active = True
        self.sampling_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.sampling_thread.start()
        
        # Update canvas position
        self.update_canvas_position()
        
        # Check session timer
        self.check_session_timer()
        
        # Show startup countdown
        self.show_startup_countdown(5)
        
        print("\n" + "="*60)
        print("SapiAgent Two-Dot Recorder v6.2.0 + VALIDATION")
        print("="*60)
        print(f"Target: {TARGET_TRAJECTORIES} trajectories (1 complete cycle)")
        print(f"Validation: {'ENABLED' if self.validation_enabled else 'DISABLED'}")
        if self.validation_enabled:
            print(f"  Start threshold: {START_VELOCITY_THRESHOLD} px/s")
            print(f"  End threshold: {END_VELOCITY_THRESHOLD} px/s")
            print(f"  Min length: {MIN_TRAJECTORY_LENGTH} points")
        print("="*60)

    def get_next_session_number(self):
        """Find the next available session number for today."""
        import glob
        pattern = f"{OUTPUT_DIR}/{self.base_session_id}_*_{SESSION_TYPE}.csv"
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            return 1
        
        numbers = []
        for filepath in existing_files:
            filename = os.path.basename(filepath)
            parts = filename.replace('.csv', '').split('_')
            try:
                numbers.append(int(parts[-2]))
            except (ValueError, IndexError):
                continue
        
        return max(numbers) + 1 if numbers else 1

    def start_recording_session(self):
        """Start recording session and open CSV file."""
        self.session_start_time = time.time() * 1000
        self.is_active = True
        self.event_count = 0
        self.flush_counter = 0
        
        filename = f"{OUTPUT_DIR}/{self.base_session_id}_{self.session_number}_{SESSION_TYPE}.csv"
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'client timestamp', 'button', 'state', 'x', 'y'
        ])
        self.csv_writer.writeheader()
        
        print(f"\n✓ Session started: {self.base_session_id}_{self.session_number}_{SESSION_TYPE}")
        self.update_ui()

    def show_startup_countdown(self, seconds_remaining):
        """Display startup countdown."""
        self.canvas.delete("all")
        
        if seconds_remaining > 0:
            self.canvas.create_text(
                self.width // 2, self.height // 2 - 50,
                text=f"GET READY",
                font=("Arial", 48, "bold"),
                fill="blue"
            )
            self.canvas.create_text(
                self.width // 2, self.height // 2 + 40,
                text=f"Recording starts in {seconds_remaining} seconds...",
                font=("Arial", 24),
                fill="gray"
            )
            self.canvas.create_text(
                self.width // 2, self.height // 2 + 100,
                text=f"Target: {TARGET_TRAJECTORIES} trajectories",
                font=("Arial", 16),
                fill="green"
            )
            
            self.master.after(1000, lambda: self.show_startup_countdown(seconds_remaining - 1))
        else:
            self.start_recording_session()
            self.spawn_initial_dots()

    def update_canvas_position(self):
        """Update canvas position for coordinate conversion (thread-safe)."""
        try:
            new_x = self.canvas.winfo_rootx()
            new_y = self.canvas.winfo_rooty()
            with self.canvas_lock:
                self.canvas_x = new_x
                self.canvas_y = new_y
        except Exception:
            self.canvas_x = 0
            self.canvas_y = 0
        self.master.after(100, self.update_canvas_position)

    def spawn_initial_dots(self):
        """Spawn first pair of dots."""
        self.dot_A = {
            'x': self.start_x,
            'y': self.start_y,
            'label': 'A (START)'
        }
        self.spawn_next_target()
        self.draw_dots()

    def spawn_next_target(self, retry_count=0, tried_from_center=False):
        """Spawn next target dot B using systematic combination cycling."""
        if retry_count > 10:
            if not tried_from_center:
                print(f"⚠️ Failed from current position, resetting to center...")
                self.dot_A = {
                    'x': self.width // 2,
                    'y': self.height // 2,
                    'label': 'A (START)'
                }
                return self.spawn_next_target(0, tried_from_center=True)
            else:
                combo = self.combinations[self.combination_index]
                print(f"❌ RARE: Skipping impossible combination: {combo[0]}px {combo[1]}")
                self.combination_index = (self.combination_index + 1) % self.total_combinations
                return self.spawn_next_target(0, tried_from_center=False)
        
        margin = 20
        start_x = self.dot_A['x']
        start_y = self.dot_A['y']
        
        threshold, orientation = self.combinations[self.combination_index]
        angle_min, angle_max = SCREEN_ANGLE_RANGES[orientation]
        angle_deg = random.uniform(angle_min, angle_max)
        angle_rad = math.radians(angle_deg)
        
        target_x = start_x + threshold * math.cos(angle_rad)
        target_y = start_y + threshold * math.sin(angle_rad)
        
        if (margin <= target_x <= self.width - margin and 
            margin <= target_y <= self.height - margin):
            
            self.dot_B = {
                'x': int(target_x),
                'y': int(target_y),
                'threshold': threshold,
                'orientation': orientation,
                'angle_deg': angle_deg,
                'label': 'B (TARGET)'
            }
            
            # DON'T advance combination_index here - only advance on VALID trajectory
            return
        
        self.spawn_next_target(retry_count + 1, tried_from_center)

    def draw_dots(self):
        """Draw both dots."""
        self.canvas.delete("dots")
        
        if self.dot_A:
            color = "green" if not self.recording_enabled else "lightgreen"
            self.canvas.create_oval(
                self.dot_A['x'] - DOT_RADIUS, self.dot_A['y'] - DOT_RADIUS,
                self.dot_A['x'] + DOT_RADIUS, self.dot_A['y'] + DOT_RADIUS,
                fill=color, outline="green", width=3, tags="dots"
            )
            self.canvas.create_text(
                self.dot_A['x'], self.dot_A['y'] - 25,
                text="A - START", font=("Arial", 10, "bold"),
                fill="lightgreen", tags="dots"
            )
        
        if self.dot_B:
            self.canvas.create_oval(
                self.dot_B['x'] - DOT_RADIUS, self.dot_B['y'] - DOT_RADIUS,
                self.dot_B['x'] + DOT_RADIUS, self.dot_B['y'] + DOT_RADIUS,
                fill="blue", outline="blue", width=3, tags="dots"
            )
            self.canvas.create_text(
                self.dot_B['x'], self.dot_B['y'] - 25,
                text="B - TARGET", font=("Arial", 10, "bold"),
                fill="cyan", tags="dots"
            )

    def update_ui(self):
        """Update timer and stats."""
        if self.is_on_break or self._updating_ui:
            return
        
        self._updating_ui = True
        self.canvas.delete("ui_text")
        
        if self.dot_B:
            recording_status = "RECORDING" if self.recording_enabled else "WAITING"
            progress = f"{self.trajectories_recorded}/{TARGET_TRAJECTORIES}"
            combo_info = f"{self.dot_B['threshold']}px {self.dot_B['orientation']}"
            
            # ⭐ NEW: Add validation stats
            if self.validation_enabled and self.validation_stats['total_attempts'] > 0:
                valid_pct = (self.validation_stats['valid_count'] / 
                           self.validation_stats['total_attempts'] * 100)
                info_text = f"{recording_status} | {progress} | {combo_info} | Valid: {valid_pct:.0f}%"
            else:
                info_text = f"{recording_status} | {progress} | {combo_info}"
            
            self.canvas.create_text(self.width // 2, 30, text=info_text,
                                   font=("Arial", 16), fill="gray", tags="ui_text")
        
        if self.session_start_time and not self.is_on_break:
            elapsed = int((time.time() * 1000 - self.session_start_time) / 1000)
            stats_text = f"Events: {self.event_count} | Rate: {self.event_count/max(elapsed, 1):.1f} Hz"
            self.canvas.create_text(self.width // 2, 60, text=stats_text,
                                   font=("Arial", 12), fill="white", tags="ui_text")
        
        self._updating_ui = False
        
        if not self.is_on_break and self.session_start_time:
            self.master.after(500, self.update_ui)

    def _sample_loop(self):
        """Poll mouse position at 125Hz (thread-safe with lock)."""
        interval = 0.008  # 125Hz
        next_time = time.perf_counter()

        while self.sampling_active:
            if (self.is_active and not self.is_on_break and 
                not self.stop_recording_movements and self.recording_enabled and self.csv_writer):
                try:
                    pos = self.mouse_controller.position

                    with self.canvas_lock:
                        offset_x = self.canvas_x
                        offset_y = self.canvas_y

                    canvas_x = pos[0] - offset_x
                    canvas_y = pos[1] - offset_y

                    if 0 <= canvas_x <= self.width and 0 <= canvas_y <= self.height:
                        timestamp = int((time.time() * 1000) - self.session_start_time)

                        # ⭐ NEW: Thread-safe CSV write
                        with self.csv_write_lock:
                            self.csv_writer.writerow({
                                'client timestamp': timestamp,
                                'button': 'NoButton',
                                'state': 'Move',
                                'x': int(canvas_x),
                                'y': int(canvas_y)
                            })
                            self.event_count += 1
                        
                        # Flush periodically
                        self.flush_counter += 1
                        if self.flush_counter >= 250:
                            with self.csv_write_lock:
                                self.csv_file.flush()
                            self.flush_counter = 0
                except Exception:
                    pass

            next_time += interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ⭐ NEW: Validation method
    def validate_last_trajectory(self):
        """
        Validate the most recently recorded trajectory.
        
        Reads from trajectory_start_position to current position,
        checks start and end velocities.
        
        Returns:
            (is_valid, rejection_reason, diagnostics)
        """
        if not self.validation_enabled:
            return True, "Validation disabled", {}
        
        csv_path = f"{OUTPUT_DIR}/{self.base_session_id}_{self.session_number}_{SESSION_TYPE}.csv"
        
        try:
            # Read trajectory from file
            with open(csv_path, 'r') as f:
                # Seek to trajectory start
                f.seek(self.trajectory_start_position)
                
                # Read all lines from start position to EOF
                lines = f.readlines()
            
            # Parse trajectory data
            trajectory = []
            for line in lines:
                if not line.strip() or line.startswith('client timestamp'):
                    continue
                parts = line.strip().split(',')
                if len(parts) == 5 and parts[2] == 'Move':
                    try:
                        trajectory.append({
                            't': int(parts[0]),
                            'x': int(parts[3]),
                            'y': int(parts[4])
                        })
                    except ValueError:
                        continue
            
            # Check minimum length
            if len(trajectory) < MIN_TRAJECTORY_LENGTH:
                reason = f"Too short: {len(trajectory)} points < {MIN_TRAJECTORY_LENGTH}"
                return False, reason, {'length': len(trajectory)}
            
            # Compute velocities
            velocities = []
            for i in range(len(trajectory) - 1):
                dt = (trajectory[i+1]['t'] - trajectory[i]['t']) / 1000.0
                dx = trajectory[i+1]['x'] - trajectory[i]['x']
                dy = trajectory[i+1]['y'] - trajectory[i]['y']
                dist = math.sqrt(dx**2 + dy**2)
                vel = dist / dt if dt > 0 else 0
                velocities.append(vel)
            
            # Check start velocities (first DWELL_CHECK_POINTS)
            check_points = min(DWELL_CHECK_POINTS, len(velocities))
            start_vels = velocities[:check_points]
            max_start_vel = max(start_vels) if start_vels else 0
            
            if max_start_vel > START_VELOCITY_THRESHOLD:
                reason = f"Bad start: max velocity {max_start_vel:.1f} px/s > {START_VELOCITY_THRESHOLD}"
                return False, reason, {
                    'start_velocities': start_vels,
                    'max_start_vel': max_start_vel
                }
            
            # Check end velocities (last DWELL_CHECK_POINTS)
            end_vels = velocities[-check_points:] if len(velocities) >= check_points else velocities
            max_end_vel = max(end_vels) if end_vels else 0
            
            if max_end_vel > END_VELOCITY_THRESHOLD:
                reason = f"Bad end: max velocity {max_end_vel:.1f} px/s > {END_VELOCITY_THRESHOLD}"
                return False, reason, {
                    'end_velocities': end_vels,
                    'max_end_vel': max_end_vel
                }
            
            # Valid!
            diagnostics = {
                'length': len(trajectory),
                'max_start_vel': max_start_vel,
                'max_end_vel': max_end_vel,
                'max_velocity': max(velocities) if velocities else 0
            }
            return True, "Valid", diagnostics
            
        except Exception as e:
            # Error during validation - assume invalid
            reason = f"Validation error: {str(e)}"
            return False, reason, {}

    # ⭐ NEW: Removal method
    def remove_last_trajectory(self):
        """
        Remove the last trajectory by truncating CSV file.
        
        Truncates file to trajectory_start_position, effectively
        removing all data written since recording started.
        """
        csv_path = f"{OUTPUT_DIR}/{self.base_session_id}_{self.session_number}_{SESSION_TYPE}.csv"
        
        try:
            # Close CSV file temporarily
            if self.csv_file:
                self.csv_file.close()
            
            # Truncate to remove trajectory
            with open(csv_path, 'r+') as f:
                f.seek(self.trajectory_start_position)
                f.truncate()
            
            # Reopen CSV file for continued writing
            self.csv_file = open(csv_path, 'a', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
                'client timestamp', 'button', 'state', 'x', 'y'
            ])
            
            print(f"  ✓ Removed invalid trajectory from CSV")
            
        except Exception as e:
            print(f"  ✗ Error removing trajectory: {e}")

    # ⭐ NEW: Visual feedback for rejection
    def show_rejection_feedback(self, reason):
        """Show red flash and rejection message."""
        # Red flash
        self.canvas.create_rectangle(
            0, 0, self.width, self.height,
            fill="darkblue", stipple="gray12", tags="rejection_flash"
        )
        
        # Rejection message
        self.canvas.create_text(
            self.width // 2, self.height // 2,
            text=f"REJECTED\n{reason}\nTry again!",
            font=("Arial", 32, "bold"),
            fill="Purple",
            tags="rejection_flash"
        )
        
        # Remove flash after 800ms
        self.master.after(800, lambda: self.canvas.delete("rejection_flash"))

    def handle_trajectory_complete(self, canvas_x, canvas_y, timestamp):
        """
        Centralized handler for trajectory completion.
        
        ⭐ MODIFIED: Now includes validation logic.
        """
        # Stop recording
        self.recording_enabled = False
        
        # Flush CSV to ensure all data on disk
        if self.csv_file:
            with self.csv_write_lock:
                self.csv_file.flush()
        
        # Write click B to CSV
        if self.csv_writer:
            with self.csv_write_lock:
                self.csv_writer.writerow({
                    'client timestamp': timestamp,
                    'button': 'Left',
                    'state': 'Pressed',
                    'x': int(canvas_x),
                    'y': int(canvas_y)
                })
                self.csv_writer.writerow({
                    'client timestamp': timestamp,
                    'button': 'Left',
                    'state': 'Released',
                    'x': int(canvas_x),
                    'y': int(canvas_y)
                })
                self.event_count += 2
                self.csv_file.flush()  # Ensure clicks written before validation
        
        # ⭐ NEW: Validate trajectory
        is_valid, reason, diagnostics = self.validate_last_trajectory()
        
        # Update validation stats
        self.validation_stats['total_attempts'] += 1
        
        if is_valid:
            # ✅ VALID TRAJECTORY
            self.validation_stats['valid_count'] += 1
            self.trajectories_recorded += 1
            
            print(f"✅ Trajectory {self.trajectories_recorded}/{TARGET_TRAJECTORIES} VALID: "
                  f"{self.dot_B['threshold']}px {self.dot_B['orientation']} | "
                  f"start_v={diagnostics.get('max_start_vel', 0):.1f}, "
                  f"end_v={diagnostics.get('max_end_vel', 0):.1f} px/s")
            
            # Check if target reached
            if self.trajectories_recorded >= TARGET_TRAJECTORIES:
                print(f"\n🎯 TARGET REACHED: {self.trajectories_recorded} trajectories!")
                self.stop_recording_movements = True
                self.is_active = False
                time.sleep(0.02)
                self.save_current_session()
                self.show_completion_screen()
                return True
            
            # Check if session time ending
            elapsed = time.time() * 1000 - self.session_start_time
            if elapsed > SESSION_DURATION_MS - 3000:
                self.stop_recording_movements = True
                return True
            
            # Continue: B becomes new A, advance combination
            self.dot_A = {
                'x': self.dot_B['x'],
                'y': self.dot_B['y'],
                'label': 'A (START)'
            }
            
            # ⭐ NOW advance combination (only on valid trajectory)
            self.combination_index = (self.combination_index + 1) % self.total_combinations
            
            if self.combination_index == 0:
                print(f"✓ Completed full cycle of {self.total_combinations} combinations!")
            
            self.spawn_next_target()
            self.draw_dots()
            return False
            
        else:
            # ❌ INVALID TRAJECTORY
            self.validation_stats['invalid_count'] += 1
            self.validation_stats['rejection_reasons'].append(reason)
            
            print(f"❌ Trajectory REJECTED: {reason}")
            
            # Remove from CSV
            self.remove_last_trajectory()
            
            # Show visual feedback
            self.show_rejection_feedback(reason)
            
            # ⭐ DON'T advance combination - respawn SAME dots
            # Just redraw current dots (A and B unchanged)
            self.draw_dots()
            
            return False

    def on_canvas_click(self, event):
        """
        THREAD-SAFE: Handle click via native Tkinter binding.
        
        ⭐ MODIFIED: Track CSV position when recording starts.
        """
        if not self.is_active or self.is_on_break:
            return
        
        canvas_x = event.x
        canvas_y = event.y
        timestamp = int((time.time() * 1000) - self.session_start_time)
        
        # Check if clicked dot A (start recording)
        if self.dot_A and not self.recording_enabled:
            dx = canvas_x - self.dot_A['x']
            dy = canvas_y - self.dot_A['y']
            
            if math.hypot(dx, dy) <= DOT_RADIUS + 3:
                # ⭐ NEW: Track where this trajectory starts
                if self.csv_file:
                    with self.csv_write_lock:
                        self.csv_file.flush()
                        self.trajectory_start_position = self.csv_file.tell()
                
                self.recording_enabled = True
                print(f"Recording STARTED (position: {self.trajectory_start_position})")
                self.draw_dots()
                return
        
        # Check if clicked dot B (stop recording)
        if self.dot_B and self.recording_enabled:
            dx = canvas_x - self.dot_B['x']
            dy = canvas_y - self.dot_B['y']
            
            if math.hypot(dx, dy) <= DOT_RADIUS + 3:
                self.handle_trajectory_complete(canvas_x, canvas_y, timestamp)
                return

    def on_return_key(self, event):
        """
        Handle Return key - same logic as click.
        
        ⭐ MODIFIED: Track CSV position when recording starts.
        """
        if not self.is_active or self.is_on_break:
            return

        pos = self.mouse_controller.position

        with self.canvas_lock:
            offset_x = self.canvas_x
            offset_y = self.canvas_y

        canvas_x = pos[0] - offset_x
        canvas_y = pos[1] - offset_y
    
        if not (0 <= canvas_x <= self.width and 0 <= canvas_y <= self.height):
            return
        
        timestamp = int((time.time() * 1000) - self.session_start_time)
        
        # Check if on dot A (start recording)
        if self.dot_A and not self.recording_enabled:
            dx = canvas_x - self.dot_A['x']
            dy = canvas_y - self.dot_A['y']
            
            if math.hypot(dx, dy) <= DOT_RADIUS + 3:
                # ⭐ NEW: Track where this trajectory starts
                if self.csv_file:
                    with self.csv_write_lock:
                        self.csv_file.flush()
                        self.trajectory_start_position = self.csv_file.tell()
                
                self.recording_enabled = True
                print(f"Recording STARTED (position: {self.trajectory_start_position})")
                self.draw_dots()
                return
        
        # Check if on dot B (stop recording)
        if self.dot_B and self.recording_enabled:
            dx = canvas_x - self.dot_B['x']
            dy = canvas_y - self.dot_B['y']
            
            if math.hypot(dx, dy) <= DOT_RADIUS + 3:
                self.handle_trajectory_complete(canvas_x, canvas_y, timestamp)
                return

    def show_completion_screen(self):
        """Display completion screen with validation stats."""
        self.canvas.delete("all")
        
        self.canvas.create_text(
            self.width // 2, self.height // 2 - 100,
            text="🎯 COMPLETE!",
            font=("Arial", 60, "bold"),
            fill="green"
        )
        self.canvas.create_text(
            self.width // 2, self.height // 2,
            text=f"{self.trajectories_recorded} trajectories recorded",
            font=("Arial", 24),
            fill="gray"
        )
        
        # ⭐ NEW: Show validation stats
        if self.validation_enabled and self.validation_stats['total_attempts'] > 0:
            valid_pct = (self.validation_stats['valid_count'] / 
                        self.validation_stats['total_attempts'] * 100)
            self.canvas.create_text(
                self.width // 2, self.height // 2 + 50,
                text=f"Validation: {self.validation_stats['valid_count']}/{self.validation_stats['total_attempts']} "
                     f"({valid_pct:.1f}% success rate)",
                font=("Arial", 18),
                fill="blue"
            )
        
        self.canvas.create_text(
            self.width // 2, self.height // 2 + 100,
            text="Press ESC to exit or wait 10 seconds...",
            font=("Arial", 16),
            fill="gray"
        )
        
        # Auto-exit after 10 seconds
        self.master.after(10000, self.quit_app)

    def check_session_timer(self):
        """Check if session time limit reached."""
        if self.is_active and not self.is_on_break and self.session_start_time:
            elapsed = int((time.time() * 1000) - self.session_start_time)
            if elapsed >= SESSION_DURATION_MS:
                self.is_active = False
                self.stop_recording_movements = True
                time.sleep(0.02)
            
                print(f"\n✓ Session {self.session_number} complete (time limit)")
                self.save_current_session()
                self.start_break()
    
        self.master.after(1000, self.check_session_timer)

    def save_current_session(self):
        """Close CSV file and report stats."""
        if not self.csv_file:
            return
        
        elapsed = time.time() * 1000 - self.session_start_time
        duration_sec = int(elapsed / 1000)
        
        self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None
        
        filename = f"{OUTPUT_DIR}/{self.base_session_id}_{self.session_number}_{SESSION_TYPE}.csv"
        
        print(f"Saved: {filename}")
        print(f"   Trajectories: {self.trajectories_recorded} | Events: {self.event_count}")
        print(f"   Duration: {duration_sec}s | Rate: {self.event_count / duration_sec:.1f} Hz")
        
        # ⭐ NEW: Print validation statistics
        if self.validation_enabled and self.validation_stats['total_attempts'] > 0:
            valid_pct = (self.validation_stats['valid_count'] / 
                        self.validation_stats['total_attempts'] * 100)
            print(f"   Validation: {self.validation_stats['valid_count']}/{self.validation_stats['total_attempts']} "
                  f"valid ({valid_pct:.1f}%)")
            
            # Print rejection reasons summary
            if self.validation_stats['rejection_reasons']:
                print(f"   Rejection reasons:")
                from collections import Counter
                reason_counts = Counter(r.split(':')[0] for r in self.validation_stats['rejection_reasons'])
                for reason, count in reason_counts.most_common():
                    print(f"     - {reason}: {count}")
        
        self.total_sessions_completed += 1

    def start_break(self):
        """Start countdown break."""
        self.is_on_break = True
        self.is_active = False
        self.recording_enabled = False
        self.canvas.delete("all")
        self.show_break_countdown(BREAK_DURATION_SEC)

    def show_break_countdown(self, seconds_remaining):
        """Display countdown."""
        self.canvas.delete("all")
        
        if seconds_remaining > 0:
            self.canvas.create_text(
                self.width // 2, self.height // 2 - 50,
                text=f"BREAK TIME",
                font=("Arial", 32, "bold"),
                fill="orange"
            )
            self.canvas.create_text(
                self.width // 2, self.height // 2 + 20,
                text=f"Next session in {seconds_remaining}s...",
                font=("Arial", 20),
                fill="gray"
            )
            
            self.master.after(1000, lambda: self.show_break_countdown(seconds_remaining - 1))
        else:
            self.start_new_session()

    def start_new_session(self):
        """Start new recording session."""
        self.is_on_break = False
        self.session_number += 1
        self.trajectories_recorded = 0
        self.stop_recording_movements = False
        self.recording_enabled = False
        self.event_count = 0
        self.flush_counter = 0
        
        # ⭐ NEW: Reset validation stats for new session
        self.validation_stats = {
            'total_attempts': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'rejection_reasons': []
        }
        
        self.start_recording_session()
        self.spawn_initial_dots()

    def quit_app(self):
        """Exit cleanly."""
        print("\n Stopping...")
        
        self.sampling_active = False
        
        if self.is_active and self.csv_file:
            print("Saving current session...")
            self.save_current_session()
        
        print(f"\n Complete | Sessions: {self.total_sessions_completed} | Dir: {OUTPUT_DIR}/")
        self.master.destroy()

def main():
    root = tk.Tk()
    app = TwoDotRecorder(root)
    root.mainloop()

if __name__ == "__main__":
    main()
