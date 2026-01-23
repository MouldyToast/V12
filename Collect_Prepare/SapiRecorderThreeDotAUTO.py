#!/usr/bin/env python3
"""
SapiAgent Three-Dot Flow Recorder
Version: 7.0.0 - CONTINUOUS FLOW WITH ANTICIPATION

==============================================================================
KEY INNOVATION
==============================================================================
Three dots visible simultaneously:
    A (green)  = Where you are / just were
    B (blue)   = Current target - move here
    C (orange) = Next target - visible so you anticipate it

This captures ANTICIPATORY MOVEMENT - humans naturally curve toward the next
target before reaching the current one. The A→B trajectory is SHAPED by
knowing where C is.

==============================================================================
RECORDING FLOW
==============================================================================
1. Session starts: A at center, B and C spawned
2. User clicks A to begin recording
3. User moves toward B (seeing C, their path curves naturally)
4. User clicks B:
   - Segment A→B saved with C context
   - Dots shift: A←B, B←C, new C spawns
   - Recording continues (no pause!)
5. User now moves toward new B (old C)
6. Repeat until target trajectories reached

==============================================================================
ML TRAINING DATA
==============================================================================
Each segment includes:
    - trajectory: the actual path points
    - vec_AB: current movement vector (distance, orientation)
    - vec_BC: NEXT movement vector (distance, orientation)
    - turn_angle: angle change from AB to BC direction

The model learns: "Given current direction and NEXT direction, 
                   here's how the trajectory should curve"

==============================================================================
OUTPUT STRUCTURE
==============================================================================
    output_dir/
    ├── session_YYYY_MM_DD_N_continuous.csv    # Raw continuous recording
    └── segments/
        ├── segment_0001.json                   # Segment metadata + trajectory
        ├── segment_0002.json
        └── ...

==============================================================================
USAGE
==============================================================================
    python SapiRecorderThreeDot.py

Controls:
    - Click A (green) to start recording
    - Click B (blue) to complete segment and shift dots
    - Press Escape to end session early
    - Session auto-ends after TARGET_SEGMENTS reached

==============================================================================
"""

import tkinter as tk
import random
import math
import csv
import json
import os
import time
import threading
from datetime import datetime
from pynput import mouse
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Visual
DOT_RADIUS = 30
DOT_A_COLOR = "#22c55e"      # Green - origin
DOT_B_COLOR = "#3b82f6"      # Blue - current target  
DOT_C_COLOR = "#f97316"      # Orange - next target (anticipation)
BACKGROUND_COLOR = "black"

# Session
TARGET_SEGMENTS = 200         # How many A→B segments to record
SESSION_DURATION_MS = 1800000 # 30 minute max (safety limit)

# Output
OUTPUT_DIR = r'D:\V12\V12_Anchors_Continuous\three_dot_flow'

# Screen constraints
SCREEN_MARGIN = 25            # Keep dots this far from edges
MAX_DISTANCE_FROM_CENTER = 690

# Sampling
SAMPLE_RATE_HZ = 125          # Mouse position polling rate

# =============================================================================
# MOVEMENT PARAMETERS
# =============================================================================

# Distance thresholds (pixels) - how far B is from A, C is from B
DISTANCE_THRESHOLDS = [93, 225, 357, 489, 621]
DISTANCE_NAMES = ["XS", "S", "M", "L", "XL"]

# Orientations (8 compass directions)
ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Screen angle ranges for each orientation
# Note: Screen coordinates have Y increasing downward
# atan2(dy, dx) gives angle where:
#   East = 0°, South = 90°, West = ±180°, North = -90°
SCREEN_ANGLE_RANGES = {
    "E":  (-22.5, 22.5),
    "SE": (22.5, 67.5),
    "S":  (67.5, 112.5),
    "SW": (112.5, 157.5),
    "W":  (157.5, 180.0),      # Also includes -180 to -157.5
    "NW": (-157.5, -112.5),
    "N":  (-112.5, -67.5),
    "NE": (-67.5, -22.5),
}

# Turn angle categories for analysis
TURN_CATEGORIES = {
    "straight": (-22.5, 22.5),
    "slight_right": (22.5, 67.5),
    "hard_right": (67.5, 135),
    "reverse_right": (135, 180),
    "slight_left": (-67.5, -22.5),
    "hard_left": (-135, -67.5),
    "reverse_left": (-180, -135),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def angle_between_vectors(v1, v2):
    """
    Compute signed angle from v1 to v2 in degrees.
    Positive = clockwise (right turn on screen)
    Negative = counter-clockwise (left turn on screen)
    """
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    
    diff = angle2 - angle1
    
    # Normalize to [-180, 180]
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    
    return math.degrees(diff)


def get_orientation_from_angle(angle_deg):
    """Convert angle in degrees to orientation string."""
    # Handle W wrapping
    if angle_deg > 157.5 or angle_deg <= -157.5:
        return "W"
    
    for orient, (lo, hi) in SCREEN_ANGLE_RANGES.items():
        if orient == "W":
            continue  # Handled above
        if lo <= angle_deg < hi:
            return orient
    
    return "E"  # Default fallback


def get_orientation_id(orient_str):
    """Convert orientation string to ID (0-7)."""
    return ORIENTATIONS.index(orient_str)


def get_distance_group(distance):
    """Get distance group ID (0-4) and name."""
    for i, threshold in enumerate(DISTANCE_THRESHOLDS):
        if i == 0:
            if distance < (threshold + DISTANCE_THRESHOLDS[1]) / 2:
                return i, DISTANCE_NAMES[i]
        elif i == len(DISTANCE_THRESHOLDS) - 1:
            return i, DISTANCE_NAMES[i]
        else:
            mid_low = (DISTANCE_THRESHOLDS[i-1] + threshold) / 2
            mid_high = (threshold + DISTANCE_THRESHOLDS[i+1]) / 2
            if mid_low <= distance < mid_high:
                return i, DISTANCE_NAMES[i]
    
    return len(DISTANCE_THRESHOLDS) - 1, DISTANCE_NAMES[-1]


def get_turn_category(turn_angle):
    """Categorize turn angle."""
    for category, (lo, hi) in TURN_CATEGORIES.items():
        if lo <= turn_angle < hi:
            return category
    return "straight"


# =============================================================================
# THREE-DOT RECORDER CLASS
# =============================================================================

class ThreeDotRecorder:
    """
    Records mouse trajectories with three visible dots for anticipatory movement.
    """
    
    def __init__(self, master):
        self.master = master
        self.master.title("SapiAgent Three-Dot Flow Recorder v7.0.0")
        
        # Window setup - large canvas
        self.width = 2500
        self.height = 1420
        self.master.geometry(f"{self.width}x{self.height}")
        
        # Canvas
        self.canvas = tk.Canvas(master, bg=BACKGROUND_COLOR, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Session identification
        self.session_id = datetime.now().strftime("session_%Y_%m_%d_%H%M%S")
        self.session_start_time = None
        
        # Session state
        self.is_active = False
        self.recording_enabled = False
        self.segments_recorded = 0
        
        # The three dots
        self.dot_A = None  # Origin (green)
        self.dot_B = None  # Current target (blue)
        self.dot_C = None  # Next target (orange) - for anticipation
        
        # Current segment tracking
        self.segment_start_time = None
        self.segment_trajectory = []  # List of (timestamp, x, y)
        
        # Thread safety
        self.canvas_lock = threading.Lock()
        self.data_lock = threading.Lock()
        
        # Canvas position (for coordinate conversion)
        self.canvas_x = 0
        self.canvas_y = 0
        
        # CSV file for continuous raw data
        self.csv_file = None
        self.csv_writer = None
        self.csv_lock = threading.Lock()
        
        # Output directories
        self.output_dir = os.path.join(OUTPUT_DIR, self.session_id)
        self.segments_dir = os.path.join(self.output_dir, 'segments')
        os.makedirs(self.segments_dir, exist_ok=True)
        
        # Mouse controller for position sampling
        self.mouse_controller = mouse.Controller()
        self.sampling_active = True
        self.sampling_thread = threading.Thread(target=self._sample_loop, daemon=True)
        
        # Statistics
        self.stats = {
            'turn_angles': [],
            'distances_AB': [],
            'distances_BC': [],
            'orientations_AB': defaultdict(int),
            'orientations_BC': defaultdict(int),
        }
        
        # Entry detection state (prevents double-triggering)
        self._inside_A = False
        self._inside_B = False
        self._last_trigger_time = 0
        self._min_trigger_interval_ms = 100  # Prevent accidental double-triggers
        
        # Bindings - keep click as backup/manual override
        self.canvas.bind("<Button-1>", self.on_click)
        self.master.bind("<Escape>", lambda e: self.end_session())
        self.master.protocol("WM_DELETE_WINDOW", self.end_session)
        
        # Start canvas position updater
        self.update_canvas_position()
        
        # Show startup screen
        self.show_startup_countdown(5)
        
        # Print session info
        print("\n" + "=" * 70)
        print("THREE-DOT FLOW RECORDER v7.0.0")
        print("=" * 70)
        print(f"Session: {self.session_id}")
        print(f"Output:  {self.output_dir}")
        print(f"Target:  {TARGET_SEGMENTS} segments")
        print("=" * 70)
        print("\nInstructions:")
        print("  1. Move cursor through GREEN dot (A) - recording starts automatically")
        print("  2. Move to BLUE dot (B) - see ORANGE dot (C) as your NEXT target")
        print("  3. Pass through BLUE dot (B) - segment completes automatically")
        print("  4. Dots shift - just keep moving fluidly!")
        print("  5. No clicking needed - just flow through the dots")
        print("  6. Press ESC to end early")
        print("=" * 70)
    
    # =========================================================================
    # CANVAS POSITION TRACKING
    # =========================================================================
    
    def update_canvas_position(self):
        """Update canvas position for coordinate conversion."""
        try:
            with self.canvas_lock:
                self.canvas_x = self.canvas.winfo_rootx()
                self.canvas_y = self.canvas.winfo_rooty()
        except Exception:
            pass
        
        if self.sampling_active:
            self.master.after(100, self.update_canvas_position)
    
    def _is_inside_dot(self, x, y, dot, radius=None):
        """Check if point (x, y) is inside a dot."""
        if dot is None:
            return False
        if radius is None:
            radius = DOT_RADIUS  # Slight forgiveness
        dx = x - dot['x']
        dy = y - dot['y']
        return (dx * dx + dy * dy) <= (radius * radius)
    
    # =========================================================================
    # STARTUP
    # =========================================================================
    
    def show_startup_countdown(self, seconds):
        """Display countdown before session starts."""
        self.canvas.delete("all")
        
        if seconds > 0:
            # Countdown display
            self.canvas.create_text(
                self.width // 2, self.height // 2 - 80,
                text="THREE-DOT FLOW RECORDER",
                font=("Arial", 36, "bold"),
                fill="white"
            )
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text=f"Starting in {seconds}...",
                font=("Arial", 48, "bold"),
                fill=DOT_B_COLOR
            )
            self.canvas.create_text(
                self.width // 2, self.height // 2 + 80,
                text="Just flow through the dots - no clicking needed!",
                font=("Arial", 20),
                fill="gray"
            )
            
            # Legend
            self.canvas.create_oval(
                self.width // 2 - 200, self.height // 2 + 140,
                self.width // 2 - 180, self.height // 2 + 160,
                fill=DOT_A_COLOR, outline=""
            )
            self.canvas.create_text(
                self.width // 2 - 160, self.height // 2 + 150,
                text="A = Pass through to start", font=("Arial", 14), fill=DOT_A_COLOR, anchor="w"
            )
            
            self.canvas.create_oval(
                self.width // 2 - 10, self.height // 2 + 140,
                self.width // 2 + 10, self.height // 2 + 160,
                fill=DOT_B_COLOR, outline=""
            )
            self.canvas.create_text(
                self.width // 2 + 30, self.height // 2 + 150,
                text="B = Current target", font=("Arial", 14), fill=DOT_B_COLOR, anchor="w"
            )
            
            self.canvas.create_oval(
                self.width // 2 + 180, self.height // 2 + 140,
                self.width // 2 + 200, self.height // 2 + 160,
                fill=DOT_C_COLOR, outline=""
            )
            self.canvas.create_text(
                self.width // 2 + 220, self.height // 2 + 150,
                text="C = Next target (look ahead!)", font=("Arial", 14), fill=DOT_C_COLOR, anchor="w"
            )
            
            self.master.after(1000, lambda: self.show_startup_countdown(seconds - 1))
        else:
            self.start_session()
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def start_session(self):
        """Initialize and start recording session."""
        self.session_start_time = time.time() * 1000
        self.is_active = True
        
        # Open CSV file for continuous recording
        csv_path = os.path.join(self.output_dir, f'{self.session_id}_continuous.csv')
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'timestamp', 'event', 'x', 'y'
        ])
        self.csv_writer.writeheader()
        
        # Spawn initial three dots
        self.spawn_initial_dots()
        self.draw_dots()
        self.update_ui()
        
        # Start sampling thread
        self.sampling_thread.start()
        
        print(f"\n✓ Session started")
        print(f"  Click the GREEN dot to begin recording")
    
    def end_session(self):
        """End the recording session and save everything."""
        print("\n" + "=" * 70)
        print("ENDING SESSION")
        print("=" * 70)
        
        self.is_active = False
        self.recording_enabled = False
        self.sampling_active = False
        
        # Wait for sampling thread
        if self.sampling_thread.is_alive():
            self.sampling_thread.join(timeout=1.0)
        
        # Close CSV
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        
        # Save session summary
        self.save_session_summary()
        
        # Show completion screen
        self.show_completion_screen()
        
        print(f"\n✓ Session complete!")
        print(f"  Segments recorded: {self.segments_recorded}")
        print(f"  Output directory:  {self.output_dir}")
    
    def save_session_summary(self):
        """Save session statistics and summary."""
        summary = {
            'session_id': self.session_id,
            'segments_recorded': self.segments_recorded,
            'target_segments': TARGET_SEGMENTS,
            'duration_seconds': (time.time() * 1000 - self.session_start_time) / 1000,
            'statistics': {
                'turn_angles': {
                    'mean': float(np.mean(self.stats['turn_angles'])) if self.stats['turn_angles'] else 0,
                    'std': float(np.std(self.stats['turn_angles'])) if self.stats['turn_angles'] else 0,
                    'min': float(min(self.stats['turn_angles'])) if self.stats['turn_angles'] else 0,
                    'max': float(max(self.stats['turn_angles'])) if self.stats['turn_angles'] else 0,
                },
                'distances_AB': {
                    'mean': float(np.mean(self.stats['distances_AB'])) if self.stats['distances_AB'] else 0,
                },
                'distances_BC': {
                    'mean': float(np.mean(self.stats['distances_BC'])) if self.stats['distances_BC'] else 0,
                },
                'orientations_AB': dict(self.stats['orientations_AB']),
                'orientations_BC': dict(self.stats['orientations_BC']),
            }
        }
        
        summary_path = os.path.join(self.output_dir, 'session_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def show_completion_screen(self):
        """Display completion screen."""
        self.canvas.delete("all")
        
        self.canvas.create_text(
            self.width // 2, self.height // 2 - 60,
            text="✓ SESSION COMPLETE",
            font=("Arial", 48, "bold"),
            fill=DOT_A_COLOR
        )
        self.canvas.create_text(
            self.width // 2, self.height // 2 + 20,
            text=f"{self.segments_recorded} segments recorded",
            font=("Arial", 24),
            fill="white"
        )
        self.canvas.create_text(
            self.width // 2, self.height // 2 + 80,
            text=f"Saved to: {self.output_dir}",
            font=("Arial", 14),
            fill="gray"
        )
        
        # Auto-close after 5 seconds
        self.master.after(5000, self.master.destroy)
    
    # =========================================================================
    # DOT SPAWNING
    # =========================================================================
    
    def spawn_initial_dots(self):
        """Spawn the initial three dots: A at center, B and C following."""
        # A starts at center
        self.dot_A = {
            'x': self.width // 2,
            'y': self.height // 2,
        }
        
        # Spawn B relative to A
        self.dot_B = self.spawn_target_from(self.dot_A)
        
        # Spawn C relative to B
        self.dot_C = self.spawn_target_from(self.dot_B)
        
        print(f"  Initial dots spawned:")
        print(f"    A: ({self.dot_A['x']}, {self.dot_A['y']})")
        print(f"    B: ({self.dot_B['x']}, {self.dot_B['y']}) - {self.dot_B['distance']}px {self.dot_B['orientation']}")
        print(f"    C: ({self.dot_C['x']}, {self.dot_C['y']}) - {self.dot_C['distance']}px {self.dot_C['orientation']}")
    
    def spawn_target_from(self, origin_dot, max_retries=50):
        """
        Spawn a new target dot at a valid position relative to origin.
        
        Returns dot dict with position and metadata.
        """
        for attempt in range(max_retries):
            # Random distance and orientation
            distance = random.choice(DISTANCE_THRESHOLDS)
            
            # Random angle within orientation
            orientation = random.choice(ORIENTATIONS)
            angle_range = SCREEN_ANGLE_RANGES[orientation]
            
            # Handle W wrapping
            if orientation == "W":
                if random.random() < 0.5:
                    angle_deg = random.uniform(157.5, 180.0)
                else:
                    angle_deg = random.uniform(-180.0, -157.5)
            else:
                angle_deg = random.uniform(angle_range[0], angle_range[1])
            
            angle_rad = math.radians(angle_deg)
            
            # Compute target position
            target_x = origin_dot['x'] + distance * math.cos(angle_rad)
            target_y = origin_dot['y'] + distance * math.sin(angle_rad)
            
            # Check bounds
            if (SCREEN_MARGIN <= target_x <= self.width - SCREEN_MARGIN and
                SCREEN_MARGIN <= target_y <= self.height - SCREEN_MARGIN):
                
                return {
                    'x': int(target_x),
                    'y': int(target_y),
                    'distance': distance,
                    'orientation': orientation,
                    'angle_deg': angle_deg,
                }
        
        # Fallback: spawn somewhere safe near center
        print(f"  WARNING: Could not spawn valid target after {max_retries} attempts")
        return {
            'x': self.width // 2 + random.randint(-100, 100),
            'y': self.height // 2 + random.randint(-100, 100),
            'distance': 100,
            'orientation': 'E',
            'angle_deg': 0,
        }
    
    def shift_dots(self):
        """
        Shift dots after B is clicked:
        - A becomes B (we just reached here)
        - B becomes C (our new target)
        - C becomes new random target
        """
        # Compute AB and BC vectors before shift (for logging)
        vec_AB = (self.dot_B['x'] - self.dot_A['x'], self.dot_B['y'] - self.dot_A['y'])
        vec_BC = (self.dot_C['x'] - self.dot_B['x'], self.dot_C['y'] - self.dot_B['y'])
        turn_angle = angle_between_vectors(vec_AB, vec_BC)
        
        # Shift
        self.dot_A = {
            'x': self.dot_B['x'],
            'y': self.dot_B['y'],
        }
        
        self.dot_B = self.dot_C.copy()
        
        self.dot_C = self.spawn_target_from(self.dot_B)
        
        return turn_angle
    
    # =========================================================================
    # DRAWING
    # =========================================================================
    
    def draw_dots(self):
        """Draw all three dots with labels."""
        self.canvas.delete("dots")
        
        # Draw connecting lines (faint, to show path)
        if self.dot_A and self.dot_B:
            self.canvas.create_line(
                self.dot_A['x'], self.dot_A['y'],
                self.dot_B['x'], self.dot_B['y'],
                fill="#333333", width=2, dash=(5, 5), tags="dots"
            )
        if self.dot_B and self.dot_C:
            self.canvas.create_line(
                self.dot_B['x'], self.dot_B['y'],
                self.dot_C['x'], self.dot_C['y'],
                fill="#222222", width=2, dash=(5, 5), tags="dots"
            )
        
        # Draw C (next target) - smallest, in back
        if self.dot_C:
            r = DOT_RADIUS - 5
            self.canvas.create_oval(
                self.dot_C['x'] - r, self.dot_C['y'] - r,
                self.dot_C['x'] + r, self.dot_C['y'] + r,
                fill=DOT_C_COLOR, outline="#ffffff", width=2, tags="dots"
            )
            self.canvas.create_text(
                self.dot_C['x'], self.dot_C['y'] - r - 15,
                text="C (next)", font=("Arial", 10, "bold"),
                fill=DOT_C_COLOR, tags="dots"
            )
        
        # Draw B (current target)
        if self.dot_B:
            r = DOT_RADIUS
            self.canvas.create_oval(
                self.dot_B['x'] - r, self.dot_B['y'] - r,
                self.dot_B['x'] + r, self.dot_B['y'] + r,
                fill=DOT_B_COLOR, outline="#ffffff", width=3, tags="dots"
            )
            self.canvas.create_text(
                self.dot_B['x'], self.dot_B['y'] - r - 15,
                text="B (target)", font=("Arial", 11, "bold"),
                fill=DOT_B_COLOR, tags="dots"
            )
        
        # Draw A (origin) - changes color based on recording state
        if self.dot_A:
            r = DOT_RADIUS
            fill_color = "#90EE90" if self.recording_enabled else DOT_A_COLOR
            self.canvas.create_oval(
                self.dot_A['x'] - r, self.dot_A['y'] - r,
                self.dot_A['x'] + r, self.dot_A['y'] + r,
                fill=fill_color, outline="#ffffff", width=3, tags="dots"
            )
            label = "A (recording...)" if self.recording_enabled else "A (pass through)"
            self.canvas.create_text(
                self.dot_A['x'], self.dot_A['y'] - r - 15,
                text=label, font=("Arial", 11, "bold"),
                fill=DOT_A_COLOR, tags="dots"
            )
    
    def update_ui(self):
        """Update status display."""
        if not self.is_active:
            return
        
        self.canvas.delete("ui")
        
        # Status bar at top
        status = "RECORDING" if self.recording_enabled else "WAITING"
        progress = f"{self.segments_recorded}/{TARGET_SEGMENTS}"
        
        info_parts = [status, progress]
        
        if self.dot_B:
            info_parts.append(f"B: {self.dot_B.get('distance', '?')}px {self.dot_B.get('orientation', '?')}")
        if self.dot_C:
            info_parts.append(f"C: {self.dot_C.get('distance', '?')}px {self.dot_C.get('orientation', '?')}")
        
        info_text = " | ".join(info_parts)
        
        self.canvas.create_text(
            self.width // 2, 25,
            text=info_text,
            font=("Arial", 14),
            fill="gray", tags="ui"
        )
        
        # Schedule next update
        if self.is_active:
            self.master.after(200, self.update_ui)
    
    # =========================================================================
    # SAMPLING LOOP (125Hz)
    # =========================================================================
    
    def _sample_loop(self):
        """Poll mouse position at 125Hz, record when enabled, detect dot entry."""
        interval = 1.0 / SAMPLE_RATE_HZ
        next_time = time.perf_counter()
        
        while self.sampling_active:
            now = time.perf_counter()
            
            if self.is_active:
                try:
                    # Get mouse position
                    pos = self.mouse_controller.position
                    
                    with self.canvas_lock:
                        canvas_x = pos[0] - self.canvas_x
                        canvas_y = pos[1] - self.canvas_y
                    
                    # Check bounds
                    if 0 <= canvas_x <= self.width and 0 <= canvas_y <= self.height:
                        timestamp = int((time.time() * 1000) - self.session_start_time)
                        
                        # === DOT ENTRY DETECTION ===
                        self._check_dot_entry(canvas_x, canvas_y, timestamp)
                        
                        # === RECORD MOVEMENT (if recording) ===
                        if self.recording_enabled:
                            # Add to current segment trajectory
                            with self.data_lock:
                                self.segment_trajectory.append((timestamp, canvas_x, canvas_y))
                            
                            # Write to CSV
                            with self.csv_lock:
                                if self.csv_writer:
                                    self.csv_writer.writerow({
                                        'timestamp': timestamp,
                                        'event': 'move',
                                        'x': int(canvas_x),
                                        'y': int(canvas_y)
                                    })
                
                except Exception as e:
                    pass  # Ignore sampling errors
            
            # Maintain timing
            next_time += interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()  # Reset if we fell behind
    
    def _check_dot_entry(self, x, y, timestamp):
        """
        Check if cursor has entered A or B dot.
        
        This enables pass-through recording - no clicking needed!
        User just flows through the dots naturally.
        """
        # Prevent rapid re-triggering (debounce)
        if timestamp - self._last_trigger_time < self._min_trigger_interval_ms:
            return
        
        # --- Check A entry (start recording) ---
        if self.dot_A and not self.recording_enabled:
            inside_A_now = self._is_inside_dot(x, y, self.dot_A)
            
            if inside_A_now and not self._inside_A:
                # Just entered A!
                self._inside_A = True
                self._last_trigger_time = timestamp
                # Schedule on main thread to avoid Tkinter issues
                self.master.after(0, lambda: self._trigger_start_recording(x, y, timestamp))
            elif not inside_A_now:
                self._inside_A = False
        
        # --- Check B entry (complete segment) ---
        if self.dot_B and self.recording_enabled:
            inside_B_now = self._is_inside_dot(x, y, self.dot_B)
            
            if inside_B_now and not self._inside_B:
                # Just entered B!
                self._inside_B = True
                self._last_trigger_time = timestamp
                # Schedule on main thread
                self.master.after(0, lambda: self._trigger_complete_segment(x, y, timestamp))
            elif not inside_B_now:
                self._inside_B = False
    
    def _trigger_start_recording(self, x, y, timestamp):
        """Called from main thread to start recording."""
        if not self.recording_enabled:  # Double-check state
            self.start_recording(x, y, timestamp)
    
    def _trigger_complete_segment(self, x, y, timestamp):
        """Called from main thread to complete segment."""
        if self.recording_enabled:  # Double-check state
            self.complete_segment(x, y, timestamp)
    
    # =========================================================================
    # CLICK HANDLING
    # =========================================================================
    
    def on_click(self, event):
        """
        Handle mouse click on canvas.
        
        This is now a BACKUP method - primary detection is via cursor entry.
        Click can be used if user prefers explicit confirmation.
        """
        if not self.is_active:
            return
        
        canvas_x = event.x
        canvas_y = event.y
        timestamp = int((time.time() * 1000) - self.session_start_time)
        
        # Debounce - prevent double-triggers
        if timestamp - self._last_trigger_time < self._min_trigger_interval_ms:
            return
        
        # Check if clicked near A (start recording)
        if self.dot_A and not self.recording_enabled:
            if self._is_inside_dot(canvas_x, canvas_y, self.dot_A):
                self._inside_A = True
                self._last_trigger_time = timestamp
                self.start_recording(canvas_x, canvas_y, timestamp)
                return
        
        # Check if clicked near B (complete segment)
        if self.dot_B and self.recording_enabled:
            if self._is_inside_dot(canvas_x, canvas_y, self.dot_B):
                self._inside_B = True
                self._last_trigger_time = timestamp
                self.complete_segment(canvas_x, canvas_y, timestamp)
                return
    
    def start_recording(self, x, y, timestamp):
        """Start recording a new segment."""
        self.recording_enabled = True
        self.segment_start_time = timestamp
        
        with self.data_lock:
            self.segment_trajectory = [(timestamp, x, y)]
        
        # Write event to CSV
        with self.csv_lock:
            if self.csv_writer:
                self.csv_writer.writerow({
                    'timestamp': timestamp,
                    'event': 'pass_A',  # User entered/passed through A
                    'x': int(x),
                    'y': int(y)
                })
        
        self.draw_dots()
        print(f"  ▶ Recording started (entered A)")
    
    def complete_segment(self, x, y, timestamp):
        """Complete current segment and shift dots - CONTINUOUS FLOW."""
        # Add final point
        with self.data_lock:
            self.segment_trajectory.append((timestamp, x, y))
            trajectory_copy = list(self.segment_trajectory)
        
        # Write click event to CSV (marks segment boundary)
        with self.csv_lock:
            if self.csv_writer:
                self.csv_writer.writerow({
                    'timestamp': timestamp,
                    'event': 'pass_B',  # Changed from click_B
                    'x': int(x),
                    'y': int(y)
                })
                self.csv_file.flush()
        
        # Save segment data BEFORE shifting dots
        self.save_segment(trajectory_copy, timestamp)
        
        # Update statistics
        self.segments_recorded += 1
        
        # Compute and log turn angle
        vec_AB = (self.dot_B['x'] - self.dot_A['x'], self.dot_B['y'] - self.dot_A['y'])
        vec_BC = (self.dot_C['x'] - self.dot_B['x'], self.dot_C['y'] - self.dot_B['y'])
        turn_angle = angle_between_vectors(vec_AB, vec_BC)
        
        # Update stats
        self.stats['turn_angles'].append(turn_angle)
        dist_AB = math.hypot(*vec_AB)
        dist_BC = math.hypot(*vec_BC)
        self.stats['distances_AB'].append(dist_AB)
        self.stats['distances_BC'].append(dist_BC)
        self.stats['orientations_AB'][self.dot_B.get('orientation', '?')] += 1
        self.stats['orientations_BC'][self.dot_C.get('orientation', '?')] += 1
        
        # Check if target reached
        if self.segments_recorded >= TARGET_SEGMENTS:
            self.recording_enabled = False
            self.end_session()
            return
        
        # === CONTINUOUS FLOW: Shift dots and IMMEDIATELY continue ===
        
        # Shift dots: A←B, B←C, new C
        self.shift_dots()
        
        # Reset entry detection state for NEW positions
        # User is now at new A (was B), so mark as inside
        self._inside_A = True   # We're already inside new A
        self._inside_B = False  # New B is elsewhere, not inside yet
        
        # Keep recording enabled! Don't stop.
        # Just reset the segment trajectory for the next segment
        self.segment_start_time = timestamp
        with self.data_lock:
            self.segment_trajectory = [(timestamp, x, y)]  # Start new segment from here
        
        # Write start of new segment to CSV
        with self.csv_lock:
            if self.csv_writer:
                self.csv_writer.writerow({
                    'timestamp': timestamp,
                    'event': 'pass_A',  # Marks start of new segment
                    'x': int(x),
                    'y': int(y)
                })
        
        self.draw_dots()
    
    def save_segment(self, trajectory, end_timestamp):
        """Save segment data to JSON file."""
        segment_num = self.segments_recorded + 1  # Before increment
        
        # Compute vectors
        vec_AB = (self.dot_B['x'] - self.dot_A['x'], self.dot_B['y'] - self.dot_A['y'])
        vec_BC = (self.dot_C['x'] - self.dot_B['x'], self.dot_C['y'] - self.dot_B['y'])
        
        # Distances
        dist_AB = math.hypot(*vec_AB)
        dist_BC = math.hypot(*vec_BC)
        
        # Angles
        angle_AB = math.degrees(math.atan2(vec_AB[1], vec_AB[0]))
        angle_BC = math.degrees(math.atan2(vec_BC[1], vec_BC[0]))
        turn_angle = angle_between_vectors(vec_AB, vec_BC)
        
        # Orientations
        orient_AB = get_orientation_from_angle(angle_AB)
        orient_BC = get_orientation_from_angle(angle_BC)
        
        # Distance groups
        dist_group_AB, dist_name_AB = get_distance_group(dist_AB)
        dist_group_BC, dist_name_BC = get_distance_group(dist_BC)
        
        # Build segment data
        segment_data = {
            'segment_id': segment_num,
            'timestamp_start': self.segment_start_time,
            'timestamp_end': end_timestamp,
            'duration_ms': end_timestamp - self.segment_start_time,
            
            # Positions
            'A': {'x': self.dot_A['x'], 'y': self.dot_A['y']},
            'B': {'x': self.dot_B['x'], 'y': self.dot_B['y']},
            'C': {'x': self.dot_C['x'], 'y': self.dot_C['y']},
            
            # Current segment (A→B)
            'AB': {
                'distance': dist_AB,
                'distance_group': dist_group_AB,
                'distance_name': dist_name_AB,
                'angle_deg': angle_AB,
                'orientation': orient_AB,
                'orientation_id': get_orientation_id(orient_AB),
            },
            
            # Next segment (B→C) - THE KEY FOR ANTICIPATION
            'BC': {
                'distance': dist_BC,
                'distance_group': dist_group_BC,
                'distance_name': dist_name_BC,
                'angle_deg': angle_BC,
                'orientation': orient_BC,
                'orientation_id': get_orientation_id(orient_BC),
            },
            
            # Turn dynamics
            'turn': {
                'angle_deg': turn_angle,
                'category': get_turn_category(turn_angle),
            },
            
            # Trajectory data
            'trajectory': {
                'length': len(trajectory),
                'x': [int(p[1]) for p in trajectory],
                'y': [int(p[2]) for p in trajectory],
                'timestamps': [p[0] for p in trajectory],
            },
        }
        
        # Save to file
        filename = f"segment_{segment_num:04d}.json"
        filepath = os.path.join(self.segments_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(segment_data, f, indent=2)


# =============================================================================
# NUMPY IMPORT (for statistics)
# =============================================================================

try:
    import numpy as np
except ImportError:
    # Minimal fallback if numpy not available
    class np:
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x):
            if not x:
                return 0
            m = sum(x) / len(x)
            return (sum((xi - m) ** 2 for xi in x) / len(x)) ** 0.5


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()
    app = ThreeDotRecorder(root)
    root.mainloop()


if __name__ == "__main__":
    main()
