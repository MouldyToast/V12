#!/usr/bin/env python3
"""
Trajectory Recorder for Cursor Path Predictor

Professional fullscreen UI for recording mouse trajectories (A->B movements).
Records cursor movements as training data for the cursor path predictor.

Features:
- Fullscreen target-based recording UI
- Thread-safe cursor tracking with velocity/acceleration
- Automatic trajectory segmentation
- JSON export for training

Usage:
    # Record 50 trajectories with fullscreen UI (recommended)
    python record_trajectories.py --targets 50
    
    # Record for 5 minutes
    python record_trajectories.py --duration 300
    
    # Console-only recording (no pygame required)
    python record_trajectories.py --console --duration 60
    
    # Custom output directory
    python record_trajectories.py --output my_trajectories/ --targets 100
"""

import numpy as np
from pathlib import Path
import argparse
import json
import math
import time
import threading
import random
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

try:
    from pynput import mouse
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RecordingConfig:
    """Configuration for recording UI."""
    # Target task
    target_radius: int = 30
    min_target_dist: int = 200
    max_target_dist: int = 800
    
    # Recording
    min_trajectory_points: int = 20
    sample_rate: int = 125  # Hz (for metadata)
    
    # UI
    screen_margin: int = 100


# =============================================================================
# THREAD-SAFE CURSOR TRACKER
# =============================================================================

class CursorTracker:
    """Thread-safe cursor state tracking with velocity/acceleration computation."""
    
    def __init__(self, max_history: int = 500):
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._positions: deque = deque(maxlen=max_history)
        self._timestamps: deque = deque(maxlen=max_history)
        self._x: float = 0.0
        self._y: float = 0.0
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._ax: float = 0.0
        self._ay: float = 0.0
        self._last_update: float = 0.0
    
    def update(self, x: float, y: float) -> None:
        """Thread-safe position update with velocity/acceleration computation."""
        now = time.perf_counter()
        
        with self._lock:
            if self._timestamps:
                dt = now - self._timestamps[-1]
                if dt > 0.0005:  # Minimum 0.5ms between updates
                    # Compute velocity
                    new_vx = (x - self._x) / dt
                    new_vy = (y - self._y) / dt
                    
                    # Compute acceleration
                    self._ax = (new_vx - self._vx) / dt
                    self._ay = (new_vy - self._vy) / dt
                    
                    self._vx = new_vx
                    self._vy = new_vy
            
            self._x = x
            self._y = y
            self._positions.append((x, y))
            self._timestamps.append(now)
            self._last_update = now
    
    def get_position(self) -> Tuple[float, float]:
        """Thread-safe position getter."""
        with self._lock:
            return (self._x, self._y)
    
    def get_state(self) -> np.ndarray:
        """Get full state vector [x, y, vx, vy, ax, ay]."""
        with self._lock:
            return np.array([
                self._x, self._y,
                self._vx, self._vy,
                self._ax, self._ay
            ], dtype=np.float32)
    
    def get_speed(self) -> float:
        """Get current speed magnitude."""
        with self._lock:
            return math.sqrt(self._vx ** 2 + self._vy ** 2)
    
    def get_recent_positions(self, n: Optional[int] = None) -> List[Tuple[float, float]]:
        """Get recent position history."""
        with self._lock:
            if n is None:
                return list(self._positions)
            return list(self._positions)[-n:]
    
    def get_recent_timestamps(self, n: Optional[int] = None) -> List[float]:
        """Get recent timestamp history."""
        with self._lock:
            if n is None:
                return list(self._timestamps)
            return list(self._timestamps)[-n:]
    
    def clear_history(self) -> None:
        """Clear position history (keeps current state)."""
        with self._lock:
            self._positions.clear()
            self._timestamps.clear()
    
    def time_since_update(self) -> float:
        """Time since last position update."""
        with self._lock:
            if self._last_update == 0:
                return float('inf')
            return time.perf_counter() - self._last_update


# =============================================================================
# TRAJECTORY RECORDING
# =============================================================================

@dataclass
class RecordedTrajectory:
    """A recorded Aâ†’B cursor movement."""
    positions: List[Tuple[float, float]]
    timestamps: List[float]
    start_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    
    @property
    def duration_ms(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[-1] - self.timestamps[0]) * 1000
    
    @property
    def point_count(self) -> int:
        return len(self.positions)
    
    @property
    def ideal_distance(self) -> float:
        if not self.positions:
            return 0.0
        start = self.positions[0]
        end = self.positions[-1]
        return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    @property
    def actual_distance(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.positions) - 1):
            dx = self.positions[i+1][0] - self.positions[i][0]
            dy = self.positions[i+1][1] - self.positions[i][1]
            total += math.sqrt(dx*dx + dy*dy)
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        x = [p[0] for p in self.positions]
        y = [p[1] for p in self.positions]
        return {
            'x': x,
            'y': y,
            'timestamps': self.timestamps,
            'start': list(self.start_pos),
            'target': list(self.target_pos),
            'ideal_distance': float(self.ideal_distance),
            'actual_distance': float(self.actual_distance),
            'duration_ms': float(self.duration_ms),
            'point_count': self.point_count,
        }


class TrajectoryRecorder:
    """Thread-safe trajectory recording with start/stop control."""
    
    def __init__(self, config: RecordingConfig):
        self.config = config
        self._lock = threading.Lock()
        self._recording = False
        self._positions: List[Tuple[float, float]] = []
        self._timestamps: List[float] = []
        self._start_pos: Optional[Tuple[float, float]] = None
        self._target_pos: Optional[Tuple[float, float]] = None
    
    def start_recording(self, start_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> None:
        """Begin recording a new trajectory."""
        with self._lock:
            self._recording = True
            self._positions = []
            self._timestamps = []
            self._start_pos = start_pos
            self._target_pos = target_pos
    
    def record_point(self, pos: Tuple[float, float]) -> None:
        """Record a position point if recording is active."""
        with self._lock:
            if self._recording:
                self._positions.append(pos)
                self._timestamps.append(time.perf_counter())
    
    def stop_recording(self) -> Optional[RecordedTrajectory]:
        """Stop recording and return the trajectory if valid."""
        with self._lock:
            self._recording = False
            
            if (len(self._positions) >= self.config.min_trajectory_points and 
                self._start_pos is not None and 
                self._target_pos is not None):
                
                traj = RecordedTrajectory(
                    positions=self._positions.copy(),
                    timestamps=self._timestamps.copy(),
                    start_pos=self._start_pos,
                    target_pos=self._target_pos,
                )
                return traj
            return None
    
    def cancel_recording(self) -> None:
        """Cancel current recording without saving."""
        with self._lock:
            self._recording = False
            self._positions = []
            self._timestamps = []
    
    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording
    
    @property
    def current_point_count(self) -> int:
        with self._lock:
            return len(self._positions)


# =============================================================================
# TARGET MANAGER
# =============================================================================

class TargetManager:
    """Manages target spawning and hit detection."""
    
    def __init__(self, config: RecordingConfig, screen_width: int, screen_height: int):
        self.config = config
        self.width = screen_width
        self.height = screen_height
        self._lock = threading.Lock()
        self._target: Optional[Tuple[float, float]] = None
        self._spawn_time: float = 0.0
    
    def spawn_target(self, from_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Spawn a new target at a random position away from the cursor."""
        with self._lock:
            # Random angle and distance
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(self.config.min_target_dist, self.config.max_target_dist)
            
            # Calculate target position
            tx = from_pos[0] + dist * math.cos(angle)
            ty = from_pos[1] + dist * math.sin(angle)
            
            # Clamp to screen bounds with margin
            margin = self.config.screen_margin
            tx = max(margin, min(self.width - margin, tx))
            ty = max(margin, min(self.height - margin, ty))
            
            self._target = (tx, ty)
            self._spawn_time = time.perf_counter()
            
            return self._target
    
    def check_hit(self, pos: Tuple[float, float]) -> bool:
        """Check if position hits the current target."""
        with self._lock:
            if self._target is None:
                return False
            
            dx = pos[0] - self._target[0]
            dy = pos[1] - self._target[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            return distance < self.config.target_radius
    
    @property
    def target(self) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._target
    
    @property
    def time_since_spawn(self) -> float:
        with self._lock:
            if self._spawn_time == 0:
                return 0.0
            return time.perf_counter() - self._spawn_time


# =============================================================================
# RECORDING SESSION
# =============================================================================

class RecordingSession:
    """Manages a complete recording session with all trajectories."""
    
    def __init__(self, output_dir: Path, config: RecordingConfig):
        self.output_dir = output_dir
        self.config = config
        self._lock = threading.Lock()
        self._trajectories: List[RecordedTrajectory] = []
        self._start_time: float = time.perf_counter()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_trajectory(self, traj: RecordedTrajectory) -> int:
        """Add a trajectory to the session. Returns trajectory count."""
        with self._lock:
            self._trajectories.append(traj)
            return len(self._trajectories)
    
    @property
    def trajectory_count(self) -> int:
        with self._lock:
            return len(self._trajectories)
    
    @property
    def elapsed_time(self) -> float:
        return time.perf_counter() - self._start_time
    
    @property
    def total_points(self) -> int:
        with self._lock:
            return sum(t.point_count for t in self._trajectories)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        with self._lock:
            if not self._trajectories:
                return {
                    'count': 0,
                    'total_points': 0,
                    'avg_points': 0,
                    'avg_duration_ms': 0,
                    'avg_distance': 0,
                }
            
            durations = [t.duration_ms for t in self._trajectories]
            distances = [t.actual_distance for t in self._trajectories]
            points = [t.point_count for t in self._trajectories]
            
            return {
                'count': len(self._trajectories),
                'total_points': sum(points),
                'avg_points': np.mean(points),
                'avg_duration_ms': np.mean(durations),
                'avg_distance': np.mean(distances),
            }
    
    def save_all(self) -> int:
        """Save all trajectories to JSON files. Returns number saved."""
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, traj in enumerate(self._trajectories):
                filename = self.output_dir / f"traj_{timestamp}_{i:04d}.json"
                
                with open(filename, 'w') as f:
                    json.dump(traj.to_dict(), f, indent=2)
            
            return len(self._trajectories)


# =============================================================================
# RECORDING UI
# =============================================================================

class RecordingUI:
    """Professional fullscreen UI for recording trajectories."""
    
    def __init__(self, config: RecordingConfig, output_dir: Path, 
                 duration: Optional[int] = None, target_count: Optional[int] = None):
        self.config = config
        self.output_dir = output_dir
        self.duration = duration
        self.target_count = target_count
        
        # Will be initialized in run()
        self.cursor_tracker: Optional[CursorTracker] = None
        self.recorder: Optional[TrajectoryRecorder] = None
        self.target_manager: Optional[TargetManager] = None
        self.session: Optional[RecordingSession] = None
        
        # Color palette (matching predict_realtime.py)
        self.COLORS = {
            'bg': (10, 12, 18),
            'grid': (22, 26, 35),
            'target': (255, 90, 90),
            'target_glow': (255, 60, 60),
            'target_hit': (90, 255, 120),
            'cursor': (255, 255, 255),
            'cursor_ring': (0, 200, 255),
            'trail': (50, 60, 80),
            'trail_recording': (255, 180, 50),
            'text': (200, 210, 230),
            'text_dim': (90, 100, 120),
            'accent': (0, 200, 255),
            'success': (90, 255, 140),
            'warning': (255, 180, 50),
            'panel_bg': (18, 22, 32, 230),
            'panel_border': (40, 50, 65),
        }
    
    def run(self) -> int:
        """Run the recording UI. Returns number of trajectories recorded."""
        if not HAS_PYGAME:
            print("ERROR: pygame required for recording UI. Install with: pip install pygame")
            return 0
        
        if not HAS_PYNPUT:
            print("ERROR: pynput required for mouse tracking. Install with: pip install pynput")
            return 0
        
        # Initialize pygame
        pygame.init()
        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h
        
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        pygame.display.set_caption("Trajectory Recorder")
        pygame.mouse.set_visible(False)
        
        # Initialize fonts
        try:
            self.font_lg = pygame.font.SysFont('Segoe UI', 22, bold=True)
            self.font_md = pygame.font.SysFont('Segoe UI', 17)
            self.font_sm = pygame.font.SysFont('Consolas', 14)
        except:
            self.font_lg = pygame.font.Font(None, 28)
            self.font_md = pygame.font.Font(None, 22)
            self.font_sm = pygame.font.Font(None, 18)
        
        # Initialize components
        self.cursor_tracker = CursorTracker()
        self.recorder = TrajectoryRecorder(self.config)
        self.target_manager = TargetManager(self.config, screen_w, screen_h)
        self.session = RecordingSession(self.output_dir, self.config)
        
        # Start mouse listener
        listener = mouse.Listener(on_move=self._on_mouse_move)
        listener.start()
        
        # Get initial position and spawn first target
        time.sleep(0.05)  # Brief delay to get initial position
        initial_pos = self.cursor_tracker.get_position()
        if initial_pos == (0, 0):
            initial_pos = (screen_w // 2, screen_h // 2)
        
        self.target_manager.spawn_target(initial_pos)
        
        # UI state
        clock = pygame.time.Clock()
        trail = deque(maxlen=150)
        hit_flash = 0
        recording_started = False
        movement_threshold = 50  # pixels/second to start recording
        
        print(f"\n{'='*60}")
        print("TRAJECTORY RECORDING")
        print(f"{'='*60}")
        print(f"Display: {screen_w}x{screen_h}")
        print(f"Output: {self.output_dir}")
        if self.target_count:
            print(f"Goal: {self.target_count} trajectories")
        elif self.duration:
            print(f"Duration: {self.duration} seconds")
        print("\nMove cursor to targets. Press ESC to finish early.")
        print("=" * 60 + "\n")
        
        running = True
        start_time = time.perf_counter()
        
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset/skip current target
                        self.recorder.cancel_recording()
                        recording_started = False
                        trail.clear()
                        pos = self.cursor_tracker.get_position()
                        self.target_manager.spawn_target(pos)
            
            # Check completion conditions
            elapsed = time.perf_counter() - start_time
            
            if self.target_count and self.session.trajectory_count >= self.target_count:
                running = False
                continue
            
            if self.duration and elapsed >= self.duration:
                running = False
                continue
            
            # Clear screen
            screen.fill(self.COLORS['bg'])
            
            # Draw grid
            for x in range(0, screen_w, 60):
                pygame.draw.line(screen, self.COLORS['grid'], (x, 0), (x, screen_h))
            for y in range(0, screen_h, 60):
                pygame.draw.line(screen, self.COLORS['grid'], (0, y), (screen_w, y))
            
            # Draw stats panel (on background layer, so targets appear on top)
            target = self.target_manager.target
            self._draw_stats_panel(screen, screen_w, screen_h, elapsed, target)
            
            # Get current state
            pos = self.cursor_tracker.get_position()
            speed = self.cursor_tracker.get_speed()
            
            # Update trail
            if not trail or (abs(pos[0] - trail[-1][0]) > 1 or abs(pos[1] - trail[-1][1]) > 1):
                trail.append(pos)
            
            # Recording logic
            if target and not recording_started and speed > movement_threshold:
                # Start recording when movement begins
                recording_started = True
                start_pos = self.cursor_tracker.get_recent_positions(1)[0] if self.cursor_tracker.get_recent_positions() else pos
                self.recorder.start_recording(start_pos, target)
                trail.clear()
            
            if self.recorder.is_recording:
                self.recorder.record_point(pos)
            
            # Check for target hit
            if target and self.target_manager.check_hit(pos):
                hit_flash = 25
                
                # Stop recording and save trajectory
                traj = self.recorder.stop_recording()
                if traj:
                    count = self.session.add_trajectory(traj)
                    print(f"  âœ“ Trajectory {count}: {traj.point_count} points, {traj.duration_ms:.0f}ms")
                
                # Spawn new target
                recording_started = False
                trail.clear()
                self.target_manager.spawn_target(pos)
            
            # Draw target
            if target:
                tx, ty = int(target[0]), int(target[1])
                
                # Glow effect
                for r in range(60, 0, -6):
                    alpha = int(50 * (1 - r / 60))
                    glow_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*self.COLORS['target_glow'], alpha), (r, r), r)
                    screen.blit(glow_surf, (tx - r, ty - r))
                
                # Target circle
                color = self.COLORS['target_hit'] if hit_flash > 0 else self.COLORS['target']
                pygame.draw.circle(screen, color, (tx, ty), self.config.target_radius, 3)
                pygame.draw.circle(screen, (255, 220, 220), (tx, ty), 6)
                
                if hit_flash > 0:
                    hit_flash -= 1
            
            # Draw trail
            if len(trail) > 1:
                pts = list(trail)
                is_recording = self.recorder.is_recording
                
                for i in range(len(pts) - 1):
                    t = i / len(pts)
                    
                    if is_recording:
                        # Orange trail while recording
                        base = self.COLORS['trail_recording']
                        c = tuple(int(base[j] * (0.3 + 0.7 * t)) for j in range(3))
                    else:
                        # Gray trail when idle
                        base = self.COLORS['trail']
                        c = tuple(int(base[j] + (140 - base[j]) * t) for j in range(3))
                    
                    p1 = (int(pts[i][0]), int(pts[i][1]))
                    p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
                    width = 2 if is_recording else 1
                    pygame.draw.line(screen, c, p1, p2, width)
            
            # Draw cursor
            cx, cy = int(pos[0]), int(pos[1])
            
            # Cursor glow
            for r in range(25, 0, -5):
                alpha = int(60 * (1 - r / 25))
                glow_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                ring_color = self.COLORS['warning'] if self.recorder.is_recording else self.COLORS['cursor_ring']
                pygame.draw.circle(glow_surf, (*ring_color, alpha), (r, r), r)
                screen.blit(glow_surf, (cx - r, cy - r))
            
            # Cursor rings
            ring_color = self.COLORS['warning'] if self.recorder.is_recording else self.COLORS['cursor_ring']
            pygame.draw.circle(screen, ring_color, (cx, cy), 12, 2)
            pygame.draw.circle(screen, self.COLORS['cursor'], (cx, cy), 4)
            
            # Draw help text
            help_text = "Move to targets  |  ESC: Finish  |  R: Skip target"
            help_surf = self.font_sm.render(help_text, True, self.COLORS['text_dim'])
            screen.blit(help_surf, (screen_w // 2 - help_surf.get_width() // 2, screen_h - 35))
            
            # Recording indicator
            if self.recorder.is_recording:
                rec_text = f"â— RECORDING ({self.recorder.current_point_count} pts)"
                rec_surf = self.font_md.render(rec_text, True, self.COLORS['warning'])
                screen.blit(rec_surf, (screen_w // 2 - rec_surf.get_width() // 2, 25))
            
            pygame.display.flip()
            clock.tick(120)
        
        # Cleanup
        listener.stop()
        pygame.mouse.set_visible(True)
        pygame.quit()
        
        # Save all trajectories
        saved_count = self.session.save_all()
        
        print(f"\n{'='*60}")
        print("RECORDING COMPLETE")
        print(f"{'='*60}")
        print(f"Trajectories: {saved_count}")
        print(f"Total points: {self.session.total_points}")
        print(f"Saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return saved_count
    
    def _on_mouse_move(self, x: float, y: float) -> None:
        """Mouse movement callback (runs in pynput thread)."""
        if self.cursor_tracker:
            self.cursor_tracker.update(x, y)
    
    def _draw_stats_panel(self, screen, screen_w: int, screen_h: int, 
                            elapsed: float, target_pos: Optional[Tuple[float, float]]) -> None:
        """Draw the statistics panel with transparency."""
        stats = self.session.get_stats() if self.session else {}
        
        # Panel dimensions - fixed top-left position
        panel_w, panel_h = 320, 190
        margin = 20
        panel_x, panel_y = margin, margin
        
        # Draw panel background with transparency
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (18, 22, 32, 160), (0, 0, panel_w, panel_h), border_radius=12)
        screen.blit(panel, (panel_x, panel_y))
        pygame.draw.rect(screen, self.COLORS['panel_border'], 
                        (panel_x, panel_y, panel_w, panel_h), 1, border_radius=12)
        
        y = panel_y + 12
        x = panel_x + 18
        
        # Title
        title = self.font_lg.render("TRAJECTORY RECORDER", True, self.COLORS['accent'])
        screen.blit(title, (x, y))
        y += 30
        
        # Progress
        if self.target_count:
            progress = f"Progress: {stats.get('count', 0)} / {self.target_count} trajectories"
            pct = (stats.get('count', 0) / self.target_count) * 100
            col = self.COLORS['success'] if pct >= 100 else self.COLORS['text']
        elif self.duration:
            remaining = max(0, self.duration - elapsed)
            progress = f"Time remaining: {remaining:.0f}s"
            col = self.COLORS['warning'] if remaining < 30 else self.COLORS['text']
        else:
            progress = f"Trajectories: {stats.get('count', 0)}"
            col = self.COLORS['text']
        
        screen.blit(self.font_md.render(progress, True, col), (x, y))
        y += 24
        
        # Stats
        screen.blit(self.font_md.render(
            f"Total points: {stats.get('total_points', 0):,}", 
            True, self.COLORS['text']), (x, y))
        y += 22
        
        if stats.get('count', 0) > 0:
            screen.blit(self.font_md.render(
                f"Avg points/traj: {stats.get('avg_points', 0):.0f}", 
                True, self.COLORS['text']), (x, y))
            y += 22
            
            screen.blit(self.font_md.render(
                f"Avg duration: {stats.get('avg_duration_ms', 0):.0f}ms", 
                True, self.COLORS['text']), (x, y))
            y += 22
            
            screen.blit(self.font_md.render(
                f"Avg distance: {stats.get('avg_distance', 0):.0f}px", 
                True, self.COLORS['text']), (x, y))
        
        # Elapsed time (bottom of panel)
        elapsed_text = f"Elapsed: {elapsed:.0f}s"
        elapsed_surf = self.font_sm.render(elapsed_text, True, self.COLORS['text_dim'])
        screen.blit(elapsed_surf, (x, panel_y + panel_h - 28))


# =============================================================================
# CONSOLE RECORDING (Legacy/Fallback)
# =============================================================================

def record_trajectories_console(output_dir: Path, duration_seconds: int = 300) -> int:
    """
    Console-based recording (no UI).
    Records mouse movements and segments them into trajectories based on idle time.
    
    Args:
        output_dir: Directory to save trajectory JSON files
        duration_seconds: Recording duration in seconds
    
    Returns:
        Number of trajectories recorded
    """
    if not HAS_PYNPUT:
        print("ERROR: pynput required for recording. Install with: pip install pynput")
        return 0
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("TRAJECTORY RECORDING (Console Mode)")
    print(f"{'='*60}")
    print(f"Duration: {duration_seconds}s")
    print(f"Output: {output_dir}")
    print("\nMove your mouse naturally. Recording starts now!")
    print("Press Ctrl+C to stop early.\n")
    
    # Thread-safe state using locks
    lock = threading.Lock()
    current_trajectory = {'x': [], 'y': [], 'timestamps': []}
    trajectories = []
    last_move_time = [time.time()]  # Using list for mutable reference
    idle_threshold = 0.5
    
    def on_move(x, y):
        nonlocal current_trajectory
        now = time.time()
        
        with lock:
            # Check for idle (new trajectory)
            if now - last_move_time[0] > idle_threshold and current_trajectory['x']:
                if len(current_trajectory['x']) >= 20:
                    trajectories.append(current_trajectory.copy())
                current_trajectory = {'x': [], 'y': [], 'timestamps': []}
            
            current_trajectory['x'].append(float(x))
            current_trajectory['y'].append(float(y))
            current_trajectory['timestamps'].append(now)
            last_move_time[0] = now
    
    listener = mouse.Listener(on_move=on_move)
    listener.start()
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            with lock:
                traj_count = len(trajectories)
            print(f"\rRecording: {elapsed:.0f}s / {duration_seconds}s | Trajectories: {traj_count}", end='')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopped early.")
    
    listener.stop()
    
    # Save final trajectory
    with lock:
        if len(current_trajectory['x']) >= 20:
            trajectories.append(current_trajectory.copy())
    
    print(f"\n\nRecorded {len(trajectories)} trajectories")
    
    # Save as JSON files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, traj in enumerate(trajectories):
        filename = output_dir / f"console_{timestamp}_{i:04d}.json"
        
        x, y = np.array(traj['x']), np.array(traj['y'])
        ideal_dist = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
        diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        actual_dist = float(np.sum(diffs))
        
        data = {
            'x': traj['x'],
            'y': traj['y'],
            'timestamps': traj['timestamps'],
            'ideal_distance': ideal_dist,
            'actual_distance': actual_dist,
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    print(f"Saved to {output_dir}")
    return len(trajectories)


# =============================================================================
# MAIN API FUNCTION
# =============================================================================

def record_trajectories(output_dir: Path, duration: Optional[int] = None, 
                        target_count: Optional[int] = None) -> int:
    """
    Record trajectories using the fullscreen UI.
    
    Args:
        output_dir: Directory to save trajectory JSON files
        duration: Recording duration in seconds (optional)
        target_count: Number of targets to hit (optional)
    
    Returns:
        Number of trajectories recorded
    """
    if not HAS_PYGAME or not HAS_PYNPUT:
        print("Pygame or pynput not available. Falling back to console mode.")
        return record_trajectories_console(output_dir, duration or 300)
    
    config = RecordingConfig()
    ui = RecordingUI(config, output_dir, duration=duration, target_count=target_count)
    return ui.run()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Record mouse trajectories for cursor path prediction training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record 50 trajectories with fullscreen UI (recommended)
    python record_trajectories.py --targets 50
    
    # Record for 5 minutes
    python record_trajectories.py --duration 300
    
    # Console-only recording (no pygame required)
    python record_trajectories.py --console --duration 60
    
    # Custom output directory
    python record_trajectories.py --output my_trajectories/ --targets 100

After recording, train the model with:
    python train_streaming_mamba.py --data recorded_trajectories/ --epochs 100 --gpu-optimized
        """
    )
    
    parser.add_argument('--output', '-o', type=str, default='recorded_trajectories',
                        help='Output directory for trajectory JSON files (default: recorded_trajectories)')
    parser.add_argument('--targets', '-t', type=int, default=None,
                        help='Number of targets to hit (recommended method)')
    parser.add_argument('--duration', '-d', type=int, default=None,
                        help='Recording duration in seconds')
    parser.add_argument('--console', '-c', action='store_true',
                        help='Use console-only recording (no pygame UI)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Check dependencies
    if not HAS_PYNPUT:
        print("ERROR: pynput required for mouse tracking.")
        print("Install with: pip install pynput")
        return
    
    if not args.console and not HAS_PYGAME:
        print("WARNING: pygame not available. Falling back to console mode.")
        print("Install pygame for fullscreen UI: pip install pygame")
        args.console = True
    
    # Run recording
    if args.console:
        duration = args.duration or 300
        count = record_trajectories_console(output_dir, duration)
    else:
        if args.targets is None and args.duration is None:
            args.targets = 50
            print("No --targets or --duration specified. Defaulting to 50 targets.")
        
        count = record_trajectories(output_dir, duration=args.duration, target_count=args.targets)
    
    # Print next steps
    if count > 0:
        print(f"\nNext step - train the model:")
        print(f"  python train_streaming_mamba.py --data {output_dir} --epochs 100 --gpu-optimized")


if __name__ == '__main__':
    main()