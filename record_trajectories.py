#!/usr/bin/env python3
"""
Trajectory Recorder for Cursor Path Predictor

Fullscreen UI for recording mouse trajectories (A->B movements).

Usage:
    python record_trajectories.py --targets 50
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
from typing import Optional, List, Tuple

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
    target_radius: int = 30
    min_target_dist: int = 200
    max_target_dist: int = 800
    screen_margin: int = 100
    min_trajectory_points: int = 20

    # Velocity thresholds
    start_velocity_threshold: float = 500.0  # px/s - must be below this to start
    end_velocity_threshold: float = 500.0    # px/s - must be below this to end
    movement_threshold: float = 5.0          # px/s - detect movement began


# =============================================================================
# CURSOR TRACKER (thread-safe - called from pynput thread)
# =============================================================================

class CursorTracker:
    """Thread-safe cursor tracking with velocity computation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._x: float = 0.0
        self._y: float = 0.0
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._last_time: float = 0.0

    def update(self, x: float, y: float) -> None:
        """Called from pynput thread."""
        now = time.perf_counter()
        with self._lock:
            if self._last_time > 0:
                dt = now - self._last_time
                if dt > 0.001:
                    self._vx = (x - self._x) / dt
                    self._vy = (y - self._y) / dt
            self._x = x
            self._y = y
            self._last_time = now

    def get_state(self) -> Tuple[float, float, float]:
        """Returns (x, y, speed). Called from main thread."""
        with self._lock:
            speed = math.sqrt(self._vx ** 2 + self._vy ** 2)
            return self._x, self._y, speed


# =============================================================================
# RECORDING UI
# =============================================================================

class RecordingUI:
    """Fullscreen UI for recording trajectories."""

    COLORS = {
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
        'panel_border': (40, 50, 65),
    }

    def __init__(self, config: RecordingConfig, output_dir: Path, target_count: int):
        self.config = config
        self.output_dir = output_dir
        self.target_count = target_count
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        """Run recording UI. Returns number of trajectories saved."""
        pygame.init()
        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h

        screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        pygame.display.set_caption("Trajectory Recorder")
        pygame.mouse.set_visible(False)

        # Fonts
        try:
            font_lg = pygame.font.SysFont('Segoe UI', 22, bold=True)
            font_md = pygame.font.SysFont('Segoe UI', 17)
            font_sm = pygame.font.SysFont('Consolas', 14)
        except:
            font_lg = pygame.font.Font(None, 28)
            font_md = pygame.font.Font(None, 22)
            font_sm = pygame.font.Font(None, 18)

        # Cursor tracker (thread-safe)
        tracker = CursorTracker()

        # Start mouse listener
        listener = mouse.Listener(on_move=lambda x, y: tracker.update(x, y))
        listener.start()
        time.sleep(0.05)

        # State
        x, y, speed = tracker.get_state()
        if x == 0 and y == 0:
            x, y = screen_w // 2, screen_h // 2

        target = self._spawn_target((x, y), screen_w, screen_h)
        trail: deque = deque(maxlen=150)
        clock = pygame.time.Clock()

        # Recording state (single source of truth)
        recording = False
        was_slow = True
        rec_positions: List[Tuple[float, float]] = []
        rec_timestamps: List[float] = []
        rec_start: Tuple[float, float] = (0, 0)
        rec_target: Tuple[float, float] = (0, 0)

        # Session stats
        saved_count = 0
        total_points = 0
        hit_flash = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*60}")
        print("TRAJECTORY RECORDING")
        print(f"{'='*60}")
        print(f"Output: {self.output_dir}")
        print(f"Goal: {self.target_count} trajectories")
        print("Move to targets. ESC=finish, R=skip")
        print("="*60 + "\n")

        running = True
        start_time = time.perf_counter()

        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        recording = False
                        was_slow = True
                        rec_positions.clear()
                        trail.clear()
                        x, y, _ = tracker.get_state()
                        target = self._spawn_target((x, y), screen_w, screen_h)

            # Check completion
            elapsed = time.perf_counter() - start_time
            if saved_count >= self.target_count:
                running = False
                continue

            # Get cursor state
            x, y, speed = tracker.get_state()
            pos = (x, y)

            # Update trail
            if not trail or abs(x - trail[-1][0]) > 1 or abs(y - trail[-1][1]) > 1:
                trail.append(pos)

            # Recording state machine
            if speed < self.config.start_velocity_threshold:
                was_slow = True

            # Start recording: was slow, now moving
            if not recording and was_slow and speed > self.config.movement_threshold:
                recording = True
                was_slow = False
                rec_positions = [pos]
                rec_timestamps = [time.perf_counter()]
                rec_start = pos
                rec_target = target
                trail.clear()

            # Record point
            if recording:
                rec_positions.append(pos)
                rec_timestamps.append(time.perf_counter())

            # Check target hit (only if recording!)
            dx, dy = x - target[0], y - target[1]
            in_target = math.sqrt(dx*dx + dy*dy) < self.config.target_radius
            settled = speed < self.config.end_velocity_threshold

            # Complete trajectory: must be recording + in target + settled
            if recording and in_target and settled:
                hit_flash = 20

                # Save if valid
                if len(rec_positions) >= self.config.min_trajectory_points:
                    self._save_trajectory(
                        rec_positions, rec_timestamps, rec_start, rec_target,
                        timestamp, saved_count
                    )
                    total_points += len(rec_positions)
                    saved_count += 1
                    print(f"  [OK] Trajectory {saved_count}: {len(rec_positions)} pts, "
                          f"{(rec_timestamps[-1] - rec_timestamps[0]) * 1000:.0f}ms")

                # Reset
                recording = False
                was_slow = True
                rec_positions.clear()
                trail.clear()
                target = self._spawn_target(pos, screen_w, screen_h)

            # === DRAWING ===
            screen.fill(self.COLORS['bg'])

            # Grid
            for gx in range(0, screen_w, 60):
                pygame.draw.line(screen, self.COLORS['grid'], (gx, 0), (gx, screen_h))
            for gy in range(0, screen_h, 60):
                pygame.draw.line(screen, self.COLORS['grid'], (0, gy), (screen_w, gy))

            # Stats panel
            self._draw_panel(screen, font_lg, font_md, font_sm, elapsed, saved_count, total_points)

            # Target
            tx, ty = int(target[0]), int(target[1])
            for r in range(60, 0, -6):
                alpha = int(50 * (1 - r / 60))
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*self.COLORS['target_glow'], alpha), (r, r), r)
                screen.blit(surf, (tx - r, ty - r))
            color = self.COLORS['target_hit'] if hit_flash > 0 else self.COLORS['target']
            pygame.draw.circle(screen, color, (tx, ty), self.config.target_radius, 3)
            pygame.draw.circle(screen, (255, 220, 220), (tx, ty), 6)
            if hit_flash > 0:
                hit_flash -= 1

            # Trail
            if len(trail) > 1:
                pts = list(trail)
                for i in range(len(pts) - 1):
                    t = i / len(pts)
                    if recording:
                        base = self.COLORS['trail_recording']
                        c = tuple(int(base[j] * (0.3 + 0.7 * t)) for j in range(3))
                        w = 2
                    else:
                        base = self.COLORS['trail']
                        c = tuple(int(base[j] + (140 - base[j]) * t) for j in range(3))
                        w = 1
                    pygame.draw.line(screen, c,
                                     (int(pts[i][0]), int(pts[i][1])),
                                     (int(pts[i+1][0]), int(pts[i+1][1])), w)

            # Cursor
            cx, cy = int(x), int(y)
            ring_color = self.COLORS['warning'] if recording else self.COLORS['cursor_ring']
            for r in range(25, 0, -5):
                alpha = int(60 * (1 - r / 25))
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*ring_color, alpha), (r, r), r)
                screen.blit(surf, (cx - r, cy - r))
            pygame.draw.circle(screen, ring_color, (cx, cy), 12, 2)
            pygame.draw.circle(screen, self.COLORS['cursor'], (cx, cy), 4)

            # Help text
            help_surf = font_sm.render("Move to targets | ESC: Finish | R: Skip",
                                       True, self.COLORS['text_dim'])
            screen.blit(help_surf, (screen_w // 2 - help_surf.get_width() // 2, screen_h - 35))

            # Recording indicator
            if recording:
                rec_surf = font_md.render(f"* RECORDING ({len(rec_positions)} pts)",
                                          True, self.COLORS['warning'])
                screen.blit(rec_surf, (screen_w // 2 - rec_surf.get_width() // 2, 25))

            pygame.display.flip()
            clock.tick(120)

        # Cleanup
        listener.stop()
        pygame.mouse.set_visible(True)
        pygame.quit()

        print(f"\n{'='*60}")
        print(f"COMPLETE: {saved_count} trajectories, {total_points} points")
        print(f"Saved to: {self.output_dir}")
        print("="*60 + "\n")

        return saved_count

    def _spawn_target(self, from_pos: Tuple[float, float],
                      screen_w: int, screen_h: int) -> Tuple[float, float]:
        """Spawn target at random position away from cursor."""
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(self.config.min_target_dist, self.config.max_target_dist)
        tx = from_pos[0] + dist * math.cos(angle)
        ty = from_pos[1] + dist * math.sin(angle)
        margin = self.config.screen_margin
        tx = max(margin, min(screen_w - margin, tx))
        ty = max(margin, min(screen_h - margin, ty))
        return (tx, ty)

    def _save_trajectory(self, positions: List[Tuple[float, float]],
                         timestamps: List[float],
                         start: Tuple[float, float],
                         target: Tuple[float, float],
                         session_ts: str, index: int) -> None:
        """Save trajectory to JSON file immediately."""
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Compute distances
        ideal_dist = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        actual_dist = sum(
            math.sqrt((xs[i+1] - xs[i])**2 + (ys[i+1] - ys[i])**2)
            for i in range(len(xs) - 1)
        )

        data = {
            'x': xs,
            'y': ys,
            'timestamps': timestamps,
            'start': list(start),
            'target': list(target),
            'ideal_distance': ideal_dist,
            'actual_distance': actual_dist,
            'duration_ms': (timestamps[-1] - timestamps[0]) * 1000,
            'point_count': len(positions),
        }

        filename = self.output_dir / f"traj_{session_ts}_{index:04d}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _draw_panel(self, screen, font_lg, font_md, font_sm,
                    elapsed: float, count: int, total_pts: int) -> None:
        """Draw stats panel."""
        panel_w, panel_h = 280, 140
        panel_x, panel_y = 20, 20

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (18, 22, 32, 180), (0, 0, panel_w, panel_h), border_radius=10)
        screen.blit(panel, (panel_x, panel_y))
        pygame.draw.rect(screen, self.COLORS['panel_border'],
                         (panel_x, panel_y, panel_w, panel_h), 1, border_radius=10)

        x, y = panel_x + 15, panel_y + 12

        screen.blit(font_lg.render("TRAJECTORY RECORDER", True, self.COLORS['accent']), (x, y))
        y += 28

        txt = f"Progress: {count} / {self.target_count}"
        col = self.COLORS['success'] if count >= self.target_count else self.COLORS['text']
        screen.blit(font_md.render(txt, True, col), (x, y))
        y += 22

        screen.blit(font_md.render(f"Total points: {total_pts:,}", True, self.COLORS['text']), (x, y))
        y += 22

        screen.blit(font_sm.render(f"Elapsed: {elapsed:.0f}s", True, self.COLORS['text_dim']), (x, y))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Record mouse trajectories for training',
        epilog="Example: python record_trajectories.py --targets 50"
    )
    parser.add_argument('--output', '-o', default='recorded_trajectories',
                        help='Output directory (default: recorded_trajectories)')
    parser.add_argument('--targets', '-t', type=int, default=50,
                        help='Number of trajectories to record (default: 50)')

    args = parser.parse_args()

    if not HAS_PYNPUT:
        print("ERROR: pip install pynput")
        return
    if not HAS_PYGAME:
        print("ERROR: pip install pygame")
        return

    config = RecordingConfig()
    ui = RecordingUI(config, Path(args.output), args.targets)
    ui.run()


if __name__ == '__main__':
    main()
