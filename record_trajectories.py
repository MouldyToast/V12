#!/usr/bin/env python3
"""
Trajectory Recorder - Simple position-based recording.

Logic:
1. Spawn target
2. Wait until cursor is IN target AND hasn't moved > X pixels for Y ms
3. Start recording at 125Hz
4. Spawn next target
5. Record until cursor is IN next target AND hasn't moved > X pixels for Y ms
6. Save, repeat
"""

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
from typing import List, Tuple

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


@dataclass
class Config:
    target_radius: int = 30
    min_target_dist: int = 200
    max_target_dist: int = 800
    screen_margin: int = 100
    min_trajectory_points: int = 20

    # Settle detection: hasn't moved > settle_pixels for settle_time
    settle_pixels: float = 3.0      # pixels
    settle_time: float = 0.05       # 50ms

    # Sampling
    sample_interval: float = 0.008  # 8ms = 125Hz


class CursorTracker:
    """Thread-safe position tracking from pynput."""

    def __init__(self):
        self._lock = threading.Lock()
        self._x: float = 0.0
        self._y: float = 0.0

    def update(self, x: float, y: float) -> None:
        with self._lock:
            self._x = x
            self._y = y

    def get_pos(self) -> Tuple[float, float]:
        with self._lock:
            return self._x, self._y


class RecordingUI:
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

    def __init__(self, config: Config, output_dir: Path, target_count: int):
        self.config = config
        self.output_dir = output_dir
        self.target_count = target_count
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _is_settled(self, pos: Tuple[float, float], anchor: Tuple[float, float],
                    anchor_time: float, now: float) -> bool:
        """Check if cursor hasn't moved > settle_pixels for settle_time."""
        dist = math.sqrt((pos[0] - anchor[0])**2 + (pos[1] - anchor[1])**2)
        if dist > self.config.settle_pixels:
            return False
        return (now - anchor_time) >= self.config.settle_time

    def _in_target(self, pos: Tuple[float, float], target: Tuple[float, float]) -> bool:
        dx, dy = pos[0] - target[0], pos[1] - target[1]
        return math.sqrt(dx*dx + dy*dy) < self.config.target_radius

    def run(self) -> int:
        pygame.init()
        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h

        screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        pygame.display.set_caption("Trajectory Recorder")
        pygame.mouse.set_visible(False)

        try:
            font_lg = pygame.font.SysFont('Segoe UI', 22, bold=True)
            font_md = pygame.font.SysFont('Segoe UI', 17)
            font_sm = pygame.font.SysFont('Consolas', 14)
        except:
            font_lg = pygame.font.Font(None, 28)
            font_md = pygame.font.Font(None, 22)
            font_sm = pygame.font.Font(None, 18)

        tracker = CursorTracker()
        listener = mouse.Listener(on_move=lambda x, y: tracker.update(x, y))
        listener.start()
        time.sleep(0.05)

        # Get initial position
        pos = tracker.get_pos()
        if pos == (0.0, 0.0):
            pos = (screen_w / 2, screen_h / 2)

        target = self._spawn_target(pos, screen_w, screen_h)
        trail: deque = deque(maxlen=150)
        clock = pygame.time.Clock()

        # State
        recording = False
        anchor_pos = pos          # Position when cursor last moved significantly
        anchor_time = time.perf_counter()

        rec_positions: List[Tuple[float, float]] = []
        rec_timestamps: List[float] = []
        rec_target: Tuple[float, float] = (0, 0)
        last_sample_time: float = 0.0

        saved_count = 0
        total_points = 0
        hit_flash = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*60}")
        print("TRAJECTORY RECORDING")
        print(f"{'='*60}")
        print(f"Output: {self.output_dir}")
        print(f"Goal: {self.target_count} trajectories")
        print(f"Settle: {self.config.settle_pixels}px for {self.config.settle_time*1000:.0f}ms")
        print("Move to targets. ESC=finish, R=skip")
        print("="*60 + "\n")

        running = True
        start_time = time.perf_counter()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        recording = False
                        rec_positions.clear()
                        trail.clear()
                        pos = tracker.get_pos()
                        anchor_pos = pos
                        anchor_time = time.perf_counter()
                        target = self._spawn_target(pos, screen_w, screen_h)

            if saved_count >= self.target_count:
                running = False
                continue

            now = time.perf_counter()
            pos = tracker.get_pos()

            # Update anchor if cursor moved significantly
            dist_from_anchor = math.sqrt((pos[0] - anchor_pos[0])**2 + (pos[1] - anchor_pos[1])**2)
            if dist_from_anchor > self.config.settle_pixels:
                anchor_pos = pos
                anchor_time = now

            # Update trail
            if not trail or abs(pos[0] - trail[-1][0]) > 1 or abs(pos[1] - trail[-1][1]) > 1:
                trail.append(pos)

            in_target = self._in_target(pos, target)
            settled = self._is_settled(pos, anchor_pos, anchor_time, now)

            # STATE MACHINE
            if not recording:
                # Waiting to start: need to be in target AND settled
                if in_target and settled:
                    recording = True
                    rec_positions = [pos]
                    rec_timestamps = [now]
                    last_sample_time = now
                    rec_target = target
                    trail.clear()
                    # Spawn next target
                    target = self._spawn_target(pos, screen_w, screen_h)
                    print(f"  [START] Recording begun at ({pos[0]:.0f}, {pos[1]:.0f})")
            else:
                # Recording: sample at 125Hz
                if now - last_sample_time >= self.config.sample_interval:
                    rec_positions.append(pos)
                    rec_timestamps.append(now)
                    last_sample_time = now

                # Check for end: in target AND settled
                if in_target and settled:
                    hit_flash = 20

                    if len(rec_positions) >= self.config.min_trajectory_points:
                        self._save_trajectory(
                            rec_positions, rec_timestamps, rec_target,
                            timestamp, saved_count
                        )
                        total_points += len(rec_positions)
                        saved_count += 1
                        duration_ms = (rec_timestamps[-1] - rec_timestamps[0]) * 1000
                        print(f"  [OK] Trajectory {saved_count}: {len(rec_positions)} pts, {duration_ms:.0f}ms")
                    else:
                        print(f"  [SKIP] Only {len(rec_positions)} pts (need {self.config.min_trajectory_points})")

                    # Reset for next
                    recording = False
                    rec_positions.clear()
                    trail.clear()
                    target = self._spawn_target(pos, screen_w, screen_h)

            # === DRAWING ===
            elapsed = now - start_time
            screen.fill(self.COLORS['bg'])

            for gx in range(0, screen_w, 60):
                pygame.draw.line(screen, self.COLORS['grid'], (gx, 0), (gx, screen_h))
            for gy in range(0, screen_h, 60):
                pygame.draw.line(screen, self.COLORS['grid'], (0, gy), (screen_w, gy))

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
            cx, cy = int(pos[0]), int(pos[1])
            ring_color = self.COLORS['warning'] if recording else self.COLORS['cursor_ring']
            for r in range(25, 0, -5):
                alpha = int(60 * (1 - r / 25))
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*ring_color, alpha), (r, r), r)
                screen.blit(surf, (cx - r, cy - r))
            pygame.draw.circle(screen, ring_color, (cx, cy), 12, 2)
            pygame.draw.circle(screen, self.COLORS['cursor'], (cx, cy), 4)

            # Status
            help_surf = font_sm.render("Move to targets | ESC: Finish | R: Skip",
                                       True, self.COLORS['text_dim'])
            screen.blit(help_surf, (screen_w // 2 - help_surf.get_width() // 2, screen_h - 35))

            if recording:
                rec_surf = font_md.render(f"● RECORDING ({len(rec_positions)} pts)",
                                          True, self.COLORS['warning'])
                screen.blit(rec_surf, (screen_w // 2 - rec_surf.get_width() // 2, 25))
            elif not recording and in_target:
                wait_surf = font_md.render("Hold still to start...", True, self.COLORS['accent'])
                screen.blit(wait_surf, (screen_w // 2 - wait_surf.get_width() // 2, 25))

            pygame.display.flip()
            clock.tick(120)

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
                         target: Tuple[float, float],
                         session_ts: str, index: int) -> None:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        ideal_dist = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        actual_dist = sum(
            math.sqrt((xs[i+1] - xs[i])**2 + (ys[i+1] - ys[i])**2)
            for i in range(len(xs) - 1)
        )

        data = {
            'x': xs,
            'y': ys,
            'timestamps': timestamps,
            'start': [xs[0], ys[0]],
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


def main():
    parser = argparse.ArgumentParser(description='Record mouse trajectories')
    parser.add_argument('--output', '-o', default='recorded_trajectories')
    parser.add_argument('--targets', '-t', type=int, default=50)
    args = parser.parse_args()

    if not HAS_PYNPUT:
        print("ERROR: pip install pynput")
        return
    if not HAS_PYGAME:
        print("ERROR: pip install pygame")
        return

    config = Config()
    ui = RecordingUI(config, Path(args.output), args.targets)
    ui.run()


if __name__ == '__main__':
    main()
