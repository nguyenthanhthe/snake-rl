"""
Snake game with random maze generation — Gym‑compatible.
Grid‑based, features/grid observations, 3 actions.

The game is played inside a randomized grid maze with loops and rooms,
matching the visual style of the user's photo.
"""
import pygame
import numpy as np
from enum import IntEnum
from collections import deque

# ── constants ──────────────────────────────────────────────────────────────
class Action(IntEnum):
    STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2

# Directions as (dx, dy) — +y is DOWN (screen coords)
_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT

# Colours (R,G,B)
BLACK = (10, 10, 15)
GREY  = (100, 100, 100)      # grey for obstacles
WHITE = (240, 240, 240)
RED   = (255, 0, 0)          # pure red for food
GREEN = (0, 255, 0)          # bright green for snake head

# ── game class ─────────────────────────────────────────────────────────────
class SnakeGame:
    """Snake game inside a randomized grid maze."""

    def __init__(self, grid_width: int = 40, grid_height: int = 22, cell_size: int = 30):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.win_width = grid_width * cell_size
        self.win_height = grid_height * cell_size
        self.max_food_dist = None

        # PyGame screen (lazy initialization)
        self._screen = None
        self._clock = None

        # Loop detection
        self._recent_positions = deque(maxlen=grid_width * grid_height)
        self.maze = None

        self.reset()

    def _generate_maze(self) -> np.ndarray:
        h, w = self.grid_height, self.grid_width
        # Start with all open space (0)
        maze = np.zeros((h, w), dtype=np.int32)
        
        # Place border walls
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1
        
        # Randomly place individual obstacle cells
        # We want a density of about 22% of the internal cells
        num_obstacles = int((h - 2) * (w - 2) * 0.22)
        
        # We must keep the center spawn area clear
        cx, cy = w // 2, h // 2
        
        count = 0
        attempts = 0
        while count < num_obstacles and attempts < 10000:
            attempts += 1
            x = np.random.randint(1, w - 1)
            y = np.random.randint(1, h - 1)
            
            # Avoid spawn area (keep 7x5 area clear around center)
            if abs(x - cx) <= 3 and abs(y - cy) <= 2:
                continue
                
            if maze[y, x] == 0:
                maze[y, x] = 1
                count += 1
                
        return maze

    def reset(self):
        """Reset game and generate a new random maze."""
        self.maze = self._generate_maze()

        # Spawn snake in the center corridor facing RIGHT
        cy = self.grid_height // 2
        cx = self.grid_width // 2
        self.snake = deque([(cx - 2, cy), (cx - 1, cy), (cx, cy)])  # tail to head
        self.dir_idx = 1  # facing RIGHT

        self._place_food()
        self.score = 0
        self._done = False
        self._recent_positions.clear()
        self._steps_since_food = 0
        return self._get_state()

    def step(self, action: int):
        if self._done:
            return self._get_state(), 0.0, True

        # 1. Turn
        self.dir_idx = self._turn(self.dir_idx, action)
        dx, dy = _DIRS[self.dir_idx]
        head = self.snake[-1]

        # Manhattan distance to food before moving
        dist_old = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        new_head = (head[0] + dx, head[1] + dy)

        # 2. Check food collision
        ate = new_head == self.food

        # 3. Move
        self.snake.append(new_head)
        if not ate:
            self.snake.popleft()
        else:
            self.score += 1
            self._steps_since_food = 0
            self._place_food()

        # 4. Check death
        self._done = self._check_death(new_head)

        # 5. Track steps since food
        self._steps_since_food += 1

        # 6. Loop detection
        loop_count = sum(1 for p in self._recent_positions if p == new_head)
        self._recent_positions.append(new_head)

        # 7. Reward shaping
        if ate:
            reward = 10.0
        elif self._done:
            reward = -10.0
        else:
            # Look up geodesic distances
            d_old = self._food_dist_grid[head[1], head[0]]
            d_new = self._food_dist_grid[new_head[1], new_head[0]]
            
            # Fallback to Manhattan if geodesic path is not found (e.g. 9999)
            if d_old == 9999 or d_new == 9999:
                d_old = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
                d_new = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
                
            if d_new < d_old:
                reward = 0.25  # reward for moving closer along corridors
            elif d_new > d_old:
                reward = -0.30  # penalty for moving away
            else:
                reward = 0.0

            # Step penalty
            reward -= 0.01

            # Loop penalty
            if loop_count >= 2:
                reward -= 0.5 * loop_count

        return self._get_state(), reward, self._done

    def render(self):
        """Draw current frame to a pygame window."""
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.win_width, self.win_height))
            pygame.display.set_caption("Maze Snake AI")
            self._clock = pygame.time.Clock()
        self._draw_frame()
        pygame.display.flip()
        self._clock.tick(15)

    def close(self):
        if self._screen:
            pygame.quit()
            self._screen = None

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """20-dim feature vector with geodesic food guidance, flood-fill safety, and tail-chase."""
        head = self.snake[-1]
        dx_food = np.sign(self.food[0] - head[0])
        dy_food = np.sign(self.food[1] - head[1])

        # Current direction
        cur_dx, cur_dy = _DIRS[self.dir_idx]
        
        # Next positions for each action
        straight_pos = (head[0] + cur_dx, head[1] + cur_dy)
        left_dir = _DIRS[(self.dir_idx - 1) % 4]
        right_dir = _DIRS[(self.dir_idx + 1) % 4]
        left_pos = (head[0] + left_dir[0], head[1] + left_dir[1])
        right_pos = (head[0] + right_dir[0], head[1] + right_dir[1])
        
        # Danger checks
        straight = self._is_danger(straight_pos)
        danger_l = self._is_danger(left_pos)
        danger_r = self._is_danger(right_pos)

        # Geodesic distances to food for next positions
        def get_geo_dist(pos):
            x, y = pos
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if self.maze[y, x] == 0:
                    return self._food_dist_grid[y, x]
            return 9999

        d_s = get_geo_dist(straight_pos)
        d_l = get_geo_dist(left_pos)
        d_r = get_geo_dist(right_pos)

        # Geodesic food guidance (which action minimizes distance to food)
        min_d = min(d_s, d_l, d_r)
        geo_s = (d_s == min_d) and (min_d < 9999)
        geo_l = (d_l == min_d) and (min_d < 9999)
        geo_r = (d_r == min_d) and (min_d < 9999)

        # ── NEW: Flood-fill safety ratio (AlphaPhoenix-inspired) ──
        # Count reachable cells from each candidate position.
        # A move that traps the snake in a small pocket is dangerous.
        # Cache body_set to avoid re-creation, and discard tail since it will move.
        body_set = set(self.snake)
        if len(self.snake) > 1:
            body_set.discard(self.snake[0])
            
        limit = max(30, len(self.snake) * 2)
        safe_s = self._flood_fill_count(straight_pos, limit, body_set) / limit if not straight else 0.0
        safe_l = self._flood_fill_count(left_pos, limit, body_set) / limit if not danger_l else 0.0
        safe_r = self._flood_fill_count(right_pos, limit, body_set) / limit if not danger_r else 0.0

        # ── NEW: Tail-chase guidance (AlphaPhoenix-inspired) ──
        # When food path is blocked, the snake should chase its own tail
        # to open up space (like following a Hamiltonian cycle back).
        # We only compute BFS to tail when we actually need it to save 95% of computations.
        need_tail_chase = (min_d == 9999) or (safe_s < 0.4 or safe_l < 0.4 or safe_r < 0.4)
        
        if need_tail_chase:
            tail = self.snake[0]
            tail_dist_grid = self._bfs_from(tail)
            
            def get_tail_dist(pos):
                x, y = pos
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    if self.maze[y, x] == 0:
                        return tail_dist_grid[y, x]
                return 9999
            
            td_s = get_tail_dist(straight_pos)
            td_l = get_tail_dist(left_pos)
            td_r = get_tail_dist(right_pos)
            
            min_td = min(td_s, td_l, td_r)
            tail_s = (td_s == min_td) and (min_td < 9999)
            tail_l = (td_l == min_td) and (min_td < 9999)
            tail_r = (td_r == min_td) and (min_td < 9999)
        else:
            tail_s = False
            tail_l = False
            tail_r = False

        return np.array([
            straight, danger_l, danger_r,
            cur_dx == 0 and cur_dy == -1,   # facing UP
            cur_dx == 1 and cur_dy == 0,    # facing RIGHT
            cur_dx == 0 and cur_dy == 1,    # facing DOWN
            cur_dx == -1 and cur_dy == 0,   # facing LEFT
            dx_food == -1,                  # food LEFT
            dx_food == 1,                   # food RIGHT
            dy_food == -1,                  # food UP
            dy_food == 1,                   # food DOWN
            geo_s, geo_l, geo_r,            # geodesic food guidance
            safe_s, safe_l, safe_r,         # flood-fill safety ratio
            tail_s, tail_l, tail_r          # tail-chase guidance
        ], dtype=np.float32)

    def _is_danger(self, pos) -> bool:
        x, y = pos
        return (x < 0 or x >= self.grid_width or
                y < 0 or y >= self.grid_height or
                self.maze[y, x] == 1 or
                pos in self.snake)

    @staticmethod
    def _turn(dir_idx: int, action: int) -> int:
        if action == Action.TURN_LEFT:
            return (dir_idx - 1) % 4
        elif action == Action.TURN_RIGHT:
            return (dir_idx + 1) % 4
        return dir_idx  # STRAIGHT

    def _place_food(self):
        h, w = self.grid_height, self.grid_width
        head = self.snake[-1]
        
        candidates = []
        for y in range(h):
            for x in range(w):
                if self.maze[y, x] == 0 and (x, y) not in self.snake:
                    candidates.append((x, y))
                    
        if not candidates:
            self.food = (w // 2, h // 2)
            self._compute_geodesic_grid()
            return
            
        np.random.shuffle(candidates)
        
        for cand in candidates:
            self.food = cand
            self._compute_geodesic_grid()
            # If the food is reachable from the head, stop and keep it
            if self._food_dist_grid[head[1], head[0]] < 9999:
                return
                
        # Fallback to the first candidate if none are reachable
        self.food = candidates[0]
        self._compute_geodesic_grid()

    def _compute_geodesic_grid(self):
        """Compute the shortest path distance from all open cells to the food using BFS."""
        h, w = self.grid_height, self.grid_width
        grid = np.full((h, w), 9999, dtype=np.int32)
        
        fx, fy = self.food
        grid[fy, fx] = 0
        
        queue = deque([(fx, fy)])
        while queue:
            cx, cy = queue.popleft()
            d = grid[cy, cx]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if self.maze[ny, nx] == 0:  # open path
                        if grid[ny, nx] == 9999:
                            grid[ny, nx] = d + 1
                            queue.append((nx, ny))
        self._food_dist_grid = grid

    def _flood_fill_count(self, start, limit: int, body_set: set) -> int:
        """Count reachable cells from start up to a limit, treating body_set as walls.
        Inspired by AlphaPhoenix: ensures the snake doesn't enter dead-end pockets,
        but stops early once sufficient space is verified to save performance."""
        x, y = start
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return 0
        if self.maze[y, x] == 1 or start in body_set:
            return 0
        
        visited = {start}
        queue = deque([start])
        count = 0
        while queue:
            cx, cy = queue.popleft()
            count += 1
            if count >= limit:
                break
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + ddx, cy + ddy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    pos = (nx, ny)
                    if pos not in visited and self.maze[ny, nx] == 0 and pos not in body_set:
                        visited.add(pos)
                        queue.append(pos)
        return count

    def _bfs_from(self, start) -> np.ndarray:
        """General BFS from any position. Returns distance grid.
        Used for tail-chase guidance (AlphaPhoenix: always know path to tail)."""
        h, w = self.grid_height, self.grid_width
        grid = np.full((h, w), 9999, dtype=np.int32)
        sx, sy = start
        if 0 <= sx < w and 0 <= sy < h and self.maze[sy, sx] == 0:
            grid[sy, sx] = 0
            queue = deque([(sx, sy)])
            while queue:
                cx, cy = queue.popleft()
                d = grid[cy, cx]
                for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + ddx, cy + ddy
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.maze[ny, nx] == 0 and grid[ny, nx] == 9999:
                            grid[ny, nx] = d + 1
                            queue.append((nx, ny))
        return grid

    def _check_death(self, head) -> bool:
        x, y = head
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if self.maze[y, x] == 1:
            return True
        # head collides with body
        return sum(1 for seg in self.snake if seg == head) > 1

    def _draw_frame(self):
        win = self._screen
        win.fill(BLACK)
        
        # Draw maze walls
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.maze[y, x] == 1:
                    # Grey squares separated by thin black lines
                    rect = (x * self.cell_size + 1, y * self.cell_size + 1,
                            self.cell_size - 2, self.cell_size - 2)
                    pygame.draw.rect(win, GREY, rect)

        # Draw food (pure red circle)
        fx, fy = self.food
        pygame.draw.circle(win, RED, 
                           (fx * self.cell_size + self.cell_size // 2, 
                            fy * self.cell_size + self.cell_size // 2), 
                           self.cell_size // 2 - 2)

        # Draw snake (green tones)
        for i, seg in enumerate(self.snake):
            sx, sy = seg
            rect = (sx * self.cell_size + 1, sy * self.cell_size + 1,
                    self.cell_size - 2, self.cell_size - 2)
            
            if i == len(self.snake) - 1:  # head
                pygame.draw.rect(win, GREEN, rect, border_radius=4)
                # Eyes
                head_x = sx * self.cell_size + self.cell_size // 2
                head_y = sy * self.cell_size + self.cell_size // 2
                dx, dy = _DIRS[self.dir_idx]
                eye1_x = head_x + dy * 5 + dx * 5
                eye1_y = head_y - dx * 5 + dy * 5
                eye2_x = head_x - dy * 5 + dx * 5
                eye2_y = head_y + dx * 5 + dy * 5
                pygame.draw.circle(win, BLACK, (int(eye1_x), int(eye1_y)), 2)
                pygame.draw.circle(win, BLACK, (int(eye2_x), int(eye2_y)), 2)
            else:  # body (varying green shades)
                g_val = int(120 + 135 * (i / len(self.snake)))
                color = (0, g_val, 0)  # Pure green tones
                pygame.draw.rect(win, color, rect, border_radius=3)
