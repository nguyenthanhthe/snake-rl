"""
Snake game — pure PyGame, Gym‑compatible.
Grid‑based, feature‑state (11‑dim vector), 3 actions.
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
BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
GREEN = (0, 200, 0)
RED   = (200, 0, 0)
BLUE  = (0, 0, 200)

# ── game class ─────────────────────────────────────────────────────────────
class SnakeGame:
    """Playable snake game.  Also serves as the rendering back‑end for RL."""

    def __init__(self, grid_size: int = 10, cell_size: int = 40):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.win_size = grid_size * cell_size
        self.max_food_dist = None  # None = any position; int = max Manhattan dist

        # PyGame init (lazy — only when render is called)
        self._screen = None
        self._clock = None

        self.reset()

    # ── public API ─────────────────────────────────────────────────────────

    def reset(self):
        """Reset game to initial state."""
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid), (mid - 1, mid), (mid - 2, mid)])  # tail → head
        self.dir_idx = 1  # facing RIGHT
        self._place_food()
        self.score = 0
        self._done = False
        return self._get_state()

    def step(self, action: int):
        """Apply action, advance one frame.  Returns (state, reward, done)."""
        if self._done:
            return self._get_state(), 0.0, True

        # 1. turn
        self.dir_idx = self._turn(self.dir_idx, action)
        dx, dy = _DIRS[self.dir_idx]
        head = self.snake[-1]

        # Manhattan distance to food BEFORE moving
        dist_old = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        new_head = (head[0] + dx, head[1] + dy)

        # 2. check food collision
        ate = new_head == self.food

        # 3. move
        self.snake.append(new_head)
        if not ate:
            self.snake.popleft()
        else:
            self.score += 1
            self._place_food()

        # 4. check death
        self._done = self._check_death(new_head)

        # 5. reward — shaped
        if ate:
            reward = 10.0
        elif self._done:
            reward = -10.0
        else:
            # Distance shaping: closer to food = positive, further = negative
            dist_new = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 1.0 if dist_new < dist_old else (-1.0 if dist_new > dist_old else 0.0)
        return self._get_state(), reward, self._done

    def render(self):
        """Draw current frame to a pygame window."""
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.win_size, self.win_size))
            pygame.display.set_caption("Snake AI")
            self._clock = pygame.time.Clock()
        self._draw_frame()
        pygame.display.flip()
        self._clock.tick(10)

    def close(self):
        if self._screen:
            pygame.quit()
            self._screen = None

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """11‑dim feature vector — see env.py docstring."""
        head = self.snake[-1]
        dx_food = np.sign(self.food[0] - head[0])
        dy_food = np.sign(self.food[1] - head[1])

        # Current direction unit vector
        cur_dx, cur_dy = _DIRS[self.dir_idx]
        # Danger checks
        straight = self._is_danger((head[0] + cur_dx, head[1] + cur_dy))
        left_dir = _DIRS[(self.dir_idx - 1) % 4]
        right_dir = _DIRS[(self.dir_idx + 1) % 4]
        danger_l = self._is_danger((head[0] + left_dir[0], head[1] + left_dir[1]))
        danger_r = self._is_danger((head[0] + right_dir[0], head[1] + right_dir[1]))

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
        ], dtype=np.float32)

    def _is_danger(self, pos) -> bool:
        x, y = pos
        return (x < 0 or x >= self.grid_size or
                y < 0 or y >= self.grid_size or
                pos in self.snake)

    @staticmethod
    def _turn(dir_idx: int, action: int) -> int:
        if action == Action.TURN_LEFT:
            return (dir_idx - 1) % 4
        elif action == Action.TURN_RIGHT:
            return (dir_idx + 1) % 4
        return dir_idx  # STRAIGHT

    def _place_food(self):
        if self.max_food_dist is not None:
            # Curriculum: food within max_food_dist Manhattan from head
            head = self.snake[-1]
            candidates = []
            gs = self.grid_size
            for x in range(max(0, head[0] - self.max_food_dist),
                           min(gs, head[0] + self.max_food_dist + 1)):
                for y in range(max(0, head[1] - self.max_food_dist),
                               min(gs, head[1] + self.max_food_dist + 1)):
                    d = abs(x - head[0]) + abs(y - head[1])
                    if 0 < d <= self.max_food_dist and (x, y) not in self.snake:
                        candidates.append((x, y))
            if candidates:
                self.food = candidates[np.random.randint(len(candidates))]
                return
        # Fallback / no curriculum
        while True:
            pos = (np.random.randint(0, self.grid_size),
                   np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                self.food = pos
                break

    def _check_death(self, head) -> bool:
        x, y = head
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # head collides with body?  Check if head appears more than once.
        return sum(1 for seg in self.snake if seg == head) > 1

    def _draw_frame(self):
        win = self._screen
        win.fill(BLACK)
        # grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(win, WHITE, (0, i * self.cell_size),
                             (self.win_size, i * self.cell_size))
            pygame.draw.line(win, WHITE, (i * self.cell_size, 0),
                             (i * self.cell_size, self.win_size))
        # food
        fx, fy = self.food
        rect = (fx * self.cell_size + 2, fy * self.cell_size + 2,
                self.cell_size - 4, self.cell_size - 4)
        pygame.draw.rect(win, RED, rect)
        # snake
        for i, seg in enumerate(self.snake):
            sx, sy = seg
            rect = (sx * self.cell_size + 2, sy * self.cell_size + 2,
                    self.cell_size - 4, self.cell_size - 4)
            colour = GREEN if i < len(self.snake) - 1 else BLUE
            pygame.draw.rect(win, colour, rect)
