"""
Grid observation builder — converts SnakeGame to 4‑channel image.

Channels:
  0 → head       (1 pixel)
  1 → body       (all segments)
  2 → food       (1 pixel)
  3 → walls      (border)
"""
import numpy as np


def make_grid_obs(head, body, food, grid_size: int, border: bool = True):
    """
    Return (4, grid_size, grid_size) float32 array.

    Parameters:
      head  — (x, y) of head  (may be out‑of‑bounds if snake just died)
      body  — iterable of (x, y) segments (including head)
      food  — (x, y) of food
    """
    obs = np.zeros((4, grid_size, grid_size), dtype=np.float32)
    # Head (clamp in case of OOB death)
    hx, hy = head[0], head[1]
    if 0 <= hx < grid_size and 0 <= hy < grid_size:
        obs[0, hy, hx] = 1.0
    # Body
    for seg in body:
        sx, sy = seg[0], seg[1]
        if 0 <= sx < grid_size and 0 <= sy < grid_size:
            obs[1, sy, sx] = 1.0
    # Food
    obs[2, food[1], food[0]] = 1.0
    # Walls
    if border:
        obs[3, 0, :] = 1.0
        obs[3, -1, :] = 1.0
        obs[3, :, 0] = 1.0
        obs[3, :, -1] = 1.0
    return obs
