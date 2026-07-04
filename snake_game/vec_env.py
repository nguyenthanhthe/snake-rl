"""
Vectorized environment — runs N SnakeGame instances in lockstep.
Supports rectangular grid shapes and grid maze observations.
"""
import numpy as np
from snake_game.env import SnakeEnv
from snake_game.grid_obs import make_grid_obs


class VecEnv:
    """N environments running synchronously."""

    def __init__(self, n_envs: int, grid_width: int = 40, grid_height: int = 22,
                 cell_size: int = 30, max_steps: int = 1000,
                 obs_mode: str = "grid"):
        self.n_envs = n_envs
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obs_mode = obs_mode
        self.envs = [SnakeEnv(grid_width, grid_height, cell_size, max_steps)
                     for _ in range(n_envs)]

    @property
    def obs_dim(self):
        if self.obs_mode == "features":
            return 20
        return (5, self.grid_height, self.grid_width)

    @property
    def obs_shape(self):
        return self.obs_dim  # alias

    # ── helpers ──────────────────────────────────────────────────────

    def _obs(self, env: SnakeEnv):
        """Return observation for a single env in the current mode."""
        if self.obs_mode == "features":
            return env.game._get_state()
        g = env.game
        # Pass rectangular dimensions and current maze layout
        return make_grid_obs(g.snake[-1], g.snake, g.food, 
                             self.grid_height, self.grid_width,
                             dir_idx=g.dir_idx, maze=g.maze)

    # ── API ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset all envs. Returns (n_envs, *obs_shape) float32 array."""
        for e in self.envs:
            e.reset()
        return np.array([self._obs(e) for e in self.envs], dtype=np.float32)

    def step(self, actions: np.ndarray):
        """
        Step all envs. Returns (states, rewards, dones, infos).
        Done envs are auto‑reset (terminal transition still returned).
        """
        states, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            s, r, d, info = env.step(int(actions[i]))
            rewards.append(r)
            dones.append(d)
            infos.append(info)
            if d:
                states.append(self._obs(env))  # obs of terminal state
                env.reset()
            else:
                states.append(self._obs(env))
        return (np.array(states, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos)

    def render_one(self, idx: int = 0):
        self.envs[idx].render()

    def get_scores(self):
        return [e.score for e in self.envs]

    def close(self):
        for e in self.envs:
            e.close()
