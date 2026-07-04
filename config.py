"""
Hyperparameters & settings for Snake PPO training.

Single control panel — modify freely.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── observation mode ────────────────────────────────────────────
    obs_mode: str = "features"   # "features" (20‑dim) or "grid" (5×H×W for CNN)

    # ── parallel environments ───────────────────────────────────────
    n_envs: int = 4
    n_steps: int = 128           # steps per update per env (rollout length)

    # ── environment ─────────────────────────────────────────────────
    grid_width: int = 40         # columns (matches wide photo)
    grid_height: int = 22        # rows (matches wide photo)
    cell_size: int = 30          # pixels per cell (smaller to fit 1600x900 screen)
    max_steps: int = 1000        # limit steps in a maze to avoid infinite loops
    show_gui: bool = False       # True = show PyGame/Viz windows, False = headless/background

    # ── PPO ─────────────────────────────────────────────────────────
    n_actions: int = 3           # STRAIGHT / LEFT / RIGHT
    lr: float = 2.5e-4           # slightly lower for stability
    lr_end: float = 0.0          # linear anneal to 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.02       # start higher for exploration in mazes
    ent_coef_end: float = 0.001  # anneal to lower
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64

    # ── network ─────────────────────────────────────────────────────
    hidden_dim: int = 256

    # ── curriculum learning ─────────────────────────────────────────
    curriculum: bool = False     # disabled for random maze training

    # ── training ────────────────────────────────────────────────────
    total_episodes: int = 15000  # mazes are harder, train longer
    log_interval: int = 1
    render_interval: int = 10
    save_interval: int = 50
    model_dir: str = "models"

    @property
    def obs_shape(self):
        if self.obs_mode == "features":
            return 20
        return (5, self.grid_height, self.grid_width)

    @property
    def total_updates_estimate(self):
        """Rough estimate of total PPO updates for LR scheduling."""
        steps_per_update = self.n_envs * self.n_steps
        # Assume ~100 steps per episode on average in maze
        total_steps = self.total_episodes * 100
        return max(total_steps // steps_per_update, 1)
