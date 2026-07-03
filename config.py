"""
Hyperparameters & settings for Snake PPO training.

Single control panel — modify freely.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── observation mode ────────────────────────────────────────────
    obs_mode: str = "grid"       # "features" (11‑dim) or "grid" (4×H×W for CNN)

    # ── parallel environments ───────────────────────────────────────
    n_envs: int = 8
    n_steps: int = 128           # steps per update per env (rollout length)

    # ── environment ─────────────────────────────────────────────────
    grid_size: int = 10          # 10×10 = max score 99
    cell_size: int = 40          # pixels per cell
    max_steps: int = 300         # truncate episode to avoid infinite loop

    # ── PPO ─────────────────────────────────────────────────────────
    n_actions: int = 3           # STRAIGHT / LEFT / RIGHT
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64

    # ── network ─────────────────────────────────────────────────────
    hidden_dim: int = 256        # ↑ từ 128 lên 256

    # ── curriculum learning ─────────────────────────────────────────
    curriculum: bool = True      # food starts close, gradually moves further
    curriculum_steps: tuple = (1000, 3000, 8000)  # episode thresholds
    curriculum_dists: tuple = (2, 5, 8)           # max Manhattan distance

    # ── training ────────────────────────────────────────────────────
    total_episodes: int = 20000
    log_interval: int = 5
    render_interval: int = 10
    save_interval: int = 50
    model_dir: str = "models"

    @property
    def obs_shape(self):
        if self.obs_mode == "features":
            return 11
        return (4, self.grid_size, self.grid_size)
