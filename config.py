"""
Hyperparameters & settings for Snake PPO training.

Modify freely — this is the single control panel.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── environment ─────────────────────────────────────────────────
    grid_size: int = 10          # 10×10 = max score 99
    cell_size: int = 40          # pixels per cell
    max_steps: int = 200         # truncate episode to avoid infinite loop

    # ── PPO ─────────────────────────────────────────────────────────
    n_inputs: int = 11           # feature vector size
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

    # ── training ────────────────────────────────────────────────────
    total_episodes: int = 5000
    log_interval: int = 100      # print stats every N episodes
    render_interval: int = 50    # show game window every N episodes
    save_interval: int = 500     # save model every N episodes
    model_dir: str = "models"

    # ── network ─────────────────────────────────────────────────────
    hidden_dim: int = 128
    n_layers: int = 2
