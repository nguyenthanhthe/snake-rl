"""
CNN Neural Network Visualizer — shows what the network "sees".

Not a chart — a live monitor of the CNN's input and output:
  - 4‑channel grid observation (head/body/food/walls) as RGB composite
  - Action probabilities as horizontal bars
  - State value + training progress
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Colour map for 4 channels: head=red, body=green, food=blue, walls=white
_CHANNEL_COLORS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                           dtype=np.float32)


def _render_composite(grid_obs: np.ndarray) -> np.ndarray:
    """Combine (4, H, W) observation into an (H, W, 3) RGB image."""
    # grid_obs: (4, H, W) with 0/1 values
    H, W = grid_obs.shape[1:3]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(4):
        for ch in range(3):
            rgb[:, :, ch] += grid_obs[c] * _CHANNEL_COLORS[c, ch]
    # Clamp to [0, 1] for display
    return np.clip(rgb, 0, 1)


class CNNVisualizer:
    """Dark‑themed live monitor for the CNN policy."""

    def __init__(self, grid_size: int, n_actions: int = 3):
        self.grid_size = grid_size
        self.n_actions = n_actions

        plt.ion()
        self.fig = plt.figure(figsize=(8, 4), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("🧠 CNN Neural Network")

        # Grid: 1 row, 3 columns
        gs = self.fig.add_gridspec(1, 3, width_ratios=[2, 1.5, 1.2],
                                    hspace=0, wspace=0.3)

        # ── Left: grid observation ─────────────────────────────────
        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_grid.set_facecolor("#16213e")
        self.ax_grid.set_title("Input Grid", color="white", fontsize=10)
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.img_display = self.ax_grid.imshow(
            np.zeros((grid_size, grid_size, 3)),
            interpolation="nearest", vmin=0, vmax=1)

        # Legend
        legend_elements = [
            mpatches.Patch(color=(1, 0, 0), label="Head"),
            mpatches.Patch(color=(0, 1, 0), label="Body"),
            mpatches.Patch(color=(0, 0, 1), label="Food"),
            mpatches.Patch(color=(1, 1, 1), label="Wall"),
        ]
        self.ax_grid.legend(handles=legend_elements, loc="upper left",
                            fontsize=6, framealpha=0.5, labelcolor="white")

        # ── Center: action probabilities ───────────────────────────
        self.ax_act = self.fig.add_subplot(gs[0, 1])
        self.ax_act.set_facecolor("#16213e")
        self.ax_act.set_xlim(0, 1)
        self.ax_act.set_ylim(-0.5, self.n_actions - 0.5)
        self.ax_act.set_title("Action Probabilities", color="white", fontsize=10)
        self.ax_act.set_yticks(range(self.n_actions))
        self.ax_act.set_yticklabels(["Straight", "Turn Left", "Turn Right"],
                                    color="white", fontsize=8)
        self.ax_act.tick_params(axis="x", colors="white", labelsize=8)
        self.ax_act.grid(True, alpha=0.2, axis="x")

        self.action_bars = self.ax_act.barh(
            range(self.n_actions), [0, 0, 0],
            color=["#00ff88", "#ff6600", "#ff3366"], height=0.6)

        # Value text
        self.value_text = self.ax_act.text(
            0.5, -0.8, "V(s) = 0.00",
            transform=self.ax_act.transAxes,
            color="#ffcc00", fontsize=12, ha="center",
            fontweight="bold")

        # ── Right: training info ───────────────────────────────────
        self.ax_info = self.fig.add_subplot(gs[0, 2])
        self.ax_info.set_facecolor("#16213e")
        self.ax_info.axis("off")
        self.info_lines = [
            self.ax_info.text(0.05, 0.90, "", color="#e0e0e0", fontsize=9,
                              fontfamily="monospace", transform=self.ax_info.transAxes),
            self.ax_info.text(0.05, 0.78, "", color="#e0e0e0", fontsize=9,
                              fontfamily="monospace", transform=self.ax_info.transAxes),
            self.ax_info.text(0.05, 0.66, "", color="#e0e0e0", fontsize=9,
                              fontfamily="monospace", transform=self.ax_info.transAxes),
            self.ax_info.text(0.05, 0.54, "", color="#e0e0e0", fontsize=9,
                              fontfamily="monospace", transform=self.ax_info.transAxes),
            self.ax_info.text(0.05, 0.42, "", color="#e0e0e0", fontsize=9,
                              fontfamily="monospace", transform=self.ax_info.transAxes),
            self.ax_info.text(0.05, 0.30, "", color="#ffcc00", fontsize=11,
                              fontweight="bold", fontfamily="monospace",
                              transform=self.ax_info.transAxes),
        ]
        self.ax_info.set_title("Status", color="white", fontsize=10)

        plt.tight_layout()
        plt.show(block=False)

    def update(self, grid_obs: np.ndarray, action_probs: np.ndarray,
               value: float, episode: int, score: int, best_score: int,
               total_episodes: int, total_steps: int):
        """Refresh all panels with new data."""
        # Grid
        rgb = _render_composite(grid_obs)
        # Add grid lines overlay
        self.img_display.set_data(rgb)

        # Action probabilities
        for bar, prob in zip(self.action_bars, action_probs):
            bar.set_width(prob)

        # Value
        self.value_text.set_text(f"V(s) = {value:.3f}")

        # Info
        self.info_lines[0].set_text(f"Episode   {episode}")
        self.info_lines[1].set_text(f"Score     {score}")
        self.info_lines[2].set_text(f"Best      {best_score}")
        self.info_lines[3].set_text(f"Total Ep  {total_episodes}")
        self.info_lines[4].set_text(f"Steps     {total_steps}")
        self.info_lines[5].set_text(
            "🧠 CNN ACTIVE" if grid_obs.shape[0] == 4 else "⚠️  FEATURE MODE")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)
