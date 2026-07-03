"""
Live training visualizer for Snake PPO (parallel envs).

Shows:
  - Pygame window — one snake game rendered periodically
  - Matplotlib chart — average episodic return + max score over updates
"""
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class LiveVisualizer:
    """Live‑updating chart for parallel‑env training."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.updates = []
        self.avg_rets = []   # average episodic return per update
        self.max_scores = []
        self.ep_lengths = []

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                                        sharex=True)
        self.fig.canvas.manager.set_window_title("Snake PPO — Training")

        self.line_ret,   = self.ax1.plot([], [], "b-", linewidth=2, label="Avg return")
        self.line_sma,   = self.ax1.plot([], [], "b--", alpha=0.5, label=f"SMA {window_size}")
        self.ax1.set_ylabel("Avg episodic return")
        self.ax1.legend(loc="upper left")
        self.ax1.grid(True, alpha=0.3)

        self.line_score, = self.ax2.plot([], [], "g-", linewidth=1.5, label="Max score")
        self.ax2.set_xlabel("Update")
        self.ax2.set_ylabel("Score")
        self.ax2.legend(loc="upper left")
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)

    def update(self, update_idx: int, avg_return: float, max_score: int):
        """Add one data point and redraw."""
        self.updates.append(update_idx)
        self.avg_rets.append(avg_return)
        self.max_scores.append(max_score)

        # SMA
        dq = deque(self.avg_rets[-self.window_size:], maxlen=self.window_size)
        sma = np.mean(dq)

        self.line_ret.set_data(self.updates, self.avg_rets)
        self.line_sma.set_data(self.updates[:len(self.avg_rets)],
                               [sma] * len(self.updates[:len(self.avg_rets)]))
        self.line_score.set_data(self.updates, self.max_scores)

        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)
