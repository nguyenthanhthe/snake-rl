"""
Live training visualizer for Snake PPO.

Shows two views simultaneously:
  1. Pygame window — snake game rendering
  2. Matplotlib chart — reward/score curves updating live
"""
import matplotlib
matplotlib.use("TkAgg")  # stable interactive backend on Windows

import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class LiveVisualizer:
    """Live-updating chart for training metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episodes = []
        self.rewards = []
        self.scores = []
        self.avg_rewards = []  # SMA over window_size

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                                        sharex=True)
        self.fig.canvas.manager.set_window_title("Snake PPO — Training")

        self.line_reward, = self.ax1.plot([], [], "b-", alpha=0.3, label="Reward")
        self.line_avg,    = self.ax1.plot([], [], "b-", linewidth=2, label=f"SMA {window_size}")
        self.ax1.set_ylabel("Reward")
        self.ax1.legend(loc="upper left")
        self.ax1.grid(True, alpha=0.3)

        self.line_score,  = self.ax2.plot([], [], "g-", linewidth=1.5, label="Score")
        self.ax2.set_xlabel("Episode")
        self.ax2.set_ylabel("Score")
        self.ax2.legend(loc="upper left")
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)

    def update(self, episode: int, reward: float, score: int):
        """Add one data point and redraw."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.scores.append(score)

        # SMA
        dq = deque(self.rewards[-self.window_size:], maxlen=self.window_size)
        self.avg_rewards.append(np.mean(dq))

        # Update lines
        self.line_reward.set_data(self.episodes, self.rewards)
        self.line_avg.set_data(self.episodes, self.avg_rewards)
        self.line_score.set_data(self.episodes, self.scores)

        # Auto‑scale axes
        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)
