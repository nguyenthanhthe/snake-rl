"""
Train PPO agent to play Snake.

Shows live visualizer:
  - Pygame window showing the snake playing (every N episodes)
  - Matplotlib chart with reward/score curves

Usage:
    conda activate snake-ai-ppo
    python train.py

Press Ctrl+C to stop early (model will still be saved).
"""
import os
import time
import torch
import numpy as np

from config import Config
from snake_game.env import SnakeEnv
from agent.ppo import PPO
from utils.vis import LiveVisualizer


def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    env = SnakeEnv(grid_size=cfg.grid_size, cell_size=cfg.cell_size,
                   max_steps=cfg.max_steps)
    agent = PPO(n_inputs=cfg.n_inputs, n_actions=cfg.n_actions, device=device,
                lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                clip_eps=cfg.clip_eps, ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
                max_grad_norm=cfg.max_grad_norm, n_epochs=cfg.n_epochs,
                batch_size=cfg.batch_size)

    os.makedirs(cfg.model_dir, exist_ok=True)

    # Visualizer
    viz = LiveVisualizer(window_size=100)

    # Stats
    episode_rewards = []
    best_score = -1
    start_time = time.time()

    try:
        for ep in range(1, cfg.total_episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            # Show game window every `render_interval` episodes
            show_game = (ep % cfg.render_interval == 0)

            while True:
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                # Render game if this episode is being shown
                if show_game:
                    env.render()
                agent.store(state, action, reward, done, log_prob, value)

                state = next_state
                ep_reward += reward

                if done:
                    agent.update(last_value=0.0)
                    break

            episode_rewards.append(ep_reward)
            if env.score > best_score:
                best_score = env.score

            # ── update live chart ────────────────────────────────────
            viz.update(ep, ep_reward, env.score)

            # ── console log ──────────────────────────────────────────
            if ep % cfg.log_interval == 0:
                avg = np.mean(episode_rewards[-cfg.log_interval:])
                elapsed = time.time() - start_time
                print(f"Ep {ep:5d} | avg reward {avg:+6.1f} | best score {best_score:2d} | "
                      f"eps/sec {cfg.log_interval / elapsed:.1f} | device {device}")
                start_time = time.time()

            # ── save checkpoint ──────────────────────────────────────
            if ep % cfg.save_interval == 0:
                path = os.path.join(cfg.model_dir, f"ppo_snake_{ep}.pt")
                torch.save(agent.net.state_dict(), path)
                print(f"💾 Saved {path}")

    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user.")

    # ── final save ───────────────────────────────────────────────────
    final_path = os.path.join(cfg.model_dir, "ppo_snake_final.pt")
    torch.save(agent.net.state_dict(), final_path)
    print(f"💾 Final model saved → {final_path}")

    # Show a final game with rendering
    print("🎮 Showing final agent performance...")
    state = env.reset()
    done = False
    while not done:
        s = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = agent.net(s.unsqueeze(0))
            action = torch.distributions.Categorical(logits=logits).sample().item()
        state, _, done, _ = env.step(action)
        env.render()
    print(f"🏆 Best score: {best_score} / {cfg.grid_size * cfg.grid_size - 1}")

    viz.close()
    env.close()


if __name__ == "__main__":
    train(Config())
