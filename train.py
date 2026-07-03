"""
Train PPO agent to play Snake — with CNN + curriculum + parallel envs.

Shows:
  - 🐍 Pygame window: game being played
  - 🧠 CNN Visualizer: what the network "sees" (grid input, action probs, value)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import numpy as np

from config import Config
from snake_game.vec_env import VecEnv
from snake_game.grid_obs import make_grid_obs
from agent.ppo import PPO
from agent.storage import RolloutBuffer
from utils.cnn_viz import CNNVisualizer


# ── curriculum ──────────────────────────────────────────────────────────────
def _curriculum_max_dist(episodes_done: int, cfg: Config) -> int:
    """Return the max Manhattan distance allowed for food placement."""
    if not cfg.curriculum:
        return cfg.grid_size * 2  # effectively unlimited
    dist = cfg.curriculum_dists[0]
    for threshold, d in zip(cfg.curriculum_steps, cfg.curriculum_dists):
        if episodes_done >= threshold:
            dist = d
    return dist


# ── train ───────────────────────────────────────────────────────────────────
def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"🔥 Device: {device}  ({gpu_name})")
    print(f"🐍 Envs: {cfg.n_envs} parallel | Grid: {cfg.grid_size}×{cfg.grid_size}")
    print(f"🧠 Mode: {cfg.obs_mode.upper()} | Hidden: {cfg.hidden_dim} | "
          f"Curriculum: {'ON' if cfg.curriculum else 'OFF'}")

    # Parallel env
    vec_env = VecEnv(cfg.n_envs, cfg.grid_size, cfg.cell_size, cfg.max_steps,
                     obs_mode=cfg.obs_mode)

    # PPO agent
    obs_shape = vec_env.obs_shape
    agent = PPO(obs_shape, cfg.n_actions, device,
                lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                clip_eps=cfg.clip_eps, ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
                max_grad_norm=cfg.max_grad_norm, n_epochs=cfg.n_epochs,
                batch_size=cfg.batch_size, hidden_dim=cfg.hidden_dim,
                grid_size=cfg.grid_size)

    # Rollout buffer
    buffer = RolloutBuffer(cfg.n_envs, cfg.n_steps, obs_shape, device)

    # CNN Visualizer (not line chart!)
    viz = CNNVisualizer(cfg.grid_size, cfg.n_actions)

    # Checkpoints
    os.makedirs(cfg.model_dir, exist_ok=True)

    # ── stats ──────────────────────────────────────────────────────
    total_episodes = 0
    best_score = -1
    episode_returns = []
    episode_scores = []
    ep_return_buf = np.zeros(cfg.n_envs, dtype=np.float32)
    start_time = time.time()
    timesteps = 0

    states = vec_env.reset()

    try:
        update_idx = 0
        while total_episodes < cfg.total_episodes:
            update_idx += 1

            # ── curriculum: set max food distance for all envs ────
            if cfg.curriculum:
                max_dist = _curriculum_max_dist(total_episodes, cfg)
                for env in vec_env.envs:
                    env.game.max_food_dist = max_dist

            # ── collect rollout ──────────────────────────────────
            for step in range(cfg.n_steps):
                s = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions, log_probs, values = agent.get_actions(s)
                next_states, rewards, dones, _ = vec_env.step(actions)

                buffer.store(states, actions, rewards, dones, log_probs, values)
                states = next_states
                timesteps += cfg.n_envs
                ep_return_buf += rewards

                for i in range(cfg.n_envs):
                    if dones[i]:
                        total_episodes += 1
                        episode_returns.append(ep_return_buf[i])
                        episode_scores.append(vec_env.get_scores()[i])
                        ep_return_buf[i] = 0.0

            # ── PPO update ────────────────────────────────────────
            last_s = torch.as_tensor(states, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, last_val = agent.net(last_s)
            agent.update(buffer, last_val)

            current_scores = vec_env.get_scores()
            best_score = max(best_score, max(current_scores))

            # ── render game ───────────────────────────────────────
            if update_idx % cfg.render_interval == 0:
                idx = np.random.randint(0, cfg.n_envs)
                vec_env.render_one(idx)

            # ── CNN visualizer ────────────────────────────────────
            if update_idx % cfg.log_interval == 0:
                v = vec_env.envs[0]
                g = v.game
                grid_obs = make_grid_obs(g.snake[-1], g.snake, g.food, cfg.grid_size)
                probs, val = agent.evaluate(grid_obs if isinstance(obs_shape, tuple)
                                            else g._get_state())
                viz.update(grid_obs, probs, val,
                           total_episodes, g.score, best_score,
                           total_episodes, timesteps)

            # ── console log ───────────────────────────────────────
            if update_idx % cfg.log_interval == 0:
                n = min(len(episode_returns), cfg.n_envs * 2) if episode_returns else 1
                avg_ret = float(np.mean(episode_returns[-n:])) if episode_returns else 0.0
                eps_sec = total_episodes / (time.time() - start_time)
                dist_str = f"dist≤{max_dist}" if cfg.curriculum else ""
                print(f"Upd {update_idx:4d} | ep {total_episodes:5d}/{cfg.total_episodes} | "
                      f"avg_ret {avg_ret:+7.2f} | best {best_score:2d} | "
                      f"{eps_sec:.1f} ep/s | {dist_str}")

            # ── save checkpoint ──────────────────────────────────
            if update_idx % cfg.save_interval == 0:
                path = os.path.join(cfg.model_dir, f"ppo_snake_upd{update_idx}.pt")
                torch.save(agent.net.state_dict(), path)
                print(f"💾 Saved {path}")

    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user.")

    # ── final save ──────────────────────────────────────────────
    final_path = os.path.join(cfg.model_dir, "ppo_snake_final.pt")
    torch.save(agent.net.state_dict(), final_path)
    print(f"💾 Final model saved → {final_path}")

    # ── final demo ──────────────────────────────────────────────
    print("🎮 Showing final agent performance...")
    env = vec_env.envs[0]
    env.game.reset()
    done = False
    while not done:
        g = env.game
        obs = make_grid_obs(g.snake[-1], g.snake, g.food, cfg.grid_size)
        probs, _ = agent.evaluate(obs)
        action = int(np.random.choice(3, p=probs))
        _, _, done, _ = env.step(action)
        # Update CNN viz during demo
        probs, val = agent.evaluate(obs)
        viz.update(obs, probs, val, total_episodes, g.score, best_score,
                   total_episodes, timesteps)
        env.render()

    print(f"🏆 Best score: {best_score} / {cfg.grid_size * cfg.grid_size - 1}")
    print(f"📊 Total episodes: {total_episodes} | Steps: {timesteps}")

    viz.close()
    vec_env.close()


if __name__ == "__main__":
    train(Config())
