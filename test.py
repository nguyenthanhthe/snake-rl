"""
Watch a trained PPO agent play Snake inside the grid obstacle board.
Runs in an infinite loop with post-crash pauses so you can watch the agent play continuously.

Usage:
    python test.py [--model models/ppo_snake_final.pt] [--delay 200]
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import numpy as np
import pygame
from config import Config
from snake_game.env import SnakeEnv
from snake_game.grid_obs import make_grid_obs
from agent.model import ActorCritic
from agent.cnn_model import CNNActorCritic


def test(model_path: str, grid_width: int, grid_height: int, obs_mode: str = "grid", delay: int = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load appropriate model based on obs_mode
    if obs_mode == "grid":
        net = CNNActorCritic(grid_height, grid_width, 3, 256).to(device)
    else:
        net = ActorCritic(20, 3, hidden_dim=256).to(device)

    # Try loading weights
    try:
        net.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True))
        print(f" Loaded weights from: {model_path}")
    except Exception as e:
        print(f"[WARN] Weight loading error (running with random weights): {e}")
    net.eval()

    # Initialize Environment (cell_size=20 for a clean compact window)
    env = SnakeEnv(grid_width=grid_width, grid_height=grid_height, cell_size=20)

    print("\n=== Starting Continuous Playback ===")
    print("Close the Pygame window or press ESC / Q in the window to exit.")

    best_score = 0
    ep = 0
    
    try:
        while True:
            ep += 1
            state = env.reset()
            total_reward = 0.0
            done = False
            step_count = 0
            
            # Show initial frame
            env.render()
            pygame.time.wait(800)
            
            while not done:
                # Handle window events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Visualizer window closed by user. Exiting...")
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                            print("Exit key pressed. Exiting...")
                            env.close()
                            return

                g = env.game
                if obs_mode == "grid":
                    obs = make_grid_obs(g.snake[-1], g.snake, g.food,
                                         grid_height, grid_width, 
                                         dir_idx=g.dir_idx, maze=g.maze)
                    s = torch.as_tensor(obs, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                else:
                    s = torch.as_tensor(state, dtype=torch.float32,
                                        device=device).unsqueeze(0)

                with torch.no_grad():
                    logits, value = net(s)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                action = int(np.argmax(probs))  # greedy choice
                state, reward, done, _ = env.step(action)
                total_reward += reward
                step_count += 1

                # Force done if stuck looping without eating food
                if g._steps_since_food >= 120:
                    done = True

                env.render()

                # Delay between steps
                if delay > 0 and not done:
                    pygame.time.wait(delay)

            best_score = max(best_score, env.game.score)
            print(f"Episode {ep:3d} Finished | Score: {env.game.score:2d} | Steps: {step_count:3d} | Best: {best_score:2d}")

            # Show crash state (collision) for 1.2 seconds before resetting the board
            pygame.time.wait(1200)

    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt...")

    env.close()


if __name__ == "__main__":
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_snake_final.pt")
    p.add_argument("--width", type=int, default=cfg.grid_width)
    p.add_argument("--height", type=int, default=cfg.grid_height)
    p.add_argument("--mode", default="grid", choices=["grid", "features"],
                   help="Observation mode: 'grid' for CNN, 'features' for MLP")
    p.add_argument("--delay", type=int, default=200,
                   help="Delay in milliseconds between steps to slow down play")
    args = p.parse_args()
    test(args.model, args.width, args.height, args.mode, args.delay)
