# Snake AI: Deep Reinforcement Learning with AlphaPhoenix-Inspired Safety Guidance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced Reinforcement Learning agent that learns to play Snake inside a high-density randomized obstacle course using Proximal Policy Optimization (PPO).

To overcome the common reinforcement learning looping behavior, where the agent circles endlessly to survive rather than actively consuming food, this project implements a hybrid architecture combining a Multi-Layer Perceptron (MLP) with Breadth-First Search (BFS) Geodesic Guidance and dynamic safety overrides.

---

## Real-Time Neural Network Visualizer

The project includes an interactive, high-fidelity neural network visualizer that runs in real-time. It provides a split-screen layout displaying the game environment on the left and the active neural network firing pathways (input activations, hidden layer weights, and output logits) on the right.

<p align="center">
  <img src="visualizer_screenshot.png" alt="Light-Theme Neural Network Visualizer" width="900" style="border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
</p>

---

## Core Architecture and Techniques

### 1. AlphaPhoenix-Inspired 20-Dimensional State Vector
Instead of processing raw screen pixels through a slow-to-converge Convolutional Neural Network (CNN), the state is represented as a structured 20-dimensional feature vector fed directly into a fast-converging MLP:

* **Local Danger (3 features)**: Checks if the cells immediately straight, left, or right relative to the snake's direction contain walls, obstacles, or the snake's own body.
* **Direction Encoding (4 features)**: A one-hot vector representing the snake's current heading direction (Up, Right, Down, Left).
* **Relative Food Vector (4 features)**: Indicates whether the food is located left, right, up, or down relative to the snake's head.
* **Geodesic Path Direction (3 features)**: Computes the shortest path to the food through the obstacle field using BFS. It indicates which of the three actions (Straight, Left, Right) minimizes the geodesic distance, allowing the network to leverage global pathfinding while learning local safety controls.
* **Flood-Fill Safety Ratio (3 features)**: Measures the reachable volume of space from each candidate action using an early-exit flood-fill algorithm. The search depth is capped at `max(30, snake_length * 2)` to minimize CPU overhead. If an action would lead the snake into a pocket smaller than its length, the corresponding safety ratio drops, guiding the policy to steer away.
* **Tail-Chase Guidance (3 features)**: Computes the BFS shortest path to the snake's tail. This pathfinder is run conditionally (only when the path to food is blocked or local safety is compromised), enabling the snake to chase its tail to safely recycle grid cells until a path opens.

### 2. High-Density Randomized Obstacle Board
* **Randomized Layout**: Renders a $40 \times 22$ grid map with independent obstacle blocks at a 22% density.
* **Safe Spawning**: Guarantees a $7 \times 5$ obstacle-free zone in the center of the board to ensure the snake can spawn safely.
* **Safe Food Spawner**: The food spawner dynamically checks candidates' neighbors, counting both walls and the snake's body segments as obstacles. Food is prevented from spawning in dead-ends or narrow corridors with $\le 1$ open exit, ensuring the food remains reachable and the snake is not led into unavoidable traps.

### 3. PPO Optimization
* **Value Function Clipping** is employed to stabilize gradient updates.
* **Linear Learning Rate and Entropy Annealing** facilitate smooth exploration-exploitation transitions.
* **Orthogonal Weight Initialization** is applied to both the actor and critic networks to improve training stability.

---

## Project Structure

```
snake-rl/
├── snake_game/
│   ├── game.py          # Core game logic (obstacle generation, safe food spawner)
│   ├── env.py           # Gym-compatible wrapper
│   ├── grid_obs.py      # Grid observation helper (optional)
│   └── vec_env.py       # Parallel environment wrapper (Vectorized Envs)
├── agent/
│   ├── model.py         # MLP Actor-Critic model (20 inputs -> 256 hidden -> 3 outputs)
│   ├── cnn_model.py     # Strided CNN Actor-Critic model (optional)
│   ├── ppo.py           # PPO algorithm implementation
│   └── storage.py       # Rollout buffer
├── docs/
│   ├── research_report_vi.md                      # Detailed Vietnamese research report
│   └── Generated with Deep Research Gemini 3.1 Pro.docx  # Extended research report
├── config.py            # Global hyperparameters and environment configuration
├── train.py             # Headless training loop
├── test.py              # Visual playback script
├── visualize_nn.py      # High-Contrast Light-Theme Neural Network Visualizer
├── run_train.py         # Training runner
└── README.md
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Agent
```bash
python run_train.py
```
*Note: The script is configured to train in the background using CUDA. Checkpoints are automatically saved in the `models/` directory.*

### 3. Run the Visualizer
```bash
python visualize_nn.py --model models/ppo_snake_final.pt --delay 100
```
*   **Resize**: Drag the window corners to dynamically scale the visualizer interface.
*   **Visual Guide**: Green lines represent positive synaptic weights, red lines represent negative weights, and active nodes breathe/expand based on activation probability.

---

## Performance and Benchmarks

The PPO agent was trained on an NVIDIA GeForce RTX 4050 Laptop GPU (via CUDA) for 50,672 episodes (24,420 updates).

| Metric | CNN Model (Baseline) | MLP + BFS Guidance | MLP + BFS + AlphaPhoenix (Latest) |
|--------|-----------------|---------------------|-----------------------------------|
| **Input Features** | Raw Grid Pixels | 14-dimensional vector | **20-dimensional vector (with Safety & Tail-Chase)** |
| **Best Score (Food Eaten)** | 1 | 29 | **51** |
| **Average Return** | -11.0 | +77.8 | **+132.43** |
| **Convergence** | Fails to converge in 15k ep | Converges under 1k ep (~30s) | **Reaches robust survival under 1k ep, optimizes to perfection** |
| **Looping Behavior** | Yes (loops to survive) | Minor deaths in tight loops | **Fully resolved (steers away or chases tail)** |
| **Food Spawn Traps** | Yes | Yes (spawns in corridors) | **Zero (guaranteed $\ge 2$ open neighbors)** |
| **Training Speed** | ~60 ep/s | ~12.2 ep/s | **~30.0 ep/s (initial) / ~0.8 ep/s (converged model)** |

---

## Graphical User Interface Enhancements

* **High-DPI Scaling**: Integrates Windows DPI scaling awareness (`ctypes.windll.shcore.SetProcessDpiAwareness(1)`) to ensure crisp rendering on high-resolution monitors.
* **Super-Sampled Anti-Aliasing (SSAA)**: Renders all graphic assets, vectors, and texts at double resolution ($2840 \times 920$), then downscales them via `pygame.transform.smoothscale()` to produce clean, anti-aliased lines and curves.
* **Light Theme Design**: Features an off-white aesthetic with charcoal text (`(33, 37, 41)`) and drop-shadow styling, ensuring maximum contrast and legibility.
* **Taskbar Process Grouping**: Sets a process-specific AppUserModelID to display the custom game icon in the Windows taskbar instead of the default Python console logo.

---

## References and Reports

* **[Detailed Vietnamese Research Report](docs/research_report_vi.md)**: Deep analysis of MDP, POMDP, State Aliasing, Potential-Based Reward Shaping (PBRS), and neural network architectures to fix looping behavior.
* **[Deep Research Gemini 3.1 Pro Docx](docs/Generated%20with%20Deep%20Research%20Gemini%203.1%20Pro.docx)**: Extended research report on loop behavior in reinforcement learning.
