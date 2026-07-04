# 🐍 Snake AI — Deep Reinforcement Learning with Geodesic Path Guidance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📺 Demonstration Video
<p align="center">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/KlxV_QhXIvg?si=ZompgL8cc3gRiKyM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</p>

*If the video frame doesn't load, you can watch it directly on YouTube here: **[Click to Watch on YouTube](https://www.youtube.com/watch?v=KlxV_QhXIvg)**.*

---

An advanced Reinforcement Learning agent that learns to play **Snake inside a high-density randomized obstacle board** using **Proximal Policy Optimization (PPO)**. 

To overcome the infamous RL loop behavior (where the snake circles endlessly to survive rather than eating food), this project implements a hybrid **MLP + BFS Geodesic Guidance** architecture that achieves rapid convergence, safety overrides, and high-performance pathfinding.

---

## 🧠 Core Architecture & Techniques

### 1. Hybrid MLP + BFS Geodesic Guidance (14-Dimensional State)
Instead of feeding raw grid images through a slow-to-converge CNN, we feed a highly structured **14-dimensional feature vector** to a fast-converging MLP:
- **Local Danger (3 features)**: Checks if the cells immediately `STRAIGHT`, `LEFT`, or `RIGHT` relative to the snake's direction contain walls, obstacles, or the snake's own body.
- **Direction Encoding (4 features)**: One-hot vector of the snake's current heading direction (`UP`, `RIGHT`, `DOWN`, `LEFT`).
- **Relative Food Vector (4 features)**: Signals whether the food is located `LEFT`, `RIGHT`, `UP`, or `DOWN` relative to the snake's head.
- **Geodesic Path Direction (3 features)**: Computes the actual shortest path to the food through the obstacle field via Breadth-First Search (BFS). Indicates which of the 3 actions (`STRAIGHT`, `LEFT`, or `RIGHT`) minimizes the geodesic distance. This allows the MLP to utilize global pathfinding guidance while learning local safety overrides.

### 2. High-Density Randomized Obstacle Board
- **Randomized Layout**: Renders a $40 \times 22$ grid map with independent obstacle blocks at **$22\%$ density**.
- **Safe Starting Zone**: Guarantees a $7 \times 5$ obstacle-free zone in the center of the board so the snake can spawn safely.
- **BFS Reachability Checks**: The food spawner dynamically runs BFS checks to guarantee that the food is always reachable from the snake's head. If a randomly picked coordinate is unreachable, it is reshuffled.

### 3. PPO Best Practices
- **Value Function Clipping** to stabilize updates.
- **Linear Learning Rate & Entropy Annealing** for optimal exploration-exploitation transitions.
- **Orthogonal Weight Initialization** for actor and critic networks.

---

## 📁 Structure

```
snake-ai-ppo/
├── snake_game/
│   ├── game.py          # Core game logic (obstacle generation, BFS pathfinder)
│   ├── env.py           # Gym-compatible wrapper
│   ├── grid_obs.py      # Grid observation helper (optional)
│   └── vec_env.py       # Parallel environment wrapper (Vectorized Envs)
├── agent/
│   ├── model.py         # MLP Actor-Critic model (14 inputs -> 256 hidden -> 3 outputs)
│   ├── cnn_model.py     # Strided CNN Actor-Critic model (optional)
│   ├── ppo.py           # PPO algorithm (annealing + clipping)
│   └── storage.py       # Rollout buffer
├── docs/
│   ├── research_report_vi.md                      # Detailed Vietnamese research report
│   └── Generated with Deep Research Gemini 3.1 Pro.docx  # Deep research report docx
├── config.py            # Global configuration (hyperparameters, env size)
├── train.py             # Headless training loop
├── test.py              # Visual playback script (Pygame GUI)
├── run_train.py         # Training runner (trains for 15,000 episodes on GPU)
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Agent
```bash
python run_train.py
```
*Note: The model trains in the background (GUI off) at ~40 episodes per second on CUDA. It completes 15,000 episodes in ~10 minutes and saves checkpoints in `models/`.*

### 3. Playback/Test the Trained Agent
```bash
# Watch the trained snake navigate the obstacle course
python test.py --model models/ppo_snake_final.pt --mode features --delay 100
```

---

## 📊 Performance Comparison

| Metric | CNN Model (Old) | MLP + BFS Geodesic Guidance (New) |
|--------|-----------------|------------------------------------|
| **Best Score (Food Eaten)** | 1 | **29** |
| **Average Return** | -11.0 | **+77.8** |
| **Convergence Speed** | Fails to converge in 15k ep | **Converges within 1k ep (~30s)** |
| **Endless Loop Bug** | Yes (Loops endlessly to survive) | **No (Actively hunts food)** |

---

## 📚 References & Reports
- **[Detailed Vietnamese Research Report](docs/research_report_vi.md)**: Deep analysis of MDP, POMDP, State Aliasing, Potential-Based Reward Shaping (PBRS), and neural network architectures to fix looping behavior.
- **[Deep Research Gemini 3.1 Pro Docx](docs/Generated%20with%20Deep%20Research%20Gemini%203.1%20Pro.docx)**: Extended research report on loop behavior in reinforcement learning.
