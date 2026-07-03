# 🐍 Snake AI — Deep Reinforcement Learning (PPO/A2C from scratch)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Build a **deep reinforcement learning** agent that plays **Snake** perfectly.
Inspired by [Code Bullet](https://youtu.be/3bhP7zuiFmQ) and [Alex Petrenko](https://www.youtube.com/watch?v=3bhP7zuiFmQ),
implemented **from scratch** using PyTorch — no Stable-Baselines, no RL libraries.

---

## 🎯 Goal

Train a neural network to play Snake **from raw pixels** (like a human would)
using **Proximal Policy Optimization (PPO)** and/or **Advantage Actor-Critic (A2C)**.

| Grid       | Target                 | Status |
|:-----------|:-----------------------|:-------|
| 6×6        | Max score (35)         | ⏳     |
| 10×10      | Max score (99)         | ⏳     |
| 20×20      | Max score (399)        | ⏳     |

---

## 🧠 Approach

### Why PPO over DQN?

- **Epsilon-greedy** in DQN is fatal for Snake — a single random move late-game almost always kills.
- **PPO/A2C** sample from a **probability distribution** over actions, so exploration is built-in naturally.
- PPO can train on **full game trajectories** instead of small (e.g. 16-step) windows, vastly improving data efficiency on larger grids.

### Architecture (planned)

```
                 ┌──────────────┐
                 │  Game State  │
                 │  (pixels or  │
                 │   features)  │
                 └──────┬───────┘
                        ↓
              ┌──────────────────┐
              │  CNN (vision)    │  ← train the network's "eyes"
              │  or feature MLP  │
              └──────────────────┘
                        ↓
              ┌──────────────────┐
              │  Actor (policy)  │──→ action probabilities
              │  Critic (value)  │──→ state value estimate
              └──────────────────┘
```

Two modes:
1. **Pixel input** — CNN processes the full game screen (harder, more general).
2. **Feature input** — hand-crafted features (direction, danger, food-relative) — faster, good baseline.

---

## 📁 Project structure

```
snake-ai-ppo/
├── snake_game/          # Snake environment (Pygame + Gym-like API)
│   ├── game.py          # Core game logic
│   └── ...
├── agent/               # RL algorithms
│   ├── ppo.py           # PPO implementation
│   ├── a2c.py           # A2C implementation
│   ├── model.py         # Neural network architectures
│   └── storage.py       # Rollout buffer / experience replay
├── train.py             # Training entry point
├── test.py              # Run a pretrained agent
├── utils/               # Visualization, logging, metrics
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting started

```bash
# Clone
git clone https://github.com/nguyenthanhthe/snake-ai-ppo.git
cd snake-ai-ppo

# Install dependencies
pip install torch pygame numpy matplotlib

# Train on 10×10 grid
python train.py --grid 10 --algo ppo

# Watch a trained agent
python test.py --checkpoint runs/ppo_10x10/best.pt
```

---

## 📈 Progress tracking

- Training metrics logged to console & CSV.
- Real-time pygame window showing agent gameplay.
- Periodic evaluation on held-out games.
- Checkpoints saved automatically.

---

## 📚 References

- [Code Bullet — AI learns to play Snake](https://youtu.be/3bhP7zuiFmQ)
- [Alex Petrenko — Snake Perfect Score PPO](https://www.youtube.com/watch?v=3bhP7zuiFmQ)
- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Asynchronous Advantage Actor-Critic (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783)
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Ilya Kostrikov — pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

---

## 📄 License

MIT — do whatever you want, just give credit.

---

*Built from scratch with ❤️ and a lot of patience.*
