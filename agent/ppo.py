"""
PPO (Proximal Policy Optimization) — from scratch.
Supports both MLP (features) and CNN (grid) backbones.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent.model import ActorCritic
from agent.cnn_model import CNNActorCritic


class PPO:
    """PPO trainer with support for MLP and CNN policies."""

    def __init__(self, obs_shape, n_actions: int, device: torch.device,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, n_epochs: int = 4, batch_size: int = 64,
                 hidden_dim: int = 256, grid_size: int = None):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.obs_shape = obs_shape

        # Choose network: CNN for grid input, MLP for features
        if isinstance(obs_shape, tuple):
            self.net = CNNActorCritic(grid_size, n_actions, hidden_dim).to(device)
        else:
            self.net = ActorCritic(obs_shape, n_actions, hidden_dim).to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def get_actions(self, states: torch.Tensor):
        """Batched forward.  Returns (actions, log_probs, values)."""
        with torch.no_grad():
            logits, values = self.net(states)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def evaluate(self, state: np.ndarray) -> tuple:
        """Single‑state evaluation (for test/viz).  Returns (action_probs, value)."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        # Handle different input shapes
        if isinstance(self.obs_shape, tuple):
            s = s.unsqueeze(0)  # (1, 4, H, W)
        else:
            s = s.unsqueeze(0)  # (1, 11)
        with torch.no_grad():
            logits, value = self.net(s)
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy(), value.item()

    def update(self, buffer, last_values: torch.Tensor):
        """Compute GAE and perform PPO mini‑batch update."""
        states, actions, old_log_probs, advantages, returns = \
            buffer.compute_gae(last_values, self.gamma, self.gae_lambda)

        n = len(states)
        indices = torch.randperm(n, device=self.device)
        for _ in range(self.n_epochs):
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_s = states[idx]
                batch_a = actions[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]
                batch_old_log = old_log_probs[idx]

                logits, values_pred = self.net(batch_s)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log)

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_pred, batch_ret)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        buffer.clear()
