"""
CNN Actor‑Critic for pixel‑based snake input.

Architecture:
  Input (4, H, W)  →  Conv layers  →  MLP trunk  →  policy head + value head

Two heads share the same CNN trunk.
"""
import torch
import torch.nn as nn


class CNNActorCritic(nn.Module):
    """Convolutional policy for grid‑based observations."""

    def __init__(self, grid_size: int, n_actions: int = 3,
                 hidden_dim: int = 256):
        super().__init__()
        self.grid_size = grid_size

        # CNN trunk
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),  # 32 × H × W
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 × H × W
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128 × H × W
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 4, grid_size, grid_size)
            feat_dim = self.conv(dummy).shape[1]

        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, 4, H, W).  Returns (logits, values)."""
        h = self.trunk(self.conv(x))
        return self.policy(h), self.value(h).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
