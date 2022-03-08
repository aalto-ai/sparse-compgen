import torch
import torch.nn as nn


class ActorCriticHead(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, n_actions, bias=False),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, n_actions, bias=False),
        )

    def forward(self, x):
        return (self.actor(x), self.critic(x))
