import torch, torch.nn as nn

class QuantileMLP(nn.Module):
    def __init__(self, in_dim, act_dim, quantiles, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, quantiles)
        )
        self.quantiles = quantiles

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)  # (B, Q)

class Actor(nn.Module):
    def __init__(self, in_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return torch.softmax(logits, dim=-1)  # simplex
