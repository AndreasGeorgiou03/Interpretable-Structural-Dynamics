import torch
import torch.nn as nn

class NODEFuncNoPhysics(nn.Module):

    def __init__(self, state_dim=8, hidden=30):
        super().__init__()
        self.state_dim = state_dim

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, state_dim)   # outputs dh/dt directly
        )

    def forward(self, t, h):
        return self.mlp(h)
