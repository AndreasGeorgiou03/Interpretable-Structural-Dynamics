import torch.nn as nn

class NSD_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 6 input features (x1, x2, x3, v1, v2, v3)
        # 20 hidden neurons, LeakyReLU activation
        self.mlp = nn.Sequential(
            nn.Linear(6, 20),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(20, 20),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(20, 1)  # Only output 1 (acceleration correction for DOF1)
        )

    def forward(self, h):
        return self.mlp(h)  # Output is a single value (accel correction for DOF1)
