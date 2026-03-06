import torch
import torch.nn as nn
from excitation import UFunFromSamples

class PhysicsOnly3DOF(nn.Module):
    def __init__(self, M, K, C, B, t_grid, u_grid, amp=1.0):
        super().__init__()
        self.M, self.K, self.C, self.B = M, K, C, B
        self.u_fun = UFunFromSamples(t_grid, u_grid)
        self.amp = float(amp)

    def forward(self, t, h):
        x = h[..., :3]
        v = h[..., 3:]
        rhs = (-self.K @ x.unsqueeze(-1) - self.C @ v.unsqueeze(-1)).squeeze(-1)
        rhs = rhs + self.B * (self.amp * self.u_fun(t))
        a = torch.linalg.solve(self.M, rhs.unsqueeze(-1)).squeeze(-1)
        dh = torch.zeros_like(h)
        dh[..., :3] = v
        dh[..., 3:] = a
        return dh
