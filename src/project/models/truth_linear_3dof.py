import torch
import torch.nn as nn

class TruthLinear3DOF(nn.Module):

    def __init__(self, M, K, C, amp):
        super().__init__()
        self.M = M
        self.K = K
        self.C = C
        self.amp = amp
        self.u_fun = None  # will represent xg_ddot(t)

    def forward(self, t, h):
        x = h[..., :3]
        v = h[..., 3:]

        rhs = (- self.K @ x.unsqueeze(-1) - self.C @ v.unsqueeze(-1)).squeeze(-1)
        a_phys = torch.linalg.solve(self.M, rhs.unsqueeze(-1)).squeeze(-1)

        xg_ddot = self.amp * self.u_fun(t)                 # scalar
        a_base  = -torch.ones_like(a_phys) * xg_ddot       # (...,3)

        a = a_phys + a_base
        return torch.cat([v, a], dim=-1)
