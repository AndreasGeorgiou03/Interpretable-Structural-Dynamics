import torch
import torch.nn as nn

class PINODEFuncLinear3DOF(nn.Module):
    def __init__(self, M, K, C, B):
        super().__init__()
        self.M = nn.Parameter(M.clone(), requires_grad=False)
        self.K = nn.Parameter(K.clone(), requires_grad=False)
        self.C = nn.Parameter(C.clone(), requires_grad=False)
        self.B = nn.Parameter(B.clone().view(3), requires_grad=False)

        self.amp = 100.0
        self.u_fun = lambda t: torch.zeros((), device=self.M.device)  # scalar tensor

        # NN1: linear (no activation), 1 hidden layer with 10 neurons
        self.mlp = nn.Sequential(
            nn.Linear(6, 10),
            nn.Linear(10, 3)
        )

    def forward(self, t, h):
        x = h[..., :3]
        v = h[..., 3:]

        rhs = (- self.K @ x.unsqueeze(-1) - self.C @ v.unsqueeze(-1)).squeeze(-1)

        a_phys = torch.linalg.solve(self.M, rhs.unsqueeze(-1)).squeeze(-1)  # (3,)
        # base excitation: -1 * xg_ddot(t) on all DOFs
        xg_ddot = self.amp * self.u_fun(t)                # scalar
        a_base = -torch.ones_like(a_phys) * xg_ddot       # (...,3)

        a_disc = self.mlp(h)               # (3,)
        a = a_phys + a_disc + a_base

        dh = torch.zeros_like(h)
        dh[..., :3] = v
        dh[..., 3:] = a
        return dh
