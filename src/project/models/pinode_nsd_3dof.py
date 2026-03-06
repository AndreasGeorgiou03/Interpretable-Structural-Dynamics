import torch
import torch.nn as nn

class PINODEFuncNSD_3DOF(nn.Module):
    def __init__(self, M, K, C, nn1_mlp, nn2_nsd):
        super().__init__()
        self.M = nn.Parameter(M.clone(), requires_grad=False)
        self.K = nn.Parameter(K.clone(), requires_grad=False)
        self.C = nn.Parameter(C.clone(), requires_grad=False)

        self.mlp = nn1_mlp
        self.nsd = nn2_nsd

        self.amp = 1.0
        self.u_fun = lambda t: torch.zeros((), device=self.M.device)

    def forward(self, t, h):
        x = h[..., :3]
        v = h[..., 3:]

        rhs = (-self.K @ x.unsqueeze(-1) - self.C @ v.unsqueeze(-1)).squeeze(-1)
        a_phys = torch.linalg.solve(self.M, rhs.unsqueeze(-1)).squeeze(-1)

        xg_ddot = self.amp * self.u_fun(t)
        a_base = -torch.ones_like(a_phys) * xg_ddot

        a_nn1 = self.mlp(h)

        a_raw = self.nsd(h)

        a_nsd_vec = torch.zeros_like(a_phys)
        a_nsd_vec[..., 0] = a_raw

        a = a_phys + a_base + a_nn1 + a_nsd_vec
        return torch.cat([v, a], dim=-1)
