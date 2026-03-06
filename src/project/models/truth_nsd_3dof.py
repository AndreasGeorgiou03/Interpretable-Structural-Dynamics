import torch
import torch.nn as nn

class TruthPhaseNSD_3DOF(nn.Module):

    def __init__(self, M, K, C, nsd_force_fun):
        super().__init__()
        self.M, self.K, self.C = M, K, C
        self.nsd_force_fun = nsd_force_fun

        self.amp = 100.0
        self.u_fun = None  # will represent xg_ddot(t)

    def forward(self, t, h):
        x = h[..., :3]
        v = h[..., 3:]

        # physics accel
        rhs = (-self.K @ x.unsqueeze(-1) - self.C @ v.unsqueeze(-1)).squeeze(-1)
        a_phys = torch.linalg.solve(self.M, rhs.unsqueeze(-1)).squeeze(-1)

        # base excitation accel (Eq. 18 bottom block)
        xg_ddot = self.amp * self.u_fun(t)                      # scalar
        a_base  = -torch.ones_like(a_phys) * xg_ddot            # (...,3)

        # NSD force on DOF1 -> convert to accel via M^{-1}
        f_nsd = self.nsd_force_fun(x[..., 0])                   # (...,)
        f_vec = torch.stack([f_nsd, torch.zeros_like(f_nsd), torch.zeros_like(f_nsd)], dim=-1)  # (...,3)
        a_nsd = torch.linalg.solve(self.M, f_vec.unsqueeze(-1)).squeeze(-1)                     # (...,3)

        a = a_phys + a_base + a_nsd
        return torch.cat([v, a], dim=-1)
