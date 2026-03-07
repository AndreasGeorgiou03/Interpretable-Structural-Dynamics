import torch
import torch.nn as nn

# PINODE scheme (physics + neural discrepancy)
# this is only for the fully physics-informed model
class PINODEFuncForcedVibration(nn.Module):

    def __init__(self, K, C, B):
        super().__init__()
        self.K = nn.Parameter(K.clone(), requires_grad=False)
        self.C = nn.Parameter(C.clone(), requires_grad=False)
        self.B = nn.Parameter(B.clone(), requires_grad=False)
        self.amp = 1.0 # this is only for initialization
        self.u_fun = lambda t: 0.0 # this is only for initialization

        # same as the previous experiment
        self.mlp = nn.Sequential(
            nn.Linear(8, 30),
            nn.Tanh(),
            nn.Linear(30, 4)
        )

    def forward(self, t, h):
        x = h[..., :4]
        v = h[..., 4:]

        dv_phys = (
            - self.K @ x.unsqueeze(-1)
            - self.C @ v.unsqueeze(-1)
        ).squeeze(-1)

        u_t = self.amp * self.u_fun(t)
        f_u = self.B * u_t
        dv_phys = dv_phys + f_u

        dv_corr = self.mlp(h)

        a = dv_phys + dv_corr

        dh = torch.zeros_like(h)
        dh[..., :4] = v
        dh[..., 4:] = a
        return dh
