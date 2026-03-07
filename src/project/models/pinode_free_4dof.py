import torch
import torch.nn as nn


class PINODEFunc4DOF(nn.Module):
    def __init__(self, K: torch.Tensor, C: torch.Tensor, scheme: int):
        super().__init__()

        if scheme not in (1, 2, 3):
            raise ValueError(f"scheme must be 1, 2, or 3, got {scheme}")

        self.scheme = scheme

        if scheme == 1:
            K_eff = torch.zeros_like(K)
            C_eff = torch.zeros_like(C)
        elif scheme == 2:
            K_eff = 0.7 * K.clone()
            C_eff = 0.7 * C.clone()
        else:  # scheme == 3
            K_eff = K.clone()
            C_eff = C.clone()

        self.K = nn.Parameter(K_eff, requires_grad=False)
        self.C = nn.Parameter(C_eff, requires_grad=False)

        self.mlp = nn.Sequential(
            nn.Linear(8, 30),
            nn.Tanh(),
            nn.Linear(30, 4),
        )

    def forward(self, t, h):
        x = h[..., :4]
        v = h[..., 4:]

        dv_phys = (
            -self.K @ x.unsqueeze(-1)
            -self.C @ v.unsqueeze(-1)
        ).squeeze(-1)

        dv_corr = self.mlp(h)

        dh = torch.zeros_like(h)
        dh[..., :4] = v
        dh[..., 4:] = dv_phys + dv_corr
        return dh
