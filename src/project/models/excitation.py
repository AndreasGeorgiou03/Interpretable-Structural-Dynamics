import torch

class UFunFromSamples:

    def __init__(self, t_grid: torch.Tensor, u_grid: torch.Tensor):
        assert t_grid.ndim == 1 and u_grid.ndim == 1, "t_grid and u_grid must be 1D"
        assert t_grid.numel() == u_grid.numel(), "t_grid and u_grid must have same length"
        self.t = t_grid
        self.u = u_grid

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # Accept scalar or vector t; return same shape
        t_in = t.reshape(-1)

        t0 = self.t[0]
        dt = self.t[1] - self.t[0]
        T = self.t.numel()

        # Convert t -> fractional index in [0, T-1)
        idx = ((t_in - t0) / dt).clamp(0, T - 1 - 1e-6)

        i0 = torch.floor(idx).long()
        i1 = (i0 + 1).clamp_max(T - 1)

        w = (idx - i0.float()).clamp(0.0, 1.0)

        out = (1.0 - w) * self.u[i0] + w * self.u[i1]
        return out.reshape(t.shape)
