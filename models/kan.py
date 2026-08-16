import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_fun=F.silu, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_fun = base_fun

        # Base linear mapping
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        # Spline parameters
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)

        # Spline weights (coefficients)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        nn.init.normal_(self.spline_weight, mean=0.0, std=scale_noise / math.sqrt(grid_size + spline_order))

        self.scale_base = scale_base
        self.scale_spline = scale_spline

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        # Recurrent B-spline evaluation (Cox-de Boor recursion formula)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:-(k+1)]) / (grid[k:-1] - grid[:-(k+1)] + 1e-8) * bases[..., :-1]
                + (grid[(k+1):] - x) / (grid[(k+1):] - grid[1:-k] + 1e-8) * bases[..., 1:]
            )
        return bases

    def forward(self, x):
        base_output = F.linear(self.base_fun(x), self.base_weight) * self.scale_base
        bases = self.b_splines(x)
        spline_output = torch.einsum('bik,oik->bo', bases, self.spline_weight) * self.scale_spline
        return base_output + spline_output

class KAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = KANLinear(9, 5)
        self.layer2 = KANLinear(5, 6)
        
        self.register_buffer('emf_mean', torch.zeros(9))
        self.register_buffer('emf_std',  torch.ones(9))
        self.register_buffer('pose_mean', torch.zeros(6))
        self.register_buffer('pose_std',  torch.ones(6))

    def set_normalization(self, emf_mean, emf_std, pose_mean=None, pose_std=None):
        self.emf_mean.copy_(torch.as_tensor(emf_mean, dtype=torch.float32))
        self.emf_std.copy_(torch.as_tensor(emf_std,  dtype=torch.float32))
        if pose_mean is not None:
            self.pose_mean.copy_(torch.as_tensor(pose_mean, dtype=torch.float32))
        if pose_std is not None:
            self.pose_std.copy_(torch.as_tensor(pose_std,  dtype=torch.float32))

    def forward(self, x, return_normalized=False):
        x = (x - self.emf_mean) / (self.emf_std + 1e-8)
        x = self.layer1(x)
        out = self.layer2(x)
        if return_normalized:
            return out
        return out * (self.pose_std + 1e-8) + self.pose_mean
