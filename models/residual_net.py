import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim=512, slope=0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(slope),
            nn.Linear(dim, dim), nn.LeakyReLU(slope),
        )
    def forward(self, x):
        return x + self.block(x)

class ResidualNet(nn.Module):
    def __init__(self, input_dim=9, output_dim=6, n_blocks=7, hidden=512, slope=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_blocks = n_blocks
        self.hidden = hidden
        self.input_layer = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, slope) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.Linear(128, output_dim),
        )
        self.register_buffer('emf_mean', torch.zeros(input_dim))
        self.register_buffer('emf_std',  torch.ones(input_dim))
        self.register_buffer('pose_mean', torch.zeros(output_dim))
        self.register_buffer('pose_std',  torch.ones(output_dim))

    def set_normalization(self, emf_mean, emf_std, pose_mean=None, pose_std=None):
        self.emf_mean.copy_(torch.as_tensor(emf_mean, dtype=torch.float32))
        self.emf_std.copy_(torch.as_tensor(emf_std,  dtype=torch.float32))
        if pose_mean is not None:
            self.pose_mean.copy_(torch.as_tensor(pose_mean, dtype=torch.float32))
        if pose_std is not None:
            self.pose_std.copy_(torch.as_tensor(pose_std,  dtype=torch.float32))

    def forward(self, x, return_normalized=False):
        x = (x - self.emf_mean) / (self.emf_std + 1e-8)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        if return_normalized:
            return out
        return out * (self.pose_std + 1e-8) + self.pose_mean
