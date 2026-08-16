import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 8),    nn.Sigmoid(),
            nn.Linear(8, 15),   nn.Sigmoid(),
            nn.Linear(15, 9),   nn.Sigmoid(),
            nn.Linear(9, 6),
        )
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
        out = self.net(x)
        if return_normalized:
            return out
        return out * (self.pose_std + 1e-8) + self.pose_mean
