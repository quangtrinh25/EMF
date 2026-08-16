import torch

def mse_loss(pred, target):
    """L_MSE = (1/N) sum(||p_i - p_hat_i||^2)"""
    return torch.mean(torch.sum((pred - target)**2, dim=1))

def rmse_loss(pred, target):
    """L_RMSE = sqrt(L_MSE)"""
    return torch.sqrt(torch.mean(torch.sum((pred - target)**2, dim=1)) + 1e-8)

def weighted_pose_mse_loss(pred, target, position_weight=1.0, orientation_weight=1.0):
    weights = torch.ones_like(target)
    weights[:, :3] *= position_weight
    weights[:, 3:] *= orientation_weight
    return torch.mean(torch.sum(weights * (pred - target) ** 2, dim=1))
