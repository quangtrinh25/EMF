import torch
import torch.nn.functional as functional


def rotation_6d_to_matrix_torch(values, eps=1e-8):
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected rotation-6D tensor with shape (N, 6), got {tuple(values.shape)}")
    a1, a2 = values[:, :3], values[:, 3:]
    b1 = functional.normalize(a1, dim=1, eps=eps)
    b2 = functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1, eps=eps)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


def so3_geodesic_radians_torch(pred_rotation_6d, target_rotation_6d, eps=1e-8):
    pred_r = rotation_6d_to_matrix_torch(pred_rotation_6d)
    target_r = rotation_6d_to_matrix_torch(target_rotation_6d)
    relative = pred_r.transpose(1, 2) @ target_r
    cosine = (relative.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) / 2.0
    # atan2 is better behaved near zero than differentiating acos directly.
    skew = torch.stack([
        relative[:, 2, 1] - relative[:, 1, 2],
        relative[:, 0, 2] - relative[:, 2, 0],
        relative[:, 1, 0] - relative[:, 0, 1],
    ], dim=1)
    sine = 0.5 * torch.sqrt(torch.sum(skew ** 2, dim=1) + eps)
    return torch.atan2(sine, cosine.clamp(-1.0, 1.0))


def pose_loss_v3(pred_normalized, target_normalized, pose_mean, pose_std,
                 position_weight=1.0, orientation_weight=1.0):
    position_loss = torch.mean((pred_normalized[:, :3] - target_normalized[:, :3]) ** 2)
    pred_raw = pred_normalized * pose_std + pose_mean
    target_raw = target_normalized * pose_std + pose_mean
    angle = so3_geodesic_radians_torch(pred_raw[:, 3:], target_raw[:, 3:])
    orientation_loss = torch.mean(angle ** 2)
    total = position_weight * position_loss + orientation_weight * orientation_loss
    return total, position_loss, orientation_loss
