"""Continuous rotation representations and SO(3) metrics.

The v3 pipeline uses the first two columns of an Rz-Ry-Rx rotation matrix as
the network target.  Gram-Schmidt maps the unconstrained six network outputs
back to SO(3), preserving signed rotations unlike cos(Euler).
"""

import numpy as np

from physics.rotations import compose_rotation


def euler_deg_to_matrix(euler_deg):
    euler = np.asarray(euler_deg, dtype=float)
    single = euler.ndim == 1
    if single:
        euler = euler[None, :]
    if euler.ndim != 2 or euler.shape[1] != 3:
        raise ValueError(f"Expected Euler angles with shape (N, 3), got {euler.shape}")
    rad = np.radians(euler)
    matrices = compose_rotation(rad[:, 0], rad[:, 1], rad[:, 2]).astype(np.float64)
    return matrices[0] if single else matrices


def matrix_to_rotation_6d(matrices):
    matrices = np.asarray(matrices, dtype=float)
    single = matrices.ndim == 2
    if single:
        matrices = matrices[None, :, :]
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (N, 3, 3), got {matrices.shape}")
    values = np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)
    return values[0].astype(np.float32) if single else values.astype(np.float32)


def rotation_6d_to_matrix(values, eps=1e-8):
    values = np.asarray(values, dtype=float)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected rotation-6D values with shape (N, 6), got {values.shape}")
    a1, a2 = values[:, :3], values[:, 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), eps)
    a2_orthogonal = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2_orthogonal / np.maximum(np.linalg.norm(a2_orthogonal, axis=1, keepdims=True), eps)
    b3 = np.cross(b1, b2)
    matrices = np.stack([b1, b2, b3], axis=2)
    return matrices[0] if single else matrices


def project_matrix_to_so3(matrices):
    """Project arbitrary 3x3 matrices to the nearest proper rotations."""
    matrices = np.asarray(matrices, dtype=float)
    single = matrices.ndim == 2
    if single:
        matrices = matrices[None, :, :]
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"Expected matrices with shape (N, 3, 3), got {matrices.shape}")
    u, _, vh = np.linalg.svd(matrices)
    rotations = u @ vh
    reflected = np.linalg.det(rotations) < 0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        rotations[reflected] = u[reflected] @ vh[reflected]
    return rotations[0] if single else rotations


def mean_rotation_6d(rotation_values, axis=0):
    """Chordal mean of rotation-6D predictions, projected back onto SO(3)."""
    values = np.asarray(rotation_values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 6:
        raise ValueError(f"Expected stacked rotation-6D values (M, N, 6), got {values.shape}")
    if axis != 0:
        values = np.moveaxis(values, axis, 0)
    matrices = np.stack([rotation_6d_to_matrix(model_values) for model_values in values])
    return matrix_to_rotation_6d(project_matrix_to_so3(np.mean(matrices, axis=0)))


def matrix_to_euler_deg(matrices):
    """Convert Rz(yaw) @ Ry(pitch) @ Rx(roll) to signed Euler degrees."""
    matrices = np.asarray(matrices, dtype=float)
    single = matrices.ndim == 2
    if single:
        matrices = matrices[None, :, :]
    r20 = np.clip(matrices[:, 2, 0], -1.0, 1.0)
    pitch = np.arcsin(-r20)
    cos_pitch = np.cos(pitch)
    regular = np.abs(cos_pitch) > 1e-7
    roll = np.where(
        regular,
        np.arctan2(matrices[:, 2, 1], matrices[:, 2, 2]),
        np.arctan2(-matrices[:, 1, 2], matrices[:, 1, 1]),
    )
    yaw = np.where(
        regular,
        np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]),
        np.zeros_like(pitch),
    )
    euler = np.degrees(np.stack([roll, pitch, yaw], axis=1))
    euler = (euler + 180.0) % 360.0 - 180.0
    return euler[0] if single else euler


def pose_deg_to_target_v3(poses):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"Expected poses with shape (N, 6), got {poses.shape}")
    rotation_6d = matrix_to_rotation_6d(euler_deg_to_matrix(poses[:, 3:]))
    return np.column_stack([poses[:, :3], rotation_6d]).astype(np.float32)


def target_v3_to_pose_deg(targets):
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 9:
        raise ValueError(f"Expected v3 targets with shape (N, 9), got {targets.shape}")
    euler = matrix_to_euler_deg(rotation_6d_to_matrix(targets[:, 3:]))
    return np.column_stack([targets[:, :3], euler]).astype(np.float32)


def so3_geodesic_deg(pred_rotation_6d, target_rotation_6d):
    pred_r = rotation_6d_to_matrix(pred_rotation_6d)
    target_r = rotation_6d_to_matrix(target_rotation_6d)
    relative = np.matmul(np.swapaxes(pred_r, 1, 2), target_r)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def wrapped_angle_difference_deg(pred, target):
    return (np.asarray(pred) - np.asarray(target) + 180.0) % 360.0 - 180.0
