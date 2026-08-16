import numpy as np

from physics.rotation_repr import (
    so3_geodesic_deg,
    target_v3_to_pose_deg,
    wrapped_angle_difference_deg,
)

def position_rmse(pred, target):
    """
    E_pos = sqrt( ||P_T - P_R||^2 / 3 )
    pred/target cols 0:3 are positions in mm.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    diff = (pred[:, :3] - target[:, :3])**2
    return float(np.sqrt(diff.sum(axis=1).mean() / 3.0))

def orientation_rmse(pred, target):
    """
    E_dir = sqrt( (da^2 + db^2 + dg^2) / 3 )
    pred/target cols 3:6 are cosine values.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    pred_cos   = np.clip(pred[:, 3:], -1.0, 1.0)
    target_cos = np.clip(target[:, 3:], -1.0, 1.0)
    pred_ang   = np.degrees(np.arccos(pred_cos))
    target_ang = np.degrees(np.arccos(target_cos))
    diff = (pred_ang - target_ang)**2
    return float(np.sqrt(diff.sum(axis=1).mean() / 3.0))


def pose_metrics_v3(pred, target):
    """Pooled metrics for XYZ + continuous rotation-6D targets.

    ``position_axis_rmse_mm`` matches the paper's per-coordinate RMSE, while
    Euclidean and percentile metrics make the physical error easier to read.
    Orientation is the signed, parameterization-independent SO(3) distance.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 9:
        raise ValueError(f"Expected matching (N, 9) v3 targets, got {pred.shape} and {target.shape}")

    delta_xyz = pred[:, :3] - target[:, :3]
    euclidean = np.linalg.norm(delta_xyz, axis=1)
    geodesic = so3_geodesic_deg(pred[:, 3:], target[:, 3:])
    pred_pose = target_v3_to_pose_deg(pred)
    target_pose = target_v3_to_pose_deg(target)
    euler_delta = wrapped_angle_difference_deg(pred_pose[:, 3:], target_pose[:, 3:])

    metrics = {
        'num_samples': int(len(target)),
        'position_axis_rmse_mm': float(np.sqrt(np.mean(delta_xyz ** 2))),
        'position_euclidean_rmse_mm': float(np.sqrt(np.mean(euclidean ** 2))),
        'position_euclidean_mean_mm': float(np.mean(euclidean)),
        'position_euclidean_median_mm': float(np.median(euclidean)),
        'position_euclidean_p95_mm': float(np.percentile(euclidean, 95)),
        'orientation_geodesic_rmse_deg': float(np.sqrt(np.mean(geodesic ** 2))),
        'orientation_geodesic_mean_deg': float(np.mean(geodesic)),
        'orientation_geodesic_median_deg': float(np.median(geodesic)),
        'orientation_geodesic_p95_deg': float(np.percentile(geodesic, 95)),
        # Same pooled algebraic form as Eq. (9) in the reference paper, but
        # retained only for protocol comparison because Euler coordinates are
        # convention/range dependent. SO(3) remains the primary v3 metric.
        'orientation_euler_component_rmse_deg': float(np.sqrt(np.mean(euler_delta ** 2))),
    }
    for index, name in enumerate(('roll', 'pitch', 'yaw')):
        metrics[f'{name}_wrapped_rmse_deg'] = float(np.sqrt(np.mean(euler_delta[:, index] ** 2)))
    return metrics
