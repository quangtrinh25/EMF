from scipy.optimize import least_squares
import numpy as np

def optimize_pose(emf_measured, forward_model_fn, tx_params, rx_params, x0=None, bounds=None):
    """
    Solve inverse problem via Trust Region Reflective minimization.
    emf_measured: (9,) array in Volts
    x0: initial guess [X, Y, Z, alpha_deg, beta_deg, gamma_deg]
    """
    if x0 is None:
        # Start at workspace center
        x0 = [0.0, 0.0, 400.0, 90.0, 90.0, 90.0]

    if bounds is None:
        bounds_lo = [-250.0, -250.0, 150.0, 0.0, 0.0, 0.0]
        bounds_hi = [ 250.0,  250.0, 650.0, 180.0, 180.0, 180.0]
    else:
        bounds_lo, bounds_hi = bounds

    def residual(p):
        emf_pred = forward_model_fn(p, tx_params, rx_params)
        return emf_measured - emf_pred

    result = least_squares(
        residual, x0,
        method='trf',
        bounds=(bounds_lo, bounds_hi),
        max_nfev=100,  # Fast evaluation
        ftol=1e-6, xtol=1e-6
    )
    return result.x

def optimize_poses_batch(emfs_measured, forward_model_fn, tx_params, rx_params, x0=None, bounds=None):
    """
    Batch optimization wrapper.
    emfs_measured: (N, 9) array
    """
    N = emfs_measured.shape[0]
    poses = np.zeros((N, 6))
    for i in range(N):
        poses[i] = optimize_pose(emfs_measured[i], forward_model_fn, tx_params, rx_params, x0, bounds)
    return poses
