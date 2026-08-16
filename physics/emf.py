import numpy as np
from physics.rotations import compose_rotation
from physics.dipole import dipole_field_B

def compute_peak_emf(pose_mm_deg, tx_params, rx_params):
    """
    Compute peak EMF amplitudes for 6-DoF pose(s).
    Implements paper Eq.(5) with support for single pose and vectorized batches.

    pose_mm_deg : (N, 6) or (6,) array of poses [X, Y, Z, alpha_deg, beta_deg, gamma_deg]
    tx_params   : dict of transmitter parameters
    rx_params   : dict of receiver parameters
    Returns     : (N, 9) or (9,) float32 array in Volts, ordered as:
                  [TX1RX1, TX1RX2, TX1RX3, TX2RX1, TX2RX2, TX2RX3, TX3RX1, TX3RX2, TX3RX3]
    """
    poses = np.asarray(pose_mm_deg, dtype=float)
    is_single = (poses.ndim == 1)
    if is_single:
        poses = poses[np.newaxis, :]  # shape (1, 6)

    N = poses.shape[0]
    alpha, beta, gamma = poses[:, 3], poses[:, 4], poses[:, 5]

    # Convert degrees to radians
    a_rad = np.radians(alpha)
    b_rad = np.radians(beta)
    g_rad = np.radians(gamma)
    
    # Compute R_CE for each pose: shape (N, 3, 3)
    R_CE = compose_rotation(a_rad, b_rad, g_rad)

    areas_m2 = np.array(rx_params['areas_mm2']) * 1e-6  # mm² -> m²
    turns = rx_params['turns']
    freqs = tx_params['frequencies']
    
    tx_pos = tx_params['positions']
    tx_ori = tx_params['orientations']
    tx_mom = tx_params['moments']
    
    # Convert tx positions/orientations if dict to list
    if isinstance(tx_pos, dict):
        tx_pos = [tx_pos[f'tx{i+1}'] for i in range(3)]
    if isinstance(tx_ori, dict):
        tx_ori = [tx_ori[f'tx{i+1}'] for i in range(3)]

    # Compute B fields and EMF
    # output EMF shape: (N, 9)
    emf = np.zeros((N, 9), dtype=np.float32)
    P = poses[:, :3]  # (N, 3)

    col_idx = 0
    for l in range(3):  # TX index
        # B_l has shape (N, 3)
        B_l = dipole_field_B(P, tx_pos[l], tx_ori[l], tx_mom[l])
        for k in range(3):  # RX index
            # Rx_k is the k-th column of R_CE: shape (N, 3)
            Rx_k = R_CE[:, :, k]  
            # Dot product B_l . Rx_k: shape (N,)
            dot_val = np.sum(B_l * Rx_k, axis=1)
            # Peak EMF: shape (N,)
            val = turns[k] * 2 * np.pi * freqs[l] * areas_m2[k] * dot_val
            emf[:, col_idx] = np.abs(val)
            col_idx += 1

    if is_single:
        return emf[0]
    return emf
