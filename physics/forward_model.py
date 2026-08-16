import numpy as np
from physics.emf import compute_peak_emf

def forward_model(pose_mm_deg, tx_params, rx_params):
    """
    Wrapper: 6-DoF pose(s) -> 9 peak EMF amplitudes.

    pose_mm_deg: [X, Y, Z, roll_deg, pitch_deg, yaw_deg] or batch (N, 6)
    returns: (9,) or (N, 9) float32 array in Volts, ordered as
             [TX1RX1, TX1RX2, TX1RX3, TX2RX1, ..., TX3RX3].
    """
    return compute_peak_emf(pose_mm_deg, tx_params, rx_params)
