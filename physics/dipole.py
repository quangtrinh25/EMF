import numpy as np

MU0 = 4 * np.pi * 1e-7  # H/m

def dipole_field_B(P_mm, P_TX_mm, D_TX, M):
    """
    Magnetic flux density B at capsule position P.
    Implements paper Eq.(4) with vectorization support.

    P_mm    : (N, 3) or (3,) capsule position in mm
    P_TX_mm : (3,) TX coil position in mm
    D_TX    : (3,) unit normal vector of TX coil
    M       : scalar magnetic moment (A·m²)
    Returns : (N, 3) or (3,) B field vector in Tesla
    """
    P = np.asarray(P_mm, dtype=float)
    P_TX = np.asarray(P_TX_mm, dtype=float)
    D = np.asarray(D_TX, dtype=float)
    D_norm = np.linalg.norm(D)
    if D_norm > 1e-12:
        D = D / D_norm  # Ensure unit vector

    # mm -> meters
    P_m = P * 1e-3
    P_TX_m = P_TX * 1e-3

    if P.ndim == 1:
        Pi = P_m - P_TX_m  # Displacement vector (3,)
        r = np.linalg.norm(Pi)
        if r < 1e-9:
            return np.zeros(3)
        B = (MU0 * M / (4 * np.pi)) * (
            3.0 * np.dot(D, Pi) * Pi / r**5 - D / r**3
        )
        return B
    else:
        Pi = P_m - P_TX_m
        r = np.linalg.norm(Pi, axis=1, keepdims=True)  # (N, 1)
        r = np.where(r < 1e-9, 1e-9, r)
        dot_D_Pi = np.sum(D * Pi, axis=1, keepdims=True)  # (N, 1)
        B = (MU0 * M / (4 * np.pi)) * (
            3.0 * dot_D_Pi * Pi / r**5 - D / r**3
        )
        return B
