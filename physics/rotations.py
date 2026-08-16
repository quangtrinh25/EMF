import numpy as np

def RotX(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def RotY(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def RotZ(g):
    c, s = np.cos(g), np.sin(g)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def compose_rotation(alpha_rad, beta_rad, gamma_rad):
    """
    R = Rz(γ) @ Ry(β) @ Rx(α) — paper Eq.(3)
    Supports both scalar inputs and numpy arrays (N,) of angles.
    If input is scalar, returns (3, 3) matrix.
    If input is array of length N, returns (N, 3, 3) matrices.
    """
    alpha_rad = np.asarray(alpha_rad)
    beta_rad = np.asarray(beta_rad)
    gamma_rad = np.asarray(gamma_rad)
    
    if alpha_rad.ndim == 0:
        return RotZ(gamma_rad) @ RotY(beta_rad) @ RotX(alpha_rad)
        
    N = len(alpha_rad)
    c_a, s_a = np.cos(alpha_rad), np.sin(alpha_rad)
    c_b, s_b = np.cos(beta_rad), np.sin(beta_rad)
    c_g, s_g = np.cos(gamma_rad), np.sin(gamma_rad)

    R = np.zeros((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = c_g * c_b
    R[:, 0, 1] = c_g * s_b * s_a - s_g * c_a
    R[:, 0, 2] = c_g * s_b * c_a + s_g * s_a

    R[:, 1, 0] = s_g * c_b
    R[:, 1, 1] = s_g * s_b * s_a + c_g * c_a
    R[:, 1, 2] = s_g * s_b * c_a - c_g * s_a

    R[:, 2, 0] = -s_b
    R[:, 2, 1] = c_b * s_a
    R[:, 2, 2] = c_b * c_a
    return R
