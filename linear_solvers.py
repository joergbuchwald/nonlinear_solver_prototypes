import numpy as np
from scipy import linalg


def solve_direct(J, rhs):
    """LU factorisation via scipy.linalg.solve."""
    return np.matrix(linalg.solve(J, rhs))


def solve_cg(J, rhs, cg_tol=1e-12, cg_max_iter=None):
    """
    Conjugate Gradient for  J * dx = rhs.

    J is symmetrised as half*(J + J^T) to satisfy the SPD requirement.
    For a strongly non-symmetric J use GMRES / BiCGSTAB instead.
    """
    n = J.shape[0]
    if cg_max_iter is None:
        cg_max_iter = n
    A = np.array(0.5 * (J + J.T))
    b = np.array(rhs).flatten()
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    for _ in range(cg_max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < cg_tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return np.matrix(x).T
