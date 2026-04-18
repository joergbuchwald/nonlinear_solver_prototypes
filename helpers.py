import numpy as np

from fd_problem import JF, F


def _phi(sol, dt):
    """Objective  phi(x) = half*||F(x)||^2  used by several solvers."""
    return 0.5 * float(np.linalg.norm(F(sol, dt)) ** 2)


def _dphi_directional(sol, dt, direction):
    """Directional derivative  phi'(0) = grad_phi . p = (J^T F) . p."""
    J = np.array(JF(sol, dt))
    f = np.array(F(sol, dt)).flatten()
    g = J.T @ f
    return float(g @ np.array(direction).flatten())


def _apply_step(sol, base, step, alpha):
    """
    Set interior solution nodes to  base + alpha * step
    without touching boundary nodes.
    """
    candidate = np.array(base).flatten() + alpha * np.array(step).flatten()
    for k in range(1, sol.Nspace - 1):
        sol.solution[sol.timestep, k] = candidate[k]
    sol.applyRB()
