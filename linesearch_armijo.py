import logging

from helpers import _apply_step, _dphi_directional, _phi

logging.basicConfig(filename="solvers_newton_armijo.log", level=logging.INFO)


def linesearch_armijo(sol, dt, update, alpha0=1.0, rho=0.5, c=1e-4, max_bt=20):
    """
    Armijo (sufficient-decrease) backtracking.

    Reduces alpha by factor rho until:
        phi(x + alpha*p) <= phi(x) + c * alpha * phi'(0)

    Parameters
    ----------
    rho  : reduction factor per step  (0 < rho < 1)
    c    : Armijo constant             (0 < c << 1, typically 1e-4)
    """
    phi0 = _phi(sol, dt)
    dphi0 = _dphi_directional(sol, dt, update)
    base = sol.solution[sol.timestep, :].copy()
    alpha = alpha0

    for _ in range(max_bt):
        _apply_step(sol, base, update, alpha)
        if _phi(sol, dt) <= phi0 + c * alpha * dphi0:
            logging.info(f"backtracking linesearch finished after {_} steps.")
            logging.info(f"alpha: {alpha} accepted.")
            return alpha
        alpha *= rho

    sol.solution[sol.timestep, :] = base
    sol.applyRB()
    return alpha
