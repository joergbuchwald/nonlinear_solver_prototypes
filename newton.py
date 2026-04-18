import logging

import numpy as np

from fd_problem import JF, F
from helpers import _apply_step
from linear_solvers import solve_cg, solve_direct
from linesearch_armijo import linesearch_armijo
from linesearch_cubic import linesearch_cubic
from linesearch_wolfe import linesearch_wolfe

logging.basicConfig(filename="solvers_newton.log", level=logging.INFO)

# ---------------------------------------------------------------------------
# Solver / linesearch identifier constants
# ---------------------------------------------------------------------------
SOLVER_DIRECT = "direct"
SOLVER_CG = "cg"
SOLVER_NCG = "nonlinear_cg"
SOLVER_BB = "barzilai_borwein"
SOLVER_TR = "trust_region"
SOLVER_ARC = "arclength"
SOLVER_CN = "cn"

LS_ARMIJO = "armijo"
LS_CUBIC = "cubic"
LS_WOLFE = "wolfe"
LS_NONE = "none"


def run_newton(
    sol,
    dt,
    linear_solver=SOLVER_DIRECT,
    linesearch=LS_ARMIJO,
    damping=1.0,
    max_iter=100,
    tol=1e-10,
    verbose=True,
):
    """
    Newton iteration with pluggable linear solver and linesearch.

    linear_solver : 'direct' | 'cg'
    linesearch    : 'armijo' | 'cubic' | 'wolfe' | 'none'
                    'none' uses fixed  damping  instead
    """
    niter = 0
    res_error = 9e9

    while (res_error > tol) and (niter <= max_iter):
        J = JF(sol, dt)
        f = F(sol, dt)
        rhs = np.matrix(-f)

        update = (
            solve_cg(J, rhs) if linear_solver == SOLVER_CG else solve_direct(J, rhs)
        )

        logging.info(f"iter {niter:3d} Linesearch method: {linesearch}")
        if linesearch == LS_ARMIJO:
            alpha = linesearch_armijo(sol, dt, update)
            logging.info(f"Linesearch alpha: {alpha}")
        elif linesearch == LS_CUBIC:
            alpha = linesearch_cubic(sol, dt, update)
            logging.info(f"Linesearch alpha: {alpha}")
        elif linesearch == LS_WOLFE:
            alpha = linesearch_wolfe(sol, dt, update)
            logging.info(f"Linesearch alpha: {alpha}")
        else:
            alpha = damping
            logging.info(f"damping alpha: {alpha}")
            base = sol.solution[sol.timestep, :].copy()
            _apply_step(sol, base, update, alpha)
            dx_error = alpha * float(np.abs(update).sum())
            res_error = float(np.linalg.norm(F(sol, dt)))
            niter += 1
            if verbose:
                print(
                    f"  iter {niter:3d}:  ||F|| = {res_error:.3e}  "
                    f"|dx| = {dx_error:.3e}  alpha = {alpha:.4f}"
                )
            continue

        dx_error = float(np.abs(update).sum())
        res_error = float(np.linalg.norm(F(sol, dt)))
        niter += 1

        if verbose:
            print(
                f"  iter {niter:3d}:  ||F|| = {res_error:.3e}  "
                f"|dx| = {dx_error:.3e}  alpha = {alpha:.4f}"
            )

    if niter > max_iter:
        print("  WARNING: Newton did not converge within max_iter.")
    return niter
