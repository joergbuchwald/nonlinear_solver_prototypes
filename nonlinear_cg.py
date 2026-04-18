import numpy as np

from fd_problem import JF, F
from linesearch_armijo import linesearch_armijo
from linesearch_cubic import linesearch_cubic
from linesearch_wolfe import linesearch_wolfe


def run_nonlinear_cg(
    sol,
    dt,
    max_iter=500,
    tol=1e-10,
    restart_every=None,
    beta_formula="PR+",
    linesearch="wolfe",
    verbose=True,
):
    """
    Nonlinear Conjugate Gradient -- minimises  phi(x) = half*||F(x)||^2.

    No Jacobian solve required; search directions built from successive
    gradients  g = J^T f.

    beta_formula : 'FR' | 'PR' | 'PR+'
    linesearch   : 'armijo' | 'cubic' | 'wolfe'
                   Strong Wolfe is the theoretically correct choice for PR+.
    """
    n = sol.Nspace
    if restart_every is None:
        restart_every = n

    def grad(s):
        J = np.array(JF(s, dt))
        f = np.array(F(s, dt)).flatten()
        return J.T @ f

    def do_ls(direction):
        p = np.matrix(direction).T
        if linesearch == "cubic":
            return linesearch_cubic(sol, dt, p)
        elif linesearch == "wolfe":
            return linesearch_wolfe(sol, dt, p)
        else:
            return linesearch_armijo(sol, dt, p)

    g = grad(sol)
    p = -g.copy()
    res_error = float(np.linalg.norm(F(sol, dt)))
    niter = 0

    for niter in range(1, max_iter + 1):
        if res_error <= tol:
            break

        if float(g @ p) >= 0:
            p = -g.copy()

        do_ls(p)
        g_new = grad(sol)

        g_norm_sq = float(g @ g)
        g_new_norm_sq = float(g_new @ g_new)

        if g_norm_sq < 1e-30:
            beta = 0.0
        elif beta_formula == "FR":
            beta = g_new_norm_sq / g_norm_sq
        elif beta_formula in ("PR", "PR+"):
            beta = float(g_new @ (g_new - g)) / g_norm_sq
            if beta_formula == "PR+":
                beta = max(beta, 0.0)
        else:
            raise ValueError(f"Unknown beta_formula '{beta_formula}'")

        if niter % restart_every == 0:
            beta = 0.0

        p = -g_new + beta * p
        g = g_new
        res_error = float(np.linalg.norm(F(sol, dt)))

        if verbose:
            print(
                f"  iter {niter:3d}:  ||F|| = {res_error:.3e}  "
                f"|g| = {np.linalg.norm(g):.3e}  beta = {beta:.4f}"
            )

    if res_error > tol:
        print("  WARNING: Nonlinear CG did not converge within max_iter.")
    return niter
