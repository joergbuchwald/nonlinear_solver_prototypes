import numpy as np

from fd_problem import JF, F
from helpers import _apply_step, _phi


def run_barzilai_borwein(
    sol, dt, max_iter=500, tol=1e-10, bb_variant="BB1", verbose=True
):
    """
    Barzilai-Borwein (BB) / Spectral Projected Gradient method.

    Minimises  phi(x) = half*||F(x)||^2  using gradient steps with a
    quasi-Newton step-length estimate. No Jacobian solve required.

    BB step-length formulas
    -----------------------
    Let  s = x_k - x_{k-1},   y = g_k - g_{k-1}

      BB1 (long step):   alpha = (s^T s) / (s^T y)   minimises ||alpha*y - s||
      BB2 (short step):  alpha = (s^T y) / (y^T y)   minimises ||s - alpha^{-1}*y||

    BB steps are used as the initial trial length; an Armijo fallback
    ensures global convergence (SPG - Spectral Projected Gradient).

    Convergence is superlinear in practice despite being formally first-order,
    because the BB step approximates a secant Hessian scaling.
    """

    def grad(s):
        J = np.array(JF(s, dt))
        f = np.array(F(s, dt)).flatten()
        return J.T @ f

    g = grad(sol)
    alpha = 1.0 / (np.linalg.norm(g) + 1e-15)  # safe initial step

    res_error = float(np.linalg.norm(F(sol, dt)))
    niter = 0

    for niter in range(1, max_iter + 1):
        if res_error <= tol:
            break

        direction = -g
        phi0 = _phi(sol, dt)
        dphi0 = float(g @ direction)
        cur_base = sol.solution[sol.timestep, :].copy()

        # Armijo fallback (SPG safety net)
        alpha_ls = alpha
        for _ in range(30):
            _apply_step(sol, cur_base, direction, alpha_ls)
            if _phi(sol, dt) <= phi0 + 1e-4 * alpha_ls * dphi0:
                break
            alpha_ls *= 0.5
        else:
            sol.solution[sol.timestep, :] = cur_base.copy()
            sol.applyRB()

        g_new = grad(sol)

        # BB step-length for next iteration
        s = sol.solution[sol.timestep, :] - cur_base
        y = g_new - g
        sy = float(s @ y)
        ss = float(s @ s)
        yy = float(y @ y)

        if abs(sy) > 1e-15:
            alpha = (ss / sy) if bb_variant == "BB1" else (sy / yy)
        alpha = float(np.clip(alpha, 1e-10, 1e3))

        g = g_new
        res_error = float(np.linalg.norm(F(sol, dt)))

        if verbose:
            print(
                f"  iter {niter:3d}:  ||F|| = {res_error:.3e}  "
                f"|g| = {np.linalg.norm(g):.3e}  alpha_BB = {alpha:.4e}"
            )

    if res_error > tol:
        print("  WARNING: Barzilai-Borwein did not converge within max_iter.")
    return niter
