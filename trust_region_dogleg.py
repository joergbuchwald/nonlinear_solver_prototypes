import numpy as np
from scipy import linalg

from fd_problem import JF, F
from helpers import _apply_step, _phi


def run_trust_region_dogleg(
    sol, dt, max_iter=100, tol=1e-10, Delta0=1.0, Delta_max=100.0, eta=0.1, verbose=True
):
    """
    Trust-region method with Powell double-dogleg step (Nocedal & Wright ch. 4).

    At each iteration solves the constrained subproblem:
        min  m(p) = phi + g^T p + half * p^T B p     s.t. ||p|| <= Delta

    where B = J^T J (Gauss-Newton Hessian) and g = J^T f.

    Dogleg step
    -----------
    The exact solution is expensive; the dogleg approximates it via a
    piecewise-linear path through two anchor points:

      p_U = -(||g||^2 / ||Bg||^2) * g   (Cauchy / steepest-descent step)
      p_N = -(J^T J)^{-1} J^T f         (full Gauss-Newton step)

    Interpolates between them to satisfy the trust-radius constraint.

    Trust-radius update rule
    ------------------------
    rho = actual_reduction / predicted_reduction

      rho < 0.25  -->  shrink Delta (step was bad)
      rho > 0.75  -->  expand Delta (model was pessimistic)
      otherwise   -->  keep   Delta

    Advantages over linesearch
    --------------------------
    - Robust when J is nearly singular (Delta acts as a natural regulariser)
    - No directional derivative needed
    - Quadratic local convergence once Delta >= ||p_N||
    """

    def grad_and_hess(s):
        J = np.array(JF(s, dt))
        f = np.array(F(s, dt)).flatten()
        return J.T @ f, J.T @ J

    Delta = Delta0
    res_error = float(np.linalg.norm(F(sol, dt)))
    niter = 0

    for niter in range(1, max_iter + 1):
        if res_error <= tol:
            break

        g, B = grad_and_hess(sol)
        phi0 = _phi(sol, dt)
        g_norm = np.linalg.norm(g)

        # Cauchy step
        Bg = B @ g
        gBg = float(g @ Bg)
        alpha_sd = (g_norm**2) / gBg if gBg > 1e-15 else Delta / (g_norm + 1e-15)
        p_U = -alpha_sd * g

        # Full Gauss-Newton step  (small Tikhonov regularisation for safety)
        try:
            p_N = -linalg.solve(B + 1e-10 * np.eye(len(g)), g)
        except linalg.LinAlgError:
            p_N = p_U.copy()

        pU_norm = np.linalg.norm(p_U)
        pN_norm = np.linalg.norm(p_N)

        # Dogleg combination
        if pN_norm <= Delta:
            p = p_N  # full Newton step fits in trust region
        elif pU_norm >= Delta:
            p = (Delta / pU_norm) * p_U  # Cauchy step scaled to boundary
        else:
            # Interpolate: p = p_U + tau*(p_N - p_U),  tau in [0, 1]
            d = p_N - p_U
            dd = float(d @ d)
            ud = float(p_U @ d)
            uu = float(p_U @ p_U)
            disc = ud**2 - dd * (uu - Delta**2)
            tau = (-ud + np.sqrt(max(disc, 0.0))) / (dd + 1e-15)
            tau = float(np.clip(tau, 0.0, 1.0))
            p = p_U + tau * d

        # Evaluate actual vs predicted reduction
        base = sol.solution[sol.timestep, :].copy()
        _apply_step(sol, base, p, 1.0)
        phi_new = _phi(sol, dt)

        m_reduction = -(float(g @ p) + 0.5 * float(p @ (B @ p)))
        actual_red = phi0 - phi_new
        rho = actual_red / (m_reduction + 1e-30)

        # Accept / reject
        if rho < eta:
            sol.solution[sol.timestep, :] = base.copy()
            sol.applyRB()

        # Update trust radius
        if rho < 0.25:
            Delta *= 0.25
        elif rho > 0.75 and abs(pN_norm - Delta) < 1e-8:
            Delta = min(2.0 * Delta, Delta_max)

        res_error = float(np.linalg.norm(F(sol, dt)))

        if verbose:
            status = "accept" if rho >= eta else "REJECT"
            print(
                f"  iter {niter:3d}:  ||F|| = {res_error:.3e}  "
                f"rho = {rho:+.3f}  Delta = {Delta:.3e}  [{status}]"
            )

    if res_error > tol:
        print("  WARNING: Trust-region dogleg did not converge within max_iter.")
    return niter
