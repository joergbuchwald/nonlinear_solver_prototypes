from helpers import _apply_step, _dphi_directional, _phi


def linesearch_wolfe(sol, dt, update, c1=1e-4, c2=0.9, alpha_max=4.0, max_iter=20):
    """
    Strong Wolfe conditions linesearch (Nocedal & Wright Algorithms 3.5/3.6).

    Enforces both conditions:
      (i)  Armijo sufficient decrease:  phi(alpha*p) <= phi(0) + c1*alpha*phi'(0)
      (ii) Curvature condition:         |phi'(alpha*p)| <= c2 * |phi'(0)|

    The curvature condition prevents alpha from being unnecessarily small,
    which is a theoretical requirement for PR+ nonlinear CG to converge.

    Parameters
    ----------
    c1       : Armijo constant    (0 < c1 < c2 < 1)
    c2       : curvature constant (c1 < c2 < 1; use 0.9 for Newton, 0.1 for NCG)
    alpha_max: upper bound on alpha during bracketing
    """
    phi0 = _phi(sol, dt)
    dphi0 = _dphi_directional(sol, dt, update)
    base = sol.solution[sol.timestep, :].copy()

    def phi_and_dphi(a):
        _apply_step(sol, base, update, a)
        p = _phi(sol, dt)
        dp = _dphi_directional(sol, dt, update)
        sol.solution[sol.timestep, :] = base.copy()
        sol.applyRB()
        return p, dp

    def zoom(alpha_lo, alpha_hi, phi_lo):
        alpha = alpha_lo  # default if max_iter == 0
        for _ in range(max_iter):
            alpha = 0.5 * (alpha_lo + alpha_hi)
            phi_a, dphi_a = phi_and_dphi(alpha)
            if phi_a > phi0 + c1 * alpha * dphi0 or phi_a >= phi_lo:
                alpha_hi = alpha
            else:
                if abs(dphi_a) <= -c2 * dphi0:
                    return alpha
                if dphi_a * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha
                phi_lo = phi_a
        return alpha

    alpha_prev, phi_prev = 0.0, phi0
    alpha = min(1.0, alpha_max)

    for i in range(max_iter):
        phi_a, dphi_a = phi_and_dphi(alpha)
        if phi_a > phi0 + c1 * alpha * dphi0 or (i > 0 and phi_a >= phi_prev):
            alpha = zoom(alpha_prev, alpha, phi_prev)
            break
        if abs(dphi_a) <= -c2 * dphi0:
            break
        if dphi_a >= 0:
            alpha = zoom(alpha, alpha_prev, phi_a)
            break
        alpha_prev = alpha
        phi_prev = phi_a
        alpha = min(2.0 * alpha, alpha_max)

    _apply_step(sol, base, update, alpha)
    print(f"Linesearch alpha: {alpha}")
    return alpha
