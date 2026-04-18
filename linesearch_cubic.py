import numpy as np

from helpers import _apply_step, _dphi_directional, _phi


def linesearch_cubic(sol, dt, update, alpha0=1.0, c=1e-4, alpha_min=1e-8, max_bt=20):
    """
    Cubic interpolation linesearch (Nocedal & Wright sec. 3.5).

    Instead of halving alpha, fits a cubic through the two most recent
    (alpha, phi(alpha)) pairs and jumps to the cubic minimum.
    Typically needs 2-3x fewer residual evaluations than pure backtracking.

    Algorithm
    ---------
    1st trial  : quadratic using phi(0), phi(alpha0), phi'(0).
    Subsequent : cubic    using phi(alpha_{i-1}), phi(alpha_i), phi'(0).
    Accept as soon as Armijo condition is met.
    """
    phi0 = _phi(sol, dt)
    dphi0 = _dphi_directional(sol, dt, update)
    base = sol.solution[sol.timestep, :].copy()

    has_prev = False
    alpha_prev = 0.0
    phi_prev = 0.0
    alpha = alpha0

    for _ in range(max_bt):
        _apply_step(sol, base, update, alpha)
        phi_a = _phi(sol, dt)

        if phi_a <= phi0 + c * alpha * dphi0:
            print(f"Backtracking linesearch converged, alpha = {alpha}")
            return alpha

        if not has_prev:
            # Quadratic interpolation
            alpha_new = -dphi0 * alpha**2 / (2.0 * (phi_a - phi0 - dphi0 * alpha))
        else:
            # Cubic interpolation
            rhs1 = phi_a - phi0 - dphi0 * alpha
            rhs2 = phi_prev - phi0 - dphi0 * alpha_prev
            d = alpha - alpha_prev
            A_ = (rhs1 / alpha**2 - rhs2 / alpha_prev**2) / d
            B_ = (-rhs1 * alpha_prev / alpha**2 + rhs2 * alpha / alpha_prev**2) / d
            disc = B_**2 - 3.0 * A_ * dphi0
            if abs(A_) < 1e-15 or disc < 0:
                alpha_new = alpha / 2.0
            else:
                alpha_new = (-B_ + np.sqrt(disc)) / (3.0 * A_)

        has_prev = True
        alpha_prev = alpha
        phi_prev = phi_a
        alpha = float(np.clip(alpha_new, 0.1 * alpha, 0.9 * alpha))
        alpha = max(alpha, alpha_min)

    sol.solution[sol.timestep, :] = base
    sol.applyRB()
    return alpha
