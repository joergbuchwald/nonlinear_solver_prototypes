import numpy as np
from scipy import linalg

from arclength_continuation import _dF_dlam, _F_lam, _JF_lam


def _cn_compute_tangent(sol, lam, dt_target, active=None):
    """
    Compute the normalised CN tangent vector at (sol, lam).

    Differentiating R[U(xi), lam(xi)] = 0 along the solution path gives
    the equations of motion (Younis et al. Eq. 7):

        J . dU/dxi  +  dF/dlam . dlam/dxi  =  0

    Setting dlam/dxi = 1 and solving one linear system:

        J . v     = dF/dlam        (load-vector RHS -- same J as Newton)
        xi_U      = -v             (spatial component before normalisation)
        xi_lam    = 1              (lambda component before normalisation)

    The pair is then normalised so that ||(xi_U, xi_lam)|| = 1.
    xi_lam is always positive, so every tangent step advances lambda.

    Parameters
    ----------
    active : bool array of shape (Nspace,), or None
        When given, solve only for the active interior subset
        (localisation). Inactive cells receive a zero tangent component.

    Returns
    -------
    xi_U   : 1-D ndarray  (length Nspace)
    xi_lam : float
    """
    n = sol.Nspace
    J = np.array(_JF_lam(sol, lam, dt_target))
    dFdlam = _dF_dlam(sol, lam, dt_target)

    if active is None:
        # Full-system solve
        try:
            v = linalg.solve(J, dFdlam)
        except linalg.LinAlgError:
            return np.zeros(n), 1.0
    else:
        # Localised solve: restrict to interior active nodes
        v = np.zeros(n)
        mask = active & (np.arange(n) > 0) & (np.arange(n) < n - 1)
        idx = np.where(mask)[0]
        if len(idx):
            try:
                v[idx] = linalg.solve(J[np.ix_(idx, idx)], dFdlam[idx])
            except linalg.LinAlgError:
                pass  # leave v = 0; tangent degrades to pure-lambda direction

    xi_U = -v
    xi_lam = 1.0
    norm = np.sqrt(np.dot(xi_U, xi_U) + xi_lam**2)
    if norm > 1e-15:
        xi_U /= norm
        xi_lam /= norm
    return xi_U, xi_lam


def _cn_select_step(
    sol,
    lam,
    dt_target,
    xi_U,
    xi_lam,
    lam_end,
    nbhd_tol,
    alpha_min,
    alpha_max,
    max_bt=10,
):
    """
    Find the *largest* alpha in [alpha_min, alpha_max] such that the
    proposed point (U + alpha*xi_U, lam + alpha*xi_lam) lies inside the
    convergence neighbourhood ||F|| <= nbhd_tol  (Algorithm 4 of the paper).

    Backtracks by halving alpha from alpha_max.  sol is left unchanged
    (all trial states are reverted before returning).

    Returns
    -------
    alpha : float   largest admissible step; alpha_min if none found
    found : bool    True when the returned alpha satisfies the check
    """
    n = sol.Nspace
    base = sol.solution[sol.timestep, :].copy()
    alpha = alpha_max

    for _ in range(max_bt):
        lam_try = float(np.clip(lam + alpha * xi_lam, 0.0, lam_end))
        for k in range(1, n - 1):
            sol.solution[sol.timestep, k] = base[k] + alpha * xi_U[k]
        sol.applyRB()
        res = float(np.linalg.norm(_F_lam(sol, lam_try, dt_target)))
        # Always restore before continuing or returning
        sol.solution[sol.timestep, :] = base
        sol.applyRB()
        if res <= nbhd_tol:
            return alpha, True
        alpha *= 0.5
        if alpha < alpha_min:
            break

    return alpha_min, False


def _cn_active_cells(sol, lam, dt_target, loc_tol):
    """
    Identify active cells for the localisation step.

    Primary-active: |R_i| > loc_tol.
    The set is expanded by one cell in each direction to capture the
    propagation of material-balance errors through the upwind graph
    (Younis et al. Appendix C).  Boundary cells are always excluded.

    Returns
    -------
    active : bool array of shape (Nspace,)
    """
    n = sol.Nspace
    f = np.abs(np.array(_F_lam(sol, lam, dt_target)).flatten())
    primary = f > loc_tol
    expanded = primary.copy()
    for i in range(1, n - 1):
        if primary[i]:
            expanded[max(i - 1, 0)] = True
            expanded[min(i + 1, n - 1)] = True
    expanded[0] = expanded[-1] = False  # boundaries are fixed by Dirichlet BCs
    return expanded


def run_cn(
    sol,
    dt_target,
    lam_start=0.0,
    lam_end=1.0,
    neighborhood_tol=1e-3,
    final_tol=1e-10,
    alpha_min=1e-4,
    alpha_max=0.5,
    max_iter=200,
    max_newton_corr=10,
    corrector_tol_frac=0.01,
    localize=False,
    loc_tol_frac=0.01,
    verbose=True,
):
    """
    Continuation-Newton (CN) method -- Younis, Tchelepi & Aziz (SPE J., 2010).

    The implicit-timestep residual R(U, lambda; U^n) = 0  is viewed as defining a
    *solution path* in the augmented (U, lambda) space.  The path starts at
    (U^n, lambda=0), where the residual is trivially zero, and ends at the solution
    (U^{n+1}, lambda=1) of the full timestep.

    Algorithm overview  (Algorithm 1 in the paper)
    -----------------------------------------------
    At each iteration:
      1. Compute the tangent to the solution path (one Jacobian solve with
         the load-vector dF/dlambda as RHS -- identical cost to one Newton step).
      2. Select the *largest* step length alpha in [alpha_min, alpha_max] such that the
         tangent step keeps the iterate inside the convergence neighbourhood
         ||F|| <= neighborhood_tol  (backtracking search, Algorithm 4).
      3. Apply the tangent step:  (U, lambda) <- (U + alpha*xi_U, lambda + alpha*xi_lam).
      4. If the new point is outside the neighbourhood (only possible when
         even alpha_min failed), revert the step and apply a Newton corrector at
         the *current* lambda to pull the iterate back to the solution path.

    After the loop, a final Newton polish drives ||F|| below final_tol at
    whatever lambda was reached.  If the loop exits before lambda = lam_end (because
    max_iter was exhausted), the last accepted iterate is a valid solution to
    a smaller known sub-step -- no computation is wasted.

    Localisation  (optional, localize=True)
    ----------------------------------------
    Implements the adaptive localisation from the paper (Section 4).
    Cells whose residual magnitude is below loc_tol = neighborhood_tol *
    loc_tol_frac are classified as inactive and excluded from the linear
    solve.  The active set is expanded by one cell in each direction to
    capture local wave propagation (Younis et al. Appendix C).
    For diffusion-dominated problems (like the heat equation here) the
    savings are modest; the feature shines for near-hyperbolic transport.

    Parameters
    ----------
    sol              : Solution  (modified in-place)
    dt_target        : float     full timestep size  (lambda=1 <-> dt_target)
    lam_start        : float     starting load level  (usually 0)
    lam_end          : float     target load level    (usually 1)
    neighborhood_tol : float     ||F|| threshold for the convergence neighbourhood
    final_tol        : float     ||F|| tolerance for the final Newton polish
    alpha_min        : float     minimum tangent step length
    alpha_max        : float     maximum tangent step length
    max_iter         : int       maximum total CN iterations
    max_newton_corr  : int       maximum Newton steps per corrector call
    corrector_tol_frac : float   corrector targets ||F|| <= neighborhood_tol * corrector_tol_frac
    localize         : bool      enable adaptive localisation
    loc_tol_frac     : float     loc_tol = neighborhood_tol * loc_tol_frac
    verbose          : bool

    Returns
    -------
    lam : float   load level reached (= lam_end on full success)
    """
    n = sol.Nspace
    lam = lam_start
    loc_tol = neighborhood_tol * loc_tol_frac
    corrector_tol = neighborhood_tol * corrector_tol_frac

    n_tangent = 0  # accepted tangent steps
    n_newton = 0  # Newton corrector steps

    if verbose:
        r0 = float(np.linalg.norm(_F_lam(sol, lam, dt_target)))
        print(
            f"  CN start:  lam = {lam:.6f}  ||F|| = {r0:.3e}"
            f"  (nbhd_tol = {neighborhood_tol:.1e})"
        )

    for it in range(1, max_iter + 1):
        if lam >= lam_end - 1e-10:
            break

        # ----------------------------------------------------------------
        # (a) Active set for localisation
        # ----------------------------------------------------------------
        active = _cn_active_cells(sol, lam, dt_target, loc_tol) if localize else None
        n_active = int(np.sum(active)) if active is not None else (n - 2)

        # ----------------------------------------------------------------
        # (b) Tangent computation  (Younis et al. Eq. 7 -- one linear solve)
        # ----------------------------------------------------------------
        xi_U, xi_lam = _cn_compute_tangent(sol, lam, dt_target, active=active)

        # ----------------------------------------------------------------
        # (c) Step-length selection  (Algorithm 4 -- backtrack from alpha_max)
        # ----------------------------------------------------------------
        alpha, step_ok = _cn_select_step(
            sol,
            lam,
            dt_target,
            xi_U,
            xi_lam,
            lam_end,
            neighborhood_tol,
            alpha_min,
            alpha_max,
        )

        # ----------------------------------------------------------------
        # (d) Apply tangent step, then check convergence neighbourhood
        # ----------------------------------------------------------------
        x_save = sol.solution[sol.timestep, :].copy()
        lam_new = float(np.clip(lam + alpha * xi_lam, 0.0, lam_end))
        for k in range(1, n - 1):
            sol.solution[sol.timestep, k] += alpha * xi_U[k]
        sol.applyRB()
        n_tangent += 1

        res = float(np.linalg.norm(_F_lam(sol, lam_new, dt_target)))
        in_nbhd = res <= neighborhood_tol

        if verbose:
            loc_str = f"  active={n_active}/{n - 2}" if localize else ""
            tag = "ok" if in_nbhd else "-> Newton corr"
            print(
                f"  iter {it:3d} [tangent]:  lam={lam_new:.6f}  "
                f"alpha={alpha:.3e}  ||F||={res:.3e}  {tag}{loc_str}"
            )

        if in_nbhd:
            lam = lam_new
            continue

        # ----------------------------------------------------------------
        # (e) Tangent step left neighbourhood -> revert and apply Newton corrector
        # ----------------------------------------------------------------
        sol.solution[sol.timestep, :] = x_save
        sol.applyRB()

        for nc in range(1, max_newton_corr + 1):
            f_vec = np.array(_F_lam(sol, lam, dt_target)).flatten()
            J_x = np.array(_JF_lam(sol, lam, dt_target))

            if localize and active is not None:
                idx = np.where(active & (np.arange(n) > 0) & (np.arange(n) < n - 1))[0]
                if len(idx):
                    try:
                        dx_loc = np.zeros(n)
                        dx_loc[idx] = linalg.solve(J_x[np.ix_(idx, idx)], -f_vec[idx])
                        for k in range(1, n - 1):
                            sol.solution[sol.timestep, k] += dx_loc[k]
                        sol.applyRB()
                    except linalg.LinAlgError:
                        break
                else:
                    break
            else:
                try:
                    dx = linalg.solve(J_x, -f_vec)
                    for k in range(1, n - 1):
                        sol.solution[sol.timestep, k] += dx[k]
                    sol.applyRB()
                except linalg.LinAlgError:
                    break
            n_newton += 1

            res_c = float(np.linalg.norm(_F_lam(sol, lam, dt_target)))
            if verbose:
                print(f"    Newton corr {nc}: ||F|| = {res_c:.3e}")
            if res_c <= corrector_tol:
                break

        if localize:
            active = _cn_active_cells(sol, lam, dt_target, loc_tol)

    # ----------------------------------------------------------------
    # Final Newton polish at the reached lam  (Algorithm 1, last line)
    # ----------------------------------------------------------------
    if verbose:
        r_pre = float(np.linalg.norm(_F_lam(sol, lam, dt_target)))
        print(
            f"\n  CN loop done: lam = {lam:.8f}  ||F|| = {r_pre:.3e}"
            f"  (tangent steps: {n_tangent},  Newton corr: {n_newton})"
        )
        print(f"  Final Newton polish to tol = {final_tol:.1e} ...")

    for fp in range(1, 51):
        f_vec = np.array(_F_lam(sol, lam, dt_target)).flatten()
        res_f = float(np.linalg.norm(f_vec))
        if verbose:
            print(f"    polish {fp}: ||F|| = {res_f:.3e}")
        if res_f <= final_tol:
            break
        J_x = np.array(_JF_lam(sol, lam, dt_target))
        try:
            dx = linalg.solve(J_x, -f_vec)
        except linalg.LinAlgError:
            break
        for k in range(1, n - 1):
            sol.solution[sol.timestep, k] += dx[k]
        sol.applyRB()

    if verbose:
        r_fin = float(np.linalg.norm(_F_lam(sol, lam, dt_target)))
        print(f"  Final: lam = {lam:.8f}  ||F|| = {r_fin:.3e}\n")

    return lam
