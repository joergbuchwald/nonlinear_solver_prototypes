import numpy as np
from scipy import linalg

from fd_problem import JF, F


def _F_lam(sol, lam, dt_target, dT=1e-6):
    """
    Residual evaluated at load level  lambda, i.e. with effective
    time step  dt_eff = lambda * dt_target.

    lambda = 0  -->  trivial (no time advance),  F = 0
    lambda = 1  -->  full target time step dt_target
    """
    return F(sol, lam * dt_target)


def _JF_lam(sol, lam, dt_target):
    """
    Spatial Jacobian  dF/dx  at load level lambda.
    Reuses the existing finite-difference Jacobian with dt_eff.
    """
    return JF(sol, lam * dt_target)


def _dF_dlam(sol, lam, dt_target, dlam=1e-7):
    """
    Derivative of residual with respect to the load parameter lambda,
    approximated by a forward finite difference:

        dF/dlam ~= [ F(x, lam+dlam) - F(x, lam) ] / dlam

    This is the 'load vector' column that enters the bordered system.
    """
    f_plus = np.array(_F_lam(sol, lam + dlam, dt_target)).flatten()
    f_0 = np.array(_F_lam(sol, lam, dt_target)).flatten()
    return (f_plus - f_0) / dlam


def run_arclength_continuation(
    sol,
    dt_target,
    lam_start=0.0,
    lam_end=1.0,
    ds=0.1,
    ds_min=1e-4,
    ds_max=0.5,
    max_steps=200,
    max_iter=20,
    tol=1e-10,
    adapt_ds=True,
    iter_target=4,
    scale_x=1.0,
    verbose=True,
):
    """
    Pseudo-arclength continuation  (Keller 1977)  along the load axis.

    Treats the time-step multiplier  lambda  as an additional unknown and
    advances the solution from  lam_start  to  lam_end  in arclength steps
    rather than fixed load increments.  This allows the path to turn corners
    (fold / limit points) that would defeat ordinary load-stepping.

    Physical interpretation for this PDE
    -------------------------------------
    The residual is  F(x, lambda) = F(x, lambda * dt_target).
    * lambda = 0  is trivially satisfied by the previous time-step's solution.
    * lambda = 1  is the full implicit time step we actually want.

    By parameterising along the arclength of the (x, lambda) curve instead of
    along lambda directly, the method handles stiff or nearly-singular steps
    that cause ordinary Newton to diverge.

    Augmented system
    ----------------
    At each continuation step we solve the  (n+1) x (n+1)  bordered system:

        | J_x     dF/dlam | | dx   |   | -F(x, lam)              |
        |                  | |      | = |                          |
        | dx_ds^T  dlam_ds | | dlam |   | Ds - N(x, lam)          |

    where  N(x, lam) = (x - x0).dx_ds + (lam - lam0).dlam_ds  is the
    arclength constraint and  (dx_ds, dlam_ds)  is the unit tangent vector
    from the previous accepted step.

    Bordered system solution  (2-solve trick)
    -----------------------------------------
    To avoid assembling the full (n+1) system, we use the Sherman-Morrison
    formula.  With  J_x * v = dF/dlam  and  J_x * w = -F  we get:

        dx   = w - dlam * v
        dlam = -(N_val + dx_ds . w) / (dlam_ds - dx_ds . v)

    This costs exactly two linear solves per Newton iteration.

    Step-length adaptation
    ----------------------
    After a successful step, ds is scaled by  sqrt(iter_target / iter_taken)
    so that well-converged steps allow a larger ds next time and
    struggling steps shrink it.

    Parameters
    ----------
    sol        : Solution   solution object (modified in-place)
    dt_target  : float      the full time-step size (lambda=1 corresponds to dt_target)
    lam_start  : float      starting load level  (usually 0)
    lam_end    : float      target load level    (usually 1)
    ds         : float      initial arclength step size
    ds_min     : float      minimum allowed ds  (signals failure if reached)
    ds_max     : float      maximum allowed ds
    max_steps  : int        maximum number of continuation steps
    max_iter   : int        Newton iterations per step
    tol        : float      Newton convergence tolerance on ||F||
    adapt_ds   : bool       adaptively rescale ds based on iteration count
    iter_target: int        desired Newton iterations per continuation step used for
                            ds adaptation:  ds_new = ds * sqrt(iter_target / niter).
                            Set to the typical Newton count you observe (e.g. 10-12
                            for a stiff problem) to prevent ds from shrinking on
                            every step.  Default 4.
    scale_x    : float      characteristic scale of the spatial state (e.g. T2 - T1).
                            The arclength distance is measured as
                            sqrt( ||dx / scale_x||^2 + dlam^2 ),
                            so that spatial and load components contribute
                            comparably.  Default 1.0 (no scaling).
    verbose    : bool

    Returns
    -------
    lam : float   final load level reached  (1.0 if successful)
    """
    n = sol.Nspace

    # ------------------------------------------------------------------
    # Internal helper: scaled dot product for the arclength metric.
    # All tangent vectors dx_ds are stored in *scaled* space (x / scale_x),
    # so the arclength norm is  sqrt( ||dx_ds||^2 + dlam_ds^2 ) = 1.
    # ------------------------------------------------------------------

    # Initial tangent vector: predictor along lambda axis only.
    # In scaled space this is (0, 1) — same as unscaled.
    dx_ds = np.zeros(n)  # stored in scaled coords (x / scale_x)
    dlam_ds = 1.0
    tang_norm = np.sqrt(np.dot(dx_ds, dx_ds) + dlam_ds**2)
    dx_ds /= tang_norm
    dlam_ds /= tang_norm

    lam = lam_start

    for step in range(1, max_steps + 1):
        # Predictor  ------------------------------------------------
        lam_pred = lam + ds * dlam_ds
        # dx_ds is in scaled coords; convert back to physical space for the update
        x_pred = sol.solution[sol.timestep, :].copy() + ds * dx_ds * scale_x

        # Clamp lambda predictor to [lam_start, lam_end]
        lam_pred = float(np.clip(lam_pred, lam_start, lam_end))

        # Apply predicted state (interior nodes only)
        x_save = sol.solution[sol.timestep, :].copy()
        for k in range(1, n - 1):
            sol.solution[sol.timestep, k] = x_pred[k]
        sol.applyRB()
        lam_cur = lam_pred

        # Store base point for arclength constraint
        x0 = x_save.copy()
        lam0 = lam

        if verbose:
            print(f"  Step {step:3d}:  lam = {lam_cur:.6f}  ds = {ds:.4e}")

        # Corrector  (Newton on the bordered system)  ---------------
        converged = False
        niter = 0
        for niter in range(1, max_iter + 1):
            f_vec = np.array(_F_lam(sol, lam_cur, dt_target)).flatten()
            J_x = np.array(_JF_lam(sol, lam_cur, dt_target))
            dFdlam = _dF_dlam(sol, lam_cur, dt_target)

            # Arclength constraint (in scaled coords):
            # N = (x - x0)/scale_x . dx_ds + (lam - lam0).dlam_ds - ds = 0
            x_cur = sol.solution[sol.timestep, :]
            N_val = (
                np.dot((x_cur - x0) / scale_x, dx_ds) + (lam_cur - lam0) * dlam_ds - ds
            )

            # ------ 2-solve bordered system (Sherman-Morrison) ------
            # Solve  J_x * v = dF/dlam
            try:
                v = linalg.solve(J_x, dFdlam)
            except linalg.LinAlgError:
                print("    Singular Jacobian — aborting step.")
                break

            # Solve  J_x * w = -F
            try:
                w = linalg.solve(J_x, -f_vec)
            except linalg.LinAlgError:
                print("    Singular Jacobian — aborting step.")
                break

            # dlam correction from arclength constraint (scaled coords).
            # Linearise N=0:  dx_ds . (dx/scale_x) + dlam_ds . dlam = -N_val
            # Substitute dx = w - dlam*v:
            #   dlam = -(N_val + dx_ds . w/scale_x) / (dlam_ds - dx_ds . v/scale_x)
            #
            # Special case: if lam_cur is already pinned at lam_end the
            # bordered-system correction cannot advance lambda further, so the
            # arclength constraint would force a non-zero dlam_corr that gets
            # clamped to zero, producing a wrong spatial step.  Instead, fall
            # back to a plain Newton step (dx = w, dlam = 0) at lam_end.
            if lam_cur >= lam_end - 1e-12:
                dx_corr = w
                dlam_corr = 0.0
            else:
                denom = dlam_ds - np.dot(dx_ds, v) / scale_x
                if abs(denom) < 1e-15:
                    print("    Tangent denominator near zero — aborting step.")
                    break
                dlam_corr = -(N_val + np.dot(dx_ds, w) / scale_x) / denom
                dx_corr = w - dlam_corr * v

            # Update
            lam_cur += dlam_corr
            # Hard clamp: lambda must stay within [0, lam_end]
            lam_cur = float(np.clip(lam_cur, 0.0, lam_end))
            for k in range(1, n - 1):
                sol.solution[sol.timestep, k] += dx_corr[k]
            sol.applyRB()

            res = float(np.linalg.norm(_F_lam(sol, lam_cur, dt_target)))
            if verbose:
                print(
                    f"    Newton {niter:2d}:  ||F|| = {res:.3e}  "
                    f"lam = {lam_cur:.6f}  dlam = {dlam_corr:+.3e}"
                )

            if res < tol:
                converged = True
                break

        # ------ Step accepted / rejected ---------------------------
        if not converged:
            # Reject: restore previous state, halve ds
            sol.solution[sol.timestep, :] = x_save.copy()
            sol.applyRB()
            ds = max(ds * 0.5, ds_min)
            if ds <= ds_min:
                print("  WARNING: ds reached minimum — arclength failed to converge.")
                return lam
            if verbose:
                print(f"  Step REJECTED — reducing ds to {ds:.4e}\n")
            continue

        # ------ Update tangent vector (secant approximation, scaled coords) -------
        dx_new = (sol.solution[sol.timestep, :] - x0) / scale_x  # scaled
        dlam_new = lam_cur - lam0
        tang_norm = np.sqrt(np.dot(dx_new, dx_new) + dlam_new**2)
        if tang_norm > 1e-15:
            dx_new = dx_new / tang_norm
            dlam_new = dlam_new / tang_norm
            # Always orient tangent to advance toward lam_end (+lambda direction)
            if dlam_new < 0:
                dx_new = -dx_new
                dlam_new = -dlam_new
            dx_ds = dx_new  # stored in scaled coords
            dlam_ds = dlam_new

        # ------ Adapt ds  ------------------------------------------
        if adapt_ds:
            ds = ds * np.sqrt(iter_target / max(niter, 1))  # type: ignore[arg-type]
            ds = float(np.clip(ds, ds_min, ds_max))

        lam = lam_cur

        if verbose:
            print(
                f"  --> accepted  lam = {lam:.6f}  "
                f"(iters = {niter})  next ds = {ds:.4e}\n"
            )

        # Check if we have reached lam_end
        if lam >= lam_end - 1e-10:
            if verbose:
                print(f"  Continuation complete at lam = {lam:.8f}")
            break

    # ------------------------------------------------------------------
    # Final Newton polish at lam_end.
    # Ensures the returned solution satisfies F(x, dt_target) = 0 to tol
    # even when the last arclength step stopped just short of lam_end
    # (e.g. because ds shrank to ds_min before the corrector could
    # converge the final tiny gap).
    # ------------------------------------------------------------------
    if verbose:
        r_pre = float(np.linalg.norm(_F_lam(sol, lam_end, dt_target)))
        print(f"\n  Final Newton polish at lam = {lam_end}  (||F|| = {r_pre:.3e}) ...")

    for _ in range(max_iter):
        f_vec = np.array(_F_lam(sol, lam_end, dt_target)).flatten()
        res_f = float(np.linalg.norm(f_vec))
        if verbose:
            print(f"    polish: ||F|| = {res_f:.3e}")
        if res_f <= tol:
            break
        J_x = np.array(_JF_lam(sol, lam_end, dt_target))
        try:
            dx = linalg.solve(J_x, -f_vec)
        except linalg.LinAlgError:
            break
        for k in range(1, n - 1):
            sol.solution[sol.timestep, k] += dx[k]
        sol.applyRB()

    lam = lam_end
    return lam
