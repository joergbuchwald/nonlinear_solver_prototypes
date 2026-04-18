# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from arclength_continuation import run_arclength_continuation
from barzilai_borwein import run_barzilai_borwein
from continuation_newton import run_cn
from newton import (
    LS_ARMIJO,
    LS_CUBIC,
    LS_NONE,
    LS_WOLFE,
    SOLVER_CG,
    SOLVER_DIRECT,
    run_newton,
)
from nonlinear_cg import run_nonlinear_cg
from solution import Solution
from trust_region_dogleg import run_trust_region_dogleg

# %% numerical definitions
Ntime = 5
Nspace = 10
dx = 0.1
dt = 0.2

# %% physical parametrization
T1 = 50  # left boundary temperature  [K]
T2 = 350  # right boundary temperature [K]

# Flat initial condition at T1 — gives non-trivial dynamics.
# A linear ramp would have zero second spatial derivative and
# therefore a zero PDE residual from the start (already at steady state).
T0 = np.full(Nspace, float(T1))

# %% solver settings
max_iter = 400
tol = 1e-10
timings = {}


def make_sol():
    """Return a fresh Solution initialised to uniform T1."""
    return Solution(T0, T1, T2, Ntime, Nspace, dx)


# %%
# ============================================================
# 1. Newton  —  direct linear solver  +  Armijo linesearch
# ============================================================
print("=" * 60)
print("Newton  (direct + Armijo)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_newton(
        sol,
        dt,
        linear_solver=SOLVER_DIRECT,
        linesearch=LS_ARMIJO,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Newton (direct + Armijo)"] = time.perf_counter() - t0
sol_newton_armijo = sol.solution.copy()
print(f"  --> {timings['Newton (direct + Armijo)']:.3f} s\n")

# %%
# ============================================================
# 2. Newton  —  direct linear solver  +  cubic linesearch
# ============================================================
print("=" * 60)
print("Newton  (direct + cubic)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_newton(
        sol,
        dt,
        linear_solver=SOLVER_DIRECT,
        linesearch=LS_CUBIC,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Newton (direct + cubic)"] = time.perf_counter() - t0
sol_newton_cubic = sol.solution.copy()
print(f"  --> {timings['Newton (direct + cubic)']:.3f} s\n")

# %%
# ============================================================
# 3. Newton  —  direct linear solver  +  strong Wolfe linesearch
# ============================================================
print("=" * 60)
print("Newton  (direct + Wolfe)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_newton(
        sol,
        dt,
        linear_solver=SOLVER_DIRECT,
        linesearch=LS_WOLFE,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Newton (direct + Wolfe)"] = time.perf_counter() - t0
sol_newton_wolfe = sol.solution.copy()
print(f"  --> {timings['Newton (direct + Wolfe)']:.3f} s\n")

# %%
# ============================================================
# 4. Newton  —  CG linear solver  +  Armijo linesearch
# ============================================================
print("=" * 60)
print("Newton  (CG + Armijo)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_newton(
        sol,
        dt,
        linear_solver=SOLVER_CG,
        linesearch=LS_ARMIJO,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Newton (CG + Armijo)"] = time.perf_counter() - t0
sol_newton_cg = sol.solution.copy()
print(f"  --> {timings['Newton (CG + Armijo)']:.3f} s\n")

# %%
# ============================================================
# 5. Newton  —  no linesearch  (fixed damping = 1)
# ============================================================
print("=" * 60)
print("Newton  (direct + no linesearch, damping = 1)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_newton(
        sol,
        dt,
        linear_solver=SOLVER_DIRECT,
        linesearch=LS_NONE,
        damping=1.0,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Newton (no LS)"] = time.perf_counter() - t0
sol_newton_none = sol.solution.copy()
print(f"  --> {timings['Newton (no LS)']:.3f} s\n")

# %%
# ============================================================
# 6. Barzilai-Borwein  (BB1 variant)
# ============================================================
print("=" * 60)
print("Barzilai-Borwein  (BB1)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_barzilai_borwein(
        sol, dt, max_iter=500, tol=tol, bb_variant="BB1", verbose=False
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Barzilai-Borwein BB1"] = time.perf_counter() - t0
sol_bb1 = sol.solution.copy()
print(f"  --> {timings['Barzilai-Borwein BB1']:.3f} s\n")

# %%
# ============================================================
# 7. Barzilai-Borwein  (BB2 variant)
# ============================================================
print("=" * 60)
print("Barzilai-Borwein  (BB2)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_barzilai_borwein(
        sol, dt, max_iter=500, tol=tol, bb_variant="BB2", verbose=False
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Barzilai-Borwein BB2"] = time.perf_counter() - t0
sol_bb2 = sol.solution.copy()
print(f"  --> {timings['Barzilai-Borwein BB2']:.3f} s\n")

# %%
# ============================================================
# 8. Trust-region dogleg
# ============================================================
print("=" * 60)
print("Trust-region dogleg")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_trust_region_dogleg(
        sol,
        dt,
        max_iter=max_iter * 6,
        tol=tol,
        Delta0=1.0,
        Delta_max=100.0,
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Trust-region dogleg"] = time.perf_counter() - t0
sol_tr = sol.solution.copy()
print(f"  --> {timings['Trust-region dogleg']:.3f} s\n")

# %%
# ============================================================
# 9. Nonlinear CG  —  Fletcher-Reeves
# ============================================================
print("=" * 60)
print("Nonlinear CG  (FR, Wolfe linesearch)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_nonlinear_cg(
        sol,
        dt,
        max_iter=500,
        tol=tol,
        beta_formula="FR",
        linesearch="wolfe",
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Nonlinear CG (FR)"] = time.perf_counter() - t0
sol_ncg_fr = sol.solution.copy()
print(f"  --> {timings['Nonlinear CG (FR)']:.3f} s\n")

# %%
# ============================================================
# 10. Nonlinear CG  —  Polak-Ribière+
# ============================================================
print("=" * 60)
print("Nonlinear CG  (PR+, Wolfe linesearch)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    niter = run_nonlinear_cg(
        sol,
        dt,
        max_iter=500,
        tol=tol,
        beta_formula="PR+",
        linesearch="wolfe",
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  converged in {niter} iters")
timings["Nonlinear CG (PR+)"] = time.perf_counter() - t0
sol_ncg_prp = sol.solution.copy()
print(f"  --> {timings['Nonlinear CG (PR+)']:.3f} s\n")

# %%
# ============================================================
# 11. Pseudo-arclength continuation  (Keller 1977)
# ============================================================
print("=" * 60)
print("Pseudo-arclength continuation")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    lam_final = run_arclength_continuation(
        sol,
        dt_target=dt,
        lam_start=0.0,
        lam_end=1.0,
        ds=0.2,
        ds_min=1e-4,
        ds_max=0.5,
        max_steps=200,
        max_iter=20,
        tol=tol,
        adapt_ds=True,
        iter_target=12,
        scale_x=float(T2 - T1),
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  lam reached = {lam_final:.6f}")
timings["Arclength continuation"] = time.perf_counter() - t0
sol_arc = sol.solution.copy()
print(f"  --> {timings['Arclength continuation']:.3f} s\n")

# %%
# ============================================================
# 12. Continuation-Newton  (Younis, Tchelepi & Aziz 2010)
# ============================================================
print("=" * 60)
print("Continuation-Newton  (CN)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    lam_final = run_cn(
        sol,
        dt_target=dt,
        lam_start=0.0,
        lam_end=1.0,
        neighborhood_tol=5.0,
        final_tol=tol,
        alpha_min=1e-4,
        alpha_max=0.5,
        max_iter=200,
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  lam reached = {lam_final:.6f}")
timings["CN"] = time.perf_counter() - t0
sol_cn = sol.solution.copy()
print(f"  --> {timings['CN']:.3f} s\n")

# %%
# ============================================================
# 13. Continuation-Newton  with localisation
# ============================================================
print("=" * 60)
print("Continuation-Newton  (CN, localised)")
print("=" * 60)

sol = make_sol()
t0 = time.perf_counter()
for step in range(Ntime - 1):
    sol.newTime()
    lam_final = run_cn(
        sol,
        dt_target=dt,
        lam_start=0.0,
        lam_end=1.0,
        neighborhood_tol=5.0,
        final_tol=tol,
        alpha_min=1e-4,
        alpha_max=0.5,
        max_iter=200,
        localize=True,
        verbose=False,
    )
    print(f"  t-step {step + 1}/{Ntime - 1}:  lam reached = {lam_final:.6f}")
timings["CN (localised)"] = time.perf_counter() - t0
sol_cn_loc = sol.solution.copy()
print(f"  --> {timings['CN (localised)']:.3f} s\n")

# %%
# ============================================================
# 14. Time-evolution plot  —  reference solver (Newton direct + Armijo)
# ============================================================
x = np.linspace(0, (Nspace - 1) * dx, Nspace)
t_vals = [i * dt for i in range(Ntime)]
colors = plt.colormaps["viridis"](np.linspace(0, 1, Ntime))

fig, ax = plt.subplots(figsize=(9, 5))
for i in range(Ntime):
    ax.plot(
        x,
        sol_newton_armijo[i, :],
        color=colors[i],
        label=f"t = {t_vals[i]:.2f} s  (step {i})",
    )
ax.set_xlabel("x  [m]")
ax.set_ylabel("T  [K]")
ax.set_title(
    f"Time evolution  —  Newton (direct + Armijo)  |  dt = {dt}, Nspace = {Nspace}"
)
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 15. Comparison plot  —  final time step, all solvers
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    x, sol_newton_armijo[-1, :], "b-", label="Newton (direct + Armijo)", linewidth=2
)
ax.plot(x, sol_newton_cubic[-1, :], "b--", label="Newton (direct + cubic)")
ax.plot(x, sol_newton_wolfe[-1, :], "b:", label="Newton (direct + Wolfe)")
ax.plot(x, sol_newton_cg[-1, :], "c-", label="Newton (CG + Armijo)")
ax.plot(x, sol_newton_none[-1, :], "c--", label="Newton (no LS)")
ax.plot(x, sol_bb1[-1, :], "g-", label="Barzilai-Borwein BB1", alpha=0.7)
ax.plot(x, sol_bb2[-1, :], "g--", label="Barzilai-Borwein BB2", alpha=0.7)
ax.plot(x, sol_tr[-1, :], "r-", label="Trust-region dogleg")
ax.plot(x, sol_ncg_fr[-1, :], "m-", label="Nonlinear CG (FR)")
ax.plot(x, sol_ncg_prp[-1, :], "m--", label="Nonlinear CG (PR+)")
ax.plot(x, sol_arc[-1, :], color="orange", label="Arclength continuation")
ax.plot(x, sol_cn[-1, :], color="brown", label="CN")
ax.plot(x, sol_cn_loc[-1, :], color="brown", label="CN (localised)", linestyle="--")

ax.set_xlabel("x  [m]")
ax.set_ylabel("T  [K]")
ax.set_title(
    f"Temperature at t = {(Ntime - 1) * dt:.2f} s  (after {Ntime - 1} time steps)"
    f"  |  dt = {dt}, Nspace = {Nspace}"
)
ax.legend(loc="best", fontsize=7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 16. Summary table  —  timing + max deviation from reference
# ============================================================
all_solutions = {
    "Newton (direct + Armijo)": sol_newton_armijo,
    "Newton (direct + cubic)": sol_newton_cubic,
    "Newton (direct + Wolfe)": sol_newton_wolfe,
    "Newton (CG + Armijo)": sol_newton_cg,
    "Newton (no LS)": sol_newton_none,
    "Barzilai-Borwein BB1": sol_bb1,
    "Barzilai-Borwein BB2": sol_bb2,
    "Trust-region dogleg": sol_tr,
    "Nonlinear CG (FR)": sol_ncg_fr,
    "Nonlinear CG (PR+)": sol_ncg_prp,
    "Arclength continuation": sol_arc,
    "CN": sol_cn,
    "CN (localised)": sol_cn_loc,
}

ref = sol_newton_armijo
print(f"{'Solver':<35}  {'time (s)':>8}  {'max|T - ref|':>14}")
print("-" * 62)
for name, S in all_solutions.items():
    err = np.max(np.abs(S - ref))
    t = timings.get(name, float("nan"))
    print(f"{name:<35}  {t:8.3f}  {err:14.3e}")
