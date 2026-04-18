"""
nonlinear_solver_prototypes
===========================
A collection of nonlinear solver prototypes for an implicit heat-equation
finite-difference problem.

Quick start
-----------
    from nonlinear_solver_prototypes.solution import Solution
    from nonlinear_solver_prototypes.fd_problem import K, F, JF
    from nonlinear_solver_prototypes.newton import run_newton, SOLVER_DIRECT, LS_ARMIJO

Public modules
--------------
solution               : Solution class
fd_problem             : K, F, JF  (heat conductivity, residual, Jacobian)
helpers                : _phi, _dphi_directional, _apply_step
linear_solvers         : solve_direct, solve_cg
linesearch_armijo      : linesearch_armijo
linesearch_cubic       : linesearch_cubic
linesearch_wolfe       : linesearch_wolfe
barzilai_borwein       : run_barzilai_borwein
trust_region_dogleg    : run_trust_region_dogleg
nonlinear_cg           : run_nonlinear_cg
newton                 : run_newton, SOLVER_*, LS_*
arclength_continuation : run_arclength_continuation, _F_lam, _JF_lam, _dF_dlam
continuation_newton    : run_cn
"""

from .arclength_continuation import (
    _dF_dlam,
    _F_lam,
    _JF_lam,
    run_arclength_continuation,
)
from .barzilai_borwein import run_barzilai_borwein
from .continuation_newton import run_cn
from .fd_problem import JF, F, K
from .helpers import _apply_step, _dphi_directional, _phi
from .linear_solvers import solve_cg, solve_direct
from .linesearch_armijo import linesearch_armijo
from .linesearch_cubic import linesearch_cubic
from .linesearch_wolfe import linesearch_wolfe
from .newton import (
    LS_ARMIJO,
    LS_CUBIC,
    LS_NONE,
    LS_WOLFE,
    SOLVER_ARC,
    SOLVER_BB,
    SOLVER_CG,
    SOLVER_CN,
    SOLVER_DIRECT,
    SOLVER_NCG,
    SOLVER_TR,
    run_newton,
)
from .nonlinear_cg import run_nonlinear_cg
from .solution import Solution
from .trust_region_dogleg import run_trust_region_dogleg

__all__ = [
    # core
    "Solution",
    "K",
    "F",
    "JF",
    # helpers
    "_phi",
    "_dphi_directional",
    "_apply_step",
    # linear solvers
    "solve_direct",
    "solve_cg",
    # linesearches
    "linesearch_armijo",
    "linesearch_cubic",
    "linesearch_wolfe",
    # iterative solvers
    "run_barzilai_borwein",
    "run_trust_region_dogleg",
    "run_nonlinear_cg",
    # Newton
    "run_newton",
    "SOLVER_DIRECT",
    "SOLVER_CG",
    "SOLVER_NCG",
    "SOLVER_BB",
    "SOLVER_TR",
    "SOLVER_ARC",
    "SOLVER_CN",
    "LS_ARMIJO",
    "LS_CUBIC",
    "LS_WOLFE",
    "LS_NONE",
    # continuation
    "run_arclength_continuation",
    "_F_lam",
    "_JF_lam",
    "_dF_dlam",
    "run_cn",
]
