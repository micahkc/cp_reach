"""Flowpipe and reachability utilities."""

from .lmi import solve_disturbance_LMI, solve_bounded_disturbance_output_LMI
from .workflows import (
    sympy_load,
    casadi_load,
    simulate_dist,
    plot_grouped,
    compute_reachable_set,
    plot_flowpipe,
    ir_load,
    analyze,
    ModelicaIRModel,
)
from .polytopic import (
    polytopic_jacobians,
    solve_time_varying_polytopic_lmi,
    eval_polynomial_metric,
    project_metric_2d,
    compute_state_bounds,
)
from .certification import certify_lipschitz_grid

__all__ = [
    # LMI solvers
    "solve_disturbance_LMI",
    "solve_bounded_disturbance_output_LMI",
    # Polytopic / time-varying
    "polytopic_jacobians",
    "solve_time_varying_polytopic_lmi",
    "eval_polynomial_metric",
    "project_metric_2d",
    "compute_state_bounds",
    # Certification
    "certify_lipschitz_grid",
    # Workflows
    "sympy_load",
    "casadi_load",
    "simulate_dist",
    "plot_grouped",
    "compute_reachable_set",
    "plot_flowpipe",
    "ir_load",
    "analyze",
    "ModelicaIRModel",
]
