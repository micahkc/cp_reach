"""Flowpipe and reachability utilities."""

from .lmi import solve_disturbance_LMI, solve_bounded_disturbance_output_LMI
from .workflows import (
    plot_grouped,
    compute_reachable_set,
    plot_flowpipe,
    analyze,
)
from cp_reach.ir.rumoca import (
    RumocaSymbolicModel,
    modelica_load,
    modelica_loads,
    rumoca_model_to_symbolic,
)
from .polytopic import (
    polytopic_jacobians,
    solve_time_varying_polytopic_lmi,
    eval_polynomial_metric,
    project_metric_2d,
    compute_state_bounds,
)
from .certification import certify_lipschitz_grid
from .cascaded import (
    load_lie_spec,
    partition_states,
    classify_remaining_states,
    identify_linear_coupling,
    se23_adjoint,
    se23_coupling_matrix,
    se23_drift_matrix,
    solve_cascaded_lmi,
    se23_expmap,
    se23_expmap_single,
    sample_ellipsoid_boundary,
    quadrotor_flatness,
    compute_omega_from_R,
    euler_equation_jacobians,
    se23_kinematic_matrices,
    se23_controller_matrices,
)

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
    "plot_grouped",
    "compute_reachable_set",
    "plot_flowpipe",
    "analyze",
    "RumocaSymbolicModel",
    "modelica_load",
    "modelica_loads",
    "rumoca_model_to_symbolic",
    # Cascaded / Lie group
    "load_lie_spec",
    "partition_states",
    "classify_remaining_states",
    "identify_linear_coupling",
    "se23_adjoint",
    "se23_coupling_matrix",
    "se23_drift_matrix",
    "solve_cascaded_lmi",
    "se23_expmap",
    "se23_expmap_single",
    "sample_ellipsoid_boundary",
    "quadrotor_flatness",
    "compute_omega_from_R",
    "euler_equation_jacobians",
    "se23_kinematic_matrices",
    "se23_controller_matrices",
]
