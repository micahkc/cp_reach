"""
Cascaded reachability analysis for systems with Lie group structure.

Implements the two-layer architecture from the paper:
  Layer 1: Rotational dynamics (polytopic LMI, 2^3 = 8 vertices)
  Layer 2: SE_2(3) kinematics (exact log-linear, 1 vertex per time point)

The key insight is that quadrotor kinematics are group-affine on SE_2(3),
so the log-linear error dynamics xi_dot = (-ad_nbar + A_C) xi + J_l^{-1} ntilde
are exact (no polytopic approximation needed for the kinematic layer).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lie group spec loader
# ---------------------------------------------------------------------------

def load_lie_spec(path: str) -> dict:
    """Load a Lie group specification JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SE_2(3) adjoint and coupling matrices (pure numpy, no cyecca dependency)
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def se23_adjoint(v: np.ndarray, a: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute the 9x9 adjoint matrix ad_n for n = (v, a, omega) in se_2(3).

    The adjoint representation of the Lie algebra element n = [v, a, omega]
    acts on xi = [xi_p, xi_v, xi_R] as:

        ad_n(xi) = [omega x xi_p + v x xi_R,
                     omega x xi_v + a x xi_R,
                     omega x xi_R]

    In matrix form (9x9), with block ordering [p, v, R]:

        ad_n = [[omega_x]  0        [v_x]  ]
               [ 0        [omega_x] [a_x]  ]
               [ 0         0        [omega_x]]

    where [.]_x denotes the skew-symmetric matrix.
    """
    Omega = _skew(omega)
    V = _skew(v)
    A = _skew(a)
    Z = np.zeros((3, 3))

    return np.block([
        [Omega, Z, V],
        [Z, Omega, A],
        [Z, Z, Omega],
    ])


def se23_coupling_matrix() -> np.ndarray:
    """
    Coupling matrix A_C encoding xi_p_dot = xi_v in the log-linear dynamics.

    A_C has I_3 in the (p, v) block:
        A_C = [[0  I  0],
               [0  0  0],
               [0  0  0]]
    """
    A_C = np.zeros((9, 9))
    A_C[0:3, 3:6] = np.eye(3)
    return A_C


def se23_drift_matrix(v_ref: np.ndarray, a_ref: np.ndarray,
                      omega_ref: np.ndarray) -> np.ndarray:
    """
    Compute A(t) = -ad_{n_bar} + A_C for the log-linear kinematic error dynamics.

    This is the exact linearization of the SE_2(3) error dynamics (no approximation).

    Parameters
    ----------
    v_ref : (3,) velocity reference (world frame) — only used if body-frame coupling matters
    a_ref : (3,) body-frame acceleration reference
    omega_ref : (3,) body-frame angular velocity reference

    Returns
    -------
    A : (9, 9) drift matrix
    """
    return -se23_adjoint(v_ref, a_ref, omega_ref) + se23_coupling_matrix()


# ---------------------------------------------------------------------------
# Automatic state partition and classification
# ---------------------------------------------------------------------------

def partition_states(state_names: List[str], lie_spec: dict) -> Dict[str, List[str]]:
    """
    Partition model states into Lie group states and remaining states
    using the Lie group specification.

    Parameters
    ----------
    state_names : list of str
        All state variable names from the Modelica model
    lie_spec : dict
        Lie group specification (from JSON file)

    Returns
    -------
    dict with keys:
        'lie_group': state names belonging to the Lie group (R, v, p)
        'remaining': all other state names
    """
    group = lie_spec['lie_groups'][0]
    lie_set = set(
        group['rotation_states']
        + group['velocity_states']
        + group['position_states']
    )

    return {
        'lie_group': [s for s in state_names if s in lie_set],
        'remaining': [s for s in state_names if s not in lie_set],
    }


def classify_remaining_states(
    ss, partition: Dict[str, List[str]], params: Dict[str, float],
) -> Dict[str, List[str]]:
    """
    Classify remaining states as linear or nonlinear by checking whether the
    Jacobian of their dynamics w.r.t. all state variables is constant.

    A state has linear error dynamics if every entry of its Jacobian row is
    free of state symbols (i.e., at most affine in x).

    Parameters
    ----------
    ss : SymbolicStateSpace
    partition : dict from partition_states()
    params : dict of parameter values

    Returns
    -------
    dict with keys 'linear' and 'nonlinear', each a list of state names
    """
    import sympy as sp

    remaining = partition['remaining']
    state_names = ss.get_state_names()
    state_syms = list(ss.state_symbols)
    state_sym_set = set(state_syms)

    param_subs = {sp.Symbol(k): v for k, v in params.items()}
    f = ss.f.subs(param_subs)

    linear, nonlinear = [], []

    for s_name in remaining:
        s_idx = state_names.index(s_name)
        f_i = f[s_idx]

        is_linear = True
        for sym in state_syms:
            deriv = sp.diff(f_i, sym)
            if deriv.free_symbols & state_sym_set:
                is_linear = False
                break

        (linear if is_linear else nonlinear).append(s_name)

    return {'linear': linear, 'nonlinear': nonlinear}


def identify_linear_coupling(
    ss, lie_spec: dict, linear_states: List[str], params: Dict[str, float],
) -> np.ndarray:
    """
    Build the coupling matrix C_int (n_lin x 9) from linear remaining states
    to the Lie algebra by inspecting the symbolic Jacobian.

    C_int encodes how Lie algebra error states drive the linear state errors:
        linear_state_dot = C_int @ [xi_p, xi_v, xi_R]

    For example, a velocity-error integrator z with z_dot = v - v_ref gives
    C_int row = [0, 0, 0, 1, 0, 0, 0, 0, 0] (couples to xi_v).

    Parameters
    ----------
    ss : SymbolicStateSpace
    lie_spec : dict
    linear_states : list of str
    params : dict of parameter values

    Returns
    -------
    C_int : (n_lin, 9) numpy array
    """
    import sympy as sp

    group = lie_spec['lie_groups'][0]
    state_names = ss.get_state_names()
    state_syms = list(ss.state_symbols)

    pos_syms = [state_syms[state_names.index(s)] for s in group['position_states']]
    vel_syms = [state_syms[state_names.index(s)] for s in group['velocity_states']]
    # Rotation: 9 matrix elements -> 3 Lie algebra dims.
    # Only check if coupling is zero; nonzero rotation coupling is unsupported.
    rot_syms = [state_syms[state_names.index(s)] for s in group['rotation_states']]

    param_subs = {sp.Symbol(k): v for k, v in params.items()}
    f = ss.f.subs(param_subs)

    n_lin = len(linear_states)
    C_int = np.zeros((n_lin, 9))

    for i, s_name in enumerate(linear_states):
        s_idx = state_names.index(s_name)
        f_i = f[s_idx]

        # xi_p block (columns 0:3)
        for j, sym in enumerate(pos_syms):
            C_int[i, j] = float(sp.diff(f_i, sym))

        # xi_v block (columns 3:6)
        for j, sym in enumerate(vel_syms):
            C_int[i, 3 + j] = float(sp.diff(f_i, sym))

        # xi_R block (columns 6:9): check that rotation coupling is zero
        rot_coeffs = [sp.diff(f_i, sym) for sym in rot_syms]
        if any(c != 0 for c in rot_coeffs):
            raise NotImplementedError(
                f"Linear state '{s_name}' couples to rotation states. "
                f"Automatic Lie algebra mapping for rotation is not supported."
            )

    return C_int


# ---------------------------------------------------------------------------
# Layer 1: Rotational dynamics (Euler's equations)
# ---------------------------------------------------------------------------

def euler_equation_jacobians(
    omega_refs: np.ndarray,
    omega_bound: float,
    Ixx: float, Iyy: float, Izz: float,
    controller_gains: Dict[str, float],
) -> List[List[np.ndarray]]:
    """
    Compute polytopic Jacobian vertices for the angular velocity error dynamics.

    The angular velocity error dynamics from Euler's equations + rate PD controller:
        omega_dot = f(omega, omega_ref) + controller + disturbance

    The Jacobian depends on omega_ref (gyroscopic coupling is bilinear in omega),
    so we create polytopic bounds at each reference point.

    Parameters
    ----------
    omega_refs : (N, 3) reference angular velocity trajectory
    omega_bound : float, bound on angular velocity error for polytopic approximation
    Ixx, Iyy, Izz : moments of inertia
    controller_gains : dict with keys 'Kw_x', 'Kw_y', 'Kw_z', 'Kr_x', 'Kr_y', 'Kr_z'

    Returns
    -------
    J_vertices_by_time : list of lists of (3, 3) arrays
    """
    N = omega_refs.shape[0]
    Kw_x = controller_gains.get('Kw_x', 1.6)
    Kw_y = controller_gains.get('Kw_y', 1.6)
    Kw_z = controller_gains.get('Kw_z', 2.0)

    J_vertices_by_time = []

    for k in range(N):
        ox_ref, oy_ref, oz_ref = omega_refs[k]

        # The gyroscopic Jacobian (from Euler's equations) at a given omega:
        #   d(omega_dot)/d(omega) has cross-coupling terms from (Iyy-Izz)*oy*oz / Ixx etc.
        # Plus the controller contributes -Kw on the diagonal.
        #
        # Euler's eqs linearized: Jacobian w.r.t. omega error:
        #   J_11 = -Kw_x
        #   J_12 = (Iyy - Izz) * oz / Ixx
        #   J_13 = (Iyy - Izz) * oy / Ixx
        #   etc.
        #
        # The nonlinearity is that oz, oy appear. We bound them at ref ± bound.

        corners = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    ox = ox_ref + s1 * omega_bound
                    oy = oy_ref + s2 * omega_bound
                    oz = oz_ref + s3 * omega_bound

                    J = np.array([
                        [-Kw_x, (Iyy - Izz) * oz / Ixx, (Iyy - Izz) * oy / Ixx],
                        [(Izz - Ixx) * oz / Iyy, -Kw_y, (Izz - Ixx) * ox / Iyy],
                        [(Ixx - Iyy) * oy / Izz, (Ixx - Iyy) * ox / Izz, -Kw_z],
                    ])
                    corners.append(J)

        J_vertices_by_time.append(corners)

    return J_vertices_by_time


# ---------------------------------------------------------------------------
# Layer 2: SE_2(3) kinematic error dynamics (exact log-linear)
# ---------------------------------------------------------------------------

def se23_controller_matrices(
    Kp: float, Kd: float, Kpq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the control input matrix B0 and gain matrix K for the SE_2(3) kinematic layer.

    The controller acts in the Lie algebra:
      u_a = -Kp * xi_p - Kd * xi_v     (position PD -> acceleration)
      u_omega = -Kpq * xi_R            (attitude P -> angular rate)

    So BK = B0 @ K maps Lie algebra error to closed-loop feedback.

    Parameters
    ----------
    Kp : float, position proportional gain
    Kd : float, velocity (derivative) gain
    Kpq : float, attitude proportional gain

    Returns
    -------
    B0 : (9, 6) control input matrix
    K : (6, 9) gain matrix
    """
    # B0: how [a(3), omega(3)] enter the 9D state [p, v, R]
    B0 = np.zeros((9, 6))
    B0[3:6, 0:3] = np.eye(3)  # acceleration -> velocity block
    B0[6:9, 3:6] = np.eye(3)  # angular rate -> rotation block

    # K: [Kp I, Kd I, 0; 0, 0, Kpq I]
    K = np.zeros((6, 9))
    K[0:3, 0:3] = Kp * np.eye(3)  # position error -> accel command
    K[0:3, 3:6] = Kd * np.eye(3)  # velocity error -> accel command
    K[3:6, 6:9] = Kpq * np.eye(3) # attitude error -> rate command

    return B0, K


def se23_kinematic_matrices(
    v_refs: np.ndarray,
    a_refs: np.ndarray,
    omega_refs: np.ndarray,
    Kp: float = 0.0,
    Kd: float = 0.0,
    Kpq: float = 0.0,
) -> List[List[np.ndarray]]:
    """
    Compute the closed-loop A(t) matrices for the kinematic layer.

    A_cl(t) = -ad_{n_bar(t)} + A_C - B0 @ K

    Since the log-linear error dynamics are exact, each time point has exactly
    ONE matrix (no polytopic vertices needed).

    Parameters
    ----------
    v_refs : (N, 3) reference velocity in world frame
    a_refs : (N, 3) reference body-frame acceleration
    omega_refs : (N, 3) reference body-frame angular velocity
    Kp : float, position proportional gain (for closed-loop)
    Kd : float, velocity derivative gain
    Kpq : float, attitude proportional gain

    Returns
    -------
    A_by_time : list of single-element lists of (9, 9) arrays
    """
    N = v_refs.shape[0]

    # Controller feedback (constant)
    if Kp > 0 or Kd > 0 or Kpq > 0:
        B0, K = se23_controller_matrices(Kp, Kd, Kpq)
        BK = B0 @ K
    else:
        BK = np.zeros((9, 9))

    A_by_time = []
    for k in range(N):
        A_k = se23_drift_matrix(v_refs[k], a_refs[k], omega_refs[k]) - BK
        A_by_time.append([A_k])  # Single vertex — exact!

    return A_by_time


def _to_diag3(gain) -> np.ndarray:
    """Convert a scalar or length-3 array-like to a 3x3 diagonal matrix."""
    g = np.asarray(gain, dtype=float).ravel()
    if g.size == 1:
        return g[0] * np.eye(3)
    if g.size == 3:
        return np.diag(g)
    raise ValueError(f"Gain must be a scalar or length-3, got shape {g.shape}")


def se23_kinematic_matrices_augmented(
    v_refs: np.ndarray,
    a_refs: np.ndarray,
    omega_refs: np.ndarray,
    C_int: np.ndarray,
    Kp=0.0,
    Kd=0.0,
    Kpq=0.0,
    Ki=0.0,
) -> List[List[np.ndarray]]:
    """
    Compute augmented closed-loop A(t) matrices: SE_2(3) + linear states.

    State vector: [xi_p(3), xi_v(3), xi_R(3), z_tilde(n_lin)]

    Parameters
    ----------
    v_refs : (N, 3) reference velocity in world frame
    a_refs : (N, 3) reference body-frame acceleration
    omega_refs : (N, 3) reference body-frame angular velocity
    C_int : (n_lin, 9) coupling matrix from identify_linear_coupling().
        Encodes how Lie algebra states drive linear state errors.
    Kp : float or (3,) array_like — position gain (per-axis or isotropic)
    Kd : float or (3,) array_like — velocity/damping gain
    Kpq : float or (3,) array_like — attitude gain
    Ki : float or (3,) array_like — integral gain (couples z_tilde -> xi_v)

    Returns
    -------
    A_by_time : list of single-element lists of (9+n_lin, 9+n_lin) arrays
    """
    N = v_refs.shape[0]
    n_lin = C_int.shape[0]
    n_aug = 9 + n_lin

    Kp_mat = _to_diag3(Kp)
    Kd_mat = _to_diag3(Kd)
    Kpq_mat = _to_diag3(Kpq)
    Ki_mat = _to_diag3(Ki)

    # Augmented controller: B0 @ K
    B0 = np.zeros((n_aug, 6))
    B0[3:6, 0:3] = np.eye(3)  # acceleration -> xi_v
    B0[6:9, 3:6] = np.eye(3)  # angular rate -> xi_R

    K = np.zeros((6, n_aug))
    K[0:3, 0:3] = Kp_mat                                            # xi_p -> accel
    K[0:3, 3:6] = Kd_mat                                            # xi_v -> accel
    K[0:3, 9:9 + min(3, n_lin)] = Ki_mat[:min(3, n_lin), :min(3, n_lin)]  # z_tilde -> accel
    K[3:6, 6:9] = Kpq_mat                                           # xi_R -> rate

    BK = B0 @ K

    A_by_time = []
    for k in range(N):
        # 9x9 exact log-linear drift from group structure
        A_lie = se23_drift_matrix(v_refs[k], a_refs[k], omega_refs[k])

        # Assemble augmented drift
        A_aug = np.zeros((n_aug, n_aug))
        A_aug[0:9, 0:9] = A_lie
        A_aug[9:9 + n_lin, 0:9] = C_int

        # Closed-loop
        A_cl = A_aug - BK
        A_by_time.append([A_cl])

    return A_by_time


# ---------------------------------------------------------------------------
# Cascaded solver
# ---------------------------------------------------------------------------

def solve_cascaded_lmi(
    times: np.ndarray,
    omega_refs: np.ndarray,
    v_refs: np.ndarray,
    a_refs: np.ndarray,
    disturbance_bounds: Dict[str, float],
    inertia: Dict[str, float],
    controller_gains: Dict[str, float],
    omega_error_bound: float = 0.5,
    polynomial_degree: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve the two-layer cascaded LMI for a quadrotor on SE_2(3).

    Layer 1: Rotational dynamics — polytopic LMI with 2^3 = 8 vertices
             Produces ellipsoidal bound on omega error.

    Layer 2: SE_2(3) kinematics — exact log-linear LMI with 1 vertex per time
             Uses omega error bound from Layer 1 as disturbance input.

    Parameters
    ----------
    times : (N,) array
        Sample times along the reference trajectory
    omega_refs : (N, 3) array
        Reference angular velocity in body frame
    v_refs : (N, 3) array
        Reference velocity in world frame (not used in adjoint for NED quadrotor,
        but included for generality — can be zeros for hover-relative)
    a_refs : (N, 3) array
        Reference body-frame acceleration (typically [0, 0, -T/m])
    disturbance_bounds : dict
        Keys: 'd_gyro' (gyro bias bound), 'd_accel' (acceleration disturbance bound)
    inertia : dict
        Keys: 'Ixx', 'Iyy', 'Izz', 'mass'
    controller_gains : dict
        Keys: 'Kw_x', 'Kw_y', 'Kw_z', 'Kr_x', 'Kr_y', 'Kr_z'
    omega_error_bound : float
        Initial guess for omega error bound (for polytopic vertices)
    polynomial_degree : int, optional
        Degree of polynomial Lyapunov function (default: len(times))
    verbose : bool

    Returns
    -------
    dict with:
        'layer1': Layer 1 results (rotational dynamics)
        'layer2': Layer 2 results (SE_2(3) kinematics)
        'omega_bounds': (3,) per-axis omega error bounds
        'xi_bounds': (9,) per-axis Lie algebra error bounds
        'xi_bounds_names': list of names for the 9 xi components
    """
    from .polytopic import (
        solve_time_varying_polytopic_lmi,
        compute_state_bounds,
    )

    N = len(times)
    if polynomial_degree is None:
        polynomial_degree = N

    Ixx = inertia['Ixx']
    Iyy = inertia['Iyy']
    Izz = inertia['Izz']

    d_gyro = disturbance_bounds.get('d_gyro', 0.0)
    d_accel = disturbance_bounds.get('d_accel', 0.0)

    # ===================================================================
    # Layer 1: Rotational dynamics
    # ===================================================================
    if verbose:
        print("=" * 60)
        print("Layer 1: Rotational dynamics (polytopic LMI, 8 vertices)")
        print("=" * 60)

    # Disturbance input matrix for angular velocity dynamics
    # Gyro bias enters through the controller: the rate PD sees (omega + d)
    # So the disturbance matrix is the identity (bias directly on omega measurement)
    B_omega = np.eye(3)

    # Compute polytopic Jacobians for Euler's equations
    J_omega_vertices = euler_equation_jacobians(
        omega_refs, omega_error_bound,
        Ixx, Iyy, Izz, controller_gains,
    )

    # Solve Layer 1 LMI
    layer1_result = solve_time_varying_polytopic_lmi(
        J_omega_vertices, B_omega, times.tolist(),
        polynomial_degree=polynomial_degree,
        verbose=verbose,
    )

    if layer1_result['status'] in ['infeasible', None]:
        raise RuntimeError(f"Layer 1 LMI failed: {layer1_result['status']}")

    # Extract omega error bounds
    omega_bounds_result = compute_state_bounds(
        layer1_result['M_coefs'],
        layer1_result['mu_physical'],
        d_gyro,
        times.tolist(),
    )
    omega_bounds = omega_bounds_result['bounds_max']

    if verbose:
        print(f"\nLayer 1 results:")
        print(f"  alpha = {layer1_result['alpha']:.4f}")
        print(f"  mu_physical = {layer1_result['mu_physical']:.4e}")
        print(f"  omega error bounds: {omega_bounds}")

    # Validate polytopic assumption
    max_omega_bound = np.max(omega_bounds)
    if max_omega_bound > omega_error_bound:
        logger.warning(
            f"Computed omega bound ({max_omega_bound:.4f}) > "
            f"polytopic assumption ({omega_error_bound}). "
            f"Consider increasing omega_error_bound."
        )

    # ===================================================================
    # Layer 2: SE_2(3) kinematics (exact log-linear)
    # ===================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Layer 2: SE_2(3) kinematics (exact log-linear, 1 vertex)")
        print("=" * 60)

    # Extract SE_2(3) controller gains from Modelica model gains
    # The position PD cascade maps roughly to:
    #   Kp_se23 ~ Kp_position * Kp_velocity (outer loop gain product)
    #   Kd_se23 ~ Kp_velocity (velocity loop P gain)
    #   Kpq_se23 ~ Kr (attitude P gain)
    Kp_se23 = controller_gains.get('Kp_pos', 0.2) * controller_gains.get('Kp_vel', 0.45)
    Kd_se23 = controller_gains.get('Kp_vel', 0.45)
    Kpq_se23 = controller_gains.get('Kr_x', 1.0)

    if verbose:
        print(f"  SE2(3) controller gains: Kp={Kp_se23:.3f}, Kd={Kd_se23:.3f}, Kpq={Kpq_se23:.3f}")

    # Compute A(t) matrices — exact, no polytopic needed!
    A_kinematic = se23_kinematic_matrices(
        v_refs, a_refs, omega_refs,
        Kp=Kp_se23, Kd=Kd_se23, Kpq=Kpq_se23,
    )

    # Disturbance input matrix for kinematic layer
    # The input mismatch is ntilde = [0, a_tilde, omega_tilde]
    # where omega_tilde is bounded by Layer 1, and a_tilde comes from
    # acceleration disturbance (e.g., thrust uncertainty).
    #
    # For the linearized log-linear dynamics: xi_dot = A(t) xi + B_n ntilde
    # B_n maps body-frame input mismatch to Lie algebra error rate.
    # At small errors, J_l^{-1} ≈ I, so B_n ≈ I_9 restricted to the
    # nonzero components of ntilde.
    #
    # ntilde = [0(3), a_tilde(3), omega_tilde(3)] in se_2(3) ordering [p, v, R]
    # So B_n picks out the v and R blocks.

    # Disturbance channels:
    # Channel 1: omega_tilde (from Layer 1) -> affects rotation block
    # Channel 2: acceleration disturbance -> affects velocity block
    B_omega_to_xi = np.zeros((9, 3))
    B_omega_to_xi[6:9, :] = np.eye(3)  # omega error -> rotation block

    B_accel_to_xi = np.zeros((9, 3))
    B_accel_to_xi[3:6, :] = np.eye(3)  # accel error -> velocity block

    # Combined disturbance: [omega_tilde(3), a_tilde(3)]
    B_kinematic = np.hstack([B_omega_to_xi, B_accel_to_xi])

    # Combined disturbance bound (max of per-axis bounds from Layer 1 for omega,
    # scalar for acceleration)
    d_combined = max(np.max(omega_bounds), d_accel) if d_accel > 0 else np.max(omega_bounds)

    # For the LMI solver, we need a single scalar disturbance bound.
    # Use the maximum across channels.
    # A more refined approach would use SE23LMIs3 with separate channels.
    d_kinematic = np.max(omega_bounds) if d_gyro > 0 else d_accel

    # If we have both disturbance types, combine them
    if d_gyro > 0 and d_accel > 0:
        # Use only the omega channel for now (dominant disturbance)
        # TODO: extend to multi-channel LMI
        B_kinematic_single = B_omega_to_xi
        d_kinematic = np.max(omega_bounds)
    elif d_gyro > 0:
        B_kinematic_single = B_omega_to_xi
        d_kinematic = np.max(omega_bounds)
    else:
        B_kinematic_single = B_accel_to_xi
        d_kinematic = d_accel

    # Solve Layer 2 LMI
    layer2_result = solve_time_varying_polytopic_lmi(
        A_kinematic, B_kinematic_single, times.tolist(),
        polynomial_degree=polynomial_degree,
        verbose=verbose,
    )

    if layer2_result['status'] in ['infeasible', None]:
        raise RuntimeError(f"Layer 2 LMI failed: {layer2_result['status']}")

    # Extract xi bounds
    xi_bounds_result = compute_state_bounds(
        layer2_result['M_coefs'],
        layer2_result['mu_physical'],
        d_kinematic,
        times.tolist(),
    )
    xi_bounds = xi_bounds_result['bounds_max']

    xi_names = ['xi_px', 'xi_py', 'xi_pz',
                'xi_vx', 'xi_vy', 'xi_vz',
                'xi_Rx', 'xi_Ry', 'xi_Rz']

    if verbose:
        print(f"\nLayer 2 results:")
        print(f"  alpha = {layer2_result['alpha']:.4f}")
        print(f"  mu_physical = {layer2_result['mu_physical']:.4e}")
        print(f"  Lie algebra error bounds:")
        for name, bound in zip(xi_names, xi_bounds):
            print(f"    {name}: ±{bound:.6f}")

    return {
        'layer1': layer1_result,
        'layer2': layer2_result,
        'omega_bounds': omega_bounds,
        'omega_bounds_over_time': omega_bounds_result['bounds_over_time'],
        'xi_bounds': xi_bounds,
        'xi_bounds_over_time': xi_bounds_result['bounds_over_time'],
        'xi_bounds_names': xi_names,
        'times': times,
        'd_kinematic': d_kinematic,
    }


# ---------------------------------------------------------------------------
# Exponential map: Lie algebra -> physical space (pure numpy)
# ---------------------------------------------------------------------------

def se23_expmap_single(xi: np.ndarray) -> np.ndarray:
    """
    Compute the SE_2(3) exponential map for a single Lie algebra element.

    Maps xi = [xi_p(3), xi_v(3), xi_R(3)] in se_2(3) to
    (delta_p, delta_v, R_error) in the Lie group.

    Uses the Rodrigues formula for SO(3) and the corresponding
    left Jacobian for the translation/velocity components.

    Parameters
    ----------
    xi : (9,) array [xi_p, xi_v, xi_R]

    Returns
    -------
    eta : (9,) array [delta_p(3), delta_v(3), euler_321(3)]
    """
    xi_p = xi[0:3]
    xi_v = xi[3:6]
    xi_R = xi[6:9]

    theta = np.linalg.norm(xi_R)

    if theta < 1e-10:
        # Small angle: R ≈ I + [xi_R]_x, J_l ≈ I
        R = np.eye(3) + _skew(xi_R)
        J_l = np.eye(3)
    else:
        # Rodrigues formula
        K = _skew(xi_R) / theta
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        # Left Jacobian of SO(3)
        J_l = (np.eye(3)
               + (1 - np.cos(theta)) / theta * K
               + (1 - np.sin(theta) / theta) * (K @ K))

    delta_p = J_l @ xi_p
    delta_v = J_l @ xi_v

    # Convert R to Euler 321 (yaw, pitch, roll)
    # R is the rotation error matrix
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0

    return np.array([delta_p[0], delta_p[1], delta_p[2],
                     delta_v[0], delta_v[1], delta_v[2],
                     roll, pitch, yaw])


def se23_expmap(Xi: np.ndarray) -> np.ndarray:
    """
    Batch exponential map: maps (9, N) Lie algebra points to (9, N) group points.

    Input columns: [xi_p(3), xi_v(3), xi_R(3)]
    Output columns: [delta_p(3), delta_v(3), euler_321(3)]
    """
    N = Xi.shape[1]
    Eta = np.zeros((9, N))
    for k in range(N):
        Eta[:, k] = se23_expmap_single(Xi[:, k])
    return Eta


def sample_ellipsoid_boundary(M: np.ndarray, n: int = 100) -> np.ndarray:
    """
    Deterministic boundary samples of {xi : xi^T M xi = 1}.

    Samples on coordinate 2-planes, then maps to ellipsoid surface.

    Parameters
    ----------
    M : (d, d) positive definite matrix
    n : int, number of samples

    Returns
    -------
    X : (d, n) points on ellipsoid boundary
    """
    d = M.shape[0]
    A = np.linalg.cholesky(M).T  # M = A^T A
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    P = len(pairs)
    per_plane = max(1, (n + P - 1) // P)

    U_list = []
    for (i, j) in pairs:
        ts = 2.0 * np.pi * np.arange(per_plane) / per_plane
        Ui = np.zeros((d, per_plane))
        Ui[i, :] = np.cos(ts)
        Ui[j, :] = np.sin(ts)
        U_list.append(Ui)
        if len(U_list) * per_plane >= n:
            break

    U = np.concatenate(U_list, axis=1)[:, :n]
    X = np.linalg.solve(A, U)
    return X


# ---------------------------------------------------------------------------
# Reference trajectory utilities for quadrotor
# ---------------------------------------------------------------------------

def quadrotor_flatness(
    p_ref: np.ndarray,
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    jerk_ref: np.ndarray,
    mass: float,
    g: float = 9.80665,
    yaw_ref: Optional[np.ndarray] = None,
    snap_ref: Optional[np.ndarray] = None,
    inertia: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute full quadrotor reference from position trajectory using differential flatness.

    Given a position trajectory and its derivatives, computes the reference
    rotation matrix, angular velocity (analytically from jerk), thrust, and
    optionally angular acceleration and moment feedforward (from snap).

    Parameters
    ----------
    p_ref : (N, 3) position reference [px, py, pz] in NED
    v_ref : (N, 3) velocity reference in NED
    a_ref : (N, 3) acceleration reference in NED
    jerk_ref : (N, 3) jerk reference in NED
    mass : float, quadrotor mass
    g : float, gravity magnitude
    yaw_ref : (N,) optional yaw reference (default: 0)
    snap_ref : (N, 3) optional snap (4th derivative) in NED, for alpha computation
    inertia : (3,) optional [Ixx, Iyy, Izz] for moment feedforward computation

    Returns
    -------
    dict with:
        'R_ref': (N, 3, 3) rotation matrices
        'omega_ref': (N, 3) body angular velocity (analytical)
        'T_ref': (N,) thrust magnitude
        'a_body_ref': (N, 3) body-frame acceleration [0, 0, -T/m]
        'v_ref': (N, 3) world-frame velocity (pass-through)
        'alpha_ref': (N, 3) angular acceleration (if snap_ref provided)
        'M_ff': (N, 3) moment feedforward (if snap_ref and inertia provided)
    """
    N = p_ref.shape[0]
    g_vec = np.array([0, 0, g])  # NED: gravity points down (+z)

    if yaw_ref is None:
        yaw_ref = np.zeros(N)

    R_ref = np.zeros((N, 3, 3))
    omega_ref = np.zeros((N, 3))
    T_ref = np.zeros(N)
    T_dot_ref = np.zeros(N)
    a_body_ref = np.zeros((N, 3))

    for k in range(N):
        # Thrust vector: F = mass * (a - g_vec), body z = -F/|F|
        thrust_vec = mass * (a_ref[k] - g_vec)
        T = np.linalg.norm(thrust_vec)
        T_ref[k] = T

        if T < 1e-8:
            R_ref[k] = np.eye(3)
            a_body_ref[k] = np.array([0, 0, 0])
            continue

        z_B = -thrust_vec / T

        # Heading direction (zero yaw rate for now)
        x_C = np.array([np.cos(yaw_ref[k]), np.sin(yaw_ref[k]), 0])

        y_B = np.cross(z_B, x_C)
        y_norm = np.linalg.norm(y_B)
        if y_norm < 1e-8:
            y_B = np.array([0, 1, 0])
        else:
            y_B = y_B / y_norm

        x_B = np.cross(y_B, z_B)
        R_ref[k] = np.column_stack([x_B, y_B, z_B])
        a_body_ref[k] = np.array([0, 0, -T / mass])

        # --- Analytical omega from jerk (Mellinger & Kumar 2011) ---
        # thrust_vec_dot = mass * jerk
        thrust_vec_dot = mass * jerk_ref[k]

        # T_dot = -z_B . thrust_vec_dot  (since thrust_vec = -T * z_B)
        T_dot = -np.dot(z_B, thrust_vec_dot)
        T_dot_ref[k] = T_dot

        # z_B_dot = -(thrust_vec_dot - (z_B . thrust_vec_dot) * z_B) / T
        #         = -(perpendicular component of thrust_vec_dot to z_B) / T
        z_B_dot = -(thrust_vec_dot + T_dot * z_B) / T

        # omega from z_B_dot = omega_y * x_B - omega_x * y_B
        omega_ref[k, 0] = -np.dot(y_B, z_B_dot)
        omega_ref[k, 1] = np.dot(x_B, z_B_dot)

        # omega_z from yaw dynamics (for constant yaw, omega_z comes from
        # the constraint that y_B stays perpendicular to x_C and z_B)
        # For zero yaw_dot: h = cross(z_B, x_C), y_B = h/|h|
        # h_dot = cross(z_B_dot, x_C)
        h = np.cross(z_B, x_C)
        h_norm = np.linalg.norm(h)
        if h_norm > 1e-8:
            h_dot = np.cross(z_B_dot, x_C)
            y_B_dot = (h_dot - y_B * np.dot(y_B, h_dot)) / h_norm
            omega_ref[k, 2] = -np.dot(x_B, y_B_dot)

    result = {
        'R_ref': R_ref,
        'omega_ref': omega_ref,
        'T_ref': T_ref,
        'a_body_ref': a_body_ref,
        'v_ref': v_ref,
    }

    # --- Analytical alpha from snap (optional) ---
    if snap_ref is not None:
        alpha_ref = np.zeros((N, 3))
        for k in range(N):
            T = T_ref[k]
            if T < 1e-8:
                continue

            z_B = R_ref[k, :, 2]
            x_B = R_ref[k, :, 0]
            y_B = R_ref[k, :, 1]

            thrust_vec_dot = mass * jerk_ref[k]
            thrust_vec_ddot = mass * snap_ref[k]
            T_dot = T_dot_ref[k]

            # z_B_dot (recompute)
            z_B_dot = -(thrust_vec_dot + T_dot * z_B) / T

            # T_ddot = -z_B_dot . thrust_vec_dot - z_B . thrust_vec_ddot
            T_ddot = -np.dot(z_B_dot, thrust_vec_dot) - np.dot(z_B, thrust_vec_ddot)

            # z_B_ddot from second derivative of z_B = -thrust_vec / T
            z_B_ddot = (
                -(thrust_vec_ddot + T_ddot * z_B + 2 * T_dot * z_B_dot) / T
                + (thrust_vec_dot + T_dot * z_B) * T_dot / (T * T)
            )

            # z_B_ddot = R * ([alpha]_x * e3 + [omega]_x^2 * e3)
            # [alpha]_x * e3 = [alpha_y, -alpha_x, 0]
            # [omega]_x^2 * e3 = [ox*oz, oy*oz, -(ox^2+oy^2)]
            # z_B_ddot = (alpha_y + ox*oz)*x_B + (-alpha_x + oy*oz)*y_B
            #          + (-(ox^2+oy^2))*z_B
            # => alpha_x = -(y_B . z_B_ddot) + oy*oz
            #    alpha_y =  (x_B . z_B_ddot) - ox*oz
            ox, oy, oz = omega_ref[k]
            alpha_ref[k, 0] = -np.dot(y_B, z_B_ddot) + oy * oz
            alpha_ref[k, 1] = np.dot(x_B, z_B_ddot) - ox * oz

            # alpha_z from yaw (analogous to omega_z computation)
            x_C = np.array([np.cos(yaw_ref[k]), np.sin(yaw_ref[k]), 0])
            h = np.cross(z_B, x_C)
            h_norm = np.linalg.norm(h)
            if h_norm > 1e-8:
                h_dot = np.cross(z_B_dot, x_C)
                y_B_dot = (h_dot - y_B * np.dot(y_B, h_dot)) / h_norm
                h_ddot = np.cross(z_B_ddot, x_C)
                # d/dt(y_B) = (h_dot - y_B*(y_B.h_dot))/|h|
                # d2/dt2 is complex; use the frame relation instead:
                # y_B_dot = -omega_z * x_B + omega_x * z_B
                # y_B_ddot = -alpha_z * x_B - omega_z * x_B_dot + alpha_x * z_B + omega_x * z_B_dot
                # x_B_dot = omega_z * y_B - omega_y * z_B
                # y_B_ddot . x_B = -alpha_z + omega_x * (z_B . x_B) = -alpha_z (if z_B perp x_B)
                # Need y_B_ddot . x_B
                # Numerically differentiate y_B_dot? No, use the relation.
                # Actually for zero yaw:
                # omega_z was from: omega_z = -x_B . y_B_dot
                # alpha_z = -x_B . y_B_ddot - x_B_dot . y_B_dot
                # x_B_dot = omega_z * y_B - omega_y * z_B
                x_B_dot = oz * y_B - oy * z_B
                # y_B_ddot from h:
                h_norm_dot = np.dot(y_B, h_dot)  # d/dt(|h|) = (h.h_dot)/|h| = y_B . h_dot
                y_B_ddot = (
                    (h_ddot - y_B * np.dot(y_B, h_ddot)) / h_norm
                    - 2 * h_norm_dot * y_B_dot / h_norm
                    - (np.dot(y_B_dot, h_dot)) * y_B / h_norm
                )
                alpha_ref[k, 2] = -np.dot(x_B, y_B_ddot) - np.dot(x_B_dot, y_B_dot)

        result['alpha_ref'] = alpha_ref

        if inertia is not None:
            Ixx, Iyy, Izz = inertia
            M_ff = np.zeros((N, 3))
            M_ff[:, 0] = Ixx * alpha_ref[:, 0]
            M_ff[:, 1] = Iyy * alpha_ref[:, 1]
            M_ff[:, 2] = Izz * alpha_ref[:, 2]
            result['M_ff'] = M_ff

    return result


def compute_omega_from_R(R_ref: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Compute body angular velocity from rotation matrix trajectory.

    omega = vee(R^T R_dot) using finite differences.

    Parameters
    ----------
    R_ref : (N, 3, 3) rotation matrices
    times : (N,) time points

    Returns
    -------
    omega_ref : (N, 3) body angular velocity
    """
    N = R_ref.shape[0]
    omega_ref = np.zeros((N, 3))

    for k in range(N):
        if k == 0:
            dt = times[1] - times[0]
            R_dot = (R_ref[1] - R_ref[0]) / dt
        elif k == N - 1:
            dt = times[-1] - times[-2]
            R_dot = (R_ref[-1] - R_ref[-2]) / dt
        else:
            dt = times[k + 1] - times[k - 1]
            R_dot = (R_ref[k + 1] - R_ref[k - 1]) / dt

        # Omega_hat = R^T R_dot (skew-symmetric)
        Omega_hat = R_ref[k].T @ R_dot

        # Extract angular velocity from skew-symmetric matrix
        omega_ref[k, 0] = Omega_hat[2, 1]  # omega_x
        omega_ref[k, 1] = Omega_hat[0, 2]  # omega_y
        omega_ref[k, 2] = Omega_hat[1, 0]  # omega_z

    return omega_ref
