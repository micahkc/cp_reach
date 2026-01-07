"""
Transfer dynamics for spacecraft rendezvous using HCW and log-linear SE₂(3) methods.

Compares:
1. Hill-Clohessy-Wiltshire (HCW) linearized relative motion (classical approach)
2. Invariant log-linear error dynamics on SE₂(3) (Lie group approach from Condie et al.)

The key difference is that HCW linearizes the gravity field around a circular reference
orbit, while the log-linear approach maintains exact error dynamics in Lie algebra
coordinates with attitude coupling.

Reference: "Comparison of Δv Planning: HCW vs. Invariant Log-Linear Error Dynamics"
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import casadi as ca
import cyecca.lie as lie


# =============================================================================
# HCW (Hill-Clohessy-Wiltshire) Transfer Dynamics
# =============================================================================

def hcw_state_matrix(n):
    """
    Construct the HCW state-space A matrix for 2D in-plane motion.

    The HCW equations (linearized relative motion for circular orbits):
        ẍ = 3n²x + 2nẏ + aₓ
        ÿ = -2nẋ + aᵧ

    State vector: [x, y, ẋ, ẏ]ᵀ

    Parameters
    ----------
    n : float
        Mean motion (rad/s), n = √(μ/r³)

    Returns
    -------
    A : ndarray (4, 4)
        State-space A matrix
    B : ndarray (4, 2)
        Input matrix for [aₓ, aᵧ]
    """
    A = np.array([
        [0,    0,    1,    0   ],
        [0,    0,    0,    1   ],
        [3*n**2, 0,  0,    2*n ],
        [0,    0,   -2*n,  0   ]
    ])

    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    return A, B


def hcw_state_transition_matrix(n, t):
    """
    Closed-form HCW state transition matrix F(t) = e^{At}.

    For state [x, y, ẋ, ẏ]ᵀ, this gives x(t) = F(t) x(0).

    Parameters
    ----------
    n : float
        Mean motion (rad/s)
    t : float
        Time (s)

    Returns
    -------
    F : ndarray (4, 4)
        State transition matrix
    """
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)

    F = np.array([
        [4 - 3*c,           0,    s/n,        2*(1 - c)/n    ],
        [6*(s - nt),        1,    2*(c - 1)/n, (4*s - 3*nt)/n],
        [3*n*s,             0,    c,          2*s            ],
        [6*n*(c - 1),       0,   -2*s,        4*c - 3        ]
    ])

    return F


def hcw_transfer_velocity(r0, rf, n, T):
    """
    Compute the transfer velocity for a two-impulse maneuver using HCW.

    Given initial position r0 and target position rf, compute the initial
    velocity v0 required to coast (no thrust) for time T and arrive at rf.

    The state transition matrix is partitioned as:
        F = [Frr  Frv]
            [Fvr  Fvv]

    Then: v0 = Frv⁻¹ (rf - Frr r0)
          vf = Fvr r0 + Fvv v0

    Parameters
    ----------
    r0 : ndarray (2,)
        Initial position [x, y] in LVLH frame (m)
    rf : ndarray (2,)
        Target position [x, y] in LVLH frame (m)
    n : float
        Mean motion (rad/s)
    T : float
        Transfer time (s)

    Returns
    -------
    v0 : ndarray (2,)
        Required initial velocity (m/s)
    vf : ndarray (2,)
        Arrival velocity (m/s)
    delta_v_total : float
        Total Δv = ||v0|| + ||vf|| (m/s)
    """
    F = hcw_state_transition_matrix(n, T)

    # Partition the STM
    Frr = F[0:2, 0:2]
    Frv = F[0:2, 2:4]
    Fvr = F[2:4, 0:2]
    Fvv = F[2:4, 2:4]

    # Compute required initial velocity
    v0 = np.linalg.solve(Frv, rf - Frr @ r0)

    # Compute arrival velocity
    vf = Fvr @ r0 + Fvv @ v0

    # Total delta-v (assuming starting and ending at rest relative to target)
    delta_v_total = np.linalg.norm(v0) + np.linalg.norm(vf)

    return v0, vf, delta_v_total


def hcw_simulate_transfer(r0, v0, n, t_span):
    """
    Simulate HCW trajectory with given initial conditions.

    Parameters
    ----------
    r0 : ndarray (2,)
        Initial position [x, y]
    v0 : ndarray (2,)
        Initial velocity [ẋ, ẏ]
    n : float
        Mean motion (rad/s)
    t_span : ndarray
        Time points to evaluate

    Returns
    -------
    trajectory : ndarray (N, 4)
        State trajectory [x, y, ẋ, ẏ] at each time point
    """
    x0 = np.concatenate([r0, v0])
    trajectory = np.zeros((len(t_span), 4))

    for i, t in enumerate(t_span):
        F = hcw_state_transition_matrix(n, t)
        trajectory[i, :] = F @ x0

    return trajectory


# =============================================================================
# Log-Linear SE₂(3) Transfer Dynamics
# =============================================================================

def se23_ad_matrix(n_bar):
    """
    Compute the adjoint matrix ad_{n̄} for n̄ ∈ se₂(3).

    The Lie algebra element n̄ = [0, ā, ω̄]ᵀ encodes the reference
    body-frame acceleration and angular velocity.

    Parameters
    ----------
    n_bar : ndarray (9,)
        Lie algebra element [ξ_p(3), ξ_v(3), ξ_R(3)] = [0, ā, ω̄]

    Returns
    -------
    ad_n_bar : ndarray (9, 9)
        Adjoint matrix
    """
    n_bar_elem = lie.se23.elem(ca.DM(n_bar))
    ad_n_bar_sx = lie.se23.adjoint(n_bar_elem)
    ad_n_bar = np.array(ca.DM(ad_n_bar_sx).full())
    return ad_n_bar


def se23_A_matrix(a_bar, omega_bar, gravity_gradient=None):
    """
    Compute the log-linear error dynamics A matrix: A = A_lg + A_G.

    From deltav-1.pdf Section 2, the hybrid log-linear error dynamics are:
        ξ̇ = (A_lg(t) + A_G) ξ

    where A_lg = -ad_{n̄} + A_C is the pure tracking formulation, and A_G
    contains the gravity gradient term G(t) that couples position error
    to velocity error derivative.

    Parameters
    ----------
    a_bar : ndarray (3,)
        Reference body-frame acceleration (m/s²)
    omega_bar : ndarray (3,)
        Reference body-frame angular velocity (rad/s)
    gravity_gradient : ndarray (3, 3), optional
        Gravity gradient tensor G(t) = (μ/||r_t||³)(3r̂_tr̂_t^T - I).
        For circular orbit in LVLH: diag([3n², 0, -n²]).
        If None, G=0 (pure tracking with gravity feedforward).

    Returns
    -------
    A : ndarray (9, 9)
        Log-linear dynamics matrix
    """
    # Construct n̄ = [0, ā, ω̄]
    n_bar = np.concatenate([np.zeros(3), a_bar, omega_bar])

    # Adjoint matrix
    ad_n_bar = se23_ad_matrix(n_bar)

    # Kinematic coupling matrix (ξ̇_p = ξ_v)
    A_C = np.zeros((9, 9))
    A_C[0:3, 3:6] = np.eye(3)

    # Log-linear dynamics: A_lg = -ad_{n̄} + A_C
    A = -ad_n_bar + A_C

    # Gravity gradient contribution (Eq. 25 in deltav-1.pdf)
    # A_G = [0   0   0]
    #       [G   0   0]
    #       [0   0   0]
    if gravity_gradient is not None:
        A[3:6, 0:3] += gravity_gradient

    return A


def circular_orbit_gravity_gradient(n):
    """
    Compute the gravity gradient tensor for a circular orbit in LVLH frame.

    From deltav-1.pdf Eq. 20, the gravity gradient tensor is:
        G = (μ/||r_t||³)(3r̂_tr̂_t^T - I)

    For a circular orbit in LVLH frame (r̂_t = [1,0,0] radial), this reduces to:
        G = diag([3n², 0, -n²])

    which gives the HCW differential gravity terms:
        ξ̈_p,x = 3n²ξ_p,x  (radial - unstable)
        ξ̈_p,y = 0         (along-track - neutral)
        ξ̈_p,z = -n²ξ_p,z  (out-of-plane - stable)

    Parameters
    ----------
    n : float
        Mean motion (rad/s)

    Returns
    -------
    G : ndarray (3, 3)
        Gravity gradient tensor in LVLH frame
    """
    return np.diag([3*n**2, 0, -n**2])


def hcw_A_matrix_9x9(n):
    """
    Build the 9x9 A matrix that exactly matches HCW dynamics.

    This constructs an A matrix in SE₂(3) error coordinates that gives
    identical STM blocks to the classical HCW equations when projected
    to the translational (ξ_p, ξ_v) subspace.

    The key differences from the general se23_A_matrix are:
    1. No position-rotation coupling (-[ω̄]×ξ_p term is absent)
    2. Uses 2n for Coriolis coupling (not n from -ad_n̄)
    3. Attitude dynamics are decoupled (ξ̇_R = 0)

    This is appropriate for coasting relative motion in LVLH coordinates
    where the frame rotates with the orbit but attitude errors don't
    affect position through frame rotation.

    Parameters
    ----------
    n : float
        Mean motion (rad/s)

    Returns
    -------
    A : ndarray (9, 9)
        HCW-compatible A matrix in SE₂(3) structure
    """
    A = np.zeros((9, 9))

    # Position dynamics: ξ̇_p = ξ_v (kinematic coupling only)
    A[0:3, 3:6] = np.eye(3)

    # Velocity dynamics: ξ̇_v = G ξ_p + Coriolis ξ_v
    # Gravity gradient
    G = circular_orbit_gravity_gradient(n)
    A[3:6, 0:3] = G

    # Coriolis coupling (2n, not n)
    # HCW: ẍ = 3n²x + 2nẏ, ÿ = -2nẋ
    A[3, 4] = 2*n   # +2n ẏ contribution to ẍ
    A[4, 3] = -2*n  # -2n ẋ contribution to ÿ

    # Attitude dynamics: decoupled (ξ̇_R = 0)
    # A[6:9, :] = 0 (already zero)

    return A


def se23_initial_error(state_ref, state_actual):
    """
    Compute initial Lie algebra error ξ₀ = Log(X̄⁻¹ X)∨.

    Parameters
    ----------
    state_ref : ndarray (10,)
        Reference state [p, v, q] where q is quaternion
    state_actual : ndarray (10,)
        Actual state [p, v, q]

    Returns
    -------
    xi_0 : ndarray (9,)
        Initial error in Lie algebra coordinates [ξ_p, ξ_v, ξ_R]
    """
    X_ref = lie.SE23Quat.elem(ca.DM(state_ref))
    X_actual = lie.SE23Quat.elem(ca.DM(state_actual))
    eta = X_ref.inverse() * X_actual
    xi_0 = np.array(ca.DM(eta.log().param).full()).flatten()
    return xi_0


def se23_error_to_group(xi):
    """
    Map Lie algebra error ξ to group element η = exp(ξ).

    Parameters
    ----------
    xi : ndarray (9,)
        Error in Lie algebra coordinates

    Returns
    -------
    eta_params : ndarray (10,)
        Group element parameters [p, v, q]
    """
    xi_elem = lie.se23.elem(ca.DM(xi))
    eta = xi_elem.exp(lie.SE23Quat)
    return np.array(ca.DM(eta.param).full()).flatten()


def se23_simulate_error(xi_0, a_bar, omega_bar, t_span, rtol=1e-9, atol=1e-12):
    """
    Simulate log-linear error dynamics: ξ̇ = A(t)ξ.

    For constant controls, A is constant (LTI system).

    Parameters
    ----------
    xi_0 : ndarray (9,)
        Initial error in Lie algebra coordinates
    a_bar : ndarray (3,) or callable
        Reference body-frame acceleration (m/s²)
    omega_bar : ndarray (3,) or callable
        Reference body-frame angular velocity (rad/s)
    t_span : ndarray
        Time points

    Returns
    -------
    xi_traj : ndarray (N, 9)
        Error trajectory in Lie algebra coordinates
    """
    def dynamics(t, xi):
        if callable(a_bar):
            a = a_bar(t)
        else:
            a = a_bar
        if callable(omega_bar):
            omega = omega_bar(t)
        else:
            omega = omega_bar
        A = se23_A_matrix(a, omega)
        return A @ xi

    sol = solve_ivp(
        dynamics,
        (t_span[0], t_span[-1]),
        xi_0,
        dense_output=True,
        rtol=rtol,
        atol=atol
    )

    return np.array([sol.sol(t) for t in t_span])


def se23_transfer_velocity_from_hcw(r0, rf, n, T, R_ref=None):
    """
    Compute transfer using HCW and express in SE₂(3) framework.

    This uses HCW to find the velocity, then converts the result
    to SE₂(3) Lie algebra coordinates for comparison.

    Parameters
    ----------
    r0 : ndarray (2,)
        Initial position [x, y] in LVLH (m)
    rf : ndarray (2,)
        Target position [x, y] in LVLH (m)
    n : float
        Mean motion (rad/s)
    T : float
        Transfer time (s)
    R_ref : ndarray (3, 3), optional
        Reference attitude (rotation matrix). Defaults to identity.

    Returns
    -------
    v0_hcw : ndarray (2,)
        HCW initial velocity (m/s)
    v0_inertial : ndarray (3,)
        Initial velocity in inertial/LVLH 3D frame
    delta_v : float
        Total Δv (m/s)
    """
    v0_hcw, vf_hcw, delta_v = hcw_transfer_velocity(r0, rf, n, T)

    # Extend to 3D (z = 0 for in-plane motion)
    v0_inertial = np.array([v0_hcw[0], v0_hcw[1], 0.0])

    return v0_hcw, v0_inertial, delta_v


def se23_continuous_thrust_transfer(xi_0, xi_f, a_bar, omega_bar, T, n_steps=100):
    """
    Compute continuous thrust transfer in SE₂(3) framework.

    Unlike HCW which uses impulsive burns, this computes the feedforward
    acceleration needed for continuous low-thrust transfer.

    For log-linear dynamics ξ̇ = Aξ + Bu with constant A, the solution is:
        ξ(T) = e^{AT} ξ(0) + ∫₀ᵀ e^{A(T-τ)} B u(τ) dτ

    Parameters
    ----------
    xi_0 : ndarray (9,)
        Initial error
    xi_f : ndarray (9,)
        Target error (typically zeros)
    a_bar : ndarray (3,)
        Reference acceleration
    omega_bar : ndarray (3,)
        Reference angular velocity
    T : float
        Transfer time (s)

    Returns
    -------
    info : dict
        Contains A matrix, eigenvalues, and controllability info
    """
    A = se23_A_matrix(a_bar, omega_bar)

    # State transition matrix
    Phi = expm(A * T)

    # For the log-linear system, the "free response" is ξ(T) = Φ ξ(0)
    xi_free = Phi @ xi_0

    # The required correction
    delta_xi = xi_f - xi_free

    # Eigenvalue analysis
    eigvals = np.linalg.eigvals(A)

    return {
        'A': A,
        'Phi': Phi,
        'xi_free': xi_free,
        'delta_xi': delta_xi,
        'eigenvalues': eigvals,
        'is_stable': np.all(np.real(eigvals) < 0)
    }


# =============================================================================
# Invariant Δv Planning (Section 1.2 of deltav.pdf)
# =============================================================================

def se23_state_transition_matrix(a_bar, omega_bar, T, gravity_gradient=None):
    """
    Compute the SE₂(3) error state transition matrix Φ(T) = e^{A*T}.

    For constant reference controls, A is constant and Φ(T) = e^{AT}.

    Parameters
    ----------
    a_bar : ndarray (3,)
        Reference body-frame acceleration (m/s²)
    omega_bar : ndarray (3,)
        Reference body-frame angular velocity (rad/s)
    T : float
        Time duration (s)
    gravity_gradient : ndarray (3, 3), optional
        Gravity gradient tensor. Use circular_orbit_gravity_gradient(n) for
        HCW-compatible dynamics. If None, assumes gravity feedforward.

    Returns
    -------
    Phi : ndarray (9, 9)
        State transition matrix
    """
    A = se23_A_matrix(a_bar, omega_bar, gravity_gradient)
    Phi = expm(A * T)
    return Phi


def se23_stm_blocks(Phi):
    """
    Partition the SE₂(3) state transition matrix into 3x3 blocks.

    Φ(T) = [Φ_pp  Φ_pv  Φ_pR]
           [Φ_vp  Φ_vv  Φ_vR]
           [Φ_Rp  Φ_Rv  Φ_RR]

    Parameters
    ----------
    Phi : ndarray (9, 9)
        State transition matrix

    Returns
    -------
    blocks : dict
        Dictionary with keys 'pp', 'pv', 'pR', 'vp', 'vv', 'vR', 'Rp', 'Rv', 'RR'
    """
    return {
        'pp': Phi[0:3, 0:3],
        'pv': Phi[0:3, 3:6],
        'pR': Phi[0:3, 6:9],
        'vp': Phi[3:6, 0:3],
        'vv': Phi[3:6, 3:6],
        'vR': Phi[3:6, 6:9],
        'Rp': Phi[6:9, 0:3],
        'Rv': Phi[6:9, 3:6],
        'RR': Phi[6:9, 6:9],
    }


def invariant_transfer_velocity(xi_0, a_bar, omega_bar, T, R_ref_0=None, R_ref_T=None,
                                gravity_gradient=None):
    """
    Compute transfer Δv using invariant log-linear error dynamics.

    This implements Section 1.2 of the deltav.pdf document, with optional
    gravity gradient from Section 2 of deltav-1.pdf for HCW compatibility.

    Given initial error ξ(0) = [ξ_p(0), ξ_v(0), ξ_R(0)]ᵀ, compute the impulsive
    velocity change Δξ_v(0) that drives terminal position error to zero: ξ_p(T) = 0.

    The boundary condition (Eq. 12):
        0 = Φ_pp(T)ξ_p(0) + Φ_pv(T)(ξ_v(0) + Δξ_v(0)) + Φ_pR(T)ξ_R(0)

    Solving for Δξ_v(0) (Eq. 13):
        Δξ_v(0) = -Φ_pv(T)⁻¹ (Φ_pp(T)ξ_p(0) + Φ_pR(T)ξ_R(0) + Φ_pv(T)ξ_v(0))

    Physical velocity change (Eq. 14):
        Δv₀ = R_ref(0) Δξ_v(0)

    Terminal braking impulse (Eq. 15):
        Δv_T = -R_ref(T) ξ_v(T⁻)

    Parameters
    ----------
    xi_0 : ndarray (9,)
        Initial error [ξ_p, ξ_v, ξ_R] in Lie algebra coordinates
    a_bar : ndarray (3,)
        Reference body-frame acceleration (m/s²)
    omega_bar : ndarray (3,)
        Reference body-frame angular velocity (rad/s)
    T : float
        Transfer time (s)
    R_ref_0 : ndarray (3, 3), optional
        Reference attitude at t=0. Defaults to identity.
    R_ref_T : ndarray (3, 3), optional
        Reference attitude at t=T. Defaults to identity.
    gravity_gradient : ndarray (3, 3), optional
        Gravity gradient tensor. Use circular_orbit_gravity_gradient(n) for
        HCW-compatible dynamics. If None, assumes gravity feedforward.

    Returns
    -------
    result : dict
        Contains:
        - delta_xi_v_0: Invariant velocity impulse at t=0
        - delta_xi_v_T: Invariant velocity impulse at t=T (braking)
        - delta_v_0: Physical velocity change at t=0 (m/s)
        - delta_v_T: Physical velocity change at t=T (m/s)
        - delta_v_total: Total Δv = ||Δv₀|| + ||Δv_T||
        - xi_T_minus: Error state just before terminal impulse
        - Phi: State transition matrix
        - blocks: STM block matrices
    """
    if R_ref_0 is None:
        R_ref_0 = np.eye(3)
    if R_ref_T is None:
        R_ref_T = np.eye(3)

    # Extract initial error components
    xi_p_0 = xi_0[0:3]
    xi_v_0 = xi_0[3:6]
    xi_R_0 = xi_0[6:9]

    # Compute state transition matrix
    Phi = se23_state_transition_matrix(a_bar, omega_bar, T, gravity_gradient)
    blocks = se23_stm_blocks(Phi)

    # Extract blocks
    Phi_pp = blocks['pp']
    Phi_pv = blocks['pv']
    Phi_pR = blocks['pR']
    Phi_vp = blocks['vp']
    Phi_vv = blocks['vv']
    Phi_vR = blocks['vR']

    # Compute required invariant velocity impulse (Eq. 13)
    # Δξ_v(0) = -Φ_pv⁻¹ (Φ_pp ξ_p(0) + Φ_pR ξ_R(0) + Φ_pv ξ_v(0))
    rhs = Phi_pp @ xi_p_0 + Phi_pR @ xi_R_0 + Phi_pv @ xi_v_0
    delta_xi_v_0 = -np.linalg.solve(Phi_pv, rhs)

    # Physical velocity change at t=0 (Eq. 14)
    delta_v_0 = R_ref_0 @ delta_xi_v_0

    # Compute error state just before terminal impulse
    # After initial impulse, ξ_v(0⁺) = ξ_v(0) + Δξ_v(0)
    xi_0_plus = xi_0.copy()
    xi_0_plus[3:6] = xi_v_0 + delta_xi_v_0

    # Propagate to terminal time: ξ(T⁻) = Φ(T) ξ(0⁺)
    xi_T_minus = Phi @ xi_0_plus

    # Extract terminal velocity error
    xi_v_T_minus = xi_T_minus[3:6]

    # Terminal braking impulse cancels remaining velocity error (Eq. 15)
    delta_xi_v_T = -xi_v_T_minus
    delta_v_T = R_ref_T @ delta_xi_v_T

    # Total Δv (Eq. 16)
    delta_v_total = np.linalg.norm(delta_v_0) + np.linalg.norm(delta_v_T)

    return {
        'delta_xi_v_0': delta_xi_v_0,
        'delta_xi_v_T': delta_xi_v_T,
        'delta_v_0': delta_v_0,
        'delta_v_T': delta_v_T,
        'delta_v_0_norm': np.linalg.norm(delta_v_0),
        'delta_v_T_norm': np.linalg.norm(delta_v_T),
        'delta_v_total': delta_v_total,
        'xi_0_plus': xi_0_plus,
        'xi_T_minus': xi_T_minus,
        'Phi': Phi,
        'blocks': blocks,
    }


def invariant_transfer_velocity_hcw(xi_0, n, T, R_ref_0=None, R_ref_T=None):
    """
    Compute transfer Δv using HCW-compatible invariant dynamics.

    This uses hcw_A_matrix_9x9(n) which exactly matches HCW dynamics
    in the translational subspace, with decoupled attitude.

    For coasting transfers between two points in LVLH frame, this gives
    identical results to hcw_transfer_velocity().

    Parameters
    ----------
    xi_0 : ndarray (9,)
        Initial error [ξ_p, ξ_v, ξ_R] in LVLH coordinates
    n : float
        Mean motion (rad/s)
    T : float
        Transfer time (s)
    R_ref_0 : ndarray (3, 3), optional
        Reference attitude at t=0. Defaults to identity.
    R_ref_T : ndarray (3, 3), optional
        Reference attitude at t=T. Defaults to identity.

    Returns
    -------
    result : dict
        Same structure as invariant_transfer_velocity()
    """
    if R_ref_0 is None:
        R_ref_0 = np.eye(3)
    if R_ref_T is None:
        R_ref_T = np.eye(3)

    # Extract initial error components
    xi_p_0 = xi_0[0:3]
    xi_v_0 = xi_0[3:6]
    xi_R_0 = xi_0[6:9]

    # Compute HCW-compatible STM
    A = hcw_A_matrix_9x9(n)
    Phi = expm(A * T)
    blocks = se23_stm_blocks(Phi)

    # Extract blocks
    Phi_pp = blocks['pp']
    Phi_pv = blocks['pv']
    Phi_pR = blocks['pR']

    # Compute required invariant velocity impulse (Eq. 13)
    rhs = Phi_pp @ xi_p_0 + Phi_pR @ xi_R_0 + Phi_pv @ xi_v_0
    delta_xi_v_0 = -np.linalg.solve(Phi_pv, rhs)

    # Physical velocity change at t=0 (Eq. 14)
    delta_v_0 = R_ref_0 @ delta_xi_v_0

    # Compute error state just before terminal impulse
    xi_0_plus = xi_0.copy()
    xi_0_plus[3:6] = xi_v_0 + delta_xi_v_0

    # Propagate to terminal time
    xi_T_minus = Phi @ xi_0_plus

    # Terminal braking impulse
    xi_v_T_minus = xi_T_minus[3:6]
    delta_xi_v_T = -xi_v_T_minus
    delta_v_T = R_ref_T @ delta_xi_v_T

    # Total Δv
    delta_v_total = np.linalg.norm(delta_v_0) + np.linalg.norm(delta_v_T)

    return {
        'delta_xi_v_0': delta_xi_v_0,
        'delta_xi_v_T': delta_xi_v_T,
        'delta_v_0': delta_v_0,
        'delta_v_T': delta_v_T,
        'delta_v_0_norm': np.linalg.norm(delta_v_0),
        'delta_v_T_norm': np.linalg.norm(delta_v_T),
        'delta_v_total': delta_v_total,
        'xi_0_plus': xi_0_plus,
        'xi_T_minus': xi_T_minus,
        'Phi': Phi,
        'blocks': blocks,
    }


def invariant_simulate_transfer(xi_0, delta_xi_v_0, a_bar, omega_bar, t_span):
    """
    Simulate invariant error trajectory after initial impulse.

    Parameters
    ----------
    xi_0 : ndarray (9,)
        Initial error before impulse
    delta_xi_v_0 : ndarray (3,)
        Invariant velocity impulse
    a_bar : ndarray (3,)
        Reference body-frame acceleration
    omega_bar : ndarray (3,)
        Reference body-frame angular velocity
    t_span : ndarray
        Time points to evaluate

    Returns
    -------
    xi_traj : ndarray (N, 9)
        Error trajectory in Lie algebra coordinates
    """
    # Apply initial impulse
    xi_0_plus = xi_0.copy()
    xi_0_plus[3:6] += delta_xi_v_0

    # Simulate from post-impulse state
    return se23_simulate_error(xi_0_plus, a_bar, omega_bar, t_span)


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.

    Parameters
    ----------
    q : ndarray (4,)
        Quaternion [w, x, y, z]

    Returns
    -------
    R : ndarray (3, 3)
        Rotation matrix
    """
    q_lie = lie.SO3Quat.elem(ca.DM(q))
    R = np.array(ca.DM(lie.SO3Dcm.from_Quat(q_lie).param).full()).reshape(3, 3).T
    return R


def rotation_matrix_after_time(omega, T, R_0=None):
    """
    Compute rotation matrix after rotating at constant angular velocity.

    R(T) = R(0) * exp([ω]× T)

    Parameters
    ----------
    omega : ndarray (3,)
        Angular velocity (rad/s)
    T : float
        Time duration (s)
    R_0 : ndarray (3, 3), optional
        Initial rotation matrix. Defaults to identity.

    Returns
    -------
    R_T : ndarray (3, 3)
        Rotation matrix at time T
    """
    if R_0 is None:
        R_0 = np.eye(3)

    # Rodrigues formula for rotation
    theta = np.linalg.norm(omega) * T
    if theta < 1e-10:
        return R_0.copy()

    axis = omega / np.linalg.norm(omega)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return R_0 @ R_delta


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_hcw_se23(r0_lvlh, rf_lvlh, n, T, mu, r_orbit, t_eval=None):
    """
    Compare HCW and SE₂(3) approaches for a transfer maneuver.

    Parameters
    ----------
    r0_lvlh : ndarray (2,)
        Initial position in LVLH [x, y] (m)
        x: radial (positive away from Earth)
        y: along-track (positive in velocity direction)
    rf_lvlh : ndarray (2,)
        Target position in LVLH [x, y] (m)
    n : float
        Mean motion (rad/s)
    T : float
        Transfer time (s)
    mu : float
        Gravitational parameter (m³/s²)
    r_orbit : float
        Orbital radius (m)
    t_eval : ndarray, optional
        Time points for trajectory evaluation

    Returns
    -------
    results : dict
        Contains HCW and SE₂(3) results for comparison
    """
    if t_eval is None:
        t_eval = np.linspace(0, T, 500)

    # --- HCW Transfer ---
    v0_hcw, vf_hcw, delta_v_hcw = hcw_transfer_velocity(r0_lvlh, rf_lvlh, n, T)
    traj_hcw = hcw_simulate_transfer(r0_lvlh, v0_hcw, n, t_eval)

    # --- SE₂(3) Analysis ---
    # Construct reference and actual states
    # Reference at origin of LVLH frame
    p_ref = np.array([r_orbit, 0, 0])  # On circular orbit
    v_ref = np.array([0, np.sqrt(mu/r_orbit), 0])  # Circular velocity
    q_ref = np.array([1, 0, 0, 0])  # Identity quaternion
    state_ref = np.concatenate([p_ref, v_ref, q_ref])

    # Actual state with LVLH offset
    # LVLH: x = radial out, y = along-track
    # Inertial: for reference at +X axis, y_lvlh -> +Y inertial
    p_actual = p_ref + np.array([r0_lvlh[0], r0_lvlh[1], 0])
    v_actual = v_ref.copy()  # Initially at rest relative to target
    q_actual = q_ref.copy()
    state_actual = np.concatenate([p_actual, v_actual, q_actual])

    # Compute initial SE₂(3) error
    xi_0 = se23_initial_error(state_ref, state_actual)

    # For the transfer, we need a reference acceleration
    # Use zero acceleration (coasting) like HCW
    a_bar = np.array([0, 0, 0])
    omega_bar = np.array([0, 0, n])  # Rotate with orbit

    # Simulate error evolution
    xi_traj = se23_simulate_error(xi_0, a_bar, omega_bar, t_eval)

    # SE₂(3) analysis
    se23_info = se23_continuous_thrust_transfer(
        xi_0, np.zeros(9), a_bar, omega_bar, T
    )

    return {
        'hcw': {
            'v0': v0_hcw,
            'vf': vf_hcw,
            'delta_v': delta_v_hcw,
            'trajectory': traj_hcw,
        },
        'se23': {
            'xi_0': xi_0,
            'xi_trajectory': xi_traj,
            'A_matrix': se23_info['A'],
            'eigenvalues': se23_info['eigenvalues'],
            'is_stable': se23_info['is_stable'],
        },
        't_eval': t_eval,
        'params': {
            'r0': r0_lvlh,
            'rf': rf_lvlh,
            'n': n,
            'T': T,
            'mu': mu,
            'r_orbit': r_orbit,
        }
    }


def orbital_params_from_radius(r, mu=3.986e14):
    """
    Compute orbital parameters from radius.

    Parameters
    ----------
    r : float
        Orbital radius (m)
    mu : float
        Gravitational parameter (m³/s²)

    Returns
    -------
    params : dict
        n (mean motion), T_orbit (period), v_circ (circular velocity)
    """
    n = np.sqrt(mu / r**3)
    T_orbit = 2 * np.pi / n
    v_circ = np.sqrt(mu / r)

    return {
        'n': n,
        'T_orbit': T_orbit,
        'v_circ': v_circ,
        'r': r,
        'mu': mu
    }
