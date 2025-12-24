"""
Log-linear error dynamics for spacecraft on SE₂(3).

Based on: "Exact Log-Linear Error Dynamics for Thrusting Spacecraft on SE(3)"
by Condie et al.
"""

import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp
import cyecca.lie as lie

class SE23Spacecraft:
    """Spacecraft dynamics on SE₂(3): state [p, v, q] where q is quaternion."""

    def __init__(self, mu=3.986e14):
        self.mu = mu  # Earth gravitational parameter

    def gravity(self, p):
        """Gravitational acceleration: g = -μ p / |p|³"""
        r = np.linalg.norm(p)
        return -self.mu * p / (r**3 + 1e-9)

    def dynamics(self, t, state, controls, p_ref_func=None):
        """
        Nonlinear dynamics: ṗ = v, v̇ = Ra + g(p_ref), q̇ = 0.5 q ⊗ [0,ω]

        If p_ref_func is None, uses g(p) (actual position).
        If p_ref_func is provided, uses g(p_ref(t)) instead.
        """
        p, v, q = state[0:3], state[3:6], state[6:10]
        q = q / np.linalg.norm(q)

        u = controls(t) if callable(controls) else controls
        a, omega = u[0:3], u[3:6]

        # Rotation matrix from quaternion
        q_lie = lie.SO3Quat.elem(ca.DM(q))
        R = np.array(ca.DM(lie.SO3Dcm.from_Quat(q_lie).param).full()).reshape(3, 3)

        # Choose which position to use for gravity
        if p_ref_func is None:
            p_for_gravity = p
        else:
            p_ref = p_ref_func(t) if callable(p_ref_func) else p_ref_func
            p_for_gravity = p_ref

        # Dynamics
        p_dot = v
        v_dot = R @ a #+ self.gravity(p_for_gravity)

        # Quaternion kinematics
        omega_quat = lie.SO3Quat.elem(ca.DM([0, *omega]))
        q_dot = np.array(ca.DM((q_lie * omega_quat).param).full()).flatten() / 2

        return np.concatenate([p_dot, v_dot, q_dot])



class LogLinearErrorDynamics:
    
    """
    Log-linear error dynamics: ξ̇ = A(t)ξ + b(t)

    where A(t) = -ad_n̄ + A_C and b(t) = gravity feedforward.
    """

    def __init__(self, spacecraft):
        self.spacecraft = spacecraft
        self.state_ref_interp = None


    def dynamics(self, t, xi, controls_ref):
        """
        Log-linear dynamics: ξ̇ = (-ad_n̄ + A_C) ξ
        xi: [ξ_p(3), ξ_v(3), ξ_R(3)] - Lie algebra error
        controls_ref: callable or array-like [a(3), ω(3)]
        """
        
        # 1) Reference controls
        u_ref = controls_ref(t) if callable(controls_ref) else controls_ref
        # print(f"u_ref: {u_ref}")
        a_bar      = np.asarray(u_ref[0:3])
        omega_bar  = np.asarray(u_ref[3:6])

        n_bar_param = ca.vertcat(
            ca.SX.zeros(3, 1),        # v_b = 0
            ca.DM(a_bar),             # a_b = ā
            ca.DM(omega_bar),         # Omega = ω̄
        )
        n_bar = lie.se23.elem(n_bar_param)

        # 3) Compute ad_{n̄} using the library's adjoint map
        #    adjoint: se23.adjoint(·) : se23 elem -> 9×9 matrix in algebra coords
        Ad_n_bar_sx = lie.se23.adjoint(n_bar)                  # CasADi SX 9×9
        Ad_n_bar    = np.array(ca.DM(Ad_n_bar_sx).full())  # convert to numpy
        Ad_n_bar[3:6,6:9] = np.eye(3)#-Ad_n_bar[3:6,6:9]
        

        A_C = np.zeros((9, 9))
        A_C[0:3,3:6] = np.eye(3)

        A = -Ad_n_bar + A_C   # 9×9
        # print(A)

        # 4) Log-linear dynamics
        xi_dot = A @ xi
        return xi_dot




def simulate_nonlinear(spacecraft, t_span, state_0, controls,
                       p_ref_func=None, rtol=1e-9, atol=1e-12):
    """
    Simulate nonlinear spacecraft dynamics.

    Parameters
    ----------
    spacecraft : SE23Spacecraft
    t_span : array
        Time points to evaluate
    state_0 : array (10,)
        Initial state [p, v, q]
    controls : callable or array
        Control function controls(t) -> [a, ω] or constant array
    p_ref_func : callable or array, optional
        If provided, gravity is evaluated at p_ref(t) instead of p.

    Returns
    -------
    state : array (n, 10)
        State trajectory at each time point
    """
    sol = solve_ivp(
        lambda t, x: spacecraft.dynamics(t, x, controls, p_ref_func),
        (t_span[0], t_span[-1]), state_0,
        dense_output=True, rtol=rtol, atol=atol
    )
    return np.array([sol.sol(t) for t in t_span])



def expmap(xi):
    """
    Map SE_2(3) Lie algebra errors ξ_k ∈ se23 to real-space errors:
      [p_err, v_err, euler_err] at each time step.

    Input:
        xi : (N, 9) array
             rows are [ξ_p(3), ξ_v(3), ξ_R(3)]

    Output:
        Eta : (N, 9) array
              columns = [p_err(3), v_err(3), euler_B321(3)]
              - p_err, v_err are in R^3
              - euler_B321 are attitude errors in B321 Euler angles
    """
    N = xi.shape[0]
    Eta = np.zeros((N, 9))

    for k in range(N):
        # Lie algebra element (ξ ∈ se23)
        xi_k = lie.se23.elem(ca.DM(xi[k, :]))

        # Lie group element η = exp(ξ) ∈ SE_2(3) with (p, v, quat)
        eta_k = xi_k.exp(lie.SE23Quat)

        # params: [p(3), v(3), quat(4)]
        params = ca.DM(eta_k.param).full().ravel()

        # quaternion part
        quat_arr = params[6:]
        quat = lie.SO3Quat.elem(ca.DM(quat_arr))

        # convert quaternion -> B321 Euler (ψ, θ, φ)
        euler = lie.group_so3.SO3EulerB321.from_Quat(quat).param
        euler_np = ca.DM(euler).full().ravel()

        # stack [p_err, v_err, euler_err]
        Eta[k, :] = np.concatenate((params[:6], euler_np))

    return Eta

def simulate_error(spacecraft, t_span, state_ref_0, state_actual_0,
                   controls_ref, controls_actual=None, rtol=1e-9, atol=1e-12):
    """
    Simulate log-linear error dynamics.

    Parameters
    ----------
    spacecraft : SE23Spacecraft
    t_span : array
        Time points to evaluate
    state_ref_0 : array (10,)
        Initial reference state
    state_actual_0 : array (10,)
        Initial actual state
    controls_ref : callable or array
        Reference controls
    controls_actual : callable or array, optional
        Actual controls (defaults to controls_ref)

    Returns
    -------
    xi : array (n, 9)
        Error trajectory in Lie algebra coordinates
    """
    if controls_actual is None:
        controls_actual = controls_ref

    # Compute initial error
    X_ref_0 = lie.SE23Quat.elem(ca.DM(state_ref_0))
    X_actual_0 = lie.SE23Quat.elem(ca.DM(state_actual_0))
    eta_0 = X_ref_0.inverse() * X_actual_0
    xi_0 = np.array(ca.DM(eta_0.log().param).full()).flatten()
    print(xi_0)

    # Simulate log-linear error dynamics
    log_linear = LogLinearErrorDynamics(spacecraft)
    sol = solve_ivp(
        lambda t, xi: log_linear.dynamics(t, xi, controls_ref),
        (t_span[0], t_span[-1]), xi_0,
        dense_output=True, rtol=rtol, atol=atol
    )
    sol_alg = np.array([sol.sol(t) for t in t_span])
    sol_group = expmap(sol_alg)
    
    return sol_alg, sol_group


def geostationary_initial_conditions():
    """Generate geostationary orbit initial state [p, v, q]."""
    r_geo = 42164e3  # meters
    v_geo = 3.07e3   # m/s
    return np.array([0, -r_geo, 0, v_geo, 0, 0, 1, 0, 0, 0])
