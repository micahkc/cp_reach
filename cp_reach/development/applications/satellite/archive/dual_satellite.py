import cyecca.lie as lie
import casadi as ca
import numpy as np


class DualSatellite(object):
    """
    Simulates uncontrolled dynamics of two satellites:
    - Chief (reference satellite)
    - Deputy (actual satellite)

    No controller is applied. Dynamics are propagated in full nonlinear state space.
    """

    def __init__(self):
        self.model = self.derive_dynamics()

    def derive_dynamics(self):
        # Chief satellite state variables (reference)
        p_c = ca.SX.sym('p_c', 3)  # position
        v_c = ca.SX.sym('v_c', 3)  # velocity
        q_c = lie.SO3Quat.elem(ca.SX.sym('q_c', 4))  # orientation quaternion
        R_c = lie.SO3Dcm.from_Quat(q_c)  # rotation matrix
        w_c = ca.SX.sym('w_c', 3)  # angular velocity in body frame

        # Deputy satellite state variables (actual)
        p_d = ca.SX.sym('p_d', 3)  # position
        v_d = ca.SX.sym('v_d', 3)  # velocity
        q_d = lie.SO3Quat.elem(ca.SX.sym('q_d', 4))  # orientation quaternion
        R_d = lie.SO3Dcm.from_Quat(q_d)  # rotation matrix
        w_d = ca.SX.sym('w_d', 3)  # angular velocity in body frame

        # Full state vector
        x_vect = ca.vertcat(
            p_c, v_c, q_c.param, w_c,
            p_d, v_d, q_d.param, w_d
        )

        # Initial conditions - geostationary orbit
        x0_dict = {
            # Chief initial conditions
            'px_c': 0,
            'py_c': -42164e3,
            'pz_c': 0,
            'vx_c': 3.07e3,
            'vy_c': 0,
            'vz_c': 0,
            'q0_c': 1,
            'q1_c': 0,
            'q2_c': 0,
            'q3_c': 0,
            'wx_c': 0,
            'wy_c': 0,
            'wz_c': 0,
            # Deputy initial conditions (same as chief by default)
            'px_d': 0,
            'py_d': -42164e3,
            'pz_d': 0,
            'vx_d': 3.07e3,
            'vy_d': 0,
            'vz_d': 0,
            'q0_d': 1,
            'q1_d': 0,
            'q2_d': 0,
            'q3_d': 0,
            'wx_d': 0,
            'wy_d': 0,
            'wz_d': 0,
        }
        x_index = {k: i for i, k in enumerate(x0_dict.keys())}

        # No inputs (uncontrolled)
        u_vect = ca.vertcat()
        u0_dict = {}
        u0_vect = ca.vertcat()
        u_index = {}

        # Parameters
        mu = ca.SX.sym('mu')  # gravitational parameter
        J = ca.SX.sym('J', 3)  # moment of inertia (assuming diagonal)

        p0_dict = {
            'mu': 3.986e14,  # Earth gravitational parameter (m^3/s^2)
            'Jx': 1.0,  # moment of inertia x (kg*m^2)
            'Jy': 1.0,  # moment of inertia y (kg*m^2)
            'Jz': 1.0,  # moment of inertia z (kg*m^2)
        }
        p_index = {k: i for i, k in enumerate(p0_dict.keys())}
        p_vect = ca.vertcat(mu, J[0], J[1], J[2])

        # Chief dynamics
        eps = 1e-9
        g_c = -mu * p_c / (ca.norm_2(p_c)**3 + eps)  # gravitational acceleration

        # Chief kinematics (no control forces or torques)
        p_c_dot = v_c
        v_c_dot = g_c
        q_c_dot = (q_c * lie.SO3Quat.elem(ca.vertcat(0, w_c[0], w_c[1], w_c[2]))).param / 2
        # Euler's equation for rigid body rotation (no external torques)
        w_c_dot = ca.vertcat(
            (J[1] - J[2]) / J[0] * w_c[1] * w_c[2],
            (J[2] - J[0]) / J[1] * w_c[2] * w_c[0],
            (J[0] - J[1]) / J[2] * w_c[0] * w_c[1]
        )

        # Deputy dynamics
        g_d = -mu * p_d / (ca.norm_2(p_d)**3 + eps)  # gravitational acceleration

        # Deputy kinematics (no control forces or torques)
        p_d_dot = v_d
        v_d_dot = g_d
        q_d_dot = (q_d * lie.SO3Quat.elem(ca.vertcat(0, w_d[0], w_d[1], w_d[2]))).param / 2
        # Euler's equation for rigid body rotation (no external torques)
        w_d_dot = ca.vertcat(
            (J[1] - J[2]) / J[0] * w_d[1] * w_d[2],
            (J[2] - J[0]) / J[1] * w_d[2] * w_d[0],
            (J[0] - J[1]) / J[2] * w_d[0] * w_d[1]
        )

        # State derivative vector
        x_dot_vect = ca.vertcat(
            p_c_dot, v_c_dot, q_c_dot, w_c_dot,
            p_d_dot, v_d_dot, q_d_dot, w_d_dot
        )

        f = ca.Function('f', [x_vect, u_vect, p_vect], [x_dot_vect], ['x', 'u', 'p'], ['x_dot'])
        return locals()

    def simulate(self, t_vect, x0_dict=None, p0_dict=None, integrator='rk'):
        """
        Simulate the dual satellite dynamics.

        Parameters
        ----------
        t_vect : array-like
            Time vector for integration
        x0_dict : dict, optional
            Initial state dictionary. If None, uses default from model.
        p0_dict : dict, optional
            Parameter dictionary. If None, uses default from model.
        integrator : str, optional
            Integration method ('rk', 'cvodes', 'idas')

        Returns
        -------
        result : dict
            Integration result with state trajectory
        """
        if x0_dict is None:
            x0_dict = self.model['x0_dict']
        if p0_dict is None:
            p0_dict = self.model['p0_dict']

        x0_vect = ca.vertcat(*[x0_dict[k] for k in self.model['x_index'].keys()])
        p0_vect = ca.vertcat(*[p0_dict[k] for k in self.model['p_index'].keys()])
        u0_vect = self.model['u0_vect']

        u_vect = self.model['u_vect']
        x_vect = self.model['x_vect']
        f = self.model['f']

        dae = {'x': x_vect, 'ode': f(x_vect, u_vect, p0_vect), 'u': u_vect}
        F = ca.integrator('F', integrator, dae, t_vect[0], t_vect)

        return F(x0=x0_vect, u=u0_vect)
