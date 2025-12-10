import cyecca.lie as lie
import casadi as ca
import numpy as np



class SatSimBurn(object):

    def __init__(self):
        self.model = self.derive_dynamics()
        
    def saturate(self, x, xmin, xmax):
        return ca.fmin(ca.fmax(x,xmin), xmax)

    def derive_dynamics(self):
        # state variables
        t = ca.SX.sym('t') # time
        p_a = ca.SX.sym('p_a', 3) # position
        v_a = ca.SX.sym('v_a', 3) # velocity
        q_a = lie.SO3Quat.elem(ca.SX.sym('q_a', 4)) # orientation
        R_a = lie.SO3Dcm.from_Quat(q_a) # rotation matrix
        X_a = lie.SE23Quat.elem(ca.vertcat(p_a, v_a, q_a.param)) # lie group state

        # reference state variables
        p_b = ca.SX.sym('p_b', 3) # position body frame
        v_b = ca.SX.sym('v_b', 3) # velocity body frame
        q_b = lie.SO3Quat.elem(ca.SX.sym('q_b', 4)) # orientation body frame
        R_b = lie.SO3Dcm.from_Quat(q_b) # rotation matrix body
        X_b = lie.SE23Quat.elem(ca.vertcat(p_b, v_b, q_b.param)) # reference lie group state

        # full state
        x_vect = ca.vertcat(t, p_a, v_a, q_a.param, p_b, v_b, q_b.param)
        
        # initial conditions, geostationary orbit
        x0_dict = {
            't': 0,
            'px_a': 0,
            'py_a': -42164e3,
            'pz_a': 0,
            'vx_a': 3.07e3,
            'vy_a': 0,
            'vz_a': 0,
            'q0_a': 1,
            'q1_a': 0,
            'q2_a': 0,
            'q3_a': 0,
            'px_b': 0,
            'py_b': -42164e3,
            'pz_b': 0,
            'vx_b': 3.07e3,
            'vy_b': 0,
            'vz_b': 0,
            'q0_b': 1,
            'q1_b': 0,
            'q2_b': 0,
            'q3_b': 0,
        }
        x_index = { k: i for i, k in enumerate(x0_dict.keys()) }

        # inputs (autonomous, no inputs)
        u_vect = ca.vertcat()
        u0_vect = ca.vertcat()
        u_index = {}

        # parameters
        mu = ca.SX.sym('mu') # gravitational parameter, G*M
        t_burn = ca.SX.sym('t_burn') # burn time
        thrust = ca.SX.sym('thrust') # thrust magnitude
        w_d_amp = ca.SX.sym('w_d_amp') # disturbance amplitude
        a_d_amp = ca.SX.sym('a_d_amp') # disturbance amplitude
        w_d_x_phase = ca.SX.sym('w_d_x_phase') # distw_burbance phase
        w_d_y_phase = ca.SX.sym('w_d_y_phase') # disturbance phase
        w_d_z_phase = ca.SX.sym('w_d_z_phase') # disturbance phase
        w_d_freq = ca.SX.sym('w_d_freq') # disturbance frequency
        Kp = ca.SX.sym('Kp')
        Kd = ca.SX.sym('Kd')
        Kpq = ca.SX.sym('Kpq')
        p0_dict = {
            'mu': 3.986e14, # gravitational parameter for Earth
            't_burn': 60, # burn time of 60 seconds
            'thrust': 30, # thrust magnitude of 50 N
            'w_d_amp': 1e-5, # disturbance amplitude of 1e-5 rad/s
            'a_d_amp': 1e-5, # disturbance amplitude of 1e-5 rad/s
            'w_d_x_phase': 0, # disturbance phase of 0 rad
            'w_d_y_phase': 0, # disturbance phase of 1 rad
            'w_d_z_phase': 0, # disturbance phase of 2 rad
            'w_d_freq': 1, # disturbance frequency of 1 Hz
            'Kp': 1, # Position gain proportional const
            'Kd': 1, # Position gain derivative term
            'Kpq': 1, # Attitude gain proportional term
        }
        p_index = { k: i for i, k in enumerate(p0_dict.keys()) }

        p_vect = ca.vertcat(mu, t_burn, thrust, w_d_amp, a_d_amp, w_d_x_phase, w_d_y_phase, w_d_z_phase, w_d_freq, Kp, Kd, Kpq)

        # disturbance
        dist_signal = ca.vertcat(
                        ca.if_else(ca.sin(2*ca.pi*w_d_freq*t + w_d_x_phase) >0, 1, -1),
                        ca.if_else(ca.sin(2*ca.pi*w_d_freq*t + w_d_y_phase) >0, 1, -1),
                        ca.if_else(ca.sin(2*ca.pi*w_d_freq*t + w_d_z_phase) >0, 1, -1))
        w_d = (w_d_amp/np.sqrt(3)) * dist_signal
        a_d = (a_d_amp/np.sqrt(3)) * dist_signal

        # reference dynamics
        eps = 1e-9
        g_b = -mu*p_b/(ca.norm_2(p_b)**3 + eps) # gravity in inertial frame
        a_b = ca.vertcat(thrust, 0, 0) # thrust in body frame
        # w_b = ca.vertcat(0, 0, 0)  # angular velocity in body frame
        # w_b is calculate orbital angular velocity so r hat points toward earth (depends on positon or velocity)
        w_b = ca.cross(p_b, v_b)/ca.dot(p_b, p_b)

        xi = (X_b.inverse() * X_a).log().param



        B0 = ca.SX([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        K = ca.SX.zeros(6,9)
        for i in range(3):
            K[i,i] = Kp
            K[i, i+3] = Kd
            K[i+3, i+6] = Kpq
        
        u = -B0@K@xi
        
        g_a = -mu*p_a/(ca.norm_2(p_a)**3 + eps) # gravity in inertial frame
        a_a = a_b + u[3:6]  # velocity control
        w_a = w_b + u[6:] + w_d


        # reference kinematics
        p_b_dot = v_b
        v_b_dot = R_b @ a_b + g_b
        q_b_dot = (q_b * lie.SO3Quat.elem(ca.vertcat(0, w_b[0], w_b[1], w_b[2]))).param / 2

        # true kinematics
        p_a_dot = v_a
        v_a_dot = R_a @ a_a + g_a + a_d
        q_a_dot = (q_a * lie.SO3Quat.elem(ca.vertcat(0, w_a[0], w_a[1], w_a[2]))).param / 2 

        # state derivatives
        x_dot_vect = ca.vertcat(
            1, p_a_dot, v_a_dot, q_a_dot,
            p_b_dot, v_b_dot, q_b_dot)
        f = ca.Function('f', [x_vect, u_vect, p_vect], [x_dot_vect], ['x', 'u', 'p'], ['x_dot'])
        return locals()
    
    


    def simulate(self, t_vect, integrator='rk'):
        x0_vect = ca.vertcat(*[self.model['x0_dict'][k] for k in self.model['x_index'].keys()])
        p0_vect = ca.vertcat(*[self.model['p0_dict'][k] for k in self.model['p_index'].keys()])
        u0_vect = ca.vertcat(*[self.model['u0_dict'][k] for k in self.model['u_index'].keys()])

        u_vect = self.model['u_vect']
        x_vect = self.model['x_vect']
        f = self.model['f']
        dae = {'x': x_vect, 'ode': f(self.model['x_vect'], u_vect, p0_vect), 'u': u_vect}
        F = ca.integrator('F', integrator, dae, t_vect[0], t_vect)
        return F(x0=x0_vect, u=u0_vect)



class SatSimCoast(object):

    def __init__(self):
        self.model = self.derive_dynamics()

    def derive_dynamics(self):
        # state variables
        t = ca.SX.sym('t') # time
        p_a = ca.SX.sym('p_a', 3) # position
        v_a = ca.SX.sym('v_a', 3) # velocity

        # full state
        x_vect = ca.vertcat(p_a, v_a)
        
        # initial conditions, geostationary orbit
        x0_dict = {
            'px_a': 0,
            'py_a': -42164e3,
            'pz_a': 0,
            'vx_a': 3.07e3,
            'vy_a': 0,
            'vz_a': 0,
        }
        x_index = { k: i for i, k in enumerate(x0_dict.keys()) }

        # inputs (autonomous, no inputs)
        u_vect = ca.vertcat()
        u0_dict = {}
        u0_vect = ca.vertcat()
        u_index = { k: i for i, k in enumerate(u0_dict.keys()) }

        # parameters
        mu = ca.SX.sym('mu') # gravitational parameter, G*M
        p0_dict = {
            'mu': 3.986e14, # gravitational parameter for Earth
        }
        p_index = { k: i for i, k in enumerate(p0_dict.keys()) }
        p_vect = ca.vertcat(mu)

        # true dynamics
        eps = 1e-9
        g_a = -mu*p_a/(ca.norm_2(p_a)**3 + eps) # gravity in inertial frame

        # true kinematics
        p_a_dot = v_a
        v_a_dot = g_a

        # state derivatives
        x_dot_vect = ca.vertcat(p_a_dot, v_a_dot)
        f = ca.Function('f', [x_vect, u_vect, p_vect], [x_dot_vect], ['x', 'u', 'p'], ['x_dot'])
        return locals()

    def simulate(self, t_vect, integrator='rk'):
        x0_vect = ca.vertcat(*[self.model['x0_dict'][k] for k in self.model['x_index'].keys()])
        p0_vect = ca.vertcat(*[self.model['p0_dict'][k] for k in self.model['p_index'].keys()])
        u0_vect = ca.vertcat(*[self.model['u0_dict'][k] for k in self.model['u_index'].keys()])
        u_vect = self.model['u_vect']
        x_vect = self.model['x_vect']
        f = self.model['f']
        dae = {'x': x_vect, 'ode': f(self.model['x_vect'], u_vect, p0_vect), 'u': u_vect}
        F = ca.integrator('F', integrator, dae, t_vect[0], t_vect)
        return F(x0=x0_vect, u=u0_vect)

