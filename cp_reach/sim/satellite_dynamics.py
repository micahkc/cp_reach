import casadi as ca
from cp_analyzer.lie.SO3 import hat


def satellite_dynamics():
    """
    Returns symbolic dynamics for a satellite using Lie group theory (SO(3)).
    Includes central gravity.
    """
    # State variables
    x = ca.MX.sym("x", 3)        # Position in inertial frame
    v = ca.MX.sym("v", 3)        # Velocity in inertial frame
    R = ca.MX.sym("R", 3, 3)     # Rotation matrix R ∈ SO(3)
    omega = ca.MX.sym("omega", 3)  # Angular velocity in body frame

    # Control inputs
    thrust = ca.MX.sym("thrust", 3)  # Thrust in body frame
    torque = ca.MX.sym("torque", 3)  # Torque in body frame

    # Parameters
    I = ca.MX.sym("I", 3)        # Inertia vector [Ixx, Iyy, Izz] (assume diagonal)
    m = ca.MX.sym("m")           # Mass
    mu = ca.MX.sym("mu")         # Gravitational parameter (km^3/s^2)

    # Normalize intertia as diagonal matrix
    I_mat = ca.diag(I)

    # Gravitational acceleration in interial frame
    r_norm = ca.norm_2(x)
    g_I = -mu * x / (r_norm**3 + 1e-8) # avoid singularities at origin

    # Translational dynamics
    a_I = (R @ thrust) / m + g_I # translational acceleration
    dx = v
    dv = a_I

    # Rotational dynamics
    R_dot = R @ hat(omega) # attitude dynamics on SO(3)
    omega_dot =(I_mat, torque - ca.cross(omega, I_mat @ omega))

    # Full dynamics function
    state = ca.vertcat(x, v, ca.reshape(R, 9, 1), omega)
    state_dot = ca.vertcat(x_dot, v_dot, ca.reshape(dR, 9, 1), omega_dot)
    u = ca.vertcat(thrust, torque)
    p = ca.vertcat(I, m, mu)

    # CasADi function
    f = ca.Function("f", [state, u, p], [state_dot], ["x", "u", "p"], ["x_dot"])
    return f

if __name__ == "__main__":
    f = satellite_dynamics()
    print(f)


