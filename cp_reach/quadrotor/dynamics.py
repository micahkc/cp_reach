#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Multirotor Reference Trajectory via Differential Flatness (CasADi)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import casadi as ca
import numpy as np

def build_f_ref(tol: float = 1e-6) -> ca.Function:
    """
    Build the CasADi function f_ref that maps flat outputs
    (position derivatives + heading) to thrust, orientation, and body rates.

    Frames / notation
    -----------------
    e : world/inertial frame
    b : body frame
    C_be : rotation matrix (columns are body axes in e-frame), maps b->e
    C_eb = C_be^T

    Inputs
    ------
    psi, psi_dot, psi_ddot : heading and derivatives (about z_e)
    v_e, a_e, j_e, s_e     : velocity, acceleration, jerk, snap in e-frame (3-vectors)
    m, g                   : mass and gravity
    J_xx, J_yy, J_zz, J_xz : inertia terms (diag + optional xz product of inertia)

    Outputs
    -------
    v_b            : velocity expressed in body frame (C_eb @ v_e)
    C_be           : body orientation (3x3)
    omega_eb_b     : body angular velocity ω (in b-frame)
    omega_dot_eb_b : body angular acceleration (in b-frame)
    M_b            : required body moment M = J ω̇ + ω × (J ω)
    T              : thrust magnitude (scalar)

    Notes
    -----
    - Thrust direction z_b is aligned with m(g ẑ - a_e).
    - Heading ψ fixes the x_c axis in the world; y_b is chosen via cross products.
    - Jerk/snap enter the rate expressions (Mellinger-style construction).
    - Tolerances avoid singularities when vectors align or T ~ 0.
    """
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Symbols and parameters
    psi      = ca.SX.sym("psi")
    psi_dot  = ca.SX.sym("psi_dot")
    psi_ddot = ca.SX.sym("psi_ddot")

    v_e = ca.SX.sym("v_e", 3)
    a_e = ca.SX.sym("a_e", 3)
    j_e = ca.SX.sym("j_e", 3)
    s_e = ca.SX.sym("s_e", 3)

    m = ca.SX.sym("m")
    g = ca.SX.sym("g")

    J_xx = ca.SX.sym("J_x")
    J_yy = ca.SX.sym("J_y")
    J_zz = ca.SX.sym("J_z")
    J_xz = ca.SX.sym("J_xz")

    # Unit vectors in e- and b-frames (here we use e-basis vectors)
    xh = ca.SX([1, 0, 0])
    yh = ca.SX([0, 1, 0])
    zh = ca.SX([0, 0, 1])

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Orientation C_be and thrust T

    # Desired thrust (world): f = m(g ẑ - a_e)
    thrust_e = m * (g * zh - a_e)
    T = ca.norm_2(thrust_e)
    T = ca.if_else(T > tol, T, tol)        # avoid division by ~0

    # Body z-axis in world
    zb_e = thrust_e / T

    # Desired world x-axis from yaw ψ (heading around z_e)
    xc_e = ca.cos(psi) * xh + ca.sin(psi) * yh

    # Form y_b and x_b via cross products, handle collinearity with tolerance
    yb_e = ca.cross(zb_e, xc_e)
    N_yb_e = ca.norm_2(yb_e)
    yb_e = ca.if_else(N_yb_e > tol, yb_e / N_yb_e, yh)  # fallback if aligned
    xb_e = ca.cross(yb_e, zb_e)

    # Rotation b->e (columns are body axes expressed in e)
    C_be = ca.hcat([xb_e, yb_e, zb_e])
    C_eb = C_be.T

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Angular velocity ω_eb_b  (body rates)

    # Jerk contribution projected to body axes:
    # set p = ω·x, q = ω·y, r = ω·z (in body frame)
    # cf. Mellinger et al. construction
    t2_e = m / T * j_e   # scaled jerk term
    p =  ca.dot(t2_e, yb_e)      # roll rate component
    q = -ca.dot(t2_e, xb_e)      # pitch rate component

    # Recover φ, θ from C_eb for heading rate coupling (ψ̇ enters r expression)
    theta = ca.asin(-C_eb[2, 0])
    phi = ca.if_else(
        ca.fabs(ca.fabs(theta) - ca.pi / 2) < tol,
        0,
        ca.atan2(C_eb[2, 1], C_eb[2, 2])
    )

    cos_phi = ca.cos(phi)
    cos_phi = ca.if_else(ca.fabs(cos_phi) > tol, cos_phi, 0)

    # r from ψ̇ and geometry (singular at φ=π/2)
    r = -q * ca.tan(phi) + ca.cos(theta) * psi_dot / cos_phi

    omega_eb_b = p * xh + q * yh + r * zh  # ω in body frame

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Angular acceleration ω̇_eb_b

    # Some auxiliary terms
    omega_eb_b_cross_zh = ca.cross(omega_eb_b, zh)

    # Ṫ from jerk projection; enter Coriolis-like terms for ṗ, q̇
    T_dot = -ca.dot(m * j_e, zb_e)
    coriolis_b = 2 * T_dot / T * omega_eb_b_cross_zh
    centrip_b  = ca.cross(omega_eb_b, omega_eb_b_cross_zh)

    # ṗ and q̇ from snap projections and ω cross-terms
    q_dot = -m / T * ca.dot(s_e, xb_e) - ca.dot(coriolis_b, xh) - ca.dot(centrip_b, xh)
    p_dot =  m / T * ca.dot(s_e, yb_e) + ca.dot(coriolis_b, yh) + ca.dot(centrip_b, yh)

    # ψ̇ coupling for φ̇, θ̇
    omega_eb_e = C_be @ omega_eb_b
    omega_ec_e = psi_dot * zh

    theta_dot = (q - ca.sin(phi) * ca.cos(theta) * psi_dot) / ca.cos(phi)
    phi_dot   =  p + ca.sin(theta) * psi_dot

    # Solve for ṙ (using constraint that z_c = z_e = ẑ and algebraic relations)
    zc_e = zh
    yc_e = ca.cross(zc_e, xc_e)
    T1 = ca.inv(ca.horzcat(xb_e, yc_e, zh))
    A  = T1 @ C_be
    b  = -T1 @ (ca.cross(omega_eb_e, phi_dot * xb_e) + ca.cross(omega_ec_e, theta_dot * yc_e))
    r_dot = (psi_ddot - A[2, 0] * p_dot - A[2, 1] * q_dot - b[2]) / A[2, 2]

    omega_dot_eb_b = p_dot * xh + q_dot * yh + r_dot * zh

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Inputs (moments) from rigid body rotational dynamics
    J = ca.SX(3, 3)
    J[0, 0] = J_xx
    J[1, 1] = J_yy
    J[2, 2] = J_zz
    J[0, 2] = J[2, 0] = J_xz   # (optional xz product of inertia)

    M_b = J @ omega_dot_eb_b + ca.cross(omega_eb_b, J @ omega_eb_b)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Velocity in body frame (handy for feedforward)
    v_b = C_eb @ v_e

    # Wrap up as a CasADi Function (same signature/order you used)
    f_ref = ca.Function(
        "f_ref",
        [psi, psi_dot, psi_ddot, v_e, a_e, j_e, s_e, m, g, J_xx, J_yy, J_zz, J_xz],
        [v_b, C_be, omega_eb_b, omega_dot_eb_b, M_b, T],
        [
            "psi", "psi_dot", "psi_ddot", "v_e", "a_e", "j_e", "s_e",
            "m", "g", "J_xx", "J_yy", "J_zz", "J_xz"
        ],
        ["v_b", "C_be", "omega_eb_b", "omega_dot_eb_b", "M_b", "T"]
    )
    return f_ref


# Optional: tiny numpy-friendly wrapper for convenience
def f_ref_numpy(f_ref_fun: ca.Function,
                psi: float, psi_dot: float, psi_ddot: float,
                v_e, a_e, j_e, s_e,
                m: float, g: float,
                J_xx: float, J_yy: float, J_zz: float, J_xz: float):
    """
    Call f_ref and return numpy arrays.
    """
    outs = f_ref_fun(psi, psi_dot, psi_ddot,
                     v_e, a_e, j_e, s_e,
                     m, g, J_xx, J_yy, J_zz, J_xz)
    return [np.array(o) for o in outs]
