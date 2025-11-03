import numpy as np
import cp_reach.physics.rigid_body as rigid_body

def solve(ang_vel_dist, ref_acceleration, pid_values, num_points=720):
    """
    Compute an SE(3) (position + orientation) over-approx reachable set using a
    Lyapunov ellipsoid on the log-coordinates, then map boundary samples to the group.

    Args:
        ang_vel_dist (float): Bound on angular-velocity magnitude used in the kinematic bound.
        ref_acceleration: Reference translational acceleration input/trajectory (passed through).
        pid_values (tuple): (kp, kd, kpq, kdq). Only kp, kd, kpq are used here.
        num_points (int): Number of ellipsoid boundary samples.

    Returns:
        dynamics_sol (dict|None)
        eta_points (ndarray): Exp-mapped boundary samples (shape depends on expmap impl).
        lower_bound (ndarray): Componentwise min of eta_points.
        upper_bound (ndarray): Componentwise max of eta_points.
        kinematics_sol (dict): Contains 'P', 'mu1', 'mu2', etc.
    """
    kp, kd, kpq, kdq = pid_values  # kdq kept for interface symmetry
    omega_dist = ang_vel_dist
    gravity_err = 0.0

    kinematics_sol = rigid_body.solve_se23_invariant_set_log_control(
            ref_acceleration, kp, kd, kpq, omega_dist, gravity_err
        )

    mu1 = kinematics_sol['mu1']  # ω-disturbance multiplier
    mu2 = kinematics_sol['mu2']  # gravity-disturbance multiplier
    P_kin = kinematics_sol['P']

    val_kin = mu1 * omega_dist**2 + mu2 * gravity_err**2
    P_kin_scaled = P_kin / val_kin

    # Sample the ellipsoid in log space and push forward via exp map
    xi_points = rigid_body.sample_ellipsoid_boundary(P_kin_scaled, n=num_points)
    eta_points = rigid_body.expmap(xi_points)

    eta_min = np.min(eta_points, axis=1)
    eta_max = np.max(eta_points, axis=1)
    bounds = np.column_stack([eta_min, eta_max])

    return xi_points, eta_points, bounds, kinematics_sol
