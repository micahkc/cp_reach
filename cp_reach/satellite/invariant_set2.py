
import numpy as np
import casadi as ca
# import cp_reach.physics.angular_acceleration  as angular_acceleration
import cp_reach.physics.coupled_dynamics as coupled_dynamics
# from cyecca.lie.group_se23 import se23, SE23Quat  # or SE23Mrp

def solve(ang_vel_dist, ref_acceleration, pid_values, sol=None, num_points= 720):
    """
    Compute over approximation of reachable sets for a satellite under bounded disturbances in translational acceleration
    and angular acceleration. This function over-approximates the reachable sets in both angular velocity
    space and SE(3) (position + orientation) space using Lyapunov-based ellipsoidal bounds.

    Parameters:
        accel_dist (float):
            Upper bound on translational acceleration disturbance (m/s²), e.g., from wind or thrust noise.
        
        ang_accel_dist (float):
            Upper bound on angular acceleration disturbance (rad/s^2), e.g., from torque uncertainty.

        ref (dict):
            Dictionary containing reference trajectories with keys:
            'ax', 'ay', 'az', 'omega1', 'omega2', 'omega3'.

        dynamics_sol (dict, optional):
            Precomputed solution to dynamics-level Lyapunov LMI.

        sol (dict, optional):
            Precomputed solution to kinematics-level Lyapunov LMI.

    Returns:
        ang_vel_points    : (3, N) ndarray
            Reachable set points in angular velocity space due to dynamic disturbance.

        lower_bound_omega : (3,) ndarray
            Lower bound of angular velocity reachable set.

        upper_bound_omega : (3,) ndarray
            Upper bound of angular velocity reachable set.

        omega_dist        : float
            Derived worst-case angular velocity magnitude used in the kinematic bound.

        dynamics_sol      : dict
            Solution to the Lyapunov LMI for angular dynamics (contains P, mu1, etc.).

        inv_points        : (6, N) ndarray
            Reachable set points in SE(3) from translational and rotational disturbances.

        lower_bound       : (6,) ndarray
            Lower bound of SE(3) reachable set.

        upper_bound       : (6,) ndarray
            Upper bound of SE(3) reachable set.

        sol    : dict
            Solution to the Lyapunov LMI for SE(3) kinematics (contains P, mu2, mu3, etc.).
    """
    # PID values:
    kp, kd, kpq, kdq = pid_values

    # Puts upper bound on position, velocity, attitude, and angular velocity error.
    gravity_err = 0

    if sol is None:
        sol = coupled_dynamics.solve_se23_invariant_set(ref_acceleration, kp, kd, kpq, kdq, ang_vel_dist, gravity_err)

    mu1 = sol['mu1'] # ang velocity disturbance
    mu2 = sol['mu2'] # gravity disturbance
    P_kin = sol['P']
    val_kin = mu1 * ang_vel_dist**2 + mu2 * gravity_err**2
    P_kin_scaled = P_kin / val_kin

    P9 = coupled_dynamics.project_ellipsoid_matrix(P_kin_scaled, [0,1,2,3,4,5,6,7,8])
    P3 = coupled_dynamics.project_ellipsoid_matrix(P_kin_scaled, [9,10,11])

    # xi_points = coupled_dynamics.walk_ellipsoid_planes(P9, n=num_points)
    xi_points = coupled_dynamics.sample_ellipsoid_surface(P9, n=num_points)
    
    omega_points = coupled_dynamics.walk_ellipsoid_planes(P3, n = 1000)
    # xi_points = coupled_dynamics.sample_ellipsoid_boundary(P_kin_scaled, n=num_points)  # e.g., 20 per 2-plane

    eta_points = coupled_dynamics.expmap(xi_points)

    
    eta_min = np.min(eta_points, axis=1)   # shape (d,)
    eta_max = np.max(eta_points, axis=1)   # shape (d,)
    omega_min = np.min(omega_points, axis=1)
    omega_max = np.max(omega_points, axis=1)

    bounds_eta = np.column_stack([eta_min, eta_max])
    bounds_omega = np.column_stack([omega_min, omega_max])

    bounds = np.vstack([bounds_eta, bounds_omega])



    return (
        xi_points,
        eta_points,
        omega_points,
        bounds,
        sol
    )
