
import numpy as np
import casadi as ca
import cp_reach.physics.angular_acceleration  as angular_acceleration
import cp_reach.physics.rigid_body as rigid_body
# from cyecca.lie.group_se23 import se23, SE23Quat  # or SE23Mrp

def solve(ang_vel_dist, ref_acceleration, pid_values, dynamics_sol=None, kinematics_sol=None, num_points= 720):
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

        kinematics_sol (dict, optional):
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

        kinematics_sol    : dict
            Solution to the Lyapunov LMI for SE(3) kinematics (contains P, mu2, mu3, etc.).
    """
    # PID values:
    kp, kd, kpq, kdq = pid_values

    # === Dynamics-level Lyapunov (angular motion) ===
    # Puts upper bound on reachable set of omega (angular velocity) error.
    if dynamics_sol is None:
        dynamics_sol = angular_acceleration.solve_inv_set(kdq, verbosity=0)

    # Get bounds from inner loop solution
    mu1 = dynamics_sol['mu1']
    P_dyn = dynamics_sol['P']
    val_dyn = mu1 * ang_vel_dist**2
    P_dyn_scaled = P_dyn / val_dyn
    
    ang_vel_points = angular_acceleration.obtain_points(P_dyn_scaled)
    lower_bound_omega = np.min(ang_vel_points, axis=1)
    upper_bound_omega = np.max(ang_vel_points, axis=1)

    # Infinity norm-based overapproximation of angular velocity error.
    # Transform P to unit ball to find worst-case angular velocity error.
    r = np.sqrt(val_dyn)
    eigvals, eigvecs = np.linalg.eig(P_dyn)
    R = np.real(eigvecs @ np.diag(1 / np.sqrt(eigvals)))
    omega_dist = r * np.max(np.linalg.norm(R, axis=1))



    
    # === Kinematics-level Lyapunov (SE(3)) ===
    # Puts upper bound on position, velocity, and attitude error.
    gravity_err = 0

    if kinematics_sol is None:
        kinematics_sol = rigid_body.solve_se23_invariant_set_log_control(ref_acceleration, kp, kd, kpq, omega_dist, gravity_err)

    mu1 = kinematics_sol['mu1'] # ang acceleration disturbance
    mu2 = kinematics_sol['mu2'] # gravity disturbance
    P_kin = kinematics_sol['P']
    val_kin = mu1 * omega_dist**2 + mu2 * gravity_err**2
    P_kin_scaled = P_kin / val_kin


    # Go through points around ellipsoid and take exponential map
    xi_points = rigid_body.sample_ellipsoid_boundary(P_kin_scaled, n=num_points)  # e.g., 20 per 2-plane
    eta_points = rigid_body.expmap(xi_points)

    lower_bound = np.min(eta_points, axis=1)
    upper_bound = np.max(eta_points, axis=1)


    return (
        ang_vel_points,
        lower_bound_omega,
        upper_bound_omega,
        omega_dist,
        dynamics_sol,
        eta_points,
        lower_bound,
        upper_bound,
        kinematics_sol,
    )
