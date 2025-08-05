
import numpy as np

import cp_reach.physics.angular_acceleration  as angular_acceleration
import cp_reach.physics.rigid_body as rigid_body


def solve(accel_dist, ang_accel_dist, ref, dynamics_sol=None, kinematics_sol=None):
    """
    Compute reachable sets for a quadrotor under bounded disturbances in translational acceleration
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


    # Extract peak disturbances from reference
    ax_max = [np.max(np.abs(ref['ax']))]
    ay_max = [np.max(np.abs(ref['ay']))]
    az_max = [np.max(9.8 - np.min(ref['az']))]  # gravity compensation

    omega1_max = [np.max(np.abs(ref['omega1']))]
    omega2_max = [np.max(np.abs(ref['omega2']))]
    omega3_max = [np.max(np.abs(ref['omega3']))]

    # === Dynamics-level Lyapunov (angular motion) ===
    if dynamics_sol is None:
        dynamics_sol = angular_acceleration.solve_inv_set(
            omega1_max, omega2_max, omega3_max
        )

    mu1 = dynamics_sol['mu1']
    P_dyn = dynamics_sol['P']
    val_dyn = mu1 * ang_accel_dist**2
    P_dyn_scaled = P_dyn / val_dyn
    r = np.sqrt(val_dyn)

    # Transform P to unit ball to find worst-case angular velocity
    eigvals, eigvecs = np.linalg.eig(P_dyn)
    R = np.real(eigvecs @ np.diag(1 / np.sqrt(eigvals)))
    ang_vel_points = angular_acceleration.obtain_points(P_dyn_scaled)
    lower_bound_omega = np.min(ang_vel_points, axis=1)
    upper_bound_omega = np.max(ang_vel_points, axis=1)

    # Infinity norm-based overapproximation of angular velocity
    omega_dist = r * np.max(np.linalg.norm(R, axis=1))

    # === Kinematics-level Lyapunov (SE(3)) ===
    if kinematics_sol is None:
        kinematics_sol = rigid_body.solve_se23_invariant_set(
            ax_max, ay_max, az_max, omega1_max, omega2_max, omega3_max, mode=0
        )

    mu2 = kinematics_sol['mu2']
    mu3 = kinematics_sol['mu3']
    P_kin = kinematics_sol['P']
    val_kin = mu2 * accel_dist**2 + mu3 * omega_dist**2
    P_kin_scaled = P_kin / val_kin

    # Project ellipsoid from se(2,3) Lie algebra
    translation_pts, _ = rigid_body.project_ellipsoid_subspace(P_kin_scaled, [0, 1, 2])
    velocity_pts, _                = rigid_body.project_ellipsoid_subspace(P_kin_scaled, [3, 4, 5])  # unused but consistent
    rotation_pts, _     = rigid_body.project_ellipsoid_subspace(P_kin_scaled, [6, 7, 8])

    # Map to SE(3) using exponential map
    inv_points = rigid_body.exp_map(translation_pts, rotation_pts)

    # Axis-aligned bounding box in SE(3)
    lower_bound = np.min(inv_points, axis=1)
    upper_bound = np.max(inv_points, axis=1)

    return (
        ang_vel_points,
        lower_bound_omega,
        upper_bound_omega,
        omega_dist,
        dynamics_sol,
        inv_points,
        lower_bound,
        upper_bound,
        kinematics_sol,
    )
