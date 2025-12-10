import numpy as np

import cp_reach.physics.angular_acceleration as angular_acceleration
import cp_reach.physics.rigid_body as rigid_body
from cp_reach.reachability import ellipsoids


def solve_nested(
    vel_dist,
    accel_dist,
    ang_accel_dist,
    ref,
    dynamics_sol=None,
    kinematics_sol=None,
    pid_values=None,
):
    """
    Compute reachable sets for a quadrotor under bounded disturbances in
    translational acceleration and angular acceleration.

    The method over-approximates:
      - the reachable set in angular velocity space (dynamics level), and
      - the reachable set in SE(2,3) (position + velocity + orientation, kinematics level),

    using Lyapunov-based ellipsoidal bounds.

    Parameters
    ----------
    vel_dist : float
        Upper bound on velocity disturbance (m/s).
    accel_dist : float
        Upper bound on translational acceleration disturbance (m/s^2).
    ang_accel_dist : float
        Upper bound on angular acceleration disturbance (rad/s^2).
    ref : dict or None
        Reference trajectory data with keys 'ax', 'ay', 'az',
        'omega1', 'omega2', 'omega3'. If None, uses hover-like defaults.
    dynamics_sol : dict, optional
        Precomputed solution to angular dynamics Lyapunov LMI.
    kinematics_sol : dict, optional
        Precomputed solution to SE(2,3) kinematics Lyapunov LMI.
    pid_values : optional
        Controller parameters (if you want to use PID-based A_cl/B_d construction).

    Returns
    -------
    dict
        {
            "angular": {
                "points":       (3, N) ndarray,
                "bounds":       (3, 2) ndarray,  # [:,0]=lower, [:,1]=upper
                "omega_dist":   float,
                "lmi_solution": dict,
            },
            "se23": {
                "points_algebra": (9, N) ndarray,  # Lie algebra (se23) coordinates
                "points_group":   (9, N) ndarray,  # Lie group (SE23) coordinates
                "bounds_algebra": (9, 2) ndarray,  # Lie algebra bounds
                "bounds_group":   (9, 2) ndarray,  # Lie group bounds
                "lmi_solution":   dict,
            },
        }
    """
    g = 9.8  # gravity

    # ===================== Reference-derived bounds =====================
    if ref is None:
        # Hover-like defaults
        acc_max = [0.0, 0.0, g]        # ax, ay, az
        omega_max = [0.0, 0.0, 0.0]    # ω1, ω2, ω3
    else:
        # Translational acceleration bounds
        ax_max = float(np.max(np.abs(ref["ax"])))
        ay_max = float(np.max(np.abs(ref["ay"])))
        az_max = float(np.max(g - np.min(ref["az"])))  # thrust deviation from gravity

        # Angular velocity bounds
        omega1 = float(np.max(np.abs(ref["omega1"])))
        omega2 = float(np.max(np.abs(ref["omega2"])))
        omega3 = float(np.max(np.abs(ref["omega3"])))

        acc_max = [ax_max, ay_max, az_max]
        omega_max = [omega1, omega2, omega3]

    # ===================== Dynamics-level (angular) =====================
    if dynamics_sol is None:
        # Solve for angular velocity invariant set
        # The angular acceleration module has solve_inv_set which takes Kdq (proportional gain)
        # Use a default proportional gain for angular velocity control
        Kdq = 10.0  # Default proportional gain for angular velocity control
        dynamics_sol = angular_acceleration.solve_inv_set(Kdq)

    # Find invariant set for disturbances
    mu1_dyn = dynamics_sol["mu1"]
    P_dyn = dynamics_sol["P"]

    # Scale so that ellipsoid is {x: x^T P_dyn_scaled x <= 1}
    val_dyn = mu1_dyn * ang_accel_dist**2
    P_dyn_scaled = P_dyn / val_dyn
    r_dyn = np.sqrt(val_dyn)

    ang_vel_points, angular_bounds, omega_dist = ellipsoids.ellipsoid_bounds_and_radius(
        P_dyn_scaled,
        r_dyn,
        angular_acceleration.obtain_points
    )

    # ===================== Kinematics-level (SE(3)) =====================
    if kinematics_sol is None:
        # solve_se23_invariant_set expects ranges for ax, ay, az, omega1, omega2, omega3
        # We'll use a simple grid around the reference values
        # ax_range = [-acc_max[0], acc_max[0]] if acc_max[0] > 0 else [0.0]
        # ay_range = [-acc_max[1], acc_max[1]] if acc_max[1] > 0 else [0.0]
        # az_range = [acc_max[2] * 0.9, acc_max[2] * 1.1] if acc_max[2] > 0 else [0.0, g]

        ax_range = [acc_max[0]]
        ay_range = [acc_max[1]]
        az_range = [acc_max[2]]

        omega1_range = [omega_max[0]] 
        omega2_range = [omega_max[1]] 
        omega3_range = [omega_max[2]] 

        kinematics_sol = rigid_body.solve_se23_invariant_set(
            ax_range, ay_range, az_range,
            omega1_range, omega2_range, omega3_range,
            mode=0  # Quadrotor mode
        )

    mu1_kin = kinematics_sol["mu1"]
    mu2_kin = kinematics_sol["mu2"]
    mu3_kin = kinematics_sol["mu3"]
    P_kin = kinematics_sol["P"]

    # Combined disturbance energy bound
    val_kin = (
        mu1_kin * vel_dist**2
        + mu2_kin * accel_dist**2
        + mu3_kin * omega_dist**2
    )
    P_kin_scaled = P_kin / val_kin

    # Sample ellipsoid boundary points in Lie algebra (se(2,3))
    num_points = 500
    xi_points = rigid_body.sample_ellipsoid_boundary(P_kin_scaled, n=num_points)

    # Map to Lie group (SE(2,3)) via exponential map
    eta_points = rigid_body.expmap(xi_points)

    # Compute axis-aligned bounds in Lie group
    eta_min = np.min(eta_points, axis=1)
    eta_max = np.max(eta_points, axis=1)
    eta_bounds = np.column_stack([eta_min, eta_max])

    # Also compute bounds in Lie algebra for reference
    xi_min = np.min(xi_points, axis=1)
    xi_max = np.max(xi_points, axis=1)
    xi_bounds = np.column_stack([xi_min, xi_max])

    # # OLD METHOD (kept for reference):
    # # Project ellipsoid from se(2,3) subspaces:
    # translation_pts, _ = rigid_body.project_ellipsoid_subspace(
    #     P_kin_scaled, [0, 1, 2]
    # )
    # rotation_pts, _ = rigid_body.project_ellipsoid_subspace(
    #     P_kin_scaled, [6, 7, 8]
    # )
    # # Map to SE(3) via the exponential map
    # inv_points = rigid_body.exp_map(translation_pts, rotation_pts)
    # # Axis-aligned bounding box in SE(3)
    # lower_bound_se3 = np.min(inv_points, axis=1)
    # upper_bound_se3 = np.max(inv_points, axis=1)
    # se3_bounds = np.stack([lower_bound_se3, upper_bound_se3], axis=1)

    return {
        "angular": {
            "points": ang_vel_points,
            "bounds": angular_bounds,   # shape (3, 2)
            "omega_dist": omega_dist,
            "lmi_solution": dynamics_sol,
        },
        "se23": {
            "points_algebra": xi_points,    # shape (9, N) - Lie algebra (se23) coordinates
            "points_group": eta_points,     # shape (9, N) - Lie group (SE23) coordinates
            "bounds_algebra": xi_bounds,    # shape (9, 2) - bounds in Lie algebra
            "bounds_group": eta_bounds,     # shape (9, 2) - bounds in Lie group
            "lmi_solution": kinematics_sol,
        },
    }
