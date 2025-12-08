import numpy as np

import cp_reach.physics.angular_acceleration as angular_acceleration
import cp_reach.physics.rigid_body as rigid_body
import cp_reach.lmi as lmi


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
      - the reachable set in SE(3) (position + orientation, kinematics level),

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
        Precomputed solution to SE(3) kinematics Lyapunov LMI.
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
            "se3": {
                "points":       (6, N) ndarray,
                "bounds":       (6, 2) ndarray,  # [:,0]=lower, [:,1]=upper
                "lmi_solution": dict,
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
        # Optional: build closed-loop A_cl, B_d using PID or LQR (if you need them)
        if pid_values is not None:
            Ac_l = angular_acceleration.derive_Acl(pid_values)
            B_d = angular_acceleration.dervice_Bd(pid_values)
        else:
            Ac_l = angular_acceleration.derive_Acl_LQR(omega_max)
            B_d = angular_acceleration.dervice_Bd_LQR(omega_max)

        dynamics_sol = LMI.solve_inv_set(Ac_l, B_d, mu_n=3)

    # Find invariant set for disturbances
    mu1_dyn = dynamics_sol["mu1"]
    P_dyn = dynamics_sol["P"]

    # Scale so that ellipsoid is {x: x^T P_dyn_scaled x <= 1}
    val_dyn = mu1_dyn * ang_accel_dist**2
    P_dyn_scaled = P_dyn / val_dyn
    r_dyn = np.sqrt(val_dyn)

    ang_vel_points, angular_bounds, omega_dist = lmi.ellipsoids.ellipsoid_bounds_and_radius(
        P_dyn_scaled,
        r_dyn,
        angular_acceleration.obtain_points
    )

    # ===================== Kinematics-level (SE(3)) =====================
    if kinematics_sol is None:
        if pid_values is not None:
            Ac_l = rigid_body.derive_Acl(pid_values)
            B_d = rigid_body.dervice_Bd(pid_values)
        else:
            Ac_l = rigid_body.derive_Acl_LQR(acc_max, omega_max)
            B_d = rigid_body.dervice_Bd_LQR(acc_max, omega_max)
            
        kinematics_sol = rigid_body.solve_se23_invariant_set(
            acc_max,
            omega_max
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

    # Project ellipsoid from se(2,3) subspaces:
    translation_pts, _ = rigid_body.project_ellipsoid_subspace(
        P_kin_scaled, [0, 1, 2]
    )
    # velocity_pts, _ = rigid_body.project_ellipsoid_subspace(
    #     P_kin_scaled, [3, 4, 5]
    # )
    rotation_pts, _ = rigid_body.project_ellipsoid_subspace(
        P_kin_scaled, [6, 7, 8]
    )

    # Map to SE(3) via the exponential map
    inv_points = rigid_body.exp_map(translation_pts, rotation_pts)

    # Axis-aligned bounding box in SE(3)
    lower_bound_se3 = np.min(inv_points, axis=1)
    upper_bound_se3 = np.max(inv_points, axis=1)

    # Pack SE(3) bounds into a single array: (6, 2)
    se3_bounds = np.stack([lower_bound_se3, upper_bound_se3], axis=1)

    return {
        "angular": {
            "points": ang_vel_points,
            "bounds": angular_bounds,   # shape (3, 2)
            "omega_dist": omega_dist,
            "lmi_solution": dynamics_sol,
        },
        "se3": {
            "points": inv_points,
            "bounds": se3_bounds,       # shape (6, 2)
            "lmi_solution": kinematics_sol,
        },
    }
