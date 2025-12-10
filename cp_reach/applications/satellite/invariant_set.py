"""
Simplified satellite invariant set computation using single-level log-linearization.

This is a simplified variant that uses log-linearization on SE(2,3) directly,
bypassing the 2-level angular dynamics + kinematics approach of invariant_set.py.

This version is faster and used in recent research notebooks (satellite-outer2/3.ipynb).
For the canonical 2-level API, see invariant_set.py.
"""

import numpy as np
import cp_reach.physics.rigid_body as rigid_body
import time

def solve(accel_dist, ref_acceleration, pid_values, num_points=720):
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
    t0 = time.perf_counter()
    kp, kd, kpq, kdq = pid_values  # kdq kept for interface symmetry

    kinematics_sol = rigid_body.solve_se23_invariant_set_log_control_simple(
            ref_acceleration, kp, kd, kpq, accel_dist
        )
    t1 = time.perf_counter()
    print(t1-t0)
    mu2 = kinematics_sol['mu']  # acceleration disturbance multiplier

    P_kin = kinematics_sol['P']

    val_kin = mu2 * accel_dist**2 
    P_kin_scaled = P_kin / val_kin

    t2 = time.perf_counter()
    print(t2-t0)
    # Sample the ellipsoid in log space and push forward via exp map
    xi_points = rigid_body.sample_ellipsoid_boundary(P_kin_scaled, n=num_points)

    eta_points = rigid_body.expmap(xi_points)

    eta_min = np.min(eta_points, axis=1)
    eta_max = np.max(eta_points, axis=1)
    bounds = np.column_stack([eta_min, eta_max])
    t3 = time.perf_counter()
    print(t3-t0)

    return xi_points, eta_points, bounds, kinematics_sol
