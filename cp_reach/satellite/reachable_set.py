
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

import cp_reach.lie.SE23 as SE23
import cp_reach.physics.angular_acceleration  as angular_acceleration
import cp_reach.physics.rigid_body as rigid_body
import cp_reach.physics.flowpipe as flowpipe
#import cp_reach.sim.multirotor_control as mr_control
import cp_reach.sim.multirotor_plan as mr_plan
import cp_reach.sim.multirotor_control as mr_control



def disturbance(gyro_disturbance, ref=None):
    """
    Compute disturbance-based reachable sets for a quadrotor.

    Inputs:
        quadrotor : dict
            Must contain keys 'thrust_disturbance' and 'gyro_disturbance'.
        ref : optional, if None will generate a reference trajectory using mr_plan.traj_3()

    Returns:
        inv_points       : (6, N) points in SE(3) group (position + orientation)
        points_algebra   : (3, N) points from dynamic ellipsoid in angular velocity space
        lower_bound      : (6,) vector of component-wise minima in SE(3)
        upper_bound      : (6,) vector of component-wise maxima in SE(3)
        kinematics_sol   : (sol)
    """
    # Get disturbances
    w_accel = quadrotor['thrust_disturbance']  # affects translational motion
    w_angular_accel = quadrotor['gyro_disturbance']    # affects angular velocity

    # Load reference trajectory if not provided
    if ref is None:
        ref = mr_plan.traj_3()

    # Extract peak accelerations and angular velocities
    ax = [np.max(np.abs(ref['ax']))]
    ay = [np.max(np.abs(ref['ay']))]
    az = [np.max(9.8 - np.min(ref['az']))]  # compensate gravity-like term
    omega1 = [np.max(np.abs(ref['omega1']))]
    omega2 = [np.max(np.abs(ref['omega2']))]
    omega3 = [np.max(np.abs(ref['omega3']))]

    # --- Inner Loop: Dynamics (angular motion, 6x6 Lyapunov)
    dynamics_sol, omega_bound = angular_acceleration.bound_dynamics(omega1, omega2, omega3, w)
    dynamics_P1 = dynamics_sol['P'] / (dynamics_sol['mu1'] * w2 ** 2)
    points_algebra = angular_acceleration.obtain_points(dynamics_P1)

    # --- Outer Loop: Kinematics (SE(2,3), 9x9 Lyapunov)
    # kinematics_sol = outer_bound.find_se23_invariant_set(ax, ay, az, omega1, omega2, omega3)
    # val = kinematics_sol['mu2'] * w1**2 + kinematics_sol['mu3'] * omega_bound**2
    # kinematics_P1 = kinematics_sol['P'] / val

    # # Project ellipsoid in se(2,3) onto relevant subspaces
    # translation_points, _ = outer_bound.project_ellipsoid_subspace(kinematics_P1, [0,1,2])
    # velocity_points, _    = outer_bound.project_ellipsoid_subspace(kinematics_P1, [3,4,5])
    # rotation_points, _    = outer_bound.project_ellipsoid_subspace(kinematics_P1, [6,7,8])

    # # Exponential map from se(3) to SE(3)
    # inv_points = outer_bound.exp_map(translation_points, rotation_points)

    # # Extract bounding box of reachable set in SE(3)
    # lower_bound = inv_points.min(axis=1)
    # upper_bound = inv_points.max(axis=1)

    # return inv_points, points_algebra, lower_bound, upper_bound, kinematics_sol, omega_bound
    return points_algebra


def plot2DInvSet(points, inv_points, ax1):
    ax1.plot(points[0, :], points[1, :], 'g', label='with Dynamic Inversion')
    ax1.set_xlabel('$\\zeta_x$, m')
    ax1.set_ylabel('$\\zeta_y$, m')

    # ax2 = plt.subplot(122)
    # ax2.plot(inv_points[0, :-1], inv_points[1, :-1], 'g', label='with Dynamic Inversion')
    
    # ax2.set_xlabel('$\\eta_x$, m')
    # ax2.set_ylabel('$\\eta_y$, m')
    # plt.grid(True)

    # plt.axis('equal')
    # plt.tight_layout()
    ax1.set_title('Invariant Set in Lie Algebra', fontsize=20)
    # ax2.set_title('Invariant Set in Lie Group', fontsize=20)

def plot3DInvSet(points, inv_points):
    plt.figure(figsize=(14,7))
    ax1 = plt.subplot(121, projection='3d', proj_type='ortho', elev=40, azim=20)
    ax1.plot3D(points[0, :], points[1, :], points[2, :],'g', label='with Dynamic Inversion')
    ax1.set_xlabel('$\\zeta_x$, m')
    ax1.set_ylabel('$\\zeta_y$, m')
    ax1.set_zlabel('$\\zeta_z$, rad', labelpad=1)
    ax1.set_title('Invariant Set in Lie Algebra', fontsize=20)

    plt.axis('auto')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    ax2 = plt.subplot(122, projection='3d', proj_type='ortho', elev=40, azim=20)

    ax2.plot3D(inv_points[0, :], inv_points[1, :], inv_points[2, :], 'g', label='with Dynamic Inversion')

    ax2.set_xlabel('$\\eta_x$, m')
    ax2.set_ylabel('$\\eta_y$, m')
    ax2.set_zlabel('$\\eta_z$, rad')
    ax2.set_title('Invariant Set in Lie Group', fontsize=20)
    plt.axis('auto')
    plt.subplots_adjust(left=0.45, right=1, top=0.5, bottom=0.08)

    plt.tight_layout()