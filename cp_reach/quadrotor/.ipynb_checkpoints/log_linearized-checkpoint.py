
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

import cp_reach.lie.SE23 as SE23
import cp_reach.flowpipe.inner_bound  as inner_bound
import cp_reach.flowpipe.outer_bound as outer_bound
import cp_reach.flowpipe.flowpipe as flowpipe
#import cp_reach.sim.multirotor_control as mr_control
import cp_reach.sim.multirotor_plan as mr_plan
import cp_reach.sim.multirotor_control as mr_control



def disturbance(quadrotor, ref):
    # Get disturbance from quadrotor configuration file.
    w1 = quadrotor['thrust_disturbance'] # disturbance for translational (impact a)  thrust disturbance for outer loop
    w2 = quadrotor['gyro_disturbance'] # disturbance for angular (impact alpha)  inner loop angular disturbance BKd

    # Obtain Reference Trajectory
    ref = mr_plan.traj_3()

    # Maximum translational acceleration and angular velocity in ref trajectory.
    ax = [np.max(ref['ax'])]
    ay = [np.max(ref['ay'])]
    az = [-np.min(ref['az'])+9.8]
    omega1 = [np.max(ref['omega1'])]
    omega2 = [np.max(ref['omega2'])]
    omega3 = [np.max(ref['omega3'])]

    # Dynamics. omega_bound is the bound on angular speed. It will feed into the Kinematics LMI.
    dynamics_P, dynamics_mu, omega_bound, max_BK = inner_bound.bound_dynamics(omega1, omega2, omega3, w2)


    # Kinematics. Find reachable set in the lie algebra for se23. This is 9 dimensional
    kinematics_sol = outer_bound.find_se23_invariant_set(ax, ay, az, omega1, omega2, omega3)
    val = kinematics_sol['mu2'] * w1**2 + kinematics_sol['mu3'] * omega_bound**2
    P_normalized = kinematics_sol['P'] / val
    #x.T(P/val)x = 1 defines an ellipsoid in a 9 dimensional space

    # Project to get reachable sets for translation, velocity, and angle.
    translation_points, _ = outer_bound.project_ellipsoid_subspace(P_normalized, [1,2,3])
    velocity_points, _ = outer_bound.project_ellipsoid_subspace(P_normalized, [4,5,6])
    rotation_points, _ = outer_bound.project_ellipsoid_subspace(P_normalized, [7,8,9])

    # Map invariant set points in Lie Algebra to the corresponding Lie Group. This is the true reachable set.
    inv_points = outer_bound.exp_map(translation_points, rotation_points)
    # find the min and max of each component. [x,y,z,thetax,thetay,thetaz]
    lower_bound = inv_points.min(axis=1)
    upper_bound = inv_points.max(axis=1)

    return inv_points, points, points_theta, omega_bound, kinematics_sol

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