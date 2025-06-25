
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

    # Maximum translational and angular acceleration in ref trajectory.
    ax = [np.max(ref['ax'])]
    ay = [np.max(ref['ay'])]
    az = [-np.min(ref['az'])+9.8]
    omega1 = [np.max(ref['omega1'])]
    omega2 = [np.max(ref['omega2'])]
    omega3 = [np.max(ref['omega3'])]

    sol, max_BK = inner_bound.find_omega_invariant_set(omega1, omega2, omega3)
    # max_BK is the maximum eigenvalue of BK
    mu_inner = sol['mu1']

    # Initial condition
    P = sol['P']
    e0 = np.array([0,0,0]) # initial error
    beta = (e0.T@P@e0) # initial Lyapnov value

    # find bound
    omegabound = inner_bound.omega_bound(omega1, omega2, omega3, w2, beta) #traj_3 result for inner bound
    print(omegabound)
    # Translational (ax,ay,az) LMI.
    sol_LMI = outer_bound.find_se23_invariant_set(ax, ay, az, omega1, omega2, omega3)
    mu_outer = sol_LMI['mu3']

    # Initial condition
    e = np.array([0,0,0,0,0,0,0,0,0]) # initial error in Lie group (nonlinear)

    # transfer initial error to Lie algebra (linear)
    e0 = ca.DM(SE23.SE23Dcm.vee(SE23.SE23Dcm.log(SE23.SE23Dcm.matrix(e))))
    e0 = np.array([e0]).reshape(9,)
    ebeta = e0.T@sol_LMI['P']@e0

    print('finding invariant set')
    # find invairant set points in Lie algebra (linear)
    points, val = outer_bound.se23_invariant_set_points(sol_LMI, 20, w1, omegabound, ebeta)
    points_theta, val = outer_bound.se23_invariant_set_points_theta(sol_LMI, 20, w1, omegabound, ebeta)

    # map invariant set points to Lie group (nonlinear)
    inv_points = outer_bound.exp_map(points, points_theta)

    mu_total = (mu_outer*mu_inner)*max_BK

    return inv_points, mu_total, points, points_theta, ebeta, omegabound, sol_LMI

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