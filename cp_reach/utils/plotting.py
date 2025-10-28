import numpy as np
import matplotlib.pyplot as plt
from cp_reach.physics.rigid_body import project_ellipsoid_subspace, exp_map

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



def rotate_point(points, angle):
    """
    Rotate a batch of 2D points counterclockwise by `angle` radians.

    Args:
        points: np.ndarray of shape (2, N)
        angle: float (radians)

    Returns:
        np.ndarray of shape (2, N): rotated points
    """
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return R @ points

def flowpipes(ref, step, vel_dist, accel_dist, omega_dist, sol, axis):
    """
    Compute flowpipes for a nonlinear system on SE(3) using LMI-based invariant set.
    Assumes:
        - No initial deviation (beta = 0)
        - t = ∞, i.e., steady-state invariant set
        - Visualization in 2D (xy or xz plane)

    Parameters:
        ref         : dict with keys 'x', 'y', 'z' (nominal trajectory)
        step        : step size
        accel_dist          : linear disturbance bound (scalar or vector)
        omega_dist  : angular velocity disturbance bound (scalar or vector)
        sol         : dict from SE23LMIs (contains 'P', 'mu1', 'mu2', 'mu3')
        axis        : 'xy' or 'xz' — determines 2D projection

    Returns:
        flowpipes     : list of np.ndarray polygons (reachable set per segment)
        intervalhull  : list of np.ndarray rectangles (bounding boxes of nominal path)
        nom           : 2D projected nominal trajectory
    """
    x_r, y_r, z_r = ref['x'], ref['y'], ref['z']

    # Choose projection direction
    if axis == 'xy':
        nom = np.vstack((x_r, y_r)).T
        proj_indices = [0, 1]  # use x, y
    elif axis == 'xz':
        nom = np.vstack((x_r, z_r)).T
        proj_indices = [0, 2]  # use x, z
    else:
        raise ValueError("axis must be 'xy' or 'xz'.")

    # Compute steady-state bound for V(∞)
    mu1, mu2, mu3 = sol['mu1'], sol['mu2'], sol['mu3']
    P = sol['P']
    val = mu1 * (np.linalg.norm(vel_dist)**2) + mu2 * (np.linalg.norm(accel_dist)**2) + mu3 * (np.linalg.norm(omega_dist)**2) + 0.01

    # Project Lyapunov ellipsoid into translational and rotational subspaces
    points, _ = project_ellipsoid_subspace(P / val, [0, 1, 2])
    points_theta, _ = project_ellipsoid_subspace(P / val, [6, 7, 8])

    # Map to SE(3) using exponential map
    inv_points = exp_map(points, points_theta)  # shape: (6, N)

    # Keep only translation part for 2D projection
    position = inv_points[0:3, :]               # x, y, z
    inv_points_2d = position[proj_indices, :]   # select x-y or x-z

    # Rotate invariant set to approximate convex hull
    inv_set = [[], []]
    for theta in np.linspace(0, np.pi, 10):
        inv_set1 = rotate_point(inv_points_2d, theta)  # shape: (2, N)
        inv_set = np.append(inv_set, inv_set1, axis=1)

    inv_poly = inv_set.T  # shape: (N, 2)


    flowpipes = []
    for i in range(0, len(nom)-1, step):
        point = nom[i]
        tangent = nom[i+1] - nom[i]
        direction = tangent / np.linalg.norm(tangent)
        normal = np.array([-direction[1], direction[0]])

        R = np.array([[normal[0], -normal[1]],
                    [normal[1],  normal[0]]])

        rotated_inv = inv_poly @ R.T
        translated = rotated_inv + point
        translated = np.vstack((translated, translated[0]))  # close polygon

        flowpipes.append(translated)


    return flowpipes, nom



def plot_flowpipes(nom, flowpipes, ax, axis='xy'):
    """
    Plot nominal trajectory and flowpipes on a given matplotlib axis.

    Parameters:
        nom        : (N, 2) array of nominal trajectory points
        flowpipes  : list of (M_i, 2) arrays representing reachable sets
        ax         : matplotlib Axes object to draw on
        axis       : 'xy' or 'xz' — used to label y-axis
    """
    # Plot flowpipes
    for i, itm in enumerate(flowpipes):
        ax.plot(itm[:, 0], itm[:, 1], 'c--', label='Flow Pipe' if i == 0 else None)

    # Plot nominal trajectory
    ax.plot(nom[:, 0], nom[:, 1], 'k-', label='Reference Trajectory')

    ax.set_title('Flowpipes')
    ax.set_xlabel('x')
    ax.set_ylabel('y' if axis == 'xy' else 'z')
    ax.axis('equal')
    ax.grid(True)

    # Avoid duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, fontsize=12, loc='upper left')