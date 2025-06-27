import numpy as np
import cvxpy as cp
import control
import itertools
import scipy
from cp_reach.lie.SE23 import *
from cp_reach.lie.se3 import *

def se23_solve_control(ax,ay,az,omega1,omega2,omega3):
    A = -ca.DM(SE23Dcm.ad_matrix(np.array([0,0,0,ax,ay,az,omega1,omega2,omega3]))+SE23Dcm.adC_matrix())
    B = np.array([[0,0,0,0], # vx
                  [0,0,0,0], # vy
                  [0,0,0,0], # vz
                  [0,0,0,0], # ax
                  [0,0,0,0], # ay
                  [1,0,0,0], # az
                  [0,1,0,0], # omega1
                  [0,0,1,0], # omega2
                  [0,0,0,1]]) # omega3 # control omega1,2,3, and az
    Q = 10*np.eye(9)  # penalize state
    R = 1*np.eye(4)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R) 
    K = -K # rescale K, set negative feedback sign
    BK = B@K
    return B, K, BK , A+B@K



def SE23LMIs(alpha, A_list, verbosity=0):
    """
    Solve Lyapunov LMI for SE(2,3) system using cvxpy.
    
    Parameters:
        alpha : float
            Lyapunov decay rate
        A_list : list of ndarray
            System matrices A_i to enforce Lyapunov inequality on
        B : ndarray
            Input matrix (not used directly here, but for compatibility)
        verbosity : int
            Logging level

    Returns:
        dict containing 'P', 'mu1', 'mu2', 'mu3', 'cost', 'alpha', and 'prob'
    """
    P = cp.Variable((9, 9), symmetric=True)
    mu1 = cp.Variable()
    mu2 = cp.Variable()
    mu3 = cp.Variable()
    gamma = mu1 + mu2 + mu3

    constraints = []

    # Block structure for Schur complement terms
    P1 = P[0:3, :]
    P2 = P[3:6, :]
    P3 = P[6:9, :]

    I3 = np.eye(3)
    zero33 = np.zeros((3,3))

    for Ai in A_list:
        LMI = cp.bmat([
            [Ai.T @ P + P @ Ai + alpha * P,   P1.T,             P2.T,             P3.T],
            [P1,                              -alpha * mu1 * I3, zero33,             zero33],
            [P2,                              zero33,             -alpha * mu2 * I3, zero33],
            [P3,                              zero33,             zero33,             -alpha * mu3 * I3]
        ])
        constraints.append(LMI << 0)

    # Enforce P >> I (numerically: P ≥ 1 * I)
    constraints += [
        P >> np.eye(9),
        mu1 >= 1e-5,
        mu2 >= 1e-5,
        mu3 >= 1e-5
    ]

    prob = cp.Problem(cp.Minimize(gamma), constraints)

    try:
        prob.solve(solver=cp.CVXOPT, verbose=(verbosity > 0))
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("LMI solver status: " + prob.status)

        return {
            'cost': gamma.value,
            'prob': prob,
            'mu1': mu1.value,
            'mu2': mu2.value,
            'mu3': mu3.value,
            'P': P.value,
            'alpha': alpha
        }

    except Exception as e:
        if verbosity > 0:
            print(f"Solver failed at alpha = {alpha:.4f}: {e}")
        return {
            'cost': np.inf,
            'prob': None,
            'mu1': None,
            'mu2': None,
            'mu3': None,
            'P': None,
            'alpha': alpha
        }


def find_se23_invariant_set(ax_range, ay_range, az_range, omega1_range, omega2_range, omega3_range, verbosity=0):
    """
    Computes a matrix P and mu1 for the SE(2,3) system 
    using a grid over angular and linear acceleration ranges.

    Parameters:
        ax_range, ay_range, az_range : iterables over linear acceleration inputs
        omega1_range, omega2_range, omega3_range : iterables over angular velocity inputs
        verbosity : int, print detail level

    Returns:
        sol : dict containing keys 'P', 'mu_1', 'alpha', 'cost', 'prob'
    """
    # Generate all combinations of [omega1, omega2, omega3, ax, ay, az]
    grid = itertools.product(omega1_range, omega2_range, omega3_range, ax_range, ay_range, az_range)
    A_matrices = []
    eigenvalues = []

    for nu in grid:
        omega1, omega2, omega3, ax, ay, az = nu

        # Define system matrices
        B_lie, K, BK, _ = se23_solve_control(0, 0, 9.8, 0, 0, 0)
        ad = SE23Dcm.ad_matrix(np.array([0, 0, 0, ax, ay, az, omega1, omega2, omega3]))
        A = -ca.DM(ad + SE23Dcm.adC_matrix()) + BK
        A_np = np.array(A)

        A_matrices.append(A_np)
        eigenvalues.append(np.linalg.eigvals(A_np))

    # Line search over alpha to find feasible contraction rate
    if verbosity > 0:
        print("Performing line search over alpha...")

    alpha_upper = -np.real(np.max(eigenvalues))  # most unstable eigenvalue (smallest real part)

    alpha_opt = scipy.optimize.fminbound(
        lambda alpha: SE23LMIs(alpha, A_matrices)['cost'],
        x1=1e-5, x2=alpha_upper,
        disp=(verbosity > 0)
    )


    sol = SE23LMIs(alpha_opt, A_matrices)

    if sol['prob'].status != 'optimal':
        raise RuntimeError("Optimization failed")

    if verbosity > 0:
        print(f"Alpha: {alpha_opt:.4f}, mu1: {sol['mu1']}, Cost: {sol['cost']:.4e}")

    return sol



def project_ellipsoid_subspace(M, indices):
    """
    Project a high-dimensional ellipsoid xᵀMx = 1 onto a 3D subspace.

    Parameters:
        P             : (n x n) ndarray, Lyapunov matrix
        indices       : list of 3 ints, indices to project onto (e.g., [0,1,2] for position)
        val           : float, scalar value (e.g., V(t)) to scale the ellipsoid
        return_matrix : bool, if True returns (R, radii) instead of points

    Returns:
        points : (3, N) ndarray of ellipsoid surface points
        val    : float, same as input (for convenience)
    """

    # Step 2: Partition P
    all_indices = np.arange(M.shape[0])
    comp_indices = np.setdiff1d(all_indices, indices)

    # Extract blocks
    A = M[np.ix_(comp_indices, comp_indices)]
    B = M[np.ix_(comp_indices, indices)]
    C = M[np.ix_(indices, comp_indices)]
    D = M[np.ix_(indices, indices)]

    # Step 3: Schur complement to eliminate complementary variables
    M_sub = D - C @ np.linalg.inv(A) @ B

    # Step 4: Ellipsoid shape
    evals, evects = np.linalg.eig(M_sub)
    evals = np.real(evals)
    evects = np.real(evects)
    radii = 1.0 / np.sqrt(evals)
    R = evects @ np.diag(radii)

    # Step 5: Sample sphere and map to ellipsoid
    n = 30
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    sphere_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)

    ellipsoid_points = R @ sphere_points

    #xTM_subx = 1 defines the ellipoid from which ellipsoid_points are taken
    return ellipsoid_points, M_sub

def exp_map(points, rotation_points):
    """
    Applies the matrix exponential map from se(3) (Lie algebra) to SE(3) (Lie group)
    for a batch of 6D twists.

    Inputs:
        points        : np.ndarray of shape (3, N)
                        Translation components (x, y, z) of N twist vectors
        rotation_points  : np.ndarray of shape (3, N)
                        Rotation components (theta_x, theta_y, theta_z) of N twist vectors

    Output:
        inv_points    : np.ndarray of shape (6, N)
                        SE(3) vector representation of transformed rigid-body poses
                        via exp(wedge(x)) for each column x = [v; w]
    """

    # Allocate space for the output — one 6D vector for each input twist
    inv_points = np.zeros((6, points.shape[1]))

    # Loop over each point (column)
    for i in range(points.shape[1]):

        # Assemble the 6D twist vector: [translation; rotation]
        twist_vec = np.array([
            points[0, i],         # x
            points[1, i],         # y
            points[2, i],         # z
            rotation_points[0, i],   # θ_x
            rotation_points[1, i],   # θ_y
            rotation_points[2, i]    # θ_z
        ])

        # Convert 6D twist vector to 4×4 matrix in se(3) Lie algebra
        Lie_matrix = SE3Dcm.wedge(twist_vec)

        # Apply matrix exponential to get SE(3) transformation matrix
        exp_matrix = SE3Dcm.exp(Lie_matrix)

        # Convert SE(3) matrix back to a 6D vector representation (e.g., [pos; axis-angle])
        exp_vector = SE3Dcm.vector(exp_matrix)

        # Convert from CasADi DM to NumPy array and reshape to (6,)
        exp_vector = np.array(ca.DM(exp_vector)).reshape(6,)

        # Store the result
        inv_points[:, i] = exp_vector

    return inv_points
