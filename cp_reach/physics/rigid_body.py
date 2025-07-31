import numpy as np
import cvxpy as cp
import control
import itertools
import scipy
from cp_reach.lie.SE23 import *
from cp_reach.lie.se3 import *

def se23_solve_control(x0, mode):
    """
    Constructs the linearized closed-loop dynamics on SE(2,3) for a rigid body.

    Parameters:
        x0   : list or array of shape (9,)
               State around which to linearize. Ordered as:
               [vx, vy, vz, ax, ay, az, ω₁, ω₂, ω₃]

        mode : int
               Specifies the actuation model:
               - mode = 0 : Drone (thrust + 3D torque)
               - mode = 1 : Satellite (3D force + 3D torque)

    Returns:
        B     : (9 x m) control input matrix (m = 4 or 6 depending on mode)
        K     : (m x 9) LQR feedback gain matrix (negative sign convention)
        BK    : (9 x 9) feedback contribution matrix (B @ K)
        A_cl  : (9 x 9) closed-loop system matrix (A + BK)
    """

    # Compute drift dynamics:
    # A = -ad(x0) + ad_C (Lie algebra plus curvature correction)
    A = -ca.DM(SE23Dcm.ad_matrix(np.array(x0)) + SE23Dcm.adC_matrix())

    # Control input matrix B depends on system type
    if mode == 0:
        # Drone Mode: 4 actuators → [az, ω₁, ω₂, ω₃]
        # Assume thrust only in z-direction and 3D torques
        B = np.array([
            [0, 0, 0, 0],  # vx
            [0, 0, 0, 0],  # vy
            [0, 0, 0, 0],  # vz
            [0, 0, 0, 0],  # ax
            [0, 0, 0, 0],  # ay
            [1, 0, 0, 0],  # az ← upward thrust
            [0, 1, 0, 0],  # ω₁ ← roll torque
            [0, 0, 1, 0],  # ω₂ ← pitch torque
            [0, 0, 0, 1],  # ω₃ ← yaw torque
        ])

    elif mode == 1:
        # Satellite Mode: 6 actuators → [ax, ay, az, ω₁, ω₂, ω₃]
        # Full 6-DoF control: translational and rotational
        B = np.array([
            [0, 0, 0, 0, 0, 0],  # vx
            [0, 0, 0, 0, 0, 0],  # vy
            [0, 0, 0, 0, 0, 0],  # vz
            [1, 0, 0, 0, 0, 0],  # ax ← force_x
            [0, 1, 0, 0, 0, 0],  # ay ← force_y
            [0, 0, 1, 0, 0, 0],  # az ← force_z
            [0, 0, 0, 1, 0, 0],  # ω₁ ← torque_x
            [0, 0, 0, 0, 1, 0],  # ω₂ ← torque_y
            [0, 0, 0, 0, 0, 1],  # ω₃ ← torque_z
        ])

    else:
        raise ValueError("Unknown mode: must be 0 (drone) or 1 (satellite)")

    # LQR cost matrices
    Q = 10 * np.eye(9)               # Penalize state deviation
    R = 1 * np.eye(B.shape[1])       # Penalize input effort

    # Solve continuous-time LQR: returns K such that u = -Kx
    K, _, _ = control.lqr(A, B, Q, R)
    K = -K  # Convert to negative feedback form

    BK = B @ K           # Feedback contribution to dynamics
    A_cl = A + BK        # Closed-loop system

    return B, K, BK, A_cl


def SE23LMIs(alpha, A_list, verbosity=0):
    """
    Solve a Lyapunov LMI for contraction analysis on an SE(2,3) system.
    
    For each system matrix Aᵢ in A_list, this function finds a symmetric
    matrix P ≻ 0 and positive scalars μ₁, μ₂, μ₃ such that:

        AᵢᵀP + P Aᵢ + αP + Schur terms ≼ 0

    The Schur terms are structured to separately penalize errors in position,
    velocity, and orientation subspaces.

    Parameters:
        alpha     : float
                    Lyapunov decay rate (contraction rate)
        A_list    : list of np.ndarray
                    List of 9x9 system matrices Aᵢ
        verbosity : int
                    Set >0 to enable solver logging

    Returns:
        dict with:
            - 'P'     : solution to Lyapunov inequality (9×9 positive definite)
            - 'mu1'   : contraction margin for position subspace
            - 'mu2'   : contraction margin for velocity subspace
            - 'mu3'   : contraction margin for orientation subspace
            - 'cost'  : scalar γ = μ₁ + μ₂ + μ₃ (to minimize)
            - 'alpha' : input contraction rate
            - 'prob'  : the solved cvxpy problem (can inspect duals, status, etc.)
    """

    # Decision variables
    P = cp.Variable((9, 9), symmetric=True)  # Lyapunov matrix
    mu1 = cp.Variable()  # contraction rate in position
    mu2 = cp.Variable()  # contraction rate in velocity
    mu3 = cp.Variable()  # contraction rate in orientation

    gamma = mu1 + mu2 + mu3  # total cost to minimize

    # Prepare LMI constraints
    constraints = []

    # Partition P into row blocks for use in Schur complement structure
    P1 = P[0:3, :]  # rows corresponding to position
    P2 = P[3:6, :]  # rows corresponding to velocity
    P3 = P[6:9, :]  # rows corresponding to orientation

    I3 = np.eye(3)
    zero33 = np.zeros((3, 3))

    # For each linearized system Aᵢ, enforce the contraction inequality
    for Ai in A_list:
        LMI = cp.bmat([
            [Ai.T @ P + P @ Ai + alpha * P,  P1.T,                P2.T,                P3.T],
            [P1,                            -alpha * mu1 * I3,    zero33,              zero33],
            [P2,                             zero33,             -alpha * mu2 * I3,    zero33],
            [P3,                             zero33,              zero33,             -alpha * mu3 * I3],
        ])
        constraints.append(LMI << 0)  # Enforce negative semidefiniteness

    # Ensure positive definiteness and numerical robustness
    constraints += [
        P >> np.eye(9),     # P ≻ I (avoid near-singularity)
        mu1 >= 1e-5,        # prevent zero contraction margin
        mu2 >= 1e-5,
        mu3 >= 1e-5,
    ]

    # Formulate and solve optimization problem
    prob = cp.Problem(cp.Minimize(gamma), constraints)

    try:
        prob.solve(solver=cp.CVXOPT, verbose=(verbosity > 0))

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"LMI solver failed: {prob.status}")

        return {
            'P': P.value,
            'mu1': mu1.value,
            'mu2': mu2.value,
            'mu3': mu3.value,
            'cost': gamma.value,
            'alpha': alpha,
            'prob': prob,
        }

    except Exception as e:
        if verbosity > 0:
            print(f"Solver failed at alpha = {alpha:.4f}: {e}")

        return {
            'P': None,
            'mu1': None,
            'mu2': None,
            'mu3': None,
            'cost': np.inf,
            'alpha': alpha,
            'prob': None,
        }



def solve_se23_invariant_set(ax_range, ay_range, az_range,
                             omega1_range, omega2_range, omega3_range,
                             mode, verbosity=0):
    """
    Computes a common Lyapunov matrix P for the SE(2,3) system
    across a grid of linear acceleration and angular velocity values.

    For each point in the grid, the system is linearized and closed-loop dynamics
    are formed using LQR feedback. A Lyapunov LMI is then solved to find a
    symmetric matrix P that satisfies:

        Aᵢᵀ P + P Aᵢ + α P + structured Schur terms ≼ 0

    for all linearized systems Aᵢ in the grid.

    Parameters:
        ax_range, ay_range, az_range             : iterable
            Linear acceleration values (m/s²) to evaluate
        omega1_range, omega2_range, omega3_range : iterable
            Angular velocity values (rad/s) to evaluate
        mode      : int
            System type:
              - 0: quadrotor model (thrust + torque)
              - 1: satellite model (full force + torque)
        verbosity : int
            If > 0, prints progress and solver output

    Returns:
        sol : dict
            {
              'P'     : Lyapunov matrix (9×9 ndarray),
              'mu1'   : margin in position subspace,
              'mu2'   : margin in velocity subspace,
              'mu3'   : margin in orientation subspace,
              'cost'  : scalar objective (mu1 + mu2 + mu3),
              'alpha' : stability decay rate,
              'prob'  : cvxpy Problem instance
            }
    """

    # Step 1: Build grid of operating points: [ω1, ω2, ω3, ax, ay, az]
    grid = itertools.product(omega1_range, omega2_range, omega3_range,
                             ax_range, ay_range, az_range)

    A_matrices = []   # Linearized closed-loop dynamics
    eigenvalues = []  # For estimating stability margin

    # Step 2: Define nominal state x₀ for control synthesis
    if mode == 0:
        # Quadrotor: hover at az = 9.8 m/s²
        x0 = [0, 0, 0, 0, 0, 9.8, 0, 0, 0]
    elif mode == 1:
        # Satellite: rest state
        x0 = [0] * 9
    else:
        raise ValueError("Invalid mode: use 0 for drone or 1 for satellite")
    
    # Use fixed LQR controller designed at x₀
    _, K, BK, _ = se23_solve_control(x0, mode)

    # Step 3: Loop over grid points and collect linearized dynamics
    for nu in grid:
        omega1, omega2, omega3, ax, ay, az = nu

        # Linearize drift at current (ω, a) using SE(2,3) Lie algebra
        ad = SE23Dcm.ad_matrix(np.array([0, 0, 0, ax, ay, az, omega1, omega2, omega3]))
        A = -ca.DM(ad + SE23Dcm.adC_matrix()) + BK

        A_matrices.append(np.array(A))
        eigenvalues.append(np.linalg.eigvals(A))

    # Step 4: Estimate maximum real part of eigenvalues for α search
    if verbosity > 0:
        print("Estimating decay rate α from closed-loop eigenvalues...")

    alpha_upper = -np.real(np.max(eigenvalues))  # Most unstable real eigenvalue

    # Step 5: Line search to minimize Lyapunov cost
    alpha_opt = scipy.optimize.fminbound(
        lambda alpha: SE23LMIs(alpha, A_matrices)['cost'],
        x1=1e-5,
        x2=alpha_upper,
        disp=(verbosity > 0)
    )

    # Step 6: Solve LMI at optimal α
    sol = SE23LMIs(alpha_opt, A_matrices)

    if sol['prob'].status != 'optimal':
        raise RuntimeError(f"Lyapunov LMI failed at α = {alpha_opt:.4f}")

    if verbosity > 0:
        print(f"[✓] Lyapunov matrix found:")
        print(f"    α = {alpha_opt:.4f}, μ₁ = {sol['mu1']:.4e}, total cost = {sol['cost']:.4e}")

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
