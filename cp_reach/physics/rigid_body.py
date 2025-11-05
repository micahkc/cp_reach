import numpy as np
import cvxpy as cp
import control
import itertools
import scipy
from cp_reach.lie.SE23 import *
from cp_reach.lie.se3 import *
import cyecca.lie as lie
from cyecca.lie.group_se23 import se23, SE23Quat
from cyecca.lie.group_se23 import SE23Quat
import casadi as ca

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
    # A = -ad(x0) + ad_C 
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



def SE23LMIs2(alpha, A_list, B1, B2, w1_max, w2_max, verbosity=0, solver=None):
    """
    Certify an invariant ellipsoid V(x)=x^T P x <= mu1*w1_max^2 + mu2*w2_max^2
    for the LTI family xdot = A_i x + B1 w1 + B2 w2 using the stacked
    single-multiplier S-procedure:

        [ A^T P + P A + α P    P B1      P B2   ]
        [   B1^T P           -α μ1 I      0     ]  <<  0  for all i
        [   B2^T P             0       -α μ2 I  ]

    Returns P, mu1, mu2 and the CVXPY Problem object.
    """
    import cvxpy as cp
    import numpy as np

    n = 9
    m1 = B1.shape[1]
    m2 = B2.shape[1]
    I_n  = np.eye(n)
    I_m1 = np.eye(m1)
    I_m2 = np.eye(m2)
    Z_12 = np.zeros((m1, m2))
    Z_21 = np.zeros((m2, m1))

    # Decision variables
    P   = cp.Variable((n, n), symmetric=True)
    mu1 = cp.Variable(nonneg=True)
    mu2 = cp.Variable(nonneg=True)

    # Objective: minimize certified radius in V
    gamma = mu1 * (w1_max**2) + mu2 * (w2_max**2)
    objective = cp.Minimize(gamma)

    constraints = [P >> I_n]

    for Ai in A_list:
        TL = Ai.T @ P + P @ Ai + alpha * P
        LMI = cp.bmat([
            [TL,           P @ B1,           P @ B2],
            [B1.T @ P,  -alpha * mu1 * I_m1,  Z_12 ],
            [B2.T @ P,       Z_21,        -alpha * mu2 * I_m2]
        ])
        constraints.append(LMI << 0)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CVXOPT, verbose=(verbosity > 0))

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"LMI solver failed: {prob.status}")

        return {
            'P': P.value,
            'mu1': mu1.value,
            'mu2': mu2.value,
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
            'cost': np.inf,
            'alpha': alpha,
            'prob': None,
        }
    
def SE23LMIs3(alpha, A_list, B1, B2, B3, w1_max, w2_max, w3_max,
              Bu=None, verbosity=0, solver=None):
    """
    Certify an invariant ellipsoid V(x)=x^T P x <= sum_j mu_j * wj_max^2
    for the LTI family:
        xdot = A_i x + B1 w1 + B2 w2 + B3 w3     (i = 1..N)

    using the stacked single-multiplier S-procedure:

        [ A^T P + P A + α P      P B1         P B2         P B3  ]
        [    B1^T P           -α μ1 I           0            0   ]  <<  0  for all i
        [    B2^T P              0           -α μ2 I          0   ]
        [    B3^T P              0              0          -α μ3 I]

    Notes:
    - `Bu` is optional and not used inside the LMI; pass it if you want to
      form closed-loop A_i = A_i + Bu @ K outside this function.

    Returns dict with:
        'P'   : ndarray (n x n)
        'mu1','mu2','mu3' : floats
        'cost': float = mu1*w1_max^2 + mu2*w2_max^2 + mu3*w3_max^2
        'alpha': alpha
        'prob' : CVXPY Problem object
    """
    import cvxpy as cp
    import numpy as np

    if not A_list:
        raise ValueError("A_list must be a non-empty list of system matrices.")

    # Infer dimensions
    n = A_list[0].shape[0]
    for Ai in A_list:
        if Ai.shape != (n, n):
            raise ValueError("All A_i must be square and of the same size.")
    m1 = B1.shape[1]
    m2 = B2.shape[1]
    m3 = B3.shape[1]

    # Identities / zeros
    I_n  = np.eye(n)
    I_m1 = np.eye(m1)
    I_m2 = np.eye(m2)
    I_m3 = np.eye(m3)
    Z_12 = np.zeros((m1, m2))
    Z_13 = np.zeros((m1, m3))
    Z_21 = np.zeros((m2, m1))
    Z_23 = np.zeros((m2, m3))
    Z_31 = np.zeros((m3, m1))
    Z_32 = np.zeros((m3, m2))

    # Decision variables
    P   = cp.Variable((n, n), symmetric=True)
    mu1 = cp.Variable(nonneg=True)
    mu2 = cp.Variable(nonneg=True)
    mu3 = cp.Variable(nonneg=True)

    # Minimize the certified radius
    gamma = mu1 * (w1_max**2) + mu2 * (w2_max**2) + mu3 * (w3_max**2)
    objective = cp.Minimize(gamma)

    # Basic Lyapunov PSD bound (tweak eps if needed)
    eps = 1.0
    constraints = [P >> eps * I_n]

    for Ai in A_list:
        TL = Ai.T @ P + P @ Ai + alpha * P
        LMI = cp.bmat([
            [TL,           P @ B1,               P @ B2,               P @ B3],
            [B1.T @ P,  -alpha * mu1 * I_m1,        Z_12,                 Z_13],
            [B2.T @ P,        Z_21,           -alpha * mu2 * I_m2,        Z_23],
            [B3.T @ P,        Z_31,                Z_32,            -alpha * mu3 * I_m3],
        ])
        constraints.append(LMI << 0)

    prob = cp.Problem(objective, constraints)

    # Solver selection (use provided if given; else fall back)
    try:
        chosen_solver = solver
        if chosen_solver is None:
            # Try a reasonable default ordering
            for s in [cp.CVXOPT, cp.MOSEK, cp.SCS]:
                try:
                    prob.solve(solver=s, verbose=(verbosity > 0))
                    chosen_solver = s
                    break
                except Exception:
                    continue
            if chosen_solver is None:
                # Final generic attempt with CVXPY default
                prob.solve(verbose=(verbosity > 0))
        else:
            prob.solve(solver=chosen_solver, verbose=(verbosity > 0))

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"LMI solver failed: {prob.status}")

        return {
            'P':   P.value,
            'mu1': float(mu1.value),
            'mu2': float(mu2.value),
            'mu3': float(mu3.value),
            'cost': float(gamma.value),
            'alpha': alpha,
            'prob': prob,
            'solver_used': getattr(chosen_solver, 'name', str(chosen_solver)) if chosen_solver else 'default'
        }

    except Exception as e:
        if verbosity > 0:
            print(f"Solver failed at alpha = {alpha:.6g}: {e}")
        return {
            'P': None,
            'mu1': None,
            'mu2': None,
            'mu3': None,
            'cost': np.inf,
            'alpha': alpha,
            'prob': None,
            'solver_used': None,
            'error': str(e),
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

def solve_se23_invariant_set_log_control(ref_acc, Kp, Kd, Kpq, omega_dist, accel_dist, gravity_err):
    """
    Computes a common Lyapunov matrix P for the SE(2,3) system
    across a grid of linear acceleration and angular velocity values.

    For each point in the grid, the system is linearized and closed-loop dynamics
    are formed using LQR feedback. A Lyapunov LMI is then solved to find a
    symmetric matrix P that satisfies:

        Aᵢᵀ P + P Aᵢ + α P + structured Schur terms ≼ 0

    for all linearized systems Aᵢ in the grid.

    Parameters:
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

    A_matrices = []   # Linearized closed-loop dynamics
    eigenvalues = []  # For estimating stability margin


    
    # This is how control input enters the system
    B0 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])

    # Gain matrix K = np.array([Kp Kd 0
    #                           0 0  Kp])
    K = np.array([
        [Kp,0,0,Kd,0,0,0,0,0],
        [0,Kp,0,0,Kd,0,0,0,0],
        [0,0,Kp,0,0,Kd,0,0,0],
        [0,0,0,0,0,0,Kpq,0,0],
        [0,0,0,0,0,0,0,Kpq,0],
        [0,0,0,0,0,0,0,0,Kpq]])
    
    # These are how disturbances enter the system (angular velocity (B1) accelerational dist (B2) and gravity (B3))
    B1 = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ])

    B2 = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,0,0],
        [0,0,0],
        [0,0,0],
    ])

    B3 = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,0,0],
        [0,0,0],
        [0,0,0],
    ])


    # We want this in the form x dot = Ax + B1 dist1 + B2 dist2 + B3 dist 3
    # A = -ad_nbar + C - B0K
    vec = np.array([0, 0, 0, ref_acc[0], ref_acc[1], ref_acc[2], 0, 0, 0])
    ad = SE23Dcm.ad_matrix(vec)
    A = -ca.DM(ad - SE23Dcm.adC_matrix()) - B0@K


    A_matrices.append(np.array(A))
    eigenvalues.append(np.linalg.eigvals(A))

    # print(eigenvalues)
    alpha_upper = -np.real(np.max(eigenvalues))  # Most unstable real eigenvalue


    # Line search to find alpha that minimizes cost (omega_dist mu1**2 + gravity_err mu2**2)
    alpha_opt = scipy.optimize.fminbound(
        lambda alpha: SE23LMIs3(alpha, A_matrices, -B1*Kpq, B2, B3, omega_dist, accel_dist, gravity_err)['cost'],
        x1=1e-5,
        x2=alpha_upper
    )

    # Solve LMI at optimal alpha
    sol = SE23LMIs3(alpha_opt, A_matrices, B1, B2, B3, omega_dist, accel_dist, gravity_err)
    

    if sol['prob'].status != 'optimal':
        raise RuntimeError(f"Lyapunov LMI failed at α = {alpha_opt:.4f}")


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


def sample_ellipsoid_boundary(M, n):
    """
    Deterministic boundary samples of {xi : xi^T M xi = 1} in R^9.
    Even-angle points are placed on every coordinate 2-plane (i,j),
    then mapped to the ellipsoid with x = A^{-1} u where M = A^T A.

    Returns X: (9 x n) with columns on the boundary.
    """
    A = np.linalg.cholesky(M).T          # M = A.T @ A
    pairs = [(i, j) for i in range(9) for j in range(i+1, 9)]
    P = len(pairs)                        # = C(9,2) = 36
    per_plane = (n + P - 1) // P          # ceil(n / P)

    U_list = []
    for (i, j) in pairs:
        ts = 2.0*np.pi*np.arange(per_plane)/per_plane
        Ui = np.zeros((9, per_plane))
        Ui[i, :] = np.cos(ts)
        Ui[j, :] = np.sin(ts)
        U_list.append(Ui)
        if len(U_list) * per_plane >= n:
            break

    U = np.concatenate(U_list, axis=1)[:, :n]  # (9 x n) on the unit sphere in each plane
    X = np.linalg.solve(A, U)                  # map to xi: xi^T M xi = 1
    return X

def expmap(X):
    n = X.shape[1]
    Eta_array = np.zeros([9,n])
    for k in range(n):
        xi = lie.se23.elem(ca.DM(X[:, k])) # Lie Algebra
        eta  = xi.exp(lie.SE23Quat) # Lie group
        params = eta.param # Extract position, velocity, and attitude (as quaternion)
        param_np = ca.DM(params).full().ravel()

        quat_arr = param_np[6:]
        quat = lie.SO3Quat.elem(ca.DM(quat_arr))
        euler = lie.group_so3.SO3EulerB321.from_Quat(quat).param
        euler_np = ca.DM(euler).full().ravel()
        Eta_array[:,k] = np.concatenate((param_np[:6], euler_np))
    
    return Eta_array

