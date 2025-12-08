import numpy as np
import cvxpy as cp
import control
import itertools
import scipy
# from cp_reach.lie.SE23 import *
# from cp_reach.lie.se3 import *
import cyecca.lie as lie
# from cyecca.lie.group_se23 import se23, SE23Quat
# from cyecca.lie.group_se23 import SE23Quat
import casadi as ca



def SE23LMIs(alpha, A_list, B1, B2, w1_max, w2_max, verbosity=0, solver=None):
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

    n = A_list[0].shape[1]
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


def solve_se23_invariant_set(ref_acc, Kp, Kd, Kpq, Kdq, omega_dist, gravity_err):
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
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])

    # Gain matrix K = np.array([Kp Kd 0
    #                           0 0  Kp])
    # print(Kp,Kd,Kpq,Kdq)
    K = np.array([
        [Kp,0,0,Kd,0,0,0,0,0,0,0,0],
        [0,Kp,0,0,Kd,0,0,0,0,0,0,0],
        [0,0,Kp,0,0,Kd,0,0,0,0,0,0],
        [0,0,0,0,0,0,Kpq,0,0,Kdq,0,0],
        [0,0,0,0,0,0,0,Kpq,0,0,Kdq,0],
        [0,0,0,0,0,0,0,0,Kpq,0,0,Kdq]])
    
    # These are how disturbances enter the system (angular velocity (B1) and gravity (B2))
    B1 = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [Kdq,0,0],
        [0,Kdq,0],
        [0,0,Kdq],
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
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ])
    


    # We want this in the form x dot = (A+B0)x + B1 dist1 + B2 dist2
    # top left of A is: -ad_nbar + C - B0K
    print(ref_acc)
    vec = np.array([0, 0, 0, ref_acc[0], ref_acc[1], ref_acc[2], 0, 0, 0])
    xi = lie.se23.elem(ca.DM(vec)) # Lie Algebra
    ad = ca.DM(lie.se23.adjoint(xi))

    # ad = ca.DM(SE23Dcm.ad_matrix(vec)).full()
    # print(ad)

    adC = np.zeros((9,9))
    adC[0, 3] = 1
    adC[1, 4] = 1
    adC[2, 5] = 1

    A_top_left = -ad + adC

    A = np.zeros((12,12))
    A[:9, :9] = A_top_left
    A[6:9,9:] = np.eye(3)

    A_prime = ca.DM(A - B0@K)


    A_matrices.append(np.array(A_prime))
    eigenvalues.append(np.linalg.eigvals(A_prime))


    alpha_upper = -np.real(np.max(eigenvalues))  # Most unstable real eigenvalue


    # Line search to find alpha that minimizes cost (omega_dist mu1**2 + gravity_err mu2**2)
    alpha_opt = scipy.optimize.fminbound(
        lambda alpha: SE23LMIs(alpha, A_matrices, B1, B2, omega_dist, gravity_err)['cost'],
        x1=1e-5,
        x2=alpha_upper
    )

    # Solve LMI at optimal alpha
    sol = SE23LMIs(alpha_opt, A_matrices, B1, B2, omega_dist, gravity_err)
    

    if sol['prob'].status != 'optimal':
        raise RuntimeError(f"Lyapunov LMI failed at α = {alpha_opt:.4f}")


    return sol


import numpy as np

def project_ellipsoid_matrix(M, indices):
    """
    Compute the projected ellipsoid matrix onto the coordinate subspace 'indices'.
    If E = {x | x^T M x = 1} in R^n and we keep coordinates y = x[indices],
    then the projection is E_sub = {y | y^T M_sub y = 1} with

        M_sub = D - C A^{-1} B,

    where M is blocked as:
        [A  B]
        [C  D]
    after reordering coordinates into (drop, keep).

    Parameters
    ----------
    M : (n,n) ndarray, symmetric positive definite
    indices : list[int], coordinates to keep (length k)

    Returns
    -------
    M_sub : (k,k) ndarray
    """
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square.")
    n = M.shape[0]
    idx_keep = np.asarray(indices, dtype=int)
    if np.any(idx_keep < 0) or np.any(idx_keep >= n):
        raise ValueError("indices out of range.")
    k = idx_keep.size

    # Complement indices
    all_idx = np.arange(n)
    idx_drop = np.setdiff1d(all_idx, idx_keep, assume_unique=False)

    # Trivial cases
    if idx_drop.size == 0:
        # projecting onto all coordinates -> same matrix
        return 0.5 * (M + M.T)
    if k == 0:
        raise ValueError("indices is empty.")

    # Block partition
    A = M[np.ix_(idx_drop, idx_drop)]
    B = M[np.ix_(idx_drop, idx_keep)]
    C = M[np.ix_(idx_keep, idx_drop)]
    D = M[np.ix_(idx_keep, idx_keep)]

    # Schur complement using solve (more stable than inv), fallback to pinv
    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(A) @ B

    M_sub = D - C @ X
    # Symmetrize to clean numerical noise
    M_sub = 0.5 * (M_sub + M_sub.T)
    return M_sub


def walk_ellipsoid_planes(M, n):
    """
    Deterministic boundary samples of {x in R^d | x^T M x = 1}.
    Places evenly-angled points on every coordinate 2-plane (i,j),
    then maps with x = M^{-1/2} u where u lies on the unit circle in that plane.

    Parameters
    ----------
    M : (d,d) ndarray, symmetric positive definite
    n : int, number of boundary samples desired

    Returns
    -------
    X : (d, n) ndarray
        Columns lie on the boundary x^T M x = 1
    """
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square.")
    d = M.shape[0]
    if n <= 0:
        return np.zeros((d, 0))

    # Symmetrize and build M^{-1/2} via eigendecomposition
    Ms = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(Ms)  # w > 0 for SPD
    # Clamp tiny eigenvalues to avoid blowups
    w = np.maximum(w, 1e-14)
    Minvhalf = V @ (np.diag(1.0 / np.sqrt(w))) @ V.T

    # Generate plane-wise unit-circle points
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    P = len(pairs)                      # number of 2-planes
    per_plane = (n + P - 1) // P        # ceil(n / P)

    U_list = []
    for (i, j) in pairs:
        t = 2.0 * np.pi * np.arange(per_plane) / per_plane
        Ui = np.zeros((d, per_plane))
        Ui[i, :] = np.cos(t)
        Ui[j, :] = np.sin(t)
        U_list.append(Ui)
        if len(U_list) * per_plane >= n:
            break

    U = np.concatenate(U_list, axis=1)[:, :n]   # (d, n), each column has ||u||=1
    X = Minvhalf @ U                            # map to boundary of x^T M x = 1
    return X



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


import numpy as np

def sample_ellipsoid_surface(P, n, method="gaussian", seed=None):
    """
    Samples points on the boundary of {xi : xi^T P xi = 1} in R^d.

    Parameters
    ----------
    P : (d,d) ndarray
        Symmetric positive definite matrix.
    n : int
        Number of surface points to generate.
    method : {"gaussian", "sobol"}
        - "gaussian": normalize Gaussian vectors (exactly uniform on S^{d-1}).
        - "sobol"   : low-discrepancy sphere mapping (needs scipy>=1.7).
    seed : int or None
        RNG seed for reproducibility (gaussian only).

    Returns
    -------
    Xi : (d, n) ndarray
        Columns lie on the ellipsoid boundary (xi^T P xi = 1).
    """
    if n <= 0:
        return np.zeros((P.shape[0], 0))

    # Symmetrize and build P^{-1/2} via eigendecomposition
    Ps = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(Ps)
    # Guard against tiny numerical negatives
    w = np.maximum(w, 1e-14)
    P_inv_half = V @ (np.diag(1.0 / np.sqrt(w))) @ V.T

    d = P.shape[0]

    # Unit-sphere samples S in R^d (columns have ||s||=1)
    if method == "gaussian":
        rng = np.random.default_rng(seed)
        S = rng.standard_normal((d, n))
        S /= np.linalg.norm(S, axis=0, keepdims=True)
    elif method == "sobol":
        try:
            from scipy.stats import qmc, norm
        except Exception as e:
            raise RuntimeError("Sobol mode requires SciPy (scipy.stats.qmc, scipy.stats.norm).") from e
        # Sobol in [0,1]^d, map to N(0,1) via inverse CDF, then normalize to the sphere
        m = int(np.ceil(np.log2(n)))
        U = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m)[:n].T  # shape (d,n)
        Z = norm.ppf(U)                          # Gaussianize
        # Replace any infs from ppf(0/1) by redraw or clip
        Z = np.where(np.isfinite(Z), Z, 0.0)
        S = Z / np.linalg.norm(Z, axis=0, keepdims=True)
    else:
        raise ValueError("method must be 'gaussian' or 'sobol'.")

    # Map sphere to ellipsoid boundary: Xi = P^{-1/2} S
    Xi = P_inv_half @ S

    # Optional tiny re-normalization onto boundary (fixes accumulated fp error)
    # scale = 1.0 / np.sqrt(np.sum(Xi * (Ps @ Xi), axis=0))
    # Xi *= scale

    return Xi


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

