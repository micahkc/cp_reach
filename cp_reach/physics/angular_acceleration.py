import logging
import numpy as np
import cvxpy as cp
import scipy

logger = logging.getLogger(__name__)

def omegaLMIs(alpha, A_list, B, verbosity=0):
    """
    Solves lyapunov LMIs using CVXPY for a list of closed-loop A matrices.

    Parameters:
        alpha : float - lyapunov rate
        A_list : list of ndarray - closed-loop system matrices (A + BK)
        B : ndarray - input matrix
        verbosity : int - logging flag

    Returns:
        dict with keys:
            'cost'  - optimal mu1 value (float)
            'mu1'   - disturbance gain (float)
            'P'     - lyapunov matrix (3x3 numpy array)
            'alpha' - input alpha
            'prob'  - CVXPY problem instance
    """
    n = B.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    mu1 = cp.Variable(nonneg=True)

    constraints = []

    for Ai in A_list:
        LMI = cp.bmat([
            [Ai.T @ P + P @ Ai + alpha * P,        P @ B],
            [B.T @ P,               -alpha * mu1 * np.eye(B.shape[1])]
        ])
        constraints.append(LMI << 0)

    constraints += [
        P >> np.eye(n),           # Positive definite
        mu1 >= 1e-5               # Small positive lower bound
    ]

    objective = cp.Minimize(mu1)
    prob = cp.Problem(objective, constraints)

    # Try different solvers in order of preference
    solvers_to_try = [cp.CVXOPT, cp.MOSEK, cp.SCS, cp.ECOS]
    solved = False

    for solver in solvers_to_try:
        try:
            prob.solve(solver=solver, verbose=(verbosity > 0))
            if prob.status in ["optimal", "optimal_inaccurate"]:
                cost = mu1.value
                P_value = P.value
                solved = True
                break
        except Exception as e:
            if verbosity > 0:
                logger.debug(f"Solver {solver} failed: {e}")
            continue

    if not solved:
        logger.warning("All solvers failed for omegaLMIs")
        cost = np.inf
        P_value = None

    return {
        'cost': cost,
        'mu1': mu1.value if mu1.value is not None else None,
        'P': P_value,  # Don't round - preserve numerical precision
        'alpha': alpha,
        'prob': prob
    }


# Force-moment LMI
def solve_inv_set(Kdq, verbosity=0):
    A_list = []
    eig_list = []

    K = Kdq*np.eye(3)
    B = np.eye(3)
    A_prime = -B@K
    B_prime = -B@K

    A_list.append(A_prime)
    eig_list.append(np.linalg.eigvals(A_prime))

    # Compute conservative upper bound on alpha for LMI line search
    alpha_upper = -np.real(np.max(eig_list))  # most unstable eigenvalue (smallest real part)

    if verbosity > 0:
        logger.info("Performing line search for optimal alpha...")

    # Minimize cost over alpha in the range (very small, conservative upper bound)
    alpha_opt = scipy.optimize.fminbound(
        lambda alpha: omegaLMIs(alpha, A_list, B_prime, verbosity=verbosity)['cost'],
        x1=1e-5, x2=alpha_upper,
        disp=(verbosity > 0)
    )

    sol = omegaLMIs(alpha_opt, A_list, B_prime)


    if sol['prob'] is None or sol['prob'].status not in ['optimal', 'optimal_inaccurate']:
        logger.error(f"Optimization failed: {sol}")
        raise RuntimeError("Optimization failed")

    if sol['P'] is None:
        raise RuntimeError("Optimization failed: P matrix is None")

    if verbosity > 0:
        logger.info(f"Alpha: {alpha_opt:.4f}, mu1: {sol['mu1']}, Cost: {sol['cost']:.4e}")
    return sol

def obtain_points(M, n=30):
    """
    Generate surface points on the ellipsoid defined by xᵀMx = 1.

    Parameters:
        M : (3x3) ndarray, symmetric positive definite matrix
        n : int, resolution of point grid (default: 30)

    Returns:
        points : (3, N) ndarray of 3D points on the ellipsoid surface
    """
    # Eigen-decomposition to get shape (radii) and orientation
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # Radii of ellipsoid are inverse sqrt of eigenvalues
    radii = 1.0 / np.sqrt(eigvals)
    R = eigvecs @ np.diag(radii)  # Transformation matrix

    # Sample unit sphere
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u, v)

    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)

    # Stack into shape (3, N)
    sphere_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)

    # Apply transformation to map unit sphere to ellipsoid
    ellipsoid_points = R @ sphere_points

    return ellipsoid_points





