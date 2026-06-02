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


def _as_3x3(x, name):
    """Coerce scalar / length-3 / 3x3 input into a 3x3 numpy matrix."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return float(arr) * np.eye(3)
    if arr.ndim == 1 and arr.size == 3:
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"{name} must be scalar, length-3, or 3x3; got shape {arr.shape}")


def solve_inv_set_pi(J, Kp, Ki, tau_dist, verbosity=0):
    """
    PI rate-loop invariant set with explicit inertia and torque disturbance.

    Plant (small-angle / hover, gyroscopic term dropped):
        J ω̇ = τ + τ_d,           ‖τ_d‖∞ ≤ tau_dist  (N·m)
    Controller:
        τ = -Kp ω - Ki z,   ż = ω
    Closed-loop on x = [ω; z] ∈ R^6:
        ẋ = [[-J⁻¹ Kp, -J⁻¹ Ki], [I, 0]] x  +  [[J⁻¹], [0]] τ_d

    Returns the SAME schema as solve_inv_set, where 'P' is the 3x3 Schur-
    complement projection of the 6x6 Lyapunov matrix onto the ω subspace
    (so downstream scaling V = mu1 * tau_dist^2 still gives a 3-D ω bound).

    Parameters
    ----------
    J  : scalar | (3,) | (3,3)  inertia matrix (kg·m²)
    Kp : scalar | (3,) | (3,3)  proportional rate gain
    Ki : scalar | (3,) | (3,3)  integral rate gain (use 0 for P-only)
    tau_dist : float            sup-norm bound on torque disturbance (N·m)
    """
    J = _as_3x3(J, "J")
    Kp = _as_3x3(Kp, "Kp")
    Ki = _as_3x3(Ki, "Ki")
    Jinv = np.linalg.inv(J)

    A_cl = np.block([
        [-Jinv @ Kp, -Jinv @ Ki],
        [np.eye(3),   np.zeros((3, 3))],
    ])
    B_d = np.vstack([Jinv, np.zeros((3, 3))])

    eigs = np.linalg.eigvals(A_cl)
    if np.any(np.real(eigs) >= 0):
        raise RuntimeError(
            f"Closed-loop unstable: eigenvalues {eigs}. Increase Kp/Ki or check J."
        )
    alpha_upper = -np.max(np.real(eigs))

    alpha_opt = scipy.optimize.fminbound(
        lambda a: omegaLMIs(a, [A_cl], B_d, verbosity=verbosity)["cost"],
        x1=1e-5,
        x2=alpha_upper,
        disp=(verbosity > 0),
    )
    sol = omegaLMIs(alpha_opt, [A_cl], B_d, verbosity=verbosity)

    if sol["P"] is None or sol["prob"].status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"PI LMI failed: {sol['prob'].status if sol['prob'] else None}")

    # Schur-complement projection P_6 -> Q_omega (3x3) on the first block.
    P6 = sol["P"]
    P11 = P6[:3, :3]
    P12 = P6[:3, 3:]
    P22 = P6[3:, 3:]
    Q_omega = P11 - P12 @ np.linalg.solve(P22, P12.T)
    Q_omega = 0.5 * (Q_omega + Q_omega.T)  # symmetrize for numerics

    if verbosity > 0:
        logger.info(
            f"PI rate loop: alpha={alpha_opt:.4f} mu1={sol['mu1']:.4e} cost={sol['cost']:.4e}"
        )

    return {
        "cost": sol["cost"],
        "mu1": sol["mu1"],
        "P": Q_omega,
        "P_full": P6,
        "alpha": alpha_opt,
        "prob": sol["prob"],
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





