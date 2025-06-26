import numpy as np
import cvxpy as cp
import control
import itertools
import scipy


def omega_solve_control_gain(omega1, omega2, omega3):
    # A = np.zeros((3,3))
    A =  -np.array([[0, -omega3, omega2],
                    [omega3, 0, -omega1],
                    [-omega2, omega1, 0]])
    B = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]) # control alpha1,2,3
    Q = 10*np.eye(3)  # penalize state
    R = 1*np.eye(3)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R) 
    #print('K', K)
    K = -K # rescale K, set negative feedback sign
    BK = B@K
    return B, K, BK , A+B@K



def omegaLMIs(alpha, A_list, B, verbosity=0):
    """
    Solves contraction LMIs using CVXPY for a list of closed-loop A matrices.

    Parameters:
        alpha : float - contraction rate
        A_list : list of ndarray - closed-loop system matrices (A + BK)
        B : ndarray - input matrix
        verbosity : int - logging flag

    Returns:
        dict with keys:
            'cost'  - optimal mu1 value (float)
            'mu1'   - disturbance gain (float)
            'P'     - contraction metric (3x3 numpy array)
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

    try:
        prob.solve(solver=cp.CVXOPT, verbose=(verbosity > 0))
        cost = mu1.value
        P_value = P.value
    except Exception as e:
        print(f"Exception during CVXPY solve: {e}")
        cost = -1
        P_value = None

    return {
        'cost': cost,
        'mu1': mu1.value if mu1.value is not None else None,
        'P': np.round(P_value, 3) if P_value is not None else None,
        'alpha': alpha,
        'prob': prob
    }


# Force-moment LMI
def find_omega_invariant_set(omega1_range, omega2_range, omega3_range, verbosity=0):
    """
    Solves for contraction metric and disturbance bound over a grid of angular velocities.
    
    Parameters:
        omega1_range, omega2_range, omega3_range : iterables of angular velocities
        verbosity : int, level of logging (0 = silent)
    
    Returns:
        sol : dict containing contraction solution, including 'P', 'mu1', etc.
        max_BK : float, maximum control gain magnitude from BK
    """
    # Generate grid of all omega combinations
    omega_grid = np.array(list(itertools.product(omega1_range, omega2_range, omega3_range)))

    A_list = []
    eig_list = []
    max_BK = 0

    for omega in omega_grid:
        w1, w2, w3 = omega
        B, K, BK, A = omega_solve_control_gain(w1, w2, w3)

        A_list.append(A)
        eig_list.append(np.linalg.eigvals(A))
        
        bk_norm = np.linalg.svd(BK).S[0]
        max_BK = max(max_BK, bk_norm)

    # Compute conservative upper bound on alpha for LMI line search
    alpha_upper = -np.real(np.max(eig_list))  # most unstable eigenvalue (smallest real part)

    if verbosity > 0:
        print("Performing line search for optimal alpha...")

    # Minimize cost over alpha in the range (very small, conservative upper bound)
    alpha_opt = scipy.optimize.fminbound(
        lambda alpha: omegaLMIs(alpha, A_list, B, verbosity=verbosity)['cost'],
        x1=1e-5, x2=alpha_upper,
        disp=(verbosity > 0)
    )

    sol = omegaLMIs(alpha_opt, A_list, B)

    if sol['prob'].status != 'optimal':
        raise RuntimeError("Optimization failed")

    if verbosity > 0:
        print(f"Alpha: {alpha_opt:.4f}, mu1: {sol['mu1']}, Cost: {sol['cost']:.4e}")

    return sol, max_BK



def bound_dynamics(omega1, omega2, omega3, dist):
    # Get contraction solution and associated gain
    sol, max_BK = find_omega_invariant_set(omega1, omega2, omega3)
    
    # Unpack metric and disturbance scaling
    P = sol['P']
    mu = sol['mu1']
    
    # Compute the radius of the ellipsoid
    val = mu * dist**2
    r = np.sqrt(val)
    
    # Eigendecomposition of P
    evals, evects = np.linalg.eig(P)
    R = np.real(evects @ np.diag(1 / np.sqrt(evals)))  # Shape transform to unit ball
    
    # Compute max coordinate (infinity norm) across all directions
    bound = r * np.max(np.linalg.norm(R, axis=1))  # exact, fast

    return P, mu, bound, max_BK



