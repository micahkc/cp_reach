import numpy as np
import scipy.optimize
import cvxpy as cp

def _solve_single_channel_LMI(alpha, A_list, B, w_max, eps=1e-3, verbosity=0, solver=None):
    """
    Certify V(x)=x^T P x <= mu * w_max^2
    for xdot = A_i x + B d using the stacked single-multiplier S-procedure:

        [ A^T P + P A + α P    P B ]
        [      B^T P        -α μ I ]  <<  0   for all A in A_list

    Returns dict with P, mu, cost, prob.
    """
    n = A_list[0].shape[0]
    m = B.shape[1]

    P  = cp.Variable((n, n), symmetric=True)
    mu = cp.Variable(nonneg=True)

    I_n = np.eye(n)
    I_m = np.eye(m)

    gamma = mu * (w_max**2)
    constraints = [P >> eps * I_n]

    for A in A_list:
        TL = A.T @ P + P @ A + alpha * P
        LMI = cp.bmat([
            [TL,        P @ B],
            [B.T @ P,  -alpha * mu * I_m]
        ])
        constraints.append(LMI << 0)

    prob = cp.Problem(cp.Minimize(gamma), constraints)

    try:
        chosen = solver
        if chosen is None:
            # Try a reasonable order of solvers
            for s in [cp.CVXOPT, cp.MOSEK, cp.SCS]:
                try:
                    prob.solve(solver=s, verbose=(verbosity > 0))
                    chosen = s
                    break
                except Exception:
                    continue
            if chosen is None:
                prob.solve(verbose=(verbosity > 0))
        else:
            prob.solve(solver=chosen, verbose=(verbosity > 0))

        return {
            'P':   None if P.value is None else P.value,
            'mu':  None if mu.value is None else float(mu.value),
            'cost': np.inf if prob.value is None else float(prob.value),
            'alpha': alpha,
            'prob': prob,
            'status': prob.status
        }
    except Exception as e:
        if verbosity > 0:
            print(f"LMI solve failed at alpha={alpha}: {e}")
        return {
            'P': None, 'mu': None, 'cost': np.inf,
            'alpha': alpha, 'prob': None, 'status': 'error'
        }


def solve_hcw_invariant_set(ref_n, w_acc_max, Kp=None, Kd=None,
                            alpha_lo=1e-6, verbosity=0, solver=None):
    """
    HCW comparison method for error dynamics: e_dot = A e + B d,
    where A is the (circular) HCW matrix and d is translational acceleration disturbance.

    State: x = [p; v] ∈ R^6  with p,v ∈ R^3.
    Dynamics:
        A_HCW = [[0, I],
                 [Ω, Γ]]  with the standard HCW structure (see below)
        B = [0; I]  (disturbance enters as acceleration)

    Optional PD closure:
        If Kp,Kd provided (scalars or 3-vectors), we apply u = -K [p; v],
        with control channel B0 = [0; I], so A_cl = A_HCW - B0 K.

    We then solve the single-channel LMI to certify V(x)=x^T P x <= mu * w_acc_max^2.

    Returns:
        {
          'P'      : (6x6) Lyapunov matrix,
          'mu'     : disturbance margin,
          'cost'   : mu * w_acc_max^2,
          'alpha'  : selected decay rate,
          'prob'   : cvxpy Problem,
          'A'      : closed-loop A used,
          'B'      : disturbance matrix used,
          'eigA'   : eigenvalues of A
        }
    """
    n = float(ref_n)

    # HCW (circular) dynamics: x = [px, py, pz, vx, vy, vz]
    A_HCW = np.array([
        [0, 0, 0, 1,   0,   0],
        [0, 0, 0, 0,   1,   0],
        [0, 0, 0, 0,   0,   1],
        [3*n**2, 0, 0, 0, 2*n, 0],
        [0, 0, 0, -2*n, 0, 0],
        [0, 0, -n**2, 0, 0, 0]
    ], dtype=float)

    # Disturbance enters as acceleration: xdot = ... + [0; I] d
    B = np.vstack([np.zeros((3,3)), np.eye(3)])

    A = A_HCW.copy()

    # Optional PD closure on the same channel B0 = [0; I]
    if Kp is not None and Kd is not None:
        Kp_vec = np.array([Kp]*3, dtype=float).ravel() if np.isscalar(Kp) else np.array(Kp, dtype=float).ravel()
        Kd_vec = np.array([Kd]*3, dtype=float).ravel() if np.isscalar(Kd) else np.array(Kd, dtype=float).ravel()
        K = np.hstack([np.diag(Kp_vec), np.diag(Kd_vec)])  # (3x6)
        A = A - B @ K

    A_list = [A]
    eigA = np.linalg.eigvals(A)

    # Choose an alpha search upper bound (needs to be positive)
    # If A is stable, -max Re(λ) is positive; otherwise give a modest cap.
    alpha_upper = float(max(1e-5, -np.real(eigA).max()))

    if verbosity > 0:
        print(f"eig(A): {eigA}")
        print(f"alpha search in [{alpha_lo}, {alpha_upper}]")

    # Line search for alpha minimizing the certified radius
    def objective(alpha):
        res = _solve_single_channel_LMI(alpha, A_list, B, w_acc_max, verbosity=0, solver=solver)
        return res['cost']

    alpha_opt = scipy.optimize.fminbound(objective, x1=alpha_lo, x2=alpha_upper, disp=verbosity > 0)

    # Final solve at alpha_opt
    sol = _solve_single_channel_LMI(alpha_opt, A_list, B, w_acc_max, verbosity=verbosity, solver=solver)

    # Basic status check
    if sol['prob'] is None or sol['prob'].status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"HCW LMI failed at α = {alpha_opt:.6g} (status: {sol['prob'].status if sol['prob'] else 'no_prob'})")

    return {
        'P': sol['P'],
        'mu': sol['mu'],
        'cost': sol['cost'],
        'alpha': alpha_opt,
        'prob': sol['prob'],
        'A': A,
        'B': B,
        'eigA': eigA,
    }



import numpy as np

def sample_ellipsoid_boundary_6(M, n):
    """
    Deterministic boundary samples of { x : x^T M x = 1 } in R^6.
    Strategy: place evenly spaced points on each coordinate 2-plane (i,j)
    on the unit circle, then map u -> x by x = A^{-1} u where M = A^T A.

    Args:
        M : (6x6) symmetric positive-definite matrix
        n : number of boundary samples to return

    Returns:
        X : (6 x n) array, each column lies on the ellipsoid boundary.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (6, 6):
        raise ValueError("M must be 6x6.")

    # Factor M = A^T A
    try:
        # Fast path (SPD): Cholesky upper so that M = R^T R with A = R
        A = np.linalg.cholesky(M).T
    except np.linalg.LinAlgError:
        # Fallback: symmetric eigendecomposition (handles near-singular gracefully)
        w, V = np.linalg.eigh(0.5*(M + M.T))
        if np.any(w <= 0):
            raise ValueError("M must be positive definite (all eigenvalues > 0).")
        A = (V * np.sqrt(w)) @ V.T  # A such that A^T A = V diag(w) V^T = M

    # Build even-angle unit-circle samples on each coordinate 2-plane
    pairs = [(i, j) for i in range(6) for j in range(i+1, 6)]  # C(6,2) = 15 planes
    P = len(pairs)
    per_plane = (n + P - 1) // P  # ceil(n / P)

    U_cols = []
    for (i, j) in pairs:
        ts = 2.0 * np.pi * np.arange(per_plane) / per_plane
        U = np.zeros((6, per_plane))
        U[i, :] = np.cos(ts)
        U[j, :] = np.sin(ts)
        U_cols.append(U)
        if len(U_cols) * per_plane >= n:
            break

    U = np.concatenate(U_cols, axis=1)[:, :n]   # (6 x n) points on unit circles in planes
    X = np.linalg.solve(A, U)                   # map to ellipsoid boundary: x^T M x = 1
    return X
