import numpy as np
import scipy.optimize
import cvxpy as cp

import itertools
import scipy
import cyecca
import casadi as ca

def th_param_bounds_from_traj(traj, mu):
    """
    traj: (N,6) chief inertial states [x,y,z,vx,vy,vz] over time grid (consistent units with mu)
    mu: gravitational parameter

    Returns dict with min/max of a=ω^2, b=ω, c=ωdot, d=k ω^(3/2), and helpers (h,k).
    """
    traj = np.asarray(traj, float)
    Rv = traj[:, :3]
    Vv = traj[:, 3:]
    R = np.linalg.norm(Rv, axis=1)
    Rhat = Rv / R[:, None]
    Rdot = (Rhat * Vv).sum(axis=1)

    hvec = np.cross(Rv, Vv)
    h = np.linalg.norm(hvec, axis=1)          # ~constant in two-body
    h_mean = float(h.mean())

    omega = h / (R**2)
    omegadot = -2.0 * (Rdot / R) * omega      # d(h/R^2)/dt with constant h
    k = mu / (h_mean**1.5)                    # k = μ / h^(3/2)
    d = k * (omega**1.5)                      # d = k ω^(3/2)

    a = omega**2
    b = omega
    c = omegadot

    print(omega, omegadot, k)

    def mm(x): return float(x.min()), float(x.max())
    return {
        'a_min': mm(a)[0], 'a_max': mm(a)[1],
        'b_min': mm(b)[0], 'b_max': mm(b)[1],
        'c_min': mm(c)[0], 'c_max': mm(c)[1],
        'd_min': mm(d)[0], 'd_max': mm(d)[1],
        'h_mean': h_mean, 'k': k
    }



def _solve_single_channel_LMI(alpha, A_list, B, w_acc_max, verbosity=0, solver=None):
    """
    Solve:  [A^T P + P A + 2 alpha P,  P B;
             B^T P,                   -mu I] << 0  for all A in A_list
    minimizing mu. Returns P, mu, cost = mu * w_acc_max^2.
    """
    n = A_list[0].shape[0]
    m = B.shape[1]
    P = cp.Variable((n, n), symmetric=True)
    mu = cp.Variable(nonneg=True)

    constraints = [P >>np.eye(n)]
    for A in A_list:
        M11 = A.T @ P + P @ A + alpha*P
        M12 = P @ B
        M21 = M12.T
        M22 = -alpha * mu * np.eye(m)
        M = cp.bmat([[M11, M12],
                     [M21, M22]])
        constraints += [M << 0]

    obj = cp.Minimize(mu * (w_acc_max**2))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbosity > 1)

    return {
        'P': P.value,
        'mu': float(mu.value) if mu.value is not None else None,
        'cost': float(mu.value) * (w_acc_max**2) if mu.value is not None else np.inf,
        'prob': prob
    }

def _A_from_abcd(a, b, c, d):
    """
    First-order LTV matrix from YA Eq. 15 (Tschauner-Hempel equations).
      a = ω^2,  b = ω,  c = ωdot,  d = k ω^(3/2)
    State X = [x, y, z, vx, vy, vz]^T (6D: position + velocity in LVLH frame)
    Disturbance enters as acceleration (B = [0; I]).
    """
    A = np.array([
        [0, 0, 0, 1,   0,    0],
        [0, 0, 0, 0,   1,    0],
        [0, 0, 0, 0,   0,    1],
        [a - d, 0,    c, 0,   0,  2*b],
        [0,   -d,    0, 0,   0,    0],
        [-c,   0, a + 2*d, -2*b, 0, 0]
    ], dtype=float)

    return A

def _B_from_abcd(a, b, c, d):
    """
    Control input matrix for TH-LTV system.
    Control enters as acceleration in LVLH frame.
    Returns B0: (6 x 3) with control entering as acceleration.
    """
    B0 = np.vstack([
        np.zeros((3, 3)),  # No direct control of position
        np.eye(3)           # Control enters as acceleration
    ])
    return B0

def solve_YA_TH_LTV_invariant_set_from_bounds(bounds, w_acc_max,
                                              Kp=None, Kd=None,
                                              alpha_lo=1e-6, verbosity=0, solver=None):
    """
    Polytopic LTV invariant-set certification for YA/TH time-domain dynamics.
    Parameters
    ----------
    bounds : dict
        Must contain min/max for the four parameters:
          a_min,a_max   (a = ω^2)
          b_min,b_max   (b = ω)
          c_min,c_max   (c = ωdot)
          d_min,d_max   (d = k ω^(3/2))
        (You can produce these via your own trajectory-preprocess, e.g., th_param_bounds_from_traj(...).)
    w_acc_max : float
        Euclidean bound on LVLH acceleration disturbance (||w||_2 <= w_acc_max).
    Kp, Kd : None | scalar | 3-vector
        Optional diagonal PD closure applied through the same channel B0 = [0; I].
        If provided, A_cl = A - B0 * [diag(Kp), diag(Kd)].
    alpha_lo : float
        Lower bound for line search on alpha (>0).
    verbosity : int
        0 quiet, 1 prints summary, >1 passes verbose to the solver.
    solver : str | None
        cvxpy solver name (e.g., "MOSEK", "SCS", etc.).

    Returns
    -------
    dict with:
        'P'        : (6x6) Lyapunov matrix
        'mu'       : disturbance gain
        'cost'     : mu * w_acc_max^2 (tube radius^2)
        'alpha'    : selected decay rate
        'prob'     : cvxpy Problem from the final solve
        'A_vertices': list of A vertex matrices used
        'B'        : (6x3) disturbance matrix used (=[0;I])
        'eigA_mid' : eigenvalues of midpoint A (diagnostic)
    """
    # Build the 2^4 vertices over (a,b,c,d)
    a_vals = [bounds['a_min'], bounds['a_max']]
    b_vals = [bounds['b_min'], bounds['b_max']]
    c_vals = [bounds['c_min'], bounds['c_max']]
    d_vals = [bounds['d_min'], bounds['d_max']]

      # Dirbance enters as acceleration
    B1 = np.vstack([np.zeros((3,3)), np.eye(3)])
    # Control B



    # PD gain matrix (always scalar Kp,Kd)
    K = np.hstack([Kp * np.eye(3), Kd * np.eye(3)])   # 3×6

    # Build all 2^4 vertices
    A_list = []
    for a, b, c, d in itertools.product(a_vals, b_vals, c_vals, d_vals):
        A = _A_from_abcd(a, b, c, d)
        B0 = _B_from_abcd(a, b, c, d)
        A_cl = A - B0 @ K     # PD closed-loop
        A_list.append(A_cl)

    # Midpoint matrix for a rough alpha bound
    amid = 0.5*(a_vals[0] + a_vals[1])
    bmid = 0.5*(b_vals[0] + b_vals[1])
    cmid = 0.5*(c_vals[0] + c_vals[1])
    dmid = 0.5*(d_vals[0] + d_vals[1])
    B0_mid = _B_from_abcd(amid, bmid, cmid, dmid)
    A_mid = _A_from_abcd(amid, bmid, cmid, dmid) - B0_mid @ K

    eigA = np.linalg.eigvals(A_mid)
    alpha_upper = float(max(1e-5, -np.real(eigA).max()))
    if alpha_upper <= alpha_lo:
        # if midpoint is unstable, give a small positive window
        alpha_upper = max(alpha_lo * 10.0, 1e-4)

    if verbosity > 0:
        print(f"eig(A_mid): {eigA}")
        print(f"alpha search in [{alpha_lo}, {alpha_upper}] using {len(A_list)} vertices")

    # 1D search over alpha to minimize cost = mu/(2*alpha) * w^2 (implemented inside as mu * w^2 objective;
    # minimizing mu suffices because alpha is fixed per call; outer search varies alpha)
    def objective(alpha):
        res = _solve_single_channel_LMI(alpha, A_list, B1, w_acc_max, verbosity=0, solver=solver)
        return res['cost']

    alpha_opt = scipy.optimize.fminbound(objective, x1=alpha_lo, x2=alpha_upper, disp=(verbosity > 0))
    sol = _solve_single_channel_LMI(alpha_opt, A_list, B1, w_acc_max, verbosity=verbosity, solver=solver)

    # Status check
    if sol['prob'] is None or sol['prob'].status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"YA/TH LMI failed at α = {alpha_opt:.6g} "
                           f"(status: {sol['prob'].status if sol['prob'] else 'no_prob'})")

    return {
        'P': sol['P'],
        'mu': sol['mu'],
        'cost': sol['cost'],
        'alpha': alpha_opt,
        'prob': sol['prob'],
        'A_vertices': A_list,
        'B': B,
        'eigA_mid': eigA,
    }


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
