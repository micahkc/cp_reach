from __future__ import annotations

import numpy as np
import scipy.optimize
import cvxpy as cp


def _parse_w_max(w_max, m: int):
    if w_max is None:
        return None  # signal optional
    if np.isscalar(w_max):
        return np.full(m, float(w_max))
    w_vec = np.array(w_max, dtype=float).ravel()
    if w_vec.size != m:
        raise ValueError(f"w_max length {w_vec.size} must match number of disturbance channels {m}")
    return w_vec


def _solve_multi_channel_LMI(alpha, A_list, B, w_max=None, verbosity=0, solver=None):
    """
    Certify V(x)=x^T P x <= sum_i mu_i * w_i^2 (or sum_i mu_i if w_max is None)
    for xdot = A_i x + B d with possibly multiple disturbance channels.
    """
    n = A_list[0].shape[0]
    m = B.shape[1]

    w_vec = _parse_w_max(w_max, m)  # None or length-m vector

    P = cp.Variable((n, n), symmetric=True)
    mu = cp.Variable(m, nonneg=True)

    I_n = np.eye(n)

    constraints = [P >>  I_n]

    for A in A_list:
        TL = A.T @ P + P @ A + alpha * P
        BR = -alpha * cp.diag(mu)
        LMI = cp.bmat([[TL, P @ B], [B.T @ P, BR]])
        constraints.append(LMI << 0)

    if w_vec is None:
        objective = cp.Minimize(cp.sum(mu))
    else:
        objective = cp.Minimize(cp.sum(cp.multiply(mu, w_vec ** 2)))

    prob = cp.Problem(objective, constraints)

    try:
        chosen = solver
        if chosen is None:
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

        mu_val = None if mu.value is None else np.array(mu.value, dtype=float)
        return {
            "P": None if P.value is None else P.value,
            "mu": mu_val,
            "cost": np.inf if prob.value is None else float(prob.value),
            "alpha": alpha,
            "prob": prob,
            "status": prob.status,
        }
    except Exception:
        return {"P": None, "mu": None, "cost": np.inf, "alpha": alpha, "prob": None, "status": "error"}


def find_feasible_alpha(A_list, B, w_max=None, alphas=None, verbosity=0, solver=None):
    """
    Scan a grid of alphas and return the first feasible solution to the disturbance LMI.

    Parameters
    ----------
    A_list : list[np.ndarray]
        List of A matrices (polytopic vertices)
    B : np.ndarray
        Disturbance input matrix (n × m)
    w_max : optional
        Disturbance bounds as in solve_disturbance_LMI
    alphas : iterable[float], optional
        Grid of alphas to try. Default: logspace from 1e-4 to 10 (40 points).
    verbosity : int
        CVXPY verbosity
    solver : optional
        CVXPY solver to use

    Returns
    -------
    tuple(alpha, sol_dict)
        The first feasible alpha and its solution dict.

    Raises
    ------
    RuntimeError
        If no feasible alpha is found on the grid.
    """
    if alphas is None:
        alphas = np.logspace(-4, 1, 40)

    for alpha in alphas:
        sol = _solve_multi_channel_LMI(alpha, A_list, B, w_max, verbosity=verbosity, solver=solver)
        if sol["status"] in ("optimal", "optimal_inaccurate"):
            return alpha, sol

    raise RuntimeError("No feasible alpha found in the provided grid.")


def solve_disturbance_LMI(
    A_list,
    B,
    w_max=None,
    alpha_bounds=(1e-6, 1.0),
    alpha_grid=None,
    verbosity: int = 0,
    solver=None,
):
    """
    Line-search over alpha to solve the disturbance LMI and return best certificate.

    w_max:
      - None: minimize sum(mu_i)
      - scalar: broadcast to all disturbance channels
      - length-m array/list: per-channel bounds
    alpha_grid:
      - Optional iterable of alpha values to scan for feasibility before line search
    """
    feasible_alpha = None
    feasible_sol = None
    if alpha_grid is not None:
        try:
            feasible_alpha, feasible_sol = find_feasible_alpha(
                A_list, B, w_max, alphas=alpha_grid, verbosity=verbosity, solver=solver
            )
        except RuntimeError:
            feasible_sol = None

    def objective(alpha):
        return _solve_multi_channel_LMI(alpha, A_list, B, w_max, verbosity=verbosity, solver=solver)["cost"]

    alpha_opt = scipy.optimize.fminbound(objective, x1=alpha_bounds[0], x2=alpha_bounds[1], disp=(verbosity > 0))
    sol = _solve_multi_channel_LMI(alpha_opt, A_list, B, w_max, verbosity=verbosity, solver=solver)

    if sol["prob"] is None or sol["status"] not in ("optimal", "optimal_inaccurate"):
        if feasible_sol is not None:
            feasible_sol["alpha"] = feasible_sol.get("alpha", feasible_alpha)
            mu_vec = feasible_sol.get("mu")
            if mu_vec is None:
                feasible_sol["radius_inf"] = np.inf
            else:
                if w_max is None:
                    feasible_sol["radius_inf"] = float(np.sqrt(np.max(mu_vec)))
                else:
                    w_vec = _parse_w_max(w_max, B.shape[1])
                    feasible_sol["radius_inf"] = float(np.max(np.sqrt(mu_vec) * w_vec))
            return feasible_sol
        raise RuntimeError(f"LMI solve failed at alpha={alpha_opt} (status={sol.get('status')})")

    sol["alpha"] = alpha_opt
    mu_vec = sol["mu"]
    if mu_vec is None:
        sol["radius_inf"] = np.inf
    else:
        if w_max is None:
            sol["radius_inf"] = float(np.sqrt(np.max(mu_vec)))
        else:
            w_vec = _parse_w_max(w_max, B.shape[1])
            sol["radius_inf"] = float(np.max(np.sqrt(mu_vec) * w_vec))
    return sol


def _solve_bounded_disturbance_output_LMI(alpha, A_list, B, C, D, eps=1e-6, verbosity=0, solver=None):
    """
    Solve the bounded-disturbance output LMIs (13.10a–b):
        [PA + AᵀP + αP, PB]            ⪯ 0
        [BᵀP           , -α μ₁ I]

        [CᵀC - P, CᵀD]                 ⪯ 0
        [DᵀC   , DᵀD - μ₂ I]

    for the system ẋ = A x + B w,  z = C x + D w. The result certifies
    an output gain bound γ = √(μ₁ + μ₂) between bounded disturbances and outputs.

    Parameters
    ----------
    alpha : float
        Lyapunov decay rate parameter α > 0
    A_list : list of np.ndarray
        List of A matrices (polytopic vertices)
    B : np.ndarray
        Input matrix (n × m)
    C : np.ndarray
        Output matrix (p × n)
    D : np.ndarray
        Feedthrough matrix (p × m)
    eps : float, optional
        Minimum eigenvalue for P >> 0
    verbosity : int, optional
        CVXPY verbosity
    solver : optional
        CVXPY solver to use

    Returns
    -------
    dict with keys P, mu1, mu2, gamma, cost, alpha, prob, status
    """
    n = A_list[0].shape[0]  # state dimension
    m = B.shape[1]          # disturbance dimension
    p = C.shape[0]          # output dimension

    if B.shape[0] != n:
        raise ValueError(f"B must have {n} rows to match A, got {B.shape[0]}")
    if D.shape != (p, m):
        raise ValueError(f"D must be {p}×{m} to match C and disturbance dimension, got {D.shape}")

    # Decision variables
    P = cp.Variable((n, n), symmetric=True)
    mu1 = cp.Variable(nonneg=True)  # scalar μ₁
    mu2 = cp.Variable(nonneg=True)  # scalar μ₂

    I_n = np.eye(n)
    I_m = np.eye(m)

    constraints = [P >> eps * I_n]

    # LMI (13.10a): stability with bounded disturbance
    for A in A_list:
        TL = A.T @ P + P @ A + alpha * P
        TR = P @ B
        BL = B.T @ P
        BR = -alpha * mu1 * I_m
        LMI_1 = cp.bmat([[TL, TR],
                         [BL, BR]])
        constraints.append(LMI_1 << 0)

    # LMI (13.10b): output bound
    TL_2 = C.T @ C - P
    TR_2 = C.T @ D
    BL_2 = D.T @ C
    BR_2 = D.T @ D - mu2 * I_m
    LMI_2 = cp.bmat([[TL_2, TR_2],
                     [BL_2, BR_2]])
    constraints.append(LMI_2 << 0)

    # Minimize μ₁ + μ₂ so that γ = √(μ₁ + μ₂) is as small as possible
    objective = cp.Minimize(mu1 + mu2)

    prob = cp.Problem(objective, constraints)

    try:
        chosen = solver
        if chosen is None:
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

        mu1_val = None if mu1.value is None else float(mu1.value)
        mu2_val = None if mu2.value is None else float(mu2.value)

        if mu1_val is not None and mu2_val is not None:
            gamma = np.sqrt(mu1_val + mu2_val)
        else:
            gamma = np.inf

        return {
            "P": None if P.value is None else P.value,
            "mu1": mu1_val,
            "mu2": mu2_val,
            "gamma": gamma,
            "cost": np.inf if prob.value is None else float(prob.value),
            "alpha": alpha,
            "prob": prob,
            "status": prob.status,
        }
    except Exception:
        return {
            "P": None,
            "mu1": None,
            "mu2": None,
            "gamma": np.inf,
            "cost": np.inf,
            "alpha": alpha,
            "prob": None,
            "status": "error"
        }


def solve_bounded_disturbance_output_LMI(
    A_list,
    B,
    C,
    D,
    alpha_bounds=(1e-6, 10.0),
    verbosity: int = 0,
    solver=None,
):
    """
    Compute output bound for system with bounded disturbance input.

    Considers the system:
        ẋ = Ax + Bw
        z = Cx + Dw

    where all eigenvalues of A have negative real part and w is bounded.

    Theorem: If there exists P > 0, α > 0, μ₁, μ₂ such that:

        [P*A + A'*P + α*P,  P*B     ] ≤ 0   (13.10a)
        [B'*P,              -α*μ₁*I ]

        [C'*C - P,  C'*D        ] ≤ 0       (13.10b)
        [D'*C,      D'*D - μ₂*I ]

    Then the output remains bounded with gain γ = √(μ₁ + μ₂) against bounded
    disturbances. When x(0) = 0 the same bound holds for all time.

    Parameters
    ----------
    A_list : list of np.ndarray or np.ndarray
        System matrix A (n × n), or list of vertices for polytopic uncertainty
    B : np.ndarray
        Input matrix (n × m)
    C : np.ndarray
        Output matrix (p × n)
    D : np.ndarray
        Feedthrough matrix (p × m)
    alpha_bounds : tuple, optional
        Bounds for line search over α parameter, default (1e-6, 10.0)
    verbosity : int, optional
        Verbosity level (0=silent, >0=verbose), default 0
    solver : optional
        CVXPY solver (e.g., cp.MOSEK, cp.CVXOPT, cp.SCS), default None (auto-select)

    Returns
    -------
    dict
        Solution dictionary containing:
        - P : np.ndarray - Lyapunov matrix (n × n)
        - mu1 : float - State disturbance bound parameter (scalar)
        - mu2 : float - Output disturbance bound parameter (scalar)
        - gamma : float - gain bound γ = √(μ₁ + μ₂)
        - alpha : float - Optimal decay rate
        - cost : float - Objective value (μ₁ + μ₂)
        - prob : cvxpy.Problem - CVXPY problem object
        - status : str - Solver status

    Raises
    ------
    RuntimeError
        If LMI solve fails

    Examples
    --------
    >>> import numpy as np
    >>> from cp_reach.reachability.lmi import solve_bounded_disturbance_output_LMI
    >>>
    >>> # Mass-spring-damper with PD feedback (single disturbance d)
    >>> A = np.array([[0, 1, 0, 0],
    ...               [-1, -0.2, 0, 0],
    ...               [0, 0, 0, 1],
    ...               [2, 0.8, -3, -1]])
    >>> B = np.array([[0], [0], [0], [1]])  # disturbance d affects last state
    >>> C = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])  # tracking errors [e, ev]
    >>> D = np.zeros((2, 1))  # no direct feedthrough
    >>>
    >>> sol = solve_bounded_disturbance_output_LMI(A, B, C, D)
    >>> print(f"Gain bound: γ = {sol['gamma']:.4f}")
    >>> print(f"Optimal α = {sol['alpha']:.4f}")

    Notes
    -----
    - This generalizes to polytopic systems by passing multiple A matrices in A_list
    - For control design, C typically represents tracking errors or regulated outputs
    - The bound holds for any bounded disturbance; smaller γ indicates better rejection
    """
    # Handle single A matrix
    if isinstance(A_list, np.ndarray):
        A_list = [A_list]

    # Validate dimensions
    n = A_list[0].shape[0]
    p = C.shape[0]

    if B.shape[0] != n:
        raise ValueError(f"B must have {n} rows to match A, got {B.shape[0]}")
    if B.shape[1] != 1:
        raise ValueError(f"B must have exactly 1 column (single disturbance), got {B.shape[1]}")
    if C.shape[1] != n:
        raise ValueError(f"C must have {n} columns to match A, got {C.shape[1]}")
    if D.shape != (p, 1):
        raise ValueError(f"D must be {p}×1 to match C and single disturbance, got {D.shape}")

    def objective(alpha):
        return _solve_bounded_disturbance_output_LMI(alpha, A_list, B, C, D, verbosity=verbosity, solver=solver)["cost"]

    alpha_opt = scipy.optimize.fminbound(
        objective,
        x1=alpha_bounds[0],
        x2=alpha_bounds[1],
        disp=(verbosity > 0)
    )

    sol = _solve_bounded_disturbance_output_LMI(alpha_opt, A_list, B, C, D, verbosity=verbosity, solver=solver)

    if sol["prob"] is None or sol["status"] not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"Bounded disturbance LMI solve failed at alpha={alpha_opt} "
            f"(status={sol.get('status')})"
        )

    return sol
