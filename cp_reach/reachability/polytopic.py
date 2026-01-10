"""
Time-varying polytopic LMI solver for nonlinear reachability analysis.

This module provides tools for computing reachable sets for nonlinear systems
using polytopic bounds on the Jacobian along a reference trajectory.

Key functions:
- polytopic_jacobians: Compute Jacobian polytope vertices at trajectory points
- solve_time_varying_polytopic_lmi: Solve for polynomial Lyapunov function M(t)
- eval_polynomial_metric: Evaluate M(t) at a given time
- project_metric_2d: Project metric to 2D subspace via Schur complement
"""

import numpy as np
import sympy as sp
import cvxpy as cp
from scipy.optimize import fminbound
from typing import Optional


def polytopic_jacobians(
    J_sym: sp.Matrix,
    ref_vals: np.ndarray | dict[sp.Symbol, np.ndarray],
    err_bounds: float | dict[sp.Symbol, float],
    state_symbols: list[sp.Symbol],
) -> list[list[np.ndarray]]:
    """
    Compute polytopic Jacobian bounds at reference trajectory points.

    For nonlinear systems, the Jacobian J(x) depends on the state. This function
    computes polytope vertices by evaluating J at all corners of the error tube
    for each nonlinear state variable.

    Parameters
    ----------
    J_sym : sympy.Matrix
        Symbolic Jacobian matrix J(x) = ∂f/∂x with parameters already substituted.
        Should still contain the nonlinear state variable(s).
    ref_vals : np.ndarray or dict[sp.Symbol, np.ndarray]
        Reference trajectory values at sample times. Either:
        - 1D array of shape (N,) for single-variable case (uses first state symbol)
        - Dict mapping state symbols to their reference trajectories, each shape (N,)
    err_bounds : float or dict[sp.Symbol, float]
        Half-width of the polytopic bounds. Either:
        - Single float for single-variable case (applies to first state symbol)
        - Dict mapping state symbols to their error bounds
    state_symbols : list[sp.Symbol]
        List of sympy symbols for states that appear nonlinearly in J.
        For backwards compatibility, if ref_vals is an array, uses first element.

    Returns
    -------
    list[list[np.ndarray]]
        List of length N, where each element is a list of Jacobian matrices
        (polytope vertices) at that time point. For n nonlinear states, returns
        2^n vertices per time point.

    Example
    -------
    Single nonlinear state (backwards compatible):
    >>> J_sym = sp.Matrix([[0, 1], [-sp.cos(theta), -c]])
    >>> theta_refs = np.array([0.0, 0.5, 1.0])
    >>> polytopes = polytopic_jacobians(J_sym, theta_refs, 0.1, [theta])
    >>> len(polytopes)  # One polytope per reference point
    3
    >>> len(polytopes[0])  # Two vertices per polytope
    2

    Multiple nonlinear states:
    >>> J_sym = sp.Matrix([[sp.sin(x1), sp.cos(x2)], [0, -1]])
    >>> refs = {x1: np.array([0.0, 0.5]), x2: np.array([1.0, 1.5])}
    >>> bounds = {x1: 0.1, x2: 0.2}
    >>> polytopes = polytopic_jacobians(J_sym, refs, bounds, [x1, x2])
    >>> len(polytopes[0])  # Four vertices (2^2) per polytope
    4
    """
    import itertools

    # Handle backwards-compatible single-variable case
    if not isinstance(ref_vals, dict):
        # Single array/list provided - use first state symbol
        nonlinear_symbols = [state_symbols[0]]
        ref_vals_dict = {state_symbols[0]: np.asarray(ref_vals)}
    else:
        ref_vals_dict = ref_vals
        nonlinear_symbols = list(ref_vals_dict.keys())

    if isinstance(err_bounds, (int, float)):
        # Single bound provided - apply to all nonlinear symbols
        err_bounds_dict = {sym: float(err_bounds) for sym in nonlinear_symbols}
    else:
        err_bounds_dict = err_bounds

    # Get number of time samples from first reference trajectory
    first_ref = next(iter(ref_vals_dict.values()))
    N = len(first_ref)

    # Build polytope vertices at each time point
    ref_jacobians = []
    n_vars = len(nonlinear_symbols)

    for i in range(N):
        J_vertices = []

        # Generate all 2^n corner combinations: each variable at ref ± err
        for signs in itertools.product([-1, 1], repeat=n_vars):
            subs_dict = {}
            for sym, sign in zip(nonlinear_symbols, signs):
                ref_val = ref_vals_dict[sym][i]
                err = err_bounds_dict[sym]
                subs_dict[sym] = ref_val + sign * err

            J_eval = J_sym.subs(subs_dict)
            J_vertices.append(np.array(J_eval, dtype=float))

        ref_jacobians.append(J_vertices)

    return ref_jacobians


def solve_time_varying_polytopic_lmi(
    J_vertices_by_time: list[list[np.ndarray]],
    B: np.ndarray,
    times: np.ndarray,
    polynomial_degree: Optional[int] = None,
    alpha_bounds: tuple[float, float] = (0.01, 1.0),
    solver: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Solve time-varying polytopic LMI for polynomial Lyapunov function.

    Parameterizes the metric as M(t) = Σ_{r=0}^{k-1} t^r M_r and solves for
    the coefficient matrices M_r that minimize the disturbance gain μ.

    For each time t_i and each Jacobian vertex J_i^(ℓ), enforces:

        [S(t_i, J_i^(ℓ)) + αM(t_i)    M(t_i)B  ]
        [B^T M(t_i)                   -αμI     ] ≼ 0

    where S(t,J) = Ṁ(t) + J^T M(t) + M(t)J is the Lyapunov derivative.

    Parameters
    ----------
    J_vertices_by_time : list[list[np.ndarray]]
        Jacobian polytope vertices at each sample time. Output from
        polytopic_jacobians().
    B : np.ndarray
        Disturbance input matrix, shape (n, m).
    times : array_like
        Sample times, shape (k,). Used for time normalization.
    polynomial_degree : int, optional
        Degree of polynomial for M(t). Defaults to len(times) (one coefficient
        per sample point).
    alpha_bounds : tuple, optional
        Search bounds for decay rate α. Default (0.01, 1.0).
    solver : str, optional
        CVXPY solver name. If None, auto-selects.
    verbose : bool, optional
        Print optimization progress.

    Returns
    -------
    dict
        {
            "M_coefs": list of (n,n) arrays [M_0, M_1, ..., M_{k-1}],
            "mu": optimal disturbance scaling (normalized time),
            "mu_physical": mu / T (physical time units),
            "alpha": optimal decay rate,
            "status": solver status string,
            "times": sample times used
        }

    Notes
    -----
    Time is normalized to [0, 1] internally for numerical stability. The
    returned mu is in normalized time; use mu_physical for bounds computation.
    """
    times = np.asarray(times, dtype=float)
    T = times[-1] - times[0]
    t_norm = (times - times[0]) / T

    B = np.asarray(B, dtype=float)
    B_scaled = T * B  # Scale B for normalized time

    n = J_vertices_by_time[0][0].shape[0]
    m = B_scaled.shape[1]
    k = polynomial_degree if polynomial_degree is not None else len(times)

    I_n = np.eye(n)
    I_m = np.eye(m)
    c_cap = 20.0  # Upper bound on M eigenvalues

    # Decision variables
    M_coefs = [cp.Variable((n, n), symmetric=True, name=f"M_{r}") for r in range(k)]
    mu = cp.Variable(nonneg=True, name="mu")
    alpha = cp.Parameter(nonneg=True, name="alpha")

    def M_of_t(ti):
        return sum((ti**r) * M_coefs[r] for r in range(k))

    def Mdot_of_t(ti):
        if k > 1:
            return sum(r * (ti**(r-1)) * M_coefs[r] for r in range(1, k))
        return np.zeros((n, n))

    # Build constraints
    constraints = []
    for i, Ji_list in enumerate(J_vertices_by_time):
        Mi = M_of_t(t_norm[i])
        Mdot = Mdot_of_t(t_norm[i])

        # M(t) bounds: I ≼ M(t) ≼ c_cap * I
        constraints.append(Mi >> I_n)
        constraints.append(Mi << c_cap * I_n)

        # LMI for each polytope vertex
        for Ji in Ji_list:
            Ji_scaled = T * np.asarray(Ji, dtype=float)
            # S = Ṁ + J^T M + M J + α M
            S = Ji_scaled.T @ Mi + Mi @ Ji_scaled + Mdot + alpha * Mi
            S = 0.5 * (S + S.T)  # Symmetrize
            MB = Mi @ B_scaled

            # Block LMI
            LMI = cp.bmat([
                [S, MB],
                [MB.T, -alpha * mu * I_m]
            ])
            constraints.append(LMI << 0)

    prob = cp.Problem(cp.Minimize(mu), constraints)

    # Select solver
    if solver is None:
        solver = cp.SCS

    def solve_for_alpha(alpha_val):
        """Solve LMI for fixed alpha, return mu."""
        alpha.value = float(alpha_val)
        try:
            prob.solve(solver=solver, warm_start=False, eps=1e-8,
                      max_iters=20000, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return float(mu.value)
        except Exception:
            pass
        return 1e6

    # Coarse sweep to find feasible range
    if verbose:
        print("Searching for optimal alpha...")

    candidates = np.logspace(np.log10(alpha_bounds[0]), np.log10(alpha_bounds[1]), 20)
    vals = np.array([solve_for_alpha(a) for a in candidates])
    finite = np.isfinite(vals) & (vals < 1e5)

    if not finite.any():
        return {
            "M_coefs": None,
            "mu": None,
            "mu_physical": None,
            "alpha": None,
            "status": "infeasible",
            "times": times,
        }

    # Refine around best point
    j = np.argmin(np.where(finite, vals, np.inf))
    a_lo = candidates[max(0, j-1)]
    a_hi = candidates[min(len(candidates)-1, j+1)]

    alpha_star = fminbound(solve_for_alpha, a_lo, a_hi, xtol=1e-3)
    mu_star = solve_for_alpha(alpha_star)

    # Extract solution
    M_coefs_val = [np.array(Mr.value) for Mr in M_coefs]

    if verbose:
        print(f"Optimal alpha: {alpha_star:.4f}")
        print(f"Optimal mu (normalized): {mu_star:.6e}")
        print(f"Optimal mu (physical): {mu_star/T:.6e}")

    return {
        "M_coefs": M_coefs_val,
        "mu": mu_star,
        "mu_physical": mu_star / T,
        "alpha": alpha_star,
        "status": prob.status,
        "times": times,
    }


def eval_polynomial_metric(
    M_coefs: list[np.ndarray],
    t: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    Evaluate polynomial metric M(t) = Σ t^r M_r at a given time.

    Parameters
    ----------
    M_coefs : list[np.ndarray]
        Polynomial coefficient matrices [M_0, M_1, ..., M_{k-1}].
    t : float
        Time at which to evaluate (in physical units).
    times : array_like
        Sample times used in LMI solve (for normalization).

    Returns
    -------
    np.ndarray
        The metric matrix M(t), shape (n, n).
    """
    times = np.asarray(times, dtype=float)
    T = times[-1] - times[0]
    t_norm = (t - times[0]) / max(T, 1e-12)

    return sum((t_norm**r) * M_coefs[r] for r in range(len(M_coefs)))


def project_metric_2d(M: np.ndarray, indices: tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Project metric to 2D subspace using Schur complement.

    For a partitioned metric:
        M = [M_11  M_12]
            [M_21  M_22]

    The projected metric is M_11 - M_12 @ M_22^{-1} @ M_21.

    Parameters
    ----------
    M : np.ndarray
        Full metric matrix, shape (n, n).
    indices : tuple[int, int], optional
        Indices of the 2D subspace to project onto. Default (0, 1).

    Returns
    -------
    np.ndarray
        Projected 2x2 metric matrix.

    Notes
    -----
    This computes the marginal ellipsoid in the selected subspace. For an
    ellipsoid {x : x^T M x ≤ 1}, the projection onto coordinates (i, j) is
    {(x_i, x_j) : [x_i, x_j]^T M_proj [x_i, x_j] ≤ 1}.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]

    # Build index sets
    idx_2d = list(indices)
    idx_rest = [i for i in range(n) if i not in idx_2d]

    if len(idx_rest) == 0:
        return M[np.ix_(idx_2d, idx_2d)]

    # Extract blocks
    M_11 = M[np.ix_(idx_2d, idx_2d)]
    M_12 = M[np.ix_(idx_2d, idx_rest)]
    M_22 = M[np.ix_(idx_rest, idx_rest)]

    # Schur complement: M_11 - M_12 @ M_22^{-1} @ M_12^T
    X = np.linalg.solve(M_22, M_12.T)
    M_proj = M_11 - M_12 @ X

    # Symmetrize to remove numerical asymmetry
    return 0.5 * (M_proj + M_proj.T)


def compute_state_bounds(
    M_coefs: list[np.ndarray],
    mu_physical: float,
    disturbance_bound: float,
    times: np.ndarray,
) -> dict:
    """
    Compute per-state error bounds from the polynomial Lyapunov solution.

    The invariant set is E(t) = { e : e^T M(t) e ≤ μ w̄² }. Per-state bounds
    are computed as radius_i = sqrt(c * [M(t)^{-1}]_{ii}) where c = μ w̄².

    Parameters
    ----------
    M_coefs : list[np.ndarray]
        Polynomial coefficient matrices from solve_time_varying_polytopic_lmi.
    mu_physical : float
        Disturbance scaling in physical time units.
    disturbance_bound : float
        Bound on disturbance magnitude w̄.
    times : array_like
        Sample times for evaluation.

    Returns
    -------
    dict
        {
            "bounds_max": max bounds over time, shape (n,),
            "bounds_over_time": bounds at each time, shape (k, n),
            "c": μ w̄² scaling factor
        }
    """
    times = np.asarray(times, dtype=float)
    c = mu_physical * disturbance_bound**2

    bounds_over_time = []
    for t in times:
        M_t = eval_polynomial_metric(M_coefs, t, times)
        eigvals = np.linalg.eigvals(M_t)
        if np.min(eigvals) < 1e-10:
            n = M_t.shape[0]
            bounds_over_time.append(np.full(n, np.nan))
            continue
        M_inv = np.linalg.inv(M_t)
        radii = np.sqrt(c * np.diag(M_inv))
        bounds_over_time.append(radii)

    bounds_over_time = np.array(bounds_over_time)
    bounds_max = np.nanmax(bounds_over_time, axis=0)

    return {
        "bounds_max": bounds_max,
        "bounds_over_time": bounds_over_time,
        "c": c,
    }
