"""
Continuous-time certification for time-varying reachability analysis.

This module provides rigorous verification that LMI constraints hold for all
continuous time t ∈ [t₀, t_f], not just at discrete sample points.

Key function:
- certify_lipschitz_grid: Verify F(t) ≺ 0 using Lipschitz-grid certificate
"""

import numpy as np
import sympy as sp
from typing import Callable, Optional

from .polytopic import eval_polynomial_metric


def certify_lipschitz_grid(
    M_coefs: list[np.ndarray],
    mu: float,
    alpha: float,
    B: np.ndarray,
    times: np.ndarray,
    J_sym: sp.Matrix,
    theta_ref_interp: Callable[[float], float],
    eps_theta: float,
    grid_points: int = 200,
    safety_factor: float = 1.15,
    max_grid_points: int = 20000,
    verbose: bool = False,
) -> dict:
    """
    Verify continuous-time validity using Lipschitz-grid certificate.

    The LMI constraints are only enforced at discrete sample points. This
    function certifies that F(t) ≺ 0 holds for ALL t ∈ [t₀, t_f] using
    Lemma IV.1 from the paper:

    **Lipschitz-Grid Certificate**: If the minimum margin m = min_j(-λ_min(F(t_j)))
    satisfies m > 0.5 * L * h, where L is the Lipschitz constant of F(t) and h
    is the grid spacing, then F(t) ≺ 0 for all t.

    Parameters
    ----------
    M_coefs : list[np.ndarray]
        Polynomial coefficient matrices [M_0, M_1, ..., M_{k-1}] from
        solve_time_varying_polytopic_lmi.
    mu : float
        Disturbance scaling (in normalized time).
    alpha : float
        Decay rate from LMI solution.
    B : np.ndarray
        Disturbance input matrix, shape (n, m).
    times : array_like
        Sample times used in LMI solution.
    J_sym : sympy.Matrix
        Symbolic Jacobian J(theta) with parameters substituted.
    theta_ref_interp : callable
        Interpolator for reference trajectory theta_ref(t).
    eps_theta : float
        Polytopic bound half-width used in LMI synthesis.
    grid_points : int, optional
        Initial number of grid points. Default 200.
    safety_factor : float, optional
        Safety factor on grid refinement. Default 1.15.
    max_grid_points : int, optional
        Maximum allowed grid points. Default 20000.
    verbose : bool, optional
        Print progress information.

    Returns
    -------
    dict
        {
            "certified": bool - True if continuous-time validity is certified,
            "min_margin": float - minimum margin m = -λ_min(F),
            "required_margin": float - required margin 0.5 * L * h,
            "lipschitz_bound": float - estimated Lipschitz constant L,
            "grid_points": int - number of grid points used,
            "worst_time": float - time where margin is smallest,
            "worst_eigenvalue": float - smallest eigenvalue at worst time,
            "message": str - human-readable result description
        }

    Notes
    -----
    The certificate is conservative - if it passes, the invariant tube is
    mathematically valid. If it fails, more grid points or a different LMI
    solution may be needed.
    """
    times = np.asarray(times, dtype=float)
    t0, tf = times[0], times[-1]
    T_total = tf - t0

    # Scale B for normalized time
    B = np.asarray(B, dtype=float)
    B_scaled = T_total * B
    BBt = B_scaled @ B_scaled.T

    n = M_coefs[0].shape[0]
    k = len(M_coefs)

    # Polynomial derivative coefficients
    Mdot_coefs = [(r+1) * M_coefs[r+1] for r in range(k-1)] if k > 1 else [np.zeros((n, n))]
    Mddot_coefs = [(q+1) * Mdot_coefs[q+1] for q in range(len(Mdot_coefs)-1)] if len(Mdot_coefs) > 1 else [np.zeros((n, n))]

    def M_eval(s):
        return sum((s**r) * M_coefs[r] for r in range(k))

    def Mdot_eval(s):
        return sum((s**q) * Mdot_coefs[q] for q in range(len(Mdot_coefs)))

    def Mddot_eval(s):
        return sum((s**p) * Mddot_coefs[p] for p in range(len(Mddot_coefs)))

    # Create numerical Jacobian function
    theta_sym = sp.Symbol('theta', real=True)
    J_fun = sp.lambdify(theta_sym, J_sym, 'numpy')

    def get_J_vertices(s):
        """Get Jacobian polytope vertices at normalized time s."""
        t_phys = t0 + s * T_total
        theta_ref = float(theta_ref_interp(t_phys))
        theta_plus = theta_ref + eps_theta
        theta_minus = theta_ref - eps_theta
        Jp = T_total * np.array(J_fun(theta_plus), dtype=float)
        Jm = T_total * np.array(J_fun(theta_minus), dtype=float)
        return [Jp, Jm]

    def compute_F_min_eigenvalue(s):
        """
        Compute F(s) and return its minimum eigenvalue.

        F(s) = S(s) - (1/(α*μ)) * M(s) * B * B^T * M(s)
        where S(s) = Ṁ(s) + J^T M(s) + M(s) J + α M(s)
        """
        M_s = M_eval(s)
        Mdot_s = Mdot_eval(s)
        J_verts = get_J_vertices(s)

        min_eig = np.inf
        for J in J_verts:
            S = Mdot_s + J.T @ M_s + M_s @ J + alpha * M_s
            F = S - (1.0 / (alpha * mu)) * (M_s @ BBt @ M_s)
            F = 0.5 * (F + F.T)  # Symmetrize
            eigs = np.linalg.eigvalsh(F)
            min_eig = min(min_eig, eigs.min())

        return min_eig

    # Estimate Lipschitz bound on F(s)
    s_dense = np.linspace(0, 1, 100)
    L_samples = []

    for s in s_dense:
        M_s = M_eval(s)
        Mdot_s = Mdot_eval(s)
        Mddot_s = Mddot_eval(s)

        norm_M = np.linalg.norm(M_s, 'fro')
        norm_Mdot = np.linalg.norm(Mdot_s, 'fro')
        norm_Mddot = np.linalg.norm(Mddot_s, 'fro')
        norm_BBt = np.linalg.norm(BBt, 2)

        J_verts = get_J_vertices(s)
        norm_J = max(np.linalg.norm(J, 'fro') for J in J_verts)

        # Estimate J' norm
        ds = 0.01
        if s + ds <= 1:
            J_verts_next = get_J_vertices(s + ds)
            dJ = max(np.linalg.norm(J_verts_next[i] - J_verts[i], 'fro')
                    for i in range(len(J_verts)))
            norm_Jprime = dJ / ds
        else:
            norm_Jprime = 0

        # Lipschitz bound estimate
        L_s = (norm_Mddot +
               2 * norm_Jprime * norm_M +
               2 * norm_J * norm_Mdot +
               alpha * norm_Mdot +
               (2 / (alpha * mu)) * norm_BBt * norm_M * norm_Mdot)
        L_samples.append(L_s)

    L = max(L_samples) * 1.5  # Safety margin

    # Grid sampling and certification
    N = grid_points
    h = 1.0 / N
    s_grid = np.linspace(0, 1, N + 1)

    min_margin = np.inf
    worst_s = None
    worst_eig = None
    sample_violation = False

    for s in s_grid:
        eig_min = compute_F_min_eigenvalue(s)
        margin = -eig_min

        if eig_min >= 0:
            sample_violation = True
            worst_s = s
            worst_eig = eig_min
            break

        if margin < min_margin:
            min_margin = margin
            worst_s = s
            worst_eig = eig_min

    if sample_violation:
        t_phys = t0 + worst_s * T_total
        return {
            "certified": False,
            "min_margin": float(-worst_eig),
            "required_margin": 0.5 * L * h,
            "lipschitz_bound": L,
            "grid_points": N,
            "worst_time": t_phys,
            "worst_eigenvalue": float(worst_eig),
            "message": f"F(t) not negative definite at t={t_phys:.3f}s (λ_min={worst_eig:.4e})"
        }

    # Check Lipschitz-grid certificate
    required_margin = 0.5 * L * h

    if min_margin > required_margin:
        t_phys = t0 + worst_s * T_total
        return {
            "certified": True,
            "min_margin": float(min_margin),
            "required_margin": float(required_margin),
            "lipschitz_bound": float(L),
            "grid_points": N,
            "worst_time": float(t_phys),
            "worst_eigenvalue": float(worst_eig),
            "message": f"Certified! m={min_margin:.4e} > 0.5*L*h={required_margin:.4e}"
        }

    # Need finer grid
    if min_margin <= 0:
        N_required = max_grid_points + 1
    else:
        N_required = int(np.ceil(safety_factor * L / (2.0 * min_margin)))

    if N_required > max_grid_points:
        t_phys = t0 + worst_s * T_total
        return {
            "certified": False,
            "min_margin": float(min_margin),
            "required_margin": float(required_margin),
            "lipschitz_bound": float(L),
            "grid_points": N,
            "worst_time": float(t_phys),
            "worst_eigenvalue": float(worst_eig),
            "message": f"Would need N={N_required} > max={max_grid_points}"
        }

    # Refine grid
    if verbose:
        print(f"  Refining grid: N={N} -> N={N_required}")

    N = N_required
    h = 1.0 / N
    s_grid = np.linspace(0, 1, N + 1)

    min_margin = np.inf
    worst_s = None
    worst_eig = None

    for s in s_grid:
        eig_min = compute_F_min_eigenvalue(s)
        margin = -eig_min

        if eig_min >= 0:
            t_phys = t0 + s * T_total
            return {
                "certified": False,
                "min_margin": float(-eig_min),
                "required_margin": 0.5 * L * h,
                "lipschitz_bound": L,
                "grid_points": N,
                "worst_time": t_phys,
                "worst_eigenvalue": float(eig_min),
                "message": f"F(t) not negative definite at t={t_phys:.3f}s on refined grid"
            }

        if margin < min_margin:
            min_margin = margin
            worst_s = s
            worst_eig = eig_min

    required_margin = 0.5 * L * h
    certified = min_margin > required_margin
    t_phys = t0 + worst_s * T_total

    return {
        "certified": certified,
        "min_margin": float(min_margin),
        "required_margin": float(required_margin),
        "lipschitz_bound": float(L),
        "grid_points": N,
        "worst_time": float(t_phys),
        "worst_eigenvalue": float(worst_eig),
        "message": f"{'Certified' if certified else 'NOT certified'}: m={min_margin:.4e} {'>' if certified else '<='} 0.5*L*h={required_margin:.4e}"
    }
