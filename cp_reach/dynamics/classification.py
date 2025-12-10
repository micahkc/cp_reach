from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

try:
    import casadi as ca  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ca = None


class DynamicsClass(str, Enum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    LIE = "lie"


def _casadi_is_linear(f) -> bool:
    """
    Heuristic: check if Jacobians w.r.t x and u are constant (no dependence on x,u).
    """
    if ca is None:
        return False
    try:
        x_sym = f.sx_in(0)
        u_sym = f.sx_in(1)
    except Exception:
        return False
    args = []
    if x_sym is not None:
        args.append(x_sym)
    if u_sym is not None:
        args.append(u_sym)
    # Ignore t/p inputs for linearity check
    expr = f(*args, *([0] * (f.n_in() - len(args))))
    Jx = ca.jacobian(expr, x_sym)
    Ju = ca.jacobian(expr, u_sym)
    return ca.is_constant(Jx) and ca.is_constant(Ju)


def _sympy_is_linear(sym_ss) -> bool:
    """
    Check linearity using a SymbolicStateSpace-style object.

    Declares linear if all second derivatives w.r.t. states/inputs are zero
    after substituting parameter defaults.
    """
    try:
        import sympy as sp  # type: ignore
    except Exception:
        return False

    if not hasattr(sym_ss, "f") or not hasattr(sym_ss, "Bu"):
        return False

    x_syms = getattr(sym_ss, "state_symbols", [])
    u_syms = getattr(sym_ss, "input_symbols", [])
    param_syms = getattr(sym_ss, "param_symbols", [])
    param_defaults = getattr(sym_ss, "param_defaults", {}) or {}
    param_subs = {sym: param_defaults.get(str(sym), sym) for sym in param_syms}

    try:
        f_sub = sym_ss.f_sub(param_defaults) if hasattr(sym_ss, "f_sub") else sym_ss.f.subs(param_subs)
        Bu_sub = sym_ss.Bu.subs(param_subs)
    except Exception:
        f_sub = sym_ss.f
        Bu_sub = sym_ss.Bu

    def _second_derivs_zero(expr):
        for xi in x_syms:
            for xj in x_syms:
                if sp.simplify(sp.diff(expr, xi, xj)) != 0:
                    return False
        for ui in u_syms:
            for uj in u_syms:
                if sp.simplify(sp.diff(expr, ui, uj)) != 0:
                    return False
        for xi in x_syms:
            for ui in u_syms:
                if sp.simplify(sp.diff(expr, xi, ui)) != 0:
                    return False
        return True

    for comp in list(f_sub) + list(Bu_sub):
        if not _second_derivs_zero(comp):
            return False

    return True


def _sympy_error_is_linear(sym_ss) -> bool:
    """
    Check linearity of measurement/error dynamics ẏ = g(x,u,…) if available.
    """
    try:
        import sympy as sp  # type: ignore
    except Exception:
        return False

    if not hasattr(sym_ss, "measurement_dynamics"):
        return False

    x_syms = getattr(sym_ss, "state_symbols", [])
    u_syms = getattr(sym_ss, "input_symbols", [])
    param_syms = getattr(sym_ss, "param_symbols", [])
    param_defaults = getattr(sym_ss, "param_defaults", {}) or {}
    param_subs = {sym: param_defaults.get(str(sym), sym) for sym in param_syms}

    try:
        g_sub = (
            sym_ss.measurement_dynamics_sub(param_defaults)
            if hasattr(sym_ss, "measurement_dynamics_sub")
            else sym_ss.measurement_dynamics.subs(param_subs)
        )
    except Exception:
        g_sub = sym_ss.measurement_dynamics

    def _second_derivs_zero(expr):
        for xi in x_syms:
            for xj in x_syms:
                if sp.simplify(sp.diff(expr, xi, xj)) != 0:
                    return False
        for ui in u_syms:
            for uj in u_syms:
                if sp.simplify(sp.diff(expr, ui, uj)) != 0:
                    return False
        for xi in x_syms:
            for ui in u_syms:
                if sp.simplify(sp.diff(expr, xi, ui)) != 0:
                    return False
        return True

    for comp in list(g_sub):
        if not _second_derivs_zero(comp):
            return False
    return True


def classify_dynamics(spec: object, casadi_backend: Optional[object] = None) -> DynamicsClass:
    """
    Classify dynamics into linear / nonlinear based on SymPy or CasADi analysis.
    """
    # Explicit metadata hint
    hint = spec.metadata.get("dynamics_type") if hasattr(spec, "metadata") else None
    if hint in ("linear", "LINEAR"):
        return DynamicsClass.LINEAR
    if hint in ("nonlinear", "NONLINEAR"):
        return DynamicsClass.NONLINEAR

    # CasADi backend heuristic
    if casadi_backend is not None and _casadi_is_linear(casadi_backend):
        return DynamicsClass.LINEAR

    # SymPy symbolic heuristic
    if hasattr(spec, "symbolic") and _sympy_is_linear(getattr(spec, "symbolic")):
        return DynamicsClass.LINEAR

    return DynamicsClass.NONLINEAR


def classify_error_dynamics(spec: object, casadi_backend: Optional[object] = None) -> DynamicsClass:
    """
    Same as classify_dynamics but checks metadata 'error_dynamics_type' first.
    """
    hint = spec.metadata.get("error_dynamics_type") if hasattr(spec, "metadata") else None
    if hint in ("linear", "LINEAR"):
        return DynamicsClass.LINEAR
    if hint in ("nonlinear", "NONLINEAR"):
        return DynamicsClass.NONLINEAR
    # SymPy measurement dynamics heuristic
    if hasattr(spec, "symbolic") and _sympy_error_is_linear(getattr(spec, "symbolic")):
        return DynamicsClass.LINEAR

    return classify_dynamics(spec, casadi_backend=casadi_backend)
