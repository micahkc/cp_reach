from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import casadi as ca  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ca = None


def _merge_params(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


class CasadiStateSpace:
    """
    Lightweight container for (possibly symbolic) state-space matrices.

    Attributes A, B, C, D can be CasADi SX/MX or numpy arrays. If provided with
    symbol vectors x_sym, u_sym, p_sym, call `evaluate(x, u, p)` to obtain
    numeric matrices.
    """

    def __init__(
        self,
        A,
        B,
        C=None,
        D=None,
        x_sym=None,
        u_sym=None,
        p_sym=None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x_sym = x_sym
        self.u_sym = u_sym
        self.p_sym = p_sym

        if ca is not None and isinstance(A, (ca.SX, ca.MX)):
            self._A_fun = ca.Function("A_fun", self._fun_inputs(), [A]) if self._fun_inputs() else None
            self._B_fun = ca.Function("B_fun", self._fun_inputs(), [B]) if self._fun_inputs() else None
            self._C_fun = ca.Function("C_fun", self._fun_inputs(), [C]) if (C is not None and self._fun_inputs()) else None
            self._D_fun = ca.Function("D_fun", self._fun_inputs(), [D]) if (D is not None and self._fun_inputs()) else None
        else:
            self._A_fun = self._B_fun = self._C_fun = self._D_fun = None

    def _fun_inputs(self):
        inputs = []
        if self.x_sym is not None:
            inputs.append(self.x_sym)
        if self.u_sym is not None:
            inputs.append(self.u_sym)
        if self.p_sym is not None:
            inputs.append(self.p_sym)
        return inputs

    def evaluate(self, x: np.ndarray, u: np.ndarray, p: Optional[np.ndarray] = None):
        """
        Evaluate A,B,C,D numerically given x,u,p if CasADi functions were built.
        Returns a dict with keys 'A','B','C','D'.
        """
        if self._A_fun is None or ca is None:
            raise RuntimeError("CasADi functions not available for evaluation.")
        args = []
        if self.x_sym is not None:
            args.append(x)
        if self.u_sym is not None:
            args.append(u)
        if self.p_sym is not None:
            args.append(np.zeros(0) if p is None else p)
        out = {"A": np.array(self._A_fun(*args)), "B": np.array(self._B_fun(*args))}
        if self._C_fun is not None:
            out["C"] = np.array(self._C_fun(*args))
        if self._D_fun is not None:
            out["D"] = np.array(self._D_fun(*args))
        return out


def casadi_linearize(f_casadi, x_sym, u_sym, p_sym=None) -> CasadiStateSpace:
    """
    Build CasadiStateSpace by differentiating a CasADi dynamics function f(x,u,p).
    """
    if ca is None:
        raise ImportError("casadi is not available")
    A = ca.jacobian(f_casadi, x_sym)
    B = ca.jacobian(f_casadi, u_sym)
    C = None
    D = None
    return CasadiStateSpace(A=A, B=B, C=C, D=D, x_sym=x_sym, u_sym=u_sym, p_sym=p_sym)


class SymbolicStateSpace:
    """
    Intuitive symbolic state-space representation with algebraic substitution.

    Represents the system:
        ẋ = f(x, p, t) + Bu(t)     (after algebraic substitution)
        y = h(x, p, t) + Du(t)
        ẏ = g(x, u, p, t)          (optional measurement dynamics)

    This class provides an intuitive interface for working with symbolic dynamics:
    - Algebraic variables are automatically substituted into derivatives
    - Control inputs are separated for clarity
    - Parameter substitution and polytopic bounds for verification

    Attributes
    ----------
    f : sympy.Matrix
        State drift dynamics f(x, p, t) with algebraic vars substituted
    Bu : sympy.Matrix
        Control input vector (how inputs affect the system)
    h : sympy.Matrix or None
        Output drift h(x, p, t)
    Du : sympy.Matrix or None
        Output control feedthrough
    measurement_dynamics : sympy.Matrix or None
        Output/measurement dynamics g(x, u, p, t); auto-derived from h if not provided
    state_symbols : list
        SymPy symbols for state variables
    input_symbols : list
        SymPy symbols for control inputs
    param_symbols : list
        SymPy symbols for parameters
    output_symbols : list or None
        SymPy symbols for outputs
    param_defaults : dict
        Default parameter values from the model

    Methods
    -------
    f_sub(params=None)
        Substitute parameters into f, returning f(x, t)
    h_sub(params=None)
        Substitute parameters into h, returning h(x, t)
    A(params=None, bounds=None)
        Compute state Jacobian as numpy array or polytope
    B()
        Compute input Jacobian as numpy array
    C(params=None, bounds=None)
        Compute output-state Jacobian as numpy array or polytope
    D()
        Compute output-input Jacobian as numpy array

    Examples
    --------
    >>> from cp_reach.ir import modelica_load
    >>>
    >>> model = modelica_load('Model.mo', output_names=['y1', 'y2'])
    >>> ss = model.symbolic
    >>>
    >>> # View symbolic dynamics
    >>> print(ss.f)      # Drift dynamics f(x, p, t)
    >>> print(ss.Bu)     # Control input effects
    >>>
    >>> # Substitute parameters
    >>> f_numeric = ss.f_sub({'m': 1.0, 'k': 2.0})  # Returns f(x, t)
    >>>
    >>> # Get numerical Jacobians
    >>> A_mat = ss.A({'m': 1.0, 'k': 2.0})          # Linear system
    >>> A_polytope = ss.A({'m': 1.0}, {'x': [0, 1]}) # Nonlinear with bounds
    """

    def __init__(
        self,
        f: Any,
        Bu: Any,
        h: Optional[Any] = None,
        Du: Optional[Any] = None,
        measurement_dynamics: Optional[Any] = None,
        state_symbols: Optional[list] = None,
        input_symbols: Optional[list] = None,
        param_symbols: Optional[list] = None,
        output_symbols: Optional[list] = None,
        param_defaults: Optional[dict] = None,
    ):
        self.state_symbols = state_symbols or []
        self.input_symbols = input_symbols or []
        self.param_symbols = param_symbols or []
        self.output_symbols = output_symbols or []
        self.param_defaults = param_defaults or {}

        self.f = f
        self.Bu = Bu
        self.h = h
        self.Du = Du
        # If measurement dynamics are not provided, attempt to derive ẏ from h and full state dynamics (includes inputs).
        self.measurement_dynamics = measurement_dynamics or self._build_measurement_dynamics_from_h()

    def f_sub(self, params: Optional[Dict[str, float]] = None):
        """
        Substitute parameters into drift dynamics.

        Parameters
        ----------
        params : dict, optional
            Parameter values to substitute. If None, uses defaults.

        Returns
        -------
        sympy.Matrix
            Drift dynamics f(x, t) with parameters substituted
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        param_vals = {**self.param_defaults, **(params or {})}
        # Use the actual Symbol objects from param_symbols, not new ones
        param_subs = {sym: param_vals[str(sym)] for sym in self.param_symbols if str(sym) in param_vals}
        return self.f.subs(param_subs)

    def h_sub(self, params: Optional[Dict[str, float]] = None):
        """
        Substitute parameters into output drift.

        Parameters
        ----------
        params : dict, optional
            Parameter values to substitute. If None, uses defaults.

        Returns
        -------
        sympy.Matrix or None
            Output drift h(x, t) with parameters substituted
        """
        if self.h is None:
            return None

        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        param_vals = {**self.param_defaults, **(params or {})}
        # Use the actual Symbol objects from param_symbols, not new ones
        param_subs = {sym: param_vals[str(sym)] for sym in self.param_symbols if str(sym) in param_vals}
        return self.h.subs(param_subs)

    def measurement_dynamics_sub(self, params: Optional[Dict[str, float]] = None):
        """
        Substitute parameters into measurement dynamics.

        Parameters
        ----------
        params : dict, optional
            Parameter values to substitute. If None, uses defaults.

        Returns
        -------
        sympy.Matrix or None
            Measurement dynamics g(y, u, p, t) with parameters substituted
        """
        if self.measurement_dynamics is None:
            return None

        try:
            import sympy as sp  # noqa: F401
        except ImportError:
            raise ImportError("sympy is required")

        param_vals = {**self.param_defaults, **(params or {})}
        param_subs = {sym: param_vals[str(sym)] for sym in self.param_symbols if str(sym) in param_vals}
        return self.measurement_dynamics.subs(param_subs)

    def _build_measurement_dynamics_from_h(self):
        """
        Construct measurement dynamics ẏ by differentiating y = h(x, t, p) using ẋ = f + Bu (includes inputs).

        Returns
        -------
        sympy.Matrix or None
            Time derivative of outputs, or None if h is not provided or sympy unavailable.
        """
        if self.h is None or not self.state_symbols:
            return None
        try:
            import sympy as sp
        except ImportError:
            return None

        t_sym = sp.Symbol("t")
        f_vec = sp.Matrix(self.f)
        if f_vec.shape[0] != len(self.state_symbols):
            raise ValueError("f must match the number of state symbols to build measurement dynamics.")

        Bu_vec = sp.Matrix(self.Bu) if self.Bu is not None else sp.zeros(*f_vec.shape)
        if Bu_vec.shape != f_vec.shape:
            raise ValueError("f and Bu must have the same shape to build measurement dynamics.")
        total_dynamics = f_vec + Bu_vec

        # Map state symbols to time-dependent functions for differentiation
        state_funcs = {sym: sp.Function(str(sym))(t_sym) for sym in self.state_symbols}
        h_of_t = self.h.subs(state_funcs)

        h_dot = h_of_t.diff(t_sym)

        # Substitute ẋ with full dynamics (includes control inputs)
        deriv_subs = {func.diff(t_sym): total_dynamics[i] for i, func in enumerate(state_funcs.values())}
        h_dot_sub = h_dot.subs(deriv_subs)

        # Replace time-dependent functions back with original symbols
        cleanup_subs = {func: sym for sym, func in state_funcs.items()}
        h_dot_clean = h_dot_sub.subs(cleanup_subs)

        return h_dot_clean

    def A(
        self,
        params: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, tuple[float, float]]] = None,
    ):
        """
        Compute state Jacobian matrix.

        For linear systems (Jacobian independent of states), returns a single
        numpy array. For nonlinear systems with bounds, returns a polytopic
        over-approximation as a list of vertex matrices.

        Parameters
        ----------
        params : dict, optional
            Parameter values. If None, uses defaults.
        bounds : dict, optional
            State/variable bounds for polytopic approximation.
            Format: {'x': (lower, upper), ...}

        Returns
        -------
        np.ndarray or list[np.ndarray]
            State Jacobian matrix or polytope vertices

        Warnings
        --------
        Warns if state variables remain in Jacobian without bounds provided.
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        # Compute Jacobian ∂f/∂x
        f_drift = self.f_sub(params)
        A_sym = f_drift.jacobian(self.state_symbols)

        # Check if there are remaining free variables (nonlinear system)
        free_vars = A_sym.free_symbols - set(self.param_symbols)

        if not free_vars:
            # Linear system - just convert to numpy
            A_func = sp.lambdify([], A_sym, 'numpy')
            return A_func()

        # Nonlinear system - need bounds for polytopic approximation
        if bounds is None:
            import warnings
            warnings.warn(
                f"State Jacobian contains free variables {free_vars}. "
                f"Provide bounds for polytopic approximation or the result may be symbolic.",
                UserWarning
            )
            # Return symbolic if no bounds
            return A_sym

        # Generate polytope vertices
        missing_bounds = free_vars - set(sp.Symbol(name) for name in bounds.keys())
        if missing_bounds:
            import warnings
            warnings.warn(
                f"Missing bounds for variables: {missing_bounds}. "
                f"Using symbolic values for these.",
                UserWarning
            )

        # Create vertex combinations
        bound_vars = [sp.Symbol(name) for name in bounds.keys() if sp.Symbol(name) in free_vars]
        if not bound_vars:
            # No bounded variables, return symbolic
            return A_sym

        vertices = []
        import itertools
        for corner in itertools.product(*[[bounds[str(var)][0], bounds[str(var)][1]] for var in bound_vars]):
            vertex_subs = dict(zip(bound_vars, corner))
            A_vertex = A_sym.subs(vertex_subs)
            A_func = sp.lambdify([], A_vertex, 'numpy')
            vertices.append(A_func())

        return vertices

    def B(self, params: Optional[Dict[str, float]] = None):
        """
        Compute input Jacobian matrix.

        Parameters
        ----------
        params : dict, optional
            Parameter values. If None, uses defaults.

        Returns
        -------
        np.ndarray
            Input Jacobian matrix
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        # Substitute parameters first
        param_vals = {**self.param_defaults, **(params or {})}
        param_subs = {sym: param_vals[str(sym)] for sym in self.param_symbols if str(sym) in param_vals}
        Bu_sub = self.Bu.subs(param_subs)

        # Bu is already linear in u, so just extract coefficients
        B_sym = Bu_sub.jacobian(self.input_symbols)
        B_func = sp.lambdify([], B_sym, 'numpy')
        return B_func()

    def C(
        self,
        params: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, tuple[float, float]]] = None,
    ):
        """
        Compute output-state Jacobian matrix.

        Similar to A(), returns numpy array for linear systems or polytope
        for nonlinear systems with bounds.

        Parameters
        ----------
        params : dict, optional
            Parameter values. If None, uses defaults.
        bounds : dict, optional
            State/variable bounds for polytopic approximation.

        Returns
        -------
        np.ndarray or list[np.ndarray] or None
            Output Jacobian matrix or polytope vertices, or None if no outputs
        """
        if self.h is None:
            return None

        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        h_drift = self.h_sub(params)
        C_sym = h_drift.jacobian(self.state_symbols)

        # Check for free variables
        free_vars = C_sym.free_symbols - set(self.param_symbols)

        if not free_vars:
            C_func = sp.lambdify([], C_sym, 'numpy')
            return C_func()

        if bounds is None:
            import warnings
            warnings.warn(
                f"Output Jacobian contains free variables {free_vars}. "
                f"Provide bounds for polytopic approximation.",
                UserWarning
            )
            return C_sym

        # Generate polytope vertices (same as A)
        bound_vars = [sp.Symbol(name) for name in bounds.keys() if sp.Symbol(name) in free_vars]
        if not bound_vars:
            return C_sym

        vertices = []
        import itertools
        for corner in itertools.product(*[[bounds[str(var)][0], bounds[str(var)][1]] for var in bound_vars]):
            vertex_subs = dict(zip(bound_vars, corner))
            C_vertex = C_sym.subs(vertex_subs)
            C_func = sp.lambdify([], C_vertex, 'numpy')
            vertices.append(C_func())

        return vertices

    def D(self, params: Optional[Dict[str, float]] = None):
        """
        Compute output-input feedthrough matrix.

        Parameters
        ----------
        params : dict, optional
            Parameter values. If None, uses defaults.

        Returns
        -------
        np.ndarray or None
            Feedthrough matrix, or None if no outputs
        """
        if self.Du is None:
            return None

        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        # Substitute parameters first
        param_vals = {**self.param_defaults, **(params or {})}
        param_subs = {sym: param_vals[str(sym)] for sym in self.param_symbols if str(sym) in param_vals}
        Du_sub = self.Du.subs(param_subs)

        D_sym = Du_sub.jacobian(self.input_symbols)
        D_func = sp.lambdify([], D_sym, 'numpy')
        return D_func()

    def E(self, params: Optional[Dict[str, float]] = None):
        """
        Compute measurement-state Jacobian for measurement dynamics.

        Returns E such that ẏ = E y (after parameter substitution). If the output
        map is invertible (square C), E is computed via chain rule using ∂g/∂x * (∂y/∂x)^{-1}.
        For non-square C, uses the Moore-Penrose pseudoinverse. If C is unavailable,
        falls back to ∂g/∂state.
        """
        if self.measurement_dynamics is None:
            return None
        if not self.state_symbols:
            raise ValueError("state_symbols must be provided to compute E.")

        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        g_sub = self.measurement_dynamics_sub(params)
        # Default: Jacobian wrt state
        E_sym = g_sub.jacobian(self.state_symbols)

        # If outputs are available, attempt chain rule to get Jacobian wrt y
        if self.output_symbols and self.h is not None:
            h_sub = self.h_sub(params)
            C_sym = h_sub.jacobian(self.state_symbols)
            try:
                if C_sym.shape[0] == C_sym.shape[1]:
                    C_inv = C_sym.inv()
                else:
                    C_inv = C_sym.pinv()
                E_sym = sp.simplify(E_sym * C_inv)
            except Exception as e:
                # If inversion fails, keep state-based Jacobian
                logger.debug(f"Could not invert output matrix C for algebraic substitution: {e}")

        # If no free variables remain, return numeric matrix
        free_vars = E_sym.free_symbols - set(self.param_symbols) - set(self.state_symbols) - set(self.input_symbols)
        if not free_vars:
            E_func = sp.lambdify([], E_sym, 'numpy')
            return E_func()
        return E_sym

    def F(self, params: Optional[Dict[str, float]] = None):
        """
        Compute measurement-input Jacobian for measurement dynamics.

        Returns F such that ẏ = E y + F u (after parameter substitution). Uses
        direct ∂g/∂u if present; otherwise falls back to chain rule C·B where
        C = ∂y/∂x and B = ∂(Bu)/∂u.
        """
        if self.measurement_dynamics is None:
            return None
        if not self.input_symbols:
            raise ValueError("input_symbols must be provided to compute F.")

        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        g_sub = self.measurement_dynamics_sub(params)
        F_sym = g_sub.jacobian(self.input_symbols)

        # If g has no explicit input dependence, use chain rule with C and B
        try:
            is_zero = all(elem.is_zero for elem in F_sym)
        except Exception:
            is_zero = False

        if is_zero and self.h is not None and self.Bu is not None:
            h_sub = self.h_sub(params)
            C_sym = h_sub.jacobian(self.state_symbols)

            # Substitute params into Bu for consistent B
            param_vals = {**self.param_defaults, **(params or {})}
            param_subs = {sym: param_vals[str(sym)] for sym in self.param_symbols if str(sym) in param_vals}
            Bu_sub = self.Bu.subs(param_subs)
            B_sym = Bu_sub.jacobian(self.input_symbols)
            F_sym = C_sym * B_sym

        free_vars = F_sym.free_symbols - set(self.param_symbols) - set(self.state_symbols) - set(self.input_symbols)
        if not free_vars:
            F_func = sp.lambdify([], F_sym, 'numpy')
            return F_func()
        return F_sym

    def get_error_dynamics(
        self,
        disturbance_inputs: Optional[list[str]] = None,
    ):
        """
        Compute symbolic error dynamics expression.

        For a system ẋ = f(x, u, d) tracking reference x_ref, the error is
        e = x - x_ref and the error dynamics are:

            ė = ẋ - ẋ_ref = f(x, u, d) - f(x_ref, u_nom, 0)

        Assuming perfect feedforward cancels the nominal dynamics at x_ref,
        this simplifies to:

            ė = f(x_ref + e, u, d) - f(x_ref, u_nom, 0)

        For linear systems: ė = A*e + B_d*d  (A, B_d constant)
        For nonlinear systems: ė = J(x)*e + B_d*d  (J varies with state)

        This method returns the symbolic expression for the error dynamics,
        which includes both the state-dependent Jacobian term and the
        disturbance input term.

        Parameters
        ----------
        disturbance_inputs : list[str], optional
            Names of inputs that are disturbances (cannot be cancelled by
            feedforward). If None, treats all inputs as disturbances.

        Returns
        -------
        dict
            Dictionary containing:
            - 'J': Jacobian matrix ∂f/∂x (symbolic, may depend on state)
            - 'B_d': Disturbance input matrix ∂f/∂d (symbolic)
            - 'f_nominal': Nominal dynamics f(x, u, 0)
            - 'f_full': Full dynamics f(x, u, d)
            - 'disturbance_symbols': List of disturbance symbols

        Examples
        --------
        >>> from cp_reach.ir import modelica_load
        >>> ss = modelica_load("Model.mo").symbolic
        >>> err = ss.get_error_dynamics(disturbance_inputs=['d'])
        >>> J = err['J']  # Jacobian (may contain sin(theta) etc.)
        >>> B_d = err['B_d']  # Disturbance matrix
        >>> # Check if J is constant (linear) or state-dependent (nonlinear)
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        # Full dynamics: ẋ = f + Bu
        full_dynamics = self.f + self.Bu

        # Identify disturbance symbols
        input_names = [str(s) for s in self.input_symbols]
        if disturbance_inputs is None:
            dist_symbols = list(self.input_symbols)
        else:
            dist_symbols = []
            for name in disturbance_inputs:
                if name not in input_names:
                    raise ValueError(
                        f"Disturbance input '{name}' not found in inputs {input_names}"
                    )
                idx = input_names.index(name)
                dist_symbols.append(self.input_symbols[idx])

        # Nominal dynamics: set disturbances to zero
        nominal_subs = {d_sym: 0 for d_sym in dist_symbols}
        nominal_dynamics = full_dynamics.subs(nominal_subs)

        # Jacobian of nominal dynamics w.r.t. states
        # This is J(x) in ė = J(x)*e + B_d*d
        J = nominal_dynamics.jacobian(self.state_symbols)

        # Disturbance input matrix: ∂f/∂d
        B_d = full_dynamics.jacobian(dist_symbols)

        return {
            'J': J,
            'B_d': B_d,
            'f_nominal': nominal_dynamics,
            'f_full': full_dynamics,
            'disturbance_symbols': dist_symbols,
        }

    def error_dynamics_are_linear(
        self,
        disturbance_inputs: Optional[list[str]] = None,
    ) -> bool:
        """
        Check if error dynamics are linear (constant Jacobian).

        The error dynamics ė = J(x)*e + B_d*d are linear if and only if
        the Jacobian J = ∂f/∂x does not depend on the state x.

        Parameters
        ----------
        disturbance_inputs : list[str], optional
            Names of inputs that are disturbances.

        Returns
        -------
        bool
            True if J is constant (linear error dynamics),
            False if J depends on state (nonlinear error dynamics)
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError("sympy is required")

        err = self.get_error_dynamics(disturbance_inputs=disturbance_inputs)
        J = err['J']

        # Substitute parameters
        param_vals = self.param_defaults or {}
        param_subs = {sym: param_vals.get(str(sym), sym) for sym in self.param_symbols}
        J_sub = J.subs(param_subs)

        # Check if any element of J depends on state variables
        state_set = set(self.state_symbols)
        for elem in J_sub:
            if elem.free_symbols & state_set:
                return False

        return True

    def linearize_error_dynamics(
        self,
        disturbance_inputs: Optional[list[str]] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """
        Extract A and B_d matrices from error dynamics.

        This evaluates the error dynamics Jacobians at equilibrium (x=0, u=0).
        For linear systems, the result is exact.
        For nonlinear systems, this gives the linearization at equilibrium.

        Parameters
        ----------
        disturbance_inputs : list[str], optional
            Names of inputs that are disturbances.
        params : dict, optional
            Parameter values. If None, uses defaults.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (A, B_d) matrices where:
            - A: State Jacobian of nominal dynamics (n x n)
            - B_d: Disturbance input matrix (n x m_dist)
        """
        import numpy as np

        # Get symbolic error dynamics
        err = self.get_error_dynamics(disturbance_inputs=disturbance_inputs)
        J_sym = err['J']
        B_d_sym = err['B_d']

        # Substitute parameters and evaluate at equilibrium
        param_vals = {**self.param_defaults, **(params or {})}
        subs_dict = {sym: param_vals[str(sym)] for sym in self.param_symbols if str(sym) in param_vals}

        # Evaluate at zero state and zero input
        for xi in self.state_symbols:
            subs_dict[xi] = 0
        for ui in self.input_symbols:
            subs_dict[ui] = 0

        A = np.array(J_sym.subs(subs_dict)).astype(float)
        B_d = np.array(B_d_sym.subs(subs_dict)).astype(float)

        return A, B_d

    def get_input_names(self) -> list[str]:
        """Return list of input names as strings."""
        return [str(s) for s in self.input_symbols]

    def get_state_names(self) -> list[str]:
        """Return list of state names as strings."""
        return [str(s) for s in self.state_symbols]
