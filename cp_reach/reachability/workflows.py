from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from cp_reach.dynamics.state_space import SymbolicStateSpace, extract_symbolic_statespace
from cp_reach.dynamics.state_space import casadi_linearize as _casadi_linearize
from cp_reach.planning import Trajectory
from cp_reach.reachability.lmi import solve_disturbance_LMI
import inspect


def _as_u_profile(u_ff: Union[np.ndarray, Callable[[float], np.ndarray]], t: np.ndarray, m: int) -> np.ndarray:
    """
    Convert a feedforward control description into a per-step array.
    """
    if callable(u_ff):
        u_arr = np.stack([np.asarray(u_ff(float(ti)), dtype=float).reshape(m) for ti in t[:-1]], axis=0)
    else:
        u_arr = np.asarray(u_ff, dtype=float)
        if u_arr.ndim == 1:
            u_arr = u_arr.reshape(-1, m)
    if u_arr.shape[0] != len(t) - 1 or u_arr.shape[1] != m:
        raise ValueError(f"u_ff must have shape {(len(t)-1, m)}, got {u_arr.shape}")
    return u_arr


# --- Modelica convenience wrappers (SymPy / CasADi via cyecca) -----------------


class ModelicaSympyModel:
    """
    Lightweight wrapper around a SymPy backend with CP_Reach-friendly accessors.
    """

    def __init__(self, backend, ss: SymbolicStateSpace):
        self.backend = backend
        self.symbolic = ss
        self.states = [s.name for s in backend.model.states]
        self.inputs = [u.name for u in backend.model.inputs]
        self.parameters = {p.name: getattr(p, "start", None) for p in backend.model.parameters}


class ModelicaCasadiModel:
    """
    Lightweight wrapper around a CasADi backend exposing a numeric RHS.
    """

    def __init__(self, backend, rhs_fun, state_names, input_names, param_defaults):
        self.backend = backend
        self.rhs_fun = rhs_fun
        self.state_names = state_names
        self.input_names = input_names
        self.param_defaults = param_defaults or {}

    def f(self, t: float, x: np.ndarray, u: np.ndarray, params: Optional[dict] = None) -> np.ndarray:
        """
        Evaluate xdot = f(x, u, p). Time is ignored (included for API symmetry).
        """
        params = params or {}
        p_vec = np.array(
            [params.get(k, self.param_defaults.get(k, 0.0)) for k in self.param_defaults.keys()], dtype=float
        )
        if p_vec.size == 0:
            p_vec = np.zeros(0)

        # Prefer explicit arity from CasADi function
        expected = None
        try:
            if hasattr(self.rhs_fun, "n_in"):
                expected = int(self.rhs_fun.n_in())
        except Exception:
            expected = None

        # If it's a Python function with named args, map them
        if expected is None:
            try:
                sig = inspect.signature(self.rhs_fun)
                kwargs = {}
                for name in sig.parameters:
                    if name in ("t", "time"):
                        kwargs[name] = t
                    elif name in ("x", "state"):
                        kwargs[name] = x
                    elif name in ("u", "input"):
                        kwargs[name] = u
                    elif name in ("p", "params"):
                        kwargs[name] = p_vec
                out = self.rhs_fun(**kwargs)
                return np.array(out, dtype=float).ravel()
            except Exception:
                expected = None

        try:
            if expected == 4:
                out = self.rhs_fun(t, x, u, p_vec)
            elif expected == 3:
                out = self.rhs_fun(x, u, p_vec)
            elif expected == 2:
                out = self.rhs_fun(x, u)
            else:
                out = self.rhs_fun(x, u, p_vec)
        except TypeError:
            # Fallback to minimal signature seen in notebooks
            out = self.rhs_fun(x, u, p_vec)
        return np.array(out, dtype=float).ravel()

    def simulate(
        self,
        t_final: float,
        dt: float,
        input_func: Callable[[float], Union[np.ndarray, dict]],
        x0: Optional[dict] = None,
        params: Optional[dict] = None,
    ):
        """
        If the backend exposes a simulate() method (as in modelica_to_casadi_feedforward),
        delegate to it. Otherwise, raise.
        """
        if not hasattr(self.backend, "simulate"):
            raise RuntimeError("Casadi backend does not provide simulate()")

        # Optionally set initial states/params on backend defaults
        if x0:
            if hasattr(self.backend, "state_defaults"):
                self.backend.state_defaults.update({k: float(v) for k, v in x0.items()})
        if params:
            if hasattr(self.backend, "param_defaults"):
                self.backend.param_defaults.update({k: float(v) for k, v in params.items()})

        # Wrap input_func to produce dicts keyed by input names if needed
        def wrapped_input_func(t: float):
            u_val = input_func(t)
            if isinstance(u_val, dict):
                return u_val
            u_arr = np.asarray(u_val, dtype=float).ravel()
            if u_arr.size != len(self.input_names):
                raise ValueError(f"input_func returned length {u_arr.size}, expected {len(self.input_names)}")
            return {name: float(u_arr[i]) for i, name in enumerate(self.input_names)}

        return self.backend.simulate(t_final=t_final, dt=dt, input_func=wrapped_input_func)


def sympy_load(model_path: str, output_names: Optional[list[str]] = None, simplify: bool = True) -> ModelicaSympyModel:
    """
    Load a Modelica file with cyecca SymPy backend and wrap as ModelicaSympyModel.
    """
    from cyecca.backends.sympy import SympyBackend

    backend = SympyBackend.from_file(model_path)
    ss = extract_symbolic_statespace(backend, output_names=output_names, simplify=simplify)
    return ModelicaSympyModel(backend=backend, ss=ss)


def casadi_load(model_path: str) -> ModelicaCasadiModel:
    """
    Load a Modelica file with cyecca CasADi backend and wrap a numeric RHS.
    """
    from cyecca.backends.casadi import CasadiBackend

    backend = CasadiBackend.from_file(model_path)
    rhs_fun = backend.get_rhs_function()
    return ModelicaCasadiModel(
        backend=backend,
        rhs_fun=rhs_fun,
        state_names=list(backend.state_names),
        input_names=list(backend.input_names),
        param_defaults=getattr(backend, "param_defaults", {}),
    )


def compute_reachable_set(
    model_sympy: ModelicaSympyModel,
    method: str = "lmi",
    dynamics: str = "error",
    dist_bound: Optional[float] = None,
    alpha_grid: Optional[Iterable[float]] = None,
    dist_input: Optional[List[str]] = None,
):
    """
    Compute an LMI disturbance bound using either state or measurement (error) dynamics.
    """
    if method != "lmi":
        raise ValueError("Only LMI method is supported")

    ss = model_sympy.symbolic
    if dynamics == "error":
        A_mat = ss.E()
        B_mat = ss.F()
    else:
        A_mat = ss.A()
        B_mat = ss.B()

    if A_mat is None or B_mat is None:
        raise ValueError("Jacobians unavailable for reachable set computation")

    # If subset of disturbance inputs provided, slice columns accordingly
    if dist_input and hasattr(model_sympy, "inputs"):
        name_to_idx = {name: i for i, name in enumerate(model_sympy.inputs)}
        idxs = []
        for name in dist_input:
            if name not in name_to_idx:
                raise ValueError(f"dist_input '{name}' not found in inputs {model_sympy.inputs}")
            idxs.append(name_to_idx[name])
        B_mat = np.asarray(B_mat, dtype=float)[:, idxs]

    if isinstance(A_mat, list):
        A_list = [np.asarray(a, dtype=float) for a in A_mat]
    else:
        A_list = [np.asarray(A_mat, dtype=float)]

    B_arr = np.asarray(B_mat, dtype=float)
    if B_arr.ndim == 1:
        B_arr = B_arr.reshape(-1, 1)

    sol = solve_disturbance_LMI(
        A_list,
        B_arr,
    )

    # Optional axis-aligned bounds per state: e^T P e <= mu * dist_bound^2
    if dist_bound is not None and sol.get("P") is not None and sol.get("mu") is not None:
        try:
            P = np.array(sol["P"], dtype=float)
            P_inv = np.linalg.inv(P)
            mu_val = sol["mu"]
            mu_scalar = float(np.max(mu_val)) if np.ndim(mu_val) else float(mu_val)
            n = P.shape[0]
            radii = []
            for idx in range(n):
                r = float(np.sqrt(mu_scalar) * dist_bound * np.sqrt(P_inv[idx, idx]))
                radii.append(r)
            sol["bounds_lower"] = -np.array(radii)
            sol["bounds_upper"] = np.array(radii)
        except Exception:
            pass
    return sol


# --- Simulation + plotting for disturbance experiments ------------------------


def simulate_dist(
    model_casadi: ModelicaCasadiModel,
    x0: dict,
    params: Optional[dict] = None,
    dist_bound: float = 0.0,
    dist_input: Optional[List[str]] = None,
    num_sims: int = 50,
    states: Optional[List[List[str]]] = None,
    input_fun: Optional[Callable[[float], np.ndarray]] = None,
    frequency_range: Optional[Tuple[float, float]] = (0.5, 5.0),
) -> Tuple[Trajectory, List[Trajectory]]:
    """
    Simulate nominal + Monte Carlo disturbances on the CasADi model.

    dist_input: list of input names that receive square disturbances in [-dist_bound, dist_bound]
    input_fun: optional callable u(t) -> ndarray of length m (defaults to zeros)
    frequency_range: optional (low, high) Hz; if provided, each trial uses a square wave
                     disturbance at a fixed random frequency in this range (phase randomized).
    """
    m = len(model_casadi.input_names)
    n = len(model_casadi.state_names)

    def u_nom(t):
        if input_fun is None:
            return np.zeros(m)
        return np.asarray(input_fun(t), dtype=float).reshape(m)

    # Map x0 dict to vector
    x0_vec = np.zeros(n)
    name_to_idx = {name: i for i, name in enumerate(model_casadi.state_names)}
    for k, v in x0.items():
        if k not in name_to_idx:
            raise ValueError(f"Unknown state '{k}' in x0")
        x0_vec[name_to_idx[k]] = float(v)

    # Simulate nominal trajectory
    t_final = 5.0
    dt = 0.01

    def input_func_nom(t):
        return u_nom(t)

    t_sim, sol_nom = model_casadi.simulate(t_final=t_final, dt=dt, input_func=input_func_nom, x0=x0, params=params)
    xs_nom = np.column_stack([sol_nom[name] for name in model_casadi.state_names])
    us_nom = np.vstack([u_nom(tt) for tt in t_sim[:-1]])
    nom = Trajectory(t=t_sim, x=xs_nom, u=us_nom, metadata={"source": "nominal"})

    # Disturbance indices
    dist_idx = []
    if dist_input:
        for name in dist_input:
            if name not in model_casadi.input_names:
                raise ValueError(f"dist_input '{name}' not found in inputs {model_casadi.input_names}")
            dist_idx.append(model_casadi.input_names.index(name))

    # Monte Carlo rollouts
    trials: List[Trajectory] = []
    rng = np.random.default_rng()
    for _ in range(num_sims):
        freq = float(rng.uniform(*frequency_range)) if frequency_range else 0.0
        period = 1.0 / freq if freq > 0 else np.inf
        phase = float(rng.uniform(0.0, period)) if freq > 0 else 0.0

        def disturbance_wave(t: float) -> float:
            t_shifted = t + phase
            cycle_position = (t_shifted % period) / period if period < np.inf else 0.0
            return dist_bound if cycle_position < 0.5 else -dist_bound

        def disturbed_input(t: float):
            base = u_nom(t).copy()
            if dist_idx:
                d_val = disturbance_wave(t)
                for idx in dist_idx:
                    base[idx] += d_val
            return base

        t_sim, sol_mc = model_casadi.simulate(
            t_final=nom.t[-1],
            dt=float(nom.t[1] - nom.t[0]),
            input_func=disturbed_input,
            x0=x0,
            params=params,
        )
        xs_mc = np.column_stack([sol_mc[name] for name in model_casadi.state_names])
        us_mc = np.vstack([disturbed_input(tt) for tt in t_sim[:-1]])
        trials.append(Trajectory(t=t_sim, x=xs_mc, u=us_mc, metadata={"source": "monte_carlo"}))

    return nom, trials


def plot_grouped(nom: Trajectory, trials: List[Trajectory], groups: List[List[str]], state_names: List[str]):
    """
    Plot groups of states on shared axes.
    groups: e.g., [['x','x_ref'], ['v','v_ref']]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    name_to_idx = {name: i for i, name in enumerate(state_names)}
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 2.8 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        mc_labeled = False
        for tr in trials:
            for name in group:
                if name not in name_to_idx:
                    continue
                idx = name_to_idx[name]
                label = "dist" if (not mc_labeled and name == group[0]) else None
                ax.plot(tr.t, tr.x[:, idx], color="0.5", alpha=0.35, linewidth=1, zorder=1, label=label)
                mc_labeled = mc_labeled or (label is not None)
        for name in group:
            if name not in name_to_idx:
                continue
            idx = name_to_idx[name]
            ax.plot(nom.t, nom.x[:, idx], label=f"{name} (nom)", color="tab:orange", linewidth=2.5, zorder=10)
        ax.set_ylabel(", ".join(group))
        ax.grid(True, linestyle=":")
        ax.legend()
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig, axes


def plot_flowpipe(
    nom: Trajectory,
    trials: List[Trajectory],
    groups: List[List[str]],
    state_names: List[str],
    error_fn: Optional[Callable[[float], Union[np.ndarray, dict, tuple]]] = None,
    error_state_names: Optional[List[str]] = None,
):
    """
    Plot nominal, Monte Carlo rollouts, and optional symmetric error bounds (flowpipe).

    error_fn: callable(t) -> either
        - array-like of length len(state_names)
        - dict mapping state_name -> bound
        - tuple (array, names) where names aligns to the array length
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    name_to_idx = {name: i for i, name in enumerate(state_names)}
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 2.8 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]

    bounds = None
    bounds_by_groups = False
    if error_fn is not None:
        samples = []
        for tt in nom.t:
            val = error_fn(tt)
            names = error_state_names
            if isinstance(val, tuple) and len(val) == 2:
                val, names = val
            if isinstance(val, dict):
                vec = np.zeros(len(state_names))
                for k, v in val.items():
                    if k in name_to_idx:
                        vec[name_to_idx[k]] = float(v)
                samples.append(vec)
            else:
                arr = np.asarray(val, dtype=float).ravel()
                if arr.size == len(state_names):
                    samples.append(arr)
                elif arr.size == len(groups):
                    samples.append(arr)
                    bounds_by_groups = True
                elif names is not None and len(names) == arr.size:
                    vec = np.zeros(len(state_names))
                    for i, nm in enumerate(names):
                        if nm in name_to_idx:
                            vec[name_to_idx[nm]] = arr[i]
                    samples.append(vec)
                else:
                    raise ValueError("error_fn output length mismatch; provide error_state_names or a dict/tuple.")
        bounds = np.vstack(samples)

    for g_idx, (ax, group) in enumerate(zip(axes, groups)):
        for tr in trials:
            for name in group:
                if name not in name_to_idx:
                    continue
                idx = name_to_idx[name]
                ax.plot(tr.t, tr.x[:, idx], color="tab:blue", alpha=0.25, linewidth=1, zorder=1)
        for name in group:
            if name not in name_to_idx:
                continue
            idx = name_to_idx[name]
            ax.plot(nom.t, nom.x[:, idx], label=f"{name} (nom)", linewidth=2.5, zorder=10)
            if bounds is not None:
                b_col = g_idx if bounds_by_groups or bounds.shape[1] == len(groups) else idx
                upper = nom.x[:, idx] + bounds[:, b_col]
                lower = nom.x[:, idx] - bounds[:, b_col]
                ax.fill_between(nom.t, lower, upper, color="tab:red", alpha=0.2, zorder=2, label="flowpipe" if name == group[0] else None)
        ax.set_ylabel(", ".join(group))
        ax.grid(True, linestyle=":")
        ax.legend()
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig, axes
