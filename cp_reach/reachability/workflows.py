from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import numpy as np

from cp_reach.config.query import ReachQuery
from cp_reach.config.uncertainty import UncertaintySpec
from cp_reach.ir.rumoca import RumocaSymbolicModel, modelica_load
from cp_reach.planning import Trajectory
from cp_reach.reachability.lmi import solve_disturbance_LMI

logger = logging.getLogger(__name__)


def compute_reachable_set(
    model_sympy: RumocaSymbolicModel,
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
        except Exception as e:
            logger.debug(f"Could not compute axis-aligned bounds: {e}")
    return sol


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


def analyze(
    modelica_path: Union[str, Path],
    model_name: Optional[str] = None,
    *,
    roots: Optional[Iterable[Union[str, Path]]] = None,
    workspace: Optional[Union[str, Path]] = None,
    uncertainty_path: Optional[Union[str, Path]] = None,
    query_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Compile a Modelica model with Rumoca 0.10 and run reachability analysis."""
    uncertainty = (
        UncertaintySpec.from_yaml(uncertainty_path) if uncertainty_path else UncertaintySpec()
    )
    query = ReachQuery.from_yaml(query_path) if query_path else ReachQuery()

    model = modelica_load(
        modelica_path,
        model_name=model_name,
        roots=roots,
        workspace=workspace,
        output_names=query.outputs or None,
    )

    for message in uncertainty.validate_against_model(model) + query.validate_against_model(model):
        warnings.warn(message)

    dist_bound = 1.0
    if query.dist_inputs:
        dist_bound = max(uncertainty.get_dist_bound(name) for name in query.dist_inputs)

    result = compute_reachable_set(
        model,
        method="lmi",
        dynamics=query.dynamics,
        dist_bound=dist_bound,
        alpha_grid=query.alpha_search.to_grid(),
        dist_input=query.dist_inputs or None,
    )

    compiled_name = getattr(model.rumoca, "name", None)
    result["model_name"] = compiled_name or Path(modelica_path).stem
    result["query_type"] = query.type
    result["dynamics"] = query.dynamics

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if query.output_format.json:
            certificate = {
                "model_name": result["model_name"],
                "status": result.get("status", "unknown"),
                "alpha": float(result.get("alpha", 0)),
                "mu": (float(np.asarray(result["mu"]).reshape(-1)[0]) if "mu" in result else None),
                "bounds_upper": (
                    np.asarray(result["bounds_upper"]).tolist()
                    if "bounds_upper" in result
                    else None
                ),
                "bounds_lower": (
                    np.asarray(result["bounds_lower"]).tolist()
                    if "bounds_lower" in result
                    else None
                ),
            }
            with (output_path / "certificate.json").open("w") as stream:
                json.dump(certificate, stream, indent=2)

    return result
