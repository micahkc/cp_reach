"""Rumoca 0.10 checked-Solve to SymPy integration.

This is the preferred Modelica boundary for CP Reach.  Rumoca performs model
compilation, structural analysis, scalarization, and algebraic causalization;
the packaged target renders that checked Solve program into native SymPy
expressions without reconstructing a second compiler representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import sympy as sp

from cp_reach.dynamics.state_space import SymbolicStateSpace


class RumocaSympyExportError(RuntimeError):
    """Raised when Rumoca cannot provide the explicit symbolic contract we need."""


@dataclass(frozen=True)
class SympySolveExport:
    """Typed CP Reach view of a generated Rumoca explicit Solve program."""

    rhs: sp.ImmutableDenseMatrix
    state_symbols: Tuple[sp.Symbol, ...]
    input_symbols: Tuple[sp.Symbol, ...]
    parameter_symbols: Tuple[sp.Symbol, ...]
    variable_expressions: Mapping[str, sp.Expr]
    state_names: Tuple[str, ...]
    input_names: Tuple[str, ...]
    parameter_names: Tuple[str, ...]
    algebraic_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    default_states: Tuple[float, ...]
    default_parameters: Tuple[float, ...]
    algebraic_assignments_complete: bool

    @property
    def parameter_defaults(self) -> Dict[str, float]:
        """Parameter defaults keyed in Rumoca's checked storage order."""
        return dict(zip(self.parameter_names, self.default_parameters))


@dataclass
class RumocaSymbolicModel:
    """Rumoca model plus the CP Reach symbolic interface derived from it."""

    rumoca: Any
    symbolic: SymbolicStateSpace
    export: SympySolveExport

    @property
    def states(self) -> list[str]:
        return list(self.export.state_names)

    @property
    def inputs(self) -> list[str]:
        return list(self.export.input_names)

    @property
    def parameters(self) -> Dict[str, float]:
        return self.export.parameter_defaults


def _target_resource():
    return resources.files("cp_reach.targets").joinpath("sympy-ode")


def _new_session(
    roots: Optional[Iterable[Union[str, Path]]],
    workspace: Optional[Union[str, Path]],
):
    try:
        import rumoca
    except ImportError as exc:
        raise ImportError(
            "Modelica loading requires Rumoca 0.10; install cp_reach with its "
            "declared dependencies"
        ) from exc

    try:
        version_text = rumoca.version()
        version = tuple(int(part) for part in version_text.split(".")[:2])
    except (AttributeError, TypeError, ValueError):
        version_text = "unknown"
        version = (0, 0)
    if version != (0, 10):
        raise ImportError("CP Reach supports Rumoca 0.10.x; " f"found {version_text}")

    session_type = getattr(rumoca, "Session", None)
    if session_type is None:
        raise ImportError(
            "Modelica loading requires Rumoca 0.10; the installed "
            "Rumoca does not provide rumoca.Session"
        )
    root_strings = [str(root) for root in roots] if roots is not None else None
    workspace_string = str(workspace) if workspace is not None else None
    return session_type(roots=root_strings, workspace=workspace_string)


def render_sympy_export(model: Any) -> SympySolveExport:
    """Render and load CP Reach's SymPy target from a compiled Rumoca model.

    ``model`` is deliberately duck-typed so the generated boundary can be
    contract-tested without invoking the compiler.
    """
    if not hasattr(model, "render"):
        raise TypeError("model must be a compiled rumoca.Model with render(target)")

    with resources.as_file(_target_resource()) as target_path:
        try:
            source = model.render(str(target_path))
        except Exception as exc:
            raise RumocaSympyExportError(
                "Rumoca could not render an explicit SymPy Solve model. "
                "The model may contain unsupported events or an implicit DAE. "
                f"Rumoca reported: {exc}"
            ) from exc

    if not isinstance(source, str) or not source.strip():
        raise RumocaSympyExportError("Rumoca returned an empty SymPy target")

    namespace: Dict[str, Any] = {"__name__": "_cp_reach_rumoca_sympy_export"}
    try:
        exec(compile(source, "<rumoca cp-reach-sympy-ode>", "exec"), namespace)
    except Exception as exc:
        raise RumocaSympyExportError("Failed to load Rumoca's generated SymPy model") from exc

    return _export_from_namespace(namespace)


def _export_from_namespace(namespace: Mapping[str, Any]) -> SympySolveExport:
    required = (
        "RHS",
        "STATE_SYMBOLS",
        "INPUT_SYMBOLS",
        "PARAMETER_SYMBOLS",
        "VARIABLE_EXPRESSIONS",
        "STATE_NAMES",
        "INPUT_NAMES",
        "PARAMETER_NAMES",
        "ALGEBRAIC_NAMES",
        "OUTPUT_NAMES",
        "DEFAULT_STATES",
        "DEFAULT_PARAMETERS",
        "ALGEBRAIC_ASSIGNMENTS_COMPLETE",
    )
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RumocaSympyExportError(
            f"Generated SymPy target is missing contract fields: {', '.join(missing)}"
        )

    state_symbols = tuple(namespace["STATE_SYMBOLS"])
    input_symbols = tuple(namespace["INPUT_SYMBOLS"])
    parameter_symbols = tuple(namespace["PARAMETER_SYMBOLS"])
    state_names = tuple(str(name) for name in namespace["STATE_NAMES"])
    input_names = tuple(str(name) for name in namespace["INPUT_NAMES"])
    parameter_names = tuple(str(name) for name in namespace["PARAMETER_NAMES"])
    default_states = tuple(float(value) for value in namespace["DEFAULT_STATES"])
    default_parameters = tuple(float(value) for value in namespace["DEFAULT_PARAMETERS"])
    rhs = sp.ImmutableDenseMatrix(namespace["RHS"])

    _require_same_length("states", state_names, state_symbols, default_states)
    _require_same_length("inputs", input_names, input_symbols)
    _require_same_length("parameters", parameter_names, parameter_symbols, default_parameters)
    if rhs.rows != len(state_symbols) or rhs.cols != 1:
        raise RumocaSympyExportError(
            f"Generated RHS has shape {rhs.shape}; expected ({len(state_symbols)}, 1)"
        )

    expressions = {
        str(name): sp.sympify(expr)
        for name, expr in dict(namespace["VARIABLE_EXPRESSIONS"]).items()
    }
    return SympySolveExport(
        rhs=rhs,
        state_symbols=state_symbols,
        input_symbols=input_symbols,
        parameter_symbols=parameter_symbols,
        variable_expressions=MappingProxyType(expressions),
        state_names=state_names,
        input_names=input_names,
        parameter_names=parameter_names,
        algebraic_names=tuple(str(name) for name in namespace["ALGEBRAIC_NAMES"]),
        output_names=tuple(str(name) for name in namespace["OUTPUT_NAMES"]),
        default_states=default_states,
        default_parameters=default_parameters,
        algebraic_assignments_complete=bool(namespace["ALGEBRAIC_ASSIGNMENTS_COMPLETE"]),
    )


def _require_same_length(label: str, *values: Sequence[Any]) -> None:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise RumocaSympyExportError(
            f"Generated {label} metadata lengths disagree: {sorted(lengths)}"
        )


def sympy_export_to_statespace(
    export: SympySolveExport,
    output_names: Optional[Iterable[str]] = None,
    simplify: bool = True,
) -> SymbolicStateSpace:
    """Convert a generated Solve export to CP Reach's analysis representation."""
    input_zero = {symbol: sp.S.Zero for symbol in export.input_symbols}
    total_rhs = sp.Matrix(export.rhs)
    drift = total_rhs.subs(input_zero, simultaneous=True)
    control = total_rhs - drift

    h = None
    feedthrough = None
    output_symbols = None
    requested_outputs = list(output_names or [])
    if requested_outputs:
        if not export.algebraic_assignments_complete:
            raise RumocaSympyExportError(
                "Rumoca did not produce a complete explicit algebraic assignment plan; "
                "named output expressions are unavailable for this implicit system"
            )
        missing = [name for name in requested_outputs if name not in export.variable_expressions]
        if missing:
            available = sorted(export.variable_expressions)
            raise ValueError(f"Unknown outputs {missing}; available variables: {available}")

        total_outputs = sp.Matrix([export.variable_expressions[name] for name in requested_outputs])
        h = total_outputs.subs(input_zero, simultaneous=True)
        feedthrough = total_outputs - h
        output_symbols = [sp.Symbol(name, real=True) for name in requested_outputs]

    if simplify:
        drift = drift.applyfunc(sp.simplify)
        control = control.applyfunc(sp.simplify)
        if h is not None:
            h = h.applyfunc(sp.simplify)
            feedthrough = feedthrough.applyfunc(sp.simplify)

    return SymbolicStateSpace(
        f=drift,
        Bu=control,
        h=h,
        Du=feedthrough,
        state_symbols=list(export.state_symbols),
        input_symbols=list(export.input_symbols),
        param_symbols=list(export.parameter_symbols),
        output_symbols=output_symbols,
        param_defaults=export.parameter_defaults,
    )


def rumoca_model_to_symbolic(
    model: Any,
    output_names: Optional[Iterable[str]] = None,
    simplify: bool = True,
) -> RumocaSymbolicModel:
    """Create a reachability-ready symbolic wrapper from ``rumoca.Model``."""
    export = render_sympy_export(model)
    symbolic = sympy_export_to_statespace(export, output_names, simplify)
    return RumocaSymbolicModel(rumoca=model, symbolic=symbolic, export=export)


def modelica_load(
    path: Union[str, Path],
    model_name: Optional[str] = None,
    *,
    roots: Optional[Iterable[Union[str, Path]]] = None,
    workspace: Optional[Union[str, Path]] = None,
    output_names: Optional[Iterable[str]] = None,
    simplify: bool = True,
    session: Any = None,
) -> RumocaSymbolicModel:
    """Compile a Modelica file with Rumoca and return a symbolic CP Reach model."""
    if session is None:
        session = _new_session(roots, workspace)

    source_path = Path(path)
    kwargs = {"model": model_name} if model_name is not None else {}
    kwargs["filename"] = source_path.name
    model = session.loads(source_path.read_text(), **kwargs)
    return rumoca_model_to_symbolic(model, output_names=output_names, simplify=simplify)


def modelica_loads(
    source: str,
    model_name: Optional[str] = None,
    *,
    filename: Optional[str] = None,
    roots: Optional[Iterable[Union[str, Path]]] = None,
    workspace: Optional[Union[str, Path]] = None,
    output_names: Optional[Iterable[str]] = None,
    simplify: bool = True,
    session: Any = None,
) -> RumocaSymbolicModel:
    """Compile Modelica source text with Rumoca and return a symbolic model."""
    if session is None:
        session = _new_session(roots, workspace)

    kwargs: Dict[str, Any] = {}
    if model_name is not None:
        kwargs["model"] = model_name
    if filename is not None:
        kwargs["filename"] = filename
    model = session.loads(source, **kwargs)
    return rumoca_model_to_symbolic(model, output_names=output_names, simplify=simplify)


__all__ = [
    "RumocaSympyExportError",
    "SympySolveExport",
    "RumocaSymbolicModel",
    "render_sympy_export",
    "sympy_export_to_statespace",
    "rumoca_model_to_symbolic",
    "modelica_load",
    "modelica_loads",
]
