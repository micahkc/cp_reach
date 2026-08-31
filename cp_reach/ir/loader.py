"""Load RuMoCA 0.9.20 DAE schema-7 JSON for CP_REACH analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

SUPPORTED_SCHEMA_VERSION = 7


@dataclass
class Component:
    """A scalar or array component from a RuMoCA DAE category."""

    name: str
    type_name: str = "Real"
    variability: str = ""
    causality: str = "local"
    shape: List[Any] = field(default_factory=list)
    start: Optional[float] = None
    role: Optional[str] = None

    def is_parameter(self) -> bool:
        """Return whether this component came from the DAE parameter category."""
        return self.variability == "parameter"

    def is_constant(self) -> bool:
        """Return whether this component came from the DAE constant category."""
        return self.variability == "constant"

    def is_input(self) -> bool:
        """Return whether RuMoCA classified this component as an input."""
        return self.causality == "input"

    def is_output(self) -> bool:
        """Return whether RuMoCA classified this component as an output."""
        return self.causality == "output"

    def is_scalar(self) -> bool:
        """Return whether this component has no array dimensions."""
        return not self.shape


@dataclass
class DaeIR:
    """The CP_REACH view of a RuMoCA 0.9.20 DAE schema-7 model."""

    model_name: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION
    states: Dict[str, Component] = field(default_factory=dict)
    algebraics: Dict[str, Component] = field(default_factory=dict)
    inputs: Dict[str, Component] = field(default_factory=dict)
    parameters: Dict[str, Component] = field(default_factory=dict)
    constants: Dict[str, Component] = field(default_factory=dict)
    equations: List[Any] = field(default_factory=list)
    initial_equations: List[Any] = field(default_factory=list)
    algebraic_equations: List[Any] = field(default_factory=list)

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        *,
        model_name: Optional[str] = None,
    ) -> "DaeIR":
        """Load DAE JSON from a file, using its stem as the model name by default."""
        source_path = Path(path)
        with source_path.open() as stream:
            data = json.load(stream)
        return cls.from_dict(data, model_name=model_name or source_path.stem)

    @classmethod
    def from_json_str(
        cls,
        json_str: str,
        *,
        model_name: Optional[str] = None,
    ) -> "DaeIR":
        """Load DAE JSON from a string."""
        return cls.from_dict(json.loads(json_str), model_name=model_name)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        model_name: Optional[str] = None,
    ) -> "DaeIR":
        """Create an IR from RuMoCA 0.9.20 DAE schema-7 data."""
        schema_version = data.get("schema_version")
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"CP_REACH requires RuMoCA DAE schema {SUPPORTED_SCHEMA_VERSION}; "
                f"received {schema_version!r}"
            )

        def parse_component(
            name: str,
            component: Dict[str, Any],
            *,
            variability: str = "",
        ) -> Component:
            return Component(
                name=name,
                variability=variability,
                causality=component.get("causality", "local"),
                shape=component.get("dims", []),
                start=_parse_start_value(component.get("start")),
            )

        states = {name: parse_component(name, value) for name, value in data["x"].items()}
        algebraics = {name: parse_component(name, value) for name, value in data["y"].items()}
        inputs = {name: parse_component(name, value) for name, value in data["u"].items()}
        parameters = {
            name: parse_component(name, value, variability="parameter")
            for name, value in data["p"].items()
        }
        constants = {
            name: parse_component(name, value, variability="constant")
            for name, value in data["constants"].items()
        }

        return cls(
            model_name=model_name or "UnnamedModel",
            schema_version=schema_version,
            states=states,
            algebraics=algebraics,
            inputs=inputs,
            parameters=parameters,
            constants=constants,
            equations=data["f_x"],
            initial_equations=data["initial_equations"],
            algebraic_equations=data["f_z"],
        )

    def infer_roles(self) -> Dict[str, str]:
        """Infer plant/controller roles from dotted component-name prefixes."""
        roles = {}
        names = [*self.states, *self.inputs, *self.algebraics]
        for name in names:
            if "." in name:
                roles[name] = name.split(".", 1)[0]
        return roles

    def get_state_names(self) -> List[str]:
        """Return state names in RuMoCA order."""
        return list(self.states)

    def get_input_names(self) -> List[str]:
        """Return input names in RuMoCA order."""
        return list(self.inputs)

    def get_parameter_names(self) -> List[str]:
        """Return parameter names in RuMoCA order."""
        return list(self.parameters)

    def get_algebraic_names(self) -> List[str]:
        """Return algebraic names in RuMoCA order."""
        return list(self.algebraics)

    def get_param_defaults(self) -> Dict[str, float]:
        """Return numeric parameter and constant defaults."""
        components = {**self.parameters, **self.constants}
        return {
            name: component.start
            for name, component in components.items()
            if component.start is not None
        }

    def n_states(self) -> int:
        """Return the number of continuous states."""
        return len(self.states)

    def n_inputs(self) -> int:
        """Return the number of inputs."""
        return len(self.inputs)

    def n_parameters(self) -> int:
        """Return the number of parameters."""
        return len(self.parameters)

    def n_algebraics(self) -> int:
        """Return the number of algebraic variables."""
        return len(self.algebraics)


def _parse_start_value(start_ast: Any) -> Optional[float]:
    """Extract a numeric start value from a RuMoCA schema-7 expression."""
    if start_ast is None:
        return None
    if "Literal" in start_ast:
        value = start_ast["Literal"]["value"]
        for tag in ("Real", "Integer"):
            if tag in value:
                return float(value[tag])
        return None
    if "Unary" in start_ast and start_ast["Unary"]["op"] == "Minus":
        value = _parse_start_value(start_ast["Unary"]["rhs"])
        return -value if value is not None else None
    return None
