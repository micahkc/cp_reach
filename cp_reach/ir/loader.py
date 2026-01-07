"""
Rumoca DAE IR JSON Loader.

This module loads Rumoca's DAE (Differential Algebraic Equations) IR format
directly from JSON files, bypassing the need for cyecca or Modelica parsing.

The DAE format follows the Modelica specification Appendix B:
    - x: continuous states (variables that appear differentiated)
    - y: algebraic variables (continuous but not differentiated)
    - u: inputs (declared with input causality)
    - p: parameters (declared with parameter variability)
    - fx: continuous-time equations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json


@dataclass
class Component:
    """
    Represents a variable/component from the Rumoca DAE IR.

    Attributes
    ----------
    name : str
        Variable name (may include component path, e.g., "plant.x")
    type_name : str
        Modelica type name (e.g., "Real", "Integer")
    variability : str
        Variable variability: "", "parameter", "constant", "discrete"
    causality : str
        Variable causality: "", "input", "output"
    shape : list[int]
        Array dimensions (empty for scalars)
    start : float or None
        Initial/default value if specified
    role : str or None
        Component role inferred from prefix (e.g., "plant", "controller")
    """

    name: str
    type_name: str = "Real"
    variability: str = ""
    causality: str = ""
    shape: List[int] = field(default_factory=list)
    start: Optional[float] = None
    role: Optional[str] = None

    def is_parameter(self) -> bool:
        """Check if this component is a parameter."""
        return self.variability == "parameter"

    def is_constant(self) -> bool:
        """Check if this component is a constant."""
        return self.variability == "constant"

    def is_input(self) -> bool:
        """Check if this component is an input."""
        return self.causality == "input"

    def is_output(self) -> bool:
        """Check if this component is an output."""
        return self.causality == "output"

    def is_scalar(self) -> bool:
        """Check if this component is a scalar (not array)."""
        return len(self.shape) == 0


@dataclass
class DaeIR:
    """
    Represents a complete DAE (Differential Algebraic Equations) model.

    This is the primary data structure for loaded Rumoca IR. It contains
    all variables categorized by their role in the DAE, plus the equations
    as parsed AST structures.

    Attributes
    ----------
    model_name : str
        Name of the compiled model
    rumoca_version : str
        Version of Rumoca that generated this IR
    states : dict[str, Component]
        State variables (x) - those that appear differentiated
    algebraics : dict[str, Component]
        Algebraic variables (y) - continuous but not differentiated
    inputs : dict[str, Component]
        Input variables (u) - declared with input causality
    parameters : dict[str, Component]
        Parameters (p) - declared with parameter variability
    constants : dict[str, Component]
        Constants (cp) - declared with constant variability
    equations : list
        Continuous-time equations (fx) as AST structures
    initial_equations : list
        Initial equations (fx_init) as AST structures
    algebraic_equations : list
        Algebraic equations (fz) as AST structures
    """

    model_name: str
    rumoca_version: str = ""
    states: Dict[str, Component] = field(default_factory=dict)
    algebraics: Dict[str, Component] = field(default_factory=dict)
    inputs: Dict[str, Component] = field(default_factory=dict)
    parameters: Dict[str, Component] = field(default_factory=dict)
    constants: Dict[str, Component] = field(default_factory=dict)
    equations: List[Any] = field(default_factory=list)
    initial_equations: List[Any] = field(default_factory=list)
    algebraic_equations: List[Any] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DaeIR":
        """
        Load a DaeIR from a Rumoca DAE JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file

        Returns
        -------
        DaeIR
            Parsed DAE representation

        Raises
        ------
        FileNotFoundError
            If the JSON file doesn't exist
        json.JSONDecodeError
            If the JSON is malformed
        KeyError
            If required fields are missing
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_json_str(cls, json_str: str) -> "DaeIR":
        """
        Load a DaeIR from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON string containing the DAE IR

        Returns
        -------
        DaeIR
            Parsed DAE representation
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DaeIR":
        """
        Create a DaeIR from a parsed JSON dictionary.

        Parameters
        ----------
        data : dict
            Parsed JSON data

        Returns
        -------
        DaeIR
            Parsed DAE representation
        """

        def parse_component(name: str, comp: dict) -> Component:
            """Parse a single component from the JSON structure."""
            # Extract start value from AST
            start = None
            start_data = comp.get("start")
            if start_data and start_data != "Empty":
                start = _parse_start_value(start_data)

            # Extract variability (parameter, constant, discrete, or empty)
            variability = ""
            var_data = comp.get("variability")
            if isinstance(var_data, dict):
                # Format: {"Parameter": {...}} or {"Constant": {...}}
                variability = list(var_data.keys())[0].lower()
            elif var_data and var_data != "Empty":
                variability = str(var_data).lower()

            # Extract causality (input, output, or empty)
            causality = ""
            caus_data = comp.get("causality")
            if isinstance(caus_data, dict):
                # Format: {"Input": {...}} or {"Output": {...}}
                causality = list(caus_data.keys())[0].lower()
            elif caus_data and caus_data != "Empty":
                causality = str(caus_data).lower()

            # Extract type name
            type_name = "Real"
            type_data = comp.get("type_name", {})
            if isinstance(type_data, dict):
                name_parts = type_data.get("name", [])
                if name_parts:
                    # Take the last part of the qualified name
                    if isinstance(name_parts[-1], dict):
                        type_name = name_parts[-1].get("text", "Real")
                    else:
                        type_name = str(name_parts[-1])

            # Extract shape (array dimensions)
            shape = comp.get("shape", [])

            return Component(
                name=name,
                type_name=type_name,
                variability=variability,
                causality=causality,
                shape=shape,
                start=start,
            )

        # Parse all component categories
        states = {k: parse_component(k, v) for k, v in data.get("x", {}).items()}
        algebraics = {k: parse_component(k, v) for k, v in data.get("y", {}).items()}
        inputs = {k: parse_component(k, v) for k, v in data.get("u", {}).items()}
        parameters = {k: parse_component(k, v) for k, v in data.get("p", {}).items()}
        constants = {k: parse_component(k, v) for k, v in data.get("cp", {}).items()}

        return cls(
            model_name=data.get("model_name", "Unknown"),
            rumoca_version=data.get("rumoca_version", ""),
            states=states,
            algebraics=algebraics,
            inputs=inputs,
            parameters=parameters,
            constants=constants,
            equations=data.get("fx", []),
            initial_equations=data.get("fx_init", []),
            algebraic_equations=data.get("fz", []),
        )

    def infer_roles(self) -> Dict[str, str]:
        """
        Infer component roles from variable name prefixes.

        Variables with dotted names like "plant.x" are assumed to come from
        a component named "plant", and the role is set to that prefix.

        Returns
        -------
        dict[str, str]
            Mapping from variable name to inferred role
        """
        roles = {}
        all_vars = (
            list(self.states.keys())
            + list(self.inputs.keys())
            + list(self.algebraics.keys())
        )
        for name in all_vars:
            if "." in name:
                prefix = name.split(".")[0]
                roles[name] = prefix
        return roles

    def get_state_names(self) -> List[str]:
        """Get ordered list of state variable names."""
        return list(self.states.keys())

    def get_input_names(self) -> List[str]:
        """Get ordered list of input variable names."""
        return list(self.inputs.keys())

    def get_parameter_names(self) -> List[str]:
        """Get ordered list of parameter names."""
        return list(self.parameters.keys())

    def get_algebraic_names(self) -> List[str]:
        """Get ordered list of algebraic variable names."""
        return list(self.algebraics.keys())

    def get_param_defaults(self) -> Dict[str, float]:
        """
        Get default parameter values.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to default value
        """
        defaults = {}
        for name, comp in self.parameters.items():
            if comp.start is not None:
                defaults[name] = comp.start
        for name, comp in self.constants.items():
            if comp.start is not None:
                defaults[name] = comp.start
        return defaults

    def n_states(self) -> int:
        """Number of state variables."""
        return len(self.states)

    def n_inputs(self) -> int:
        """Number of input variables."""
        return len(self.inputs)

    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def n_algebraics(self) -> int:
        """Number of algebraic variables."""
        return len(self.algebraics)


def _parse_start_value(start_ast: Any) -> Optional[float]:
    """
    Extract numeric start value from AST.

    Parameters
    ----------
    start_ast : dict
        AST node representing the start value expression

    Returns
    -------
    float or None
        Numeric value if parseable, None otherwise
    """
    if not isinstance(start_ast, dict):
        return None

    # Handle Terminal nodes (literals)
    if "Terminal" in start_ast:
        terminal = start_ast["Terminal"]
        token = terminal.get("token", {})
        text = token.get("text", "")
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    # Handle unary minus
    if "Unary" in start_ast:
        unary = start_ast["Unary"]
        op = unary.get("op", {})
        if "Neg" in op or "Minus" in op:
            operand = unary.get("operand", {})
            val = _parse_start_value(operand)
            if val is not None:
                return -val
        return None

    return None
