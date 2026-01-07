"""
Uncertainty Specification Parser.

This module defines the YAML schema for uncertainty specifications and
provides loaders for uncertainty models used in reachability analysis.

Example YAML format:
    disturbances:
      d:
        type: bounded
        bound: 1.0
      w:
        type: gaussian
        covariance: [[0.1, 0], [0, 0.1]]

    parameters:
      m:
        nominal: 1.0
        bounds: [0.9, 1.1]

    initial_conditions:
      e:
        type: zero
      x:
        type: ellipsoid
        center: [0, 0]
        shape: [[0.01, 0], [0, 0.01]]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


@dataclass
class BoundedDisturbance:
    """
    Bounded disturbance specification: ||d(t)|| <= bound.

    Attributes
    ----------
    bound : float
        Maximum magnitude of the disturbance
    norm : str
        Norm type: "inf" (L-infinity), "2" (L2), default "inf"
    """

    bound: float
    norm: str = "inf"

    def __post_init__(self):
        if self.bound < 0:
            raise ValueError(f"Disturbance bound must be non-negative, got {self.bound}")
        if self.norm not in ("inf", "2", "1"):
            raise ValueError(f"Unknown norm type: {self.norm}")


@dataclass
class GaussianDisturbance:
    """
    Gaussian disturbance specification: d ~ N(0, covariance).

    Attributes
    ----------
    covariance : np.ndarray
        Covariance matrix (n x n for n-dimensional disturbance)
    """

    covariance: np.ndarray

    def __post_init__(self):
        self.covariance = np.atleast_2d(self.covariance)
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError(
                f"Covariance must be square, got shape {self.covariance.shape}"
            )

    @property
    def dimension(self) -> int:
        """Dimension of the disturbance."""
        return self.covariance.shape[0]


@dataclass
class ParameterUncertainty:
    """
    Parametric uncertainty specification.

    Attributes
    ----------
    nominal : float
        Nominal parameter value
    bounds : tuple[float, float]
        Lower and upper bounds on the parameter
    """

    nominal: float
    bounds: tuple

    def __post_init__(self):
        if len(self.bounds) != 2:
            raise ValueError(f"Bounds must have exactly 2 elements, got {len(self.bounds)}")
        if self.bounds[0] > self.bounds[1]:
            raise ValueError(
                f"Lower bound {self.bounds[0]} > upper bound {self.bounds[1]}"
            )
        if not (self.bounds[0] <= self.nominal <= self.bounds[1]):
            raise ValueError(
                f"Nominal value {self.nominal} not in bounds {self.bounds}"
            )

    @property
    def lower(self) -> float:
        """Lower bound."""
        return self.bounds[0]

    @property
    def upper(self) -> float:
        """Upper bound."""
        return self.bounds[1]

    @property
    def range(self) -> float:
        """Range of uncertainty."""
        return self.bounds[1] - self.bounds[0]


@dataclass
class InitialCondition:
    """
    Initial condition uncertainty specification.

    Attributes
    ----------
    type : str
        Type of IC: "zero", "point", "ellipsoid", "box"
    center : np.ndarray or None
        Center point (for non-zero types)
    shape : np.ndarray or None
        Shape matrix (for ellipsoid) or bounds (for box)
    """

    type: str
    center: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.type not in ("zero", "point", "ellipsoid", "box"):
            raise ValueError(f"Unknown initial condition type: {self.type}")

        if self.center is not None:
            self.center = np.atleast_1d(self.center)

        if self.shape is not None:
            self.shape = np.atleast_2d(self.shape)

    @property
    def dimension(self) -> int:
        """Dimension of the initial condition."""
        if self.center is not None:
            return len(self.center)
        if self.shape is not None:
            return self.shape.shape[0]
        return 0


@dataclass
class UncertaintySpec:
    """
    Complete uncertainty specification for a system.

    This bundles all uncertainty sources: disturbances, parametric uncertainty,
    and initial condition uncertainty.

    Attributes
    ----------
    disturbances : dict[str, BoundedDisturbance | GaussianDisturbance]
        Disturbance specifications by input name
    parameters : dict[str, ParameterUncertainty]
        Parameter uncertainty by parameter name
    initial_conditions : dict[str, InitialCondition]
        Initial condition uncertainty by state name
    """

    disturbances: Dict[str, Union[BoundedDisturbance, GaussianDisturbance]] = field(
        default_factory=dict
    )
    parameters: Dict[str, ParameterUncertainty] = field(default_factory=dict)
    initial_conditions: Dict[str, InitialCondition] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "UncertaintySpec":
        """
        Load uncertainty specification from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file

        Returns
        -------
        UncertaintySpec
            Parsed uncertainty specification

        Raises
        ------
        ImportError
            If PyYAML is not installed
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the YAML is malformed
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML parsing. Install with: pip install pyyaml"
            )

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data or {})

    @classmethod
    def from_yaml_str(cls, yaml_str: str) -> "UncertaintySpec":
        """Load from a YAML string."""
        if yaml is None:
            raise ImportError("PyYAML is required")
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintySpec":
        """
        Create UncertaintySpec from a parsed dictionary.

        Parameters
        ----------
        data : dict
            Parsed YAML data

        Returns
        -------
        UncertaintySpec
        """
        disturbances = {}
        parameters = {}
        initial_conditions = {}

        # Parse disturbances
        for name, dist_data in data.get("disturbances", {}).items():
            dist_type = dist_data.get("type", "bounded")

            if dist_type == "bounded":
                disturbances[name] = BoundedDisturbance(
                    bound=float(dist_data.get("bound", 1.0)),
                    norm=dist_data.get("norm", "inf"),
                )
            elif dist_type == "gaussian":
                cov_data = dist_data.get("covariance", [[1.0]])
                disturbances[name] = GaussianDisturbance(
                    covariance=np.array(cov_data, dtype=float)
                )
            else:
                raise ValueError(f"Unknown disturbance type: {dist_type}")

        # Parse parameters
        for name, param_data in data.get("parameters", {}).items():
            parameters[name] = ParameterUncertainty(
                nominal=float(param_data.get("nominal", 1.0)),
                bounds=tuple(param_data.get("bounds", [0.9, 1.1])),
            )

        # Parse initial conditions
        for name, ic_data in data.get("initial_conditions", {}).items():
            ic_type = ic_data.get("type", "zero")
            center = ic_data.get("center")
            shape = ic_data.get("shape")

            if center is not None:
                center = np.array(center, dtype=float)
            if shape is not None:
                shape = np.array(shape, dtype=float)

            initial_conditions[name] = InitialCondition(
                type=ic_type,
                center=center,
                shape=shape,
            )

        return cls(
            disturbances=disturbances,
            parameters=parameters,
            initial_conditions=initial_conditions,
        )

    def get_dist_bound(self, input_name: str) -> float:
        """
        Get the disturbance bound for a specific input.

        Parameters
        ----------
        input_name : str
            Name of the input/disturbance channel

        Returns
        -------
        float
            Disturbance bound (inf for unbounded)
        """
        if input_name not in self.disturbances:
            return float("inf")

        dist = self.disturbances[input_name]
        if isinstance(dist, BoundedDisturbance):
            return dist.bound
        elif isinstance(dist, GaussianDisturbance):
            # For Gaussian, use 3-sigma bound as approximation
            return 3.0 * np.sqrt(np.max(np.diag(dist.covariance)))
        return float("inf")

    def get_dist_bounds_vector(self, input_names: List[str]) -> np.ndarray:
        """
        Get disturbance bounds as a vector.

        Parameters
        ----------
        input_names : list[str]
            Ordered list of input names

        Returns
        -------
        np.ndarray
            Vector of disturbance bounds
        """
        return np.array([self.get_dist_bound(name) for name in input_names])

    def get_param_nominal(self, param_name: str) -> float:
        """Get nominal value for a parameter."""
        if param_name in self.parameters:
            return self.parameters[param_name].nominal
        return 1.0

    def get_param_bounds(self, param_name: str) -> tuple:
        """Get bounds for a parameter."""
        if param_name in self.parameters:
            return self.parameters[param_name].bounds
        return (1.0, 1.0)  # No uncertainty

    def get_param_defaults(self) -> Dict[str, float]:
        """Get all nominal parameter values as a dict."""
        return {name: p.nominal for name, p in self.parameters.items()}

    def validate_against_ir(self, ir: "DaeIR") -> List[str]:
        """
        Validate uncertainty spec against a DaeIR.

        Checks that all referenced variables exist in the IR.

        Parameters
        ----------
        ir : DaeIR
            DAE IR to validate against

        Returns
        -------
        list[str]
            List of warning messages (empty if all valid)
        """
        warnings = []

        # Check disturbances
        for name in self.disturbances:
            if name not in ir.inputs:
                warnings.append(f"Disturbance '{name}' not found in IR inputs")

        # Check parameters
        for name in self.parameters:
            if name not in ir.parameters:
                warnings.append(f"Parameter '{name}' not found in IR parameters")

        # Check initial conditions
        for name in self.initial_conditions:
            if name not in ir.states and name not in ir.algebraics:
                warnings.append(f"Initial condition '{name}' not found in IR")

        return warnings
