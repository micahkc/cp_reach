"""
Reachability Query Specification Parser.

This module defines the YAML schema for reachability queries that specify
what guarantees to compute and how to compute them.

Example YAML format:
    query:
      type: flowpipe
      dynamics: error
      outputs: [e, ev]

      # Reference trajectory (optional, for flowpipe analysis)
      trajectory:
        # Option 1: Load from file (CSV or NPY)
        source: file
        path: "trajectory.csv"
        # CSV format: t, x1, x2, ..., u1, u2, ...
        # NPY format: dict with 't', 'x', 'u' keys

        # Option 2: Generate with polynomial planner
        source: polynomial
        waypoints:             # Position waypoints (n_waypoints x n_dim)
          - [0, 0]
          - [1, 0.5]
          - [2, 0]
        velocities:            # Velocity at waypoints (null = free)
          - [0, 0]
          - null
          - [0, 0]
        min_deriv: 4           # Minimize snap (4) or jerk (3)
        poly_deg: 5            # Polynomial degree per segment
        duration: auto         # Total time, or 'auto' to optimize

      alpha_search:
        min: 1.0e-4
        max: 10.0
        num_points: 40

      output_format:
        certificate: true
        bounds: true
        plots: true
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
class AlphaSearch:
    """
    Specification for alpha (decay rate) search in LMI solving.

    Attributes
    ----------
    min : float
        Minimum alpha value to search
    max : float
        Maximum alpha value to search
    num_points : int
        Number of grid points for search
    scale : str
        Scale type: "log" or "linear"
    """

    min: float = 1e-4
    max: float = 10.0
    num_points: int = 40
    scale: str = "log"

    def __post_init__(self):
        if self.min <= 0:
            raise ValueError(f"Alpha min must be positive, got {self.min}")
        if self.max <= self.min:
            raise ValueError(f"Alpha max must be > min, got max={self.max}, min={self.min}")
        if self.num_points < 2:
            raise ValueError(f"Need at least 2 grid points, got {self.num_points}")
        if self.scale not in ("log", "linear"):
            raise ValueError(f"Unknown scale type: {self.scale}")

    def to_grid(self) -> np.ndarray:
        """
        Generate the alpha search grid.

        Returns
        -------
        np.ndarray
            Array of alpha values to search
        """
        if self.scale == "log":
            return np.logspace(np.log10(self.min), np.log10(self.max), self.num_points)
        else:
            return np.linspace(self.min, self.max, self.num_points)


@dataclass
class OutputFormat:
    """
    Specification for output format.

    Attributes
    ----------
    certificate : bool
        Whether to output the Lyapunov certificate (P matrix)
    bounds : bool
        Whether to output state bounds
    plots : bool
        Whether to generate plots
    json : bool
        Whether to output results as JSON
    """

    certificate: bool = True
    bounds: bool = True
    plots: bool = True
    json: bool = True


@dataclass
class TrajectorySpec:
    """
    Specification for a reference trajectory.

    The trajectory can be loaded from a file or generated using the
    polynomial path planner.

    Attributes
    ----------
    source : str
        Source type: "file" or "polynomial"
    path : str or None
        Path to trajectory file (for source="file")
    waypoints : list or None
        Position waypoints for polynomial planner (n_waypoints x n_dim)
    velocities : list or None
        Velocity constraints at waypoints (None entries are free)
    accelerations : list or None
        Acceleration constraints at waypoints (None entries are free)
    min_deriv : int
        Derivative order to minimize (3=jerk, 4=snap)
    poly_deg : int
        Polynomial degree per segment
    duration : float or str
        Total trajectory duration, or "auto" to optimize
    n_points : int
        Number of time points to sample
    """

    source: str = "file"
    path: Optional[str] = None
    waypoints: Optional[List[List[float]]] = None
    velocities: Optional[List[Optional[List[float]]]] = None
    accelerations: Optional[List[Optional[List[float]]]] = None
    min_deriv: int = 4
    poly_deg: int = 5
    duration: Union[float, str] = "auto"
    n_points: int = 100

    def __post_init__(self):
        valid_sources = ("file", "polynomial")
        if self.source not in valid_sources:
            raise ValueError(f"Unknown trajectory source: {self.source}. Valid: {valid_sources}")

        if self.source == "file" and not self.path:
            raise ValueError("Trajectory source='file' requires 'path' to be specified")

        if self.source == "polynomial" and not self.waypoints:
            raise ValueError("Trajectory source='polynomial' requires 'waypoints' to be specified")

    def load(self, base_path: Optional[Path] = None) -> "Trajectory":
        """
        Load or generate the trajectory.

        Parameters
        ----------
        base_path : Path, optional
            Base path for resolving relative file paths

        Returns
        -------
        Trajectory
            The loaded or generated trajectory
        """
        from cp_reach.planning.trajectory import Trajectory

        if self.source == "file":
            return self._load_from_file(base_path)
        else:
            return self._generate_polynomial()

    def _load_from_file(self, base_path: Optional[Path] = None) -> "Trajectory":
        """Load trajectory from CSV or NPY file."""
        from cp_reach.planning.trajectory import Trajectory

        file_path = Path(self.path)
        if base_path and not file_path.is_absolute():
            file_path = base_path / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".npy" or suffix == ".npz":
            # NumPy format: expects dict with 't', 'x', 'u' keys
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                # Single array - assume it's a dict saved with allow_pickle
                data = data.item()
            return Trajectory(
                t=np.asarray(data["t"]),
                x=np.asarray(data["x"]),
                u=np.asarray(data.get("u", np.zeros((len(data["t"]), 0)))),
                metadata={"source": "file", "path": str(file_path)},
            )

        elif suffix == ".csv":
            # CSV format: t, x1, x2, ..., xn, u1, u2, ..., um
            # First row is header with column names
            import csv

            with open(file_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                data = np.array([list(map(float, row)) for row in reader])

            t = data[:, 0]

            # Parse header to find state and input columns
            x_cols = [i for i, h in enumerate(header) if h.startswith("x") or h.startswith("state")]
            u_cols = [i for i, h in enumerate(header) if h.startswith("u") or h.startswith("input")]

            if not x_cols:
                # Assume columns 1 to n are states, rest are inputs
                # Try to infer from column count
                n_cols = data.shape[1] - 1  # Exclude time column
                x_cols = list(range(1, n_cols + 1))
                u_cols = []

            x = data[:, x_cols] if x_cols else data[:, 1:]
            u = data[:, u_cols] if u_cols else np.zeros((len(t), 0))

            return Trajectory(
                t=t,
                x=x,
                u=u,
                metadata={"source": "file", "path": str(file_path), "header": header},
            )

        else:
            raise ValueError(f"Unknown trajectory file format: {suffix}. Use .csv, .npy, or .npz")

    def _generate_polynomial(self) -> "Trajectory":
        """Generate trajectory using polynomial planner.

        The polynomial planner requires: poly_deg = 2 * bc_deriv - 1
        For poly_deg=5, bc_deriv=3 (pos, vel, acc).
        For poly_deg=7, bc_deriv=4 (pos, vel, acc, jerk).

        All boundary conditions must be specified (no free rows optimization).
        If velocities/accelerations are not provided at a waypoint, zero is used.
        """
        from cp_reach.planning import plan_minimum_derivative_trajectory
        from cp_reach.planning.trajectory import Trajectory

        waypoints = np.array(self.waypoints)  # (n_waypoints, n_dim)
        n_waypoints, n_dim = waypoints.shape
        n_legs = n_waypoints - 1

        # Given poly_deg, compute the corresponding bc_deriv.
        bc_deriv = (self.poly_deg + 1) // 2

        # Build boundary conditions array: (bc_deriv, n_waypoints, n_dim)
        bc = np.zeros((bc_deriv, n_waypoints, n_dim))

        # Set positions (always required)
        bc[0, :, :] = waypoints

        # Set velocities if provided and bc_deriv >= 2
        if bc_deriv >= 2 and self.velocities:
            for i, vel in enumerate(self.velocities):
                if vel is not None:
                    bc[1, i, :] = vel

        # Set accelerations if provided and bc_deriv >= 3
        if bc_deriv >= 3 and self.accelerations:
            for i, acc in enumerate(self.accelerations):
                if acc is not None:
                    bc[2, i, :] = acc

        # Generate trajectory
        traj = plan_minimum_derivative_trajectory(
            bc=bc,
            min_deriv=self.min_deriv,
            poly_deg=self.poly_deg,
            k_time=0.1 if self.duration == "auto" else 0.0,
            T_guess=None if self.duration == "auto" else np.ones(n_legs) * self.duration / n_legs,
            rows_free=None,  # All constraints specified
            bc_deriv=bc_deriv,
            control_dim=n_dim,
        )

        return traj

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectorySpec":
        """Create TrajectorySpec from a parsed dictionary."""
        return cls(
            source=data.get("source", "file"),
            path=data.get("path"),
            waypoints=data.get("waypoints"),
            velocities=data.get("velocities"),
            accelerations=data.get("accelerations"),
            min_deriv=int(data.get("min_deriv", 4)),
            poly_deg=int(data.get("poly_deg", 5)),
            duration=data.get("duration", "auto"),
            n_points=int(data.get("n_points", 100)),
        )


@dataclass
class ReachQuery:
    """
    Complete reachability query specification.

    This specifies what analysis to perform and what outputs to produce.

    Attributes
    ----------
    type : str
        Query type: "bounded_reachable_set", "output_bound", "invariant_set", "flowpipe"
    dynamics : str
        Which dynamics to analyze: "error", "full_state", "estimation_error"
    outputs : list[str]
        Which output variables to compute bounds for
    horizon : float or None
        Time horizon for time-varying analysis (None for infinite horizon)
    dist_inputs : list[str]
        Which inputs are treated as disturbances
    trajectory : TrajectorySpec or None
        Reference trajectory specification (for flowpipe analysis)
    alpha_search : AlphaSearch
        Parameters for alpha (decay rate) optimization
    output_format : OutputFormat
        What outputs to produce
    """

    type: str = "bounded_reachable_set"
    dynamics: str = "error"
    outputs: List[str] = field(default_factory=list)
    horizon: Optional[float] = None
    dist_inputs: List[str] = field(default_factory=list)
    trajectory: Optional[TrajectorySpec] = None
    alpha_search: AlphaSearch = field(default_factory=AlphaSearch)
    output_format: OutputFormat = field(default_factory=OutputFormat)

    def __post_init__(self):
        valid_types = ("bounded_reachable_set", "output_bound", "invariant_set", "flowpipe")
        if self.type not in valid_types:
            raise ValueError(f"Unknown query type: {self.type}. Valid types: {valid_types}")

        valid_dynamics = ("error", "full_state", "estimation_error", "combined")
        if self.dynamics not in valid_dynamics:
            raise ValueError(
                f"Unknown dynamics type: {self.dynamics}. Valid types: {valid_dynamics}"
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ReachQuery":
        """
        Load query specification from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file

        Returns
        -------
        ReachQuery
            Parsed query specification
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
    def from_yaml_str(cls, yaml_str: str) -> "ReachQuery":
        """Load from a YAML string."""
        if yaml is None:
            raise ImportError("PyYAML is required")
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReachQuery":
        """
        Create ReachQuery from a parsed dictionary.

        Parameters
        ----------
        data : dict
            Parsed YAML data (expects a "query" key or direct fields)

        Returns
        -------
        ReachQuery
        """
        # Handle nested "query" key
        if "query" in data:
            data = data["query"]

        # Parse alpha search
        alpha_data = data.get("alpha_search", {})
        alpha_search = AlphaSearch(
            min=float(alpha_data.get("min", 1e-4)),
            max=float(alpha_data.get("max", 10.0)),
            num_points=int(alpha_data.get("num_points", 40)),
            scale=alpha_data.get("scale", "log"),
        )

        # Parse output format
        format_data = data.get("output_format", {})
        output_format = OutputFormat(
            certificate=format_data.get("certificate", True),
            bounds=format_data.get("bounds", True),
            plots=format_data.get("plots", True),
            json=format_data.get("json", True),
        )

        # Parse trajectory (optional)
        traj_data = data.get("trajectory")
        trajectory = TrajectorySpec.from_dict(traj_data) if traj_data else None

        # Parse main fields
        return cls(
            type=data.get("type", "bounded_reachable_set"),
            dynamics=data.get("dynamics", "error"),
            outputs=data.get("outputs", []),
            horizon=data.get("horizon"),
            dist_inputs=data.get("dist_inputs", data.get("disturbances", [])),
            trajectory=trajectory,
            alpha_search=alpha_search,
            output_format=output_format,
        )

    def to_compute_args(self) -> Dict[str, Any]:
        """
        Convert query to arguments for compute_reachable_set().

        Returns
        -------
        dict
            Keyword arguments for the compute function
        """
        args = {
            "method": "lmi",
            "dynamics": self.dynamics,
            "alpha_grid": self.alpha_search.to_grid(),
        }

        if self.dist_inputs:
            args["dist_input"] = self.dist_inputs

        return args

    def validate_against_ir(self, ir: "DaeIR") -> List[str]:
        """
        Validate query against a DaeIR.

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

        # Check outputs
        available_outputs = set(ir.algebraics.keys()) | set(ir.states.keys())
        for name in self.outputs:
            if name not in available_outputs:
                warnings.append(f"Output '{name}' not found in IR")

        # Check disturbance inputs
        for name in self.dist_inputs:
            if name not in ir.inputs:
                warnings.append(f"Disturbance input '{name}' not found in IR inputs")

        return warnings
