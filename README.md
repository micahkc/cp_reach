# cp_reach

**cp_reach** is a Python library for computing reachable sets and performing reachability analysis on cyber-physical systems (CPS). It uses rigorous mathematical methods including Linear Matrix Inequalities (LMIs), Lyapunov theory, and Lie group structures to provide formal guarantees for nonlinear dynamics.

## Features

- **Generalizable Model Support**: Reachable sets for systems written in Modelica utilizing [RuMoCA](https://rumoca.dev)
- **Domain-Specific Analysis**: Reachable sets for Rover, Quadrotor, and Satellite utilizing log-linearized Lie groups
- **LMI-Based Over-Approximation**: Compute ellipsoidal reachable set over-approximations using Lyapunov theory
- **Polytopic Uncertainty**: Handle time-varying and uncertain systems via polytopic LMI formulations
- **Monte Carlo Simulation**: Simulate trajectories with bounded disturbances for validation
- **Trajectory Planning**: Polynomial trajectory generation for reference tracking
- **Multiple Backends**: Symbolic (SymPy) and numeric (CasADi) computation engines
- **Visualization**: Built-in plotting for error bounds, trajectories, and flowpipes

## Installation

### From PyPI (when published)

```bash
pip install cp_reach
```

### From Source

```bash
git clone https://github.com/CogniPilot/cp_reach.git
cd cp_reach
pip install -e .
```

### Dependencies

The library requires:
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `cvxpy`: Convex optimization and LMI solving
- `casadi`: Symbolic framework and numerical integration
- `sympy`: Symbolic mathematics for dynamics classification
- `cyecca`: Lie group operations (SO(3), SE(2,3))

## Quick Start

Here's a simple example computing reachable sets for a mass-spring-damper system from Modelica:

```python
import numpy as np
from cp_reach.reachability import (
    sympy_load,
    casadi_load,
    simulate_dist,
    compute_reachable_set,
    plot_flowpipe,
)

# Load Modelica model with SymPy backend
model_sympy = sympy_load("MassSpringPD.mo", output_names=["e", "ev"])

# Load CasADi backend for simulation
model_casadi = casadi_load("MassSpringPD.mo")

# Define initial conditions and disturbance bound
x0 = {"x": 1, "x_ref": 1}
dist_bound = 1.0

# Simulate nominal trajectory and Monte Carlo trials
nom_res, mc_res = simulate_dist(
    model_casadi,
    x0=x0,
    dist_bound=dist_bound,
    dist_input=["d"],
    num_sims=100,
)

# Compute ellipsoidal reachable set bounds via LMI
sol = compute_reachable_set(
    model_sympy,
    method="lmi",
    dynamics="error",
    dist_input=["d"],
    dist_bound=dist_bound,
    alpha_grid=np.logspace(-4, 1, 40),
)

# Visualize flowpipe with bounds
plot_flowpipe(
    nom_res,
    mc_res,
    groups=[["x"], ["v"]],
    state_names=model_casadi.state_names,
    error_fn=lambda t: sol["bounds_upper"],
)
```

See [examples/](examples/) for more comprehensive examples.

## Architecture

cp_reach is organized into several modules:

- **`ir`**: Intermediate representation loading from RuMoCA JSON output
- **`dynamics`**: Symbolic state space representations and dynamics classification
- **`reachability`**: Core reachability analysis (LMI solving, simulation workflows)
- **`plotting`**: Visualization utilities for error bounds, trajectories, and flowpipes
- **`planning`**: Trajectory generation (polynomial paths, waypoint planning)
- **`physics`**: Rigid body dynamics, angular acceleration, and SE(2,3) kinematics
- **`applications`**: Domain-specific modules for satellite, quadrotor, and rover systems

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Examples

See the [examples/](examples/) directory for:
- **General**: Linear and nonlinear Modelica model imports with structured workflow
- **Satellite**: HCW, TH-LTV, and SE(2,3) reachability analysis
- **Quadrotor**: Log-linearized error dynamics and flowpipe computation
- **Rover**: EMI disturbance and rollover analysis

## Supported Systems

### Linear Systems (Modelica)
- **Trajectory Planning**: Polynomial reference trajectory generation
- **Monte Carlo Simulation**: Disturbance simulation with bounded inputs
- **Reachability Analysis**: LMI-based ellipsoidal over-approximation

### Nonlinear Systems (Modelica)
- **Polytopic LMI**: Jacobian bounds over state ranges for nonlinear dynamics
- **Structured Workflow**: Systematic analysis via `SymbolicStateSpace`

### Domain-Specific Modules

**Satellite**:
- HCW (Hill-Clohessy-Wiltshire) linearized orbital dynamics
- TH-LTV (Tschauner-Hempel) time-varying linearized dynamics
- SE(2,3) log-linearized error dynamics

**Quadrotor**:
- Log-linearized error dynamics on SE(2,3)

**Rover**:
- EMI disturbance analysis
- Rollover reachability

## Testing

Run the test suite with pytest:

```bash
pytest tests/ -v
```

The test suite covers LMI solvers, dynamics classification, IR loading, and integration workflows.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

For major changes, please open an issue first to discuss proposed changes.

## Contact

For questions or support, please:
- Open an issue on GitHub
- Contact [Micah Condie](mailto:mkcondie01@gmail.com)
