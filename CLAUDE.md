# CLAUDE.md - CP_Reach Development Guide

## Project Overview

cp_reach is a Python library for computing reachable sets on cyber-physical systems using LMIs, Lyapunov theory, and Lie group structures. It provides formal guarantees on state deviations under bounded disturbances.

## Quick Reference

- **Language**: Python 3.8+
- **Version**: 0.2.0
- **License**: Apache-2.0
- **Author**: Micah Condie

## Build & Install

```bash
pip install -e .              # Basic install
pip install -e ".[dev]"       # With dev tools (pytest, black, etc.)
pip install -e ".[satellite]" # With cyecca for Lie group applications
pip install -e ".[all]"       # Everything
```

## Testing

```bash
pytest tests/ -v                          # Run all tests
pytest tests/ -v --cov=cp_reach           # With coverage
pytest tests/test_lmi.py -v               # Just LMI tests
pytest -m "not slow" tests/ -v            # Skip slow tests
```

Test files: `test_lmi.py` (LMI solvers), `test_ir_loader.py` (IR parsing), `test_integration.py` (end-to-end), `test_dynamics.py` (classification).

## Linting & Formatting

```bash
black --line-length 100 cp_reach/        # Format code
isort --profile black cp_reach/          # Sort imports
flake8 cp_reach/                         # Lint
mypy cp_reach/                           # Type check
```

Configuration is in `pyproject.toml`. Line length: 100. Target Python: 3.8-3.11.

## Project Structure

```
cp_reach/
├── ir/              # Modelica JSON IR loading (DaeIR, ir_to_symbolic_statespace)
├── dynamics/        # State-space representations (SymPy/CasADi backends)
├── reachability/    # Core analysis: LMI solvers, polytopic, cascaded Lie group
├── config/          # YAML schema for uncertainty.yaml and reach_query.yaml
├── planning/        # Polynomial trajectory generation
├── plotting/        # Flowpipe and trajectory visualization
├── physics/         # Rigid body dynamics utilities
└── development/     # Domain-specific apps (satellite, quadrotor, rover)
```

## Key Modules & Entry Points

### IR Loading
- `cp_reach.ir.DaeIR` — DAE intermediate representation from Rumoca JSON
- `cp_reach.ir.ir_to_symbolic_statespace()` — Convert IR to SymbolicStateSpace

### Reachability Analysis
- `solve_disturbance_LMI(A_list, B, w_max)` — Core LMI solver for linear systems
- `polytopic_jacobians()` — Jacobian polytope vertices for nonlinear systems
- `solve_time_varying_polytopic_lmi()` — Time-varying bounds for nonlinear systems
- `solve_cascaded_lmi()` — Two-layer cascaded LMI for SE_2(3) Lie group systems

### Workflows
- `ir_load(path)` — Load Modelica JSON to ModelicaIRModel
- `analyze()` — Full pipeline: load + solve + plot
- `simulate_dist()` — Monte Carlo disturbance simulation
- `plot_flowpipe()` — Visualization with bounds

## Architecture Patterns

- **Lazy loading**: Submodules loaded via `__getattr__` to avoid circular imports
- **Dual backends**: SymPy for symbolic analysis, CasADi for fast numeric evaluation
- **Polytopic uncertainty**: Nonlinear systems handled via Jacobian bounds at polytope vertices
- **Cascaded LMI**: Lie group systems split into rotational (polytopic) + kinematic (exact) layers

## Data Flow

```
Modelica (.mo) → rumoca → JSON IR → DaeIR → SymbolicStateSpace
                                                    ↓
                              uncertainty.yaml + reach_query.yaml
                                                    ↓
                                    LMI Solver (cvxpy) → P matrix + bounds
                                                    ↓
                              simulate_dist() → Monte Carlo validation
                                                    ↓
                                    plot_flowpipe() → visualization
```

## Dependencies

Core: numpy, cvxpy, casadi, sympy, scipy, matplotlib, control, pandas
Optional: cyecca (Lie groups), rumoca (Modelica compiler)

## CI/CD

GitHub Actions (`.github/workflows/test.yml`): runs on Ubuntu/macOS/Windows with Python 3.8-3.11. Coverage uploaded to Codecov on Ubuntu + Python 3.11.

## Examples

Located in `examples/`:
- `general/structured_workflow.ipynb` — Linear mass-spring-damper (start here)
- `general/structured_nonlinear.ipynb` — Nonlinear pendulum with polytopic LMI
- `general/structured_quadrotor.ipynb` — Quadrotor on SE_2(3) with cascaded LMI
- `satellite/` — HCW, TH-LTV, SE(2,3) orbital dynamics
- `quadrotor/` — Log-linearized SE(2,3) attitude dynamics
- `rover/` — EMI disturbance and rollover analysis

## Conventions

- All reachability functions accept raw numpy matrices, not model objects (separation of concerns)
- YAML configs define disturbance bounds (`uncertainty.yaml`) and analysis parameters (`reach_query.yaml`)
- New LMI formulations go in `reachability/lmi.py`; new Lie group methods go in `reachability/cascaded.py`
- Domain-specific application code lives under `development/applications/<domain>/`
