# CP_Reach Architecture

This document describes the internal architecture of cp_reach, a library for computing reachable sets and performing reachability analysis on cyber-physical systems.

## Overview

CP_Reach uses Linear Matrix Inequalities (LMIs) and Lyapunov theory to compute ellipsoidal over-approximations of reachable sets. The library supports both linear systems (imported from Modelica via Rumoca) and nonlinear systems (via polytopic LMI methods).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  ir_load()      │  │  sympy_load()   │  │  analyze()                  │  │
│  │  (IR JSON)      │  │  (Modelica)     │  │  (IR + YAML config)         │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
└───────────┼────────────────────┼─────────────────────────┼──────────────────┘
            │                    │                         │
            ▼                    ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Pipeline                                      │
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐  │
│   │    IR    │───▶│   dynamics   │───▶│ reachability│───▶│   plotting   │  │
│   │  loader  │    │  state_space │    │     LMI     │    │   flowpipe   │  │
│   └──────────┘    └──────────────┘    └─────────────┘    └──────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

CP_Reach is organized into 8 main modules:

```
cp_reach/
├── __init__.py          # Package init with lazy loading
├── cli.py               # Command-line interface
│
├── config/              # Configuration management
│   ├── uncertainty.py   # Disturbance bound specifications
│   └── query.py         # Analysis query configuration
│
├── dynamics/            # State space representations
│   ├── state_space.py   # CasADi/SymPy backends
│   └── classification.py# Dynamics classification
│
├── ir/                  # Intermediate representation (Modelica)
│   ├── loader.py        # JSON IR loading from Rumoca
│   ├── ast_parser.py    # AST parsing for DAE systems
│   └── state_space.py   # IR to state space conversion
│
├── physics/             # Rigid body dynamics
│   ├── rigid_body.py    # Rigid body models
│   ├── angular_acceleration.py
│   └── coupled_dynamics.py
│
├── planning/            # Trajectory generation
│   ├── polynomial.py    # Polynomial trajectory planning
│   └── trajectory.py    # Trajectory data structure
│
├── plotting/            # Visualization
│   └── plotting.py      # Flowpipe and trajectory plots
│
├── reachability/        # Core analysis (main module)
│   ├── lmi.py           # LMI solvers
│   ├── polytopic.py     # Time-varying polytopic LMIs
│   ├── certification.py # Continuous-time verification
│   ├── ellipsoids.py    # Ellipsoid operations
│   └── workflows.py     # High-level analysis workflows
│
└── development/         # Domain-specific applications
    └── applications/
        ├── satellite/   # HCW, TH-LTV, SE(2,3) dynamics
        ├── quadrotor/   # Log-linearized SE(2,3)
        └── rover/       # Ground vehicle analysis
```

## Core Modules

### 1. `ir` - Intermediate Representation

Loads Modelica models compiled to JSON by [Rumoca](https://rumoca.dev). This provides a cyecca-free path for simpler deployments.

**Key classes:**
- `DaeIR`: Represents a DAE system from Rumoca JSON
- `Component`: Individual components within the DAE

**Data flow:**
```
Modelica (.mo) → Rumoca → JSON IR → DaeIR → SymbolicStateSpace
```

### 2. `dynamics` - State Space

Provides symbolic and numeric state space representations with automatic Jacobian computation.

**Key classes:**
- `SymbolicStateSpace`: SymPy-based state space with A, B, C, D, E, F matrices
- `CasadiStateSpace`: CasADi-based numeric state space

**Matrix semantics:**
- `A, B`: State dynamics (ẋ = Ax + Bu)
- `C, D`: Output equation (y = Cx + Du)
- `E, F`: Error dynamics for tracking (ė = Ee + Fd)

### 3. `reachability` - Core Analysis

The main computational module. Solves LMI problems to compute guaranteed bounds on system behavior.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `solve_disturbance_LMI()` | Main LMI solver for disturbance rejection |
| `solve_bounded_disturbance_output_LMI()` | Output-focused LMI with gain bounds |
| `polytopic_jacobians()` | Compute Jacobians at polytope vertices |
| `solve_time_varying_polytopic_lmi()` | Time-varying bounds for nonlinear systems |
| `compute_reachable_set()` | High-level wrapper for LMI analysis |
| `simulate_dist()` | Monte Carlo validation with disturbances |

**LMI formulation:**

For a stable system ẋ = Ax + Bw with bounded disturbance ||w|| ≤ w_max, we find P > 0 and μ such that:

```
[A'P + PA + αP,  PB  ]
[    B'P,      -αμI  ] ≤ 0
```

This certifies that V(x) = x'Px remains bounded, yielding ellipsoidal reachable set bounds.

### 4. `config` - Configuration

YAML-based configuration for structured workflows.

**Files:**
- `uncertainty.yaml`: Disturbance bounds, parameter ranges
- `reach_query.yaml`: Analysis type, outputs, solver settings

### 5. `planning` - Trajectory Generation

Polynomial trajectory planning for reference tracking scenarios.

**Key class:**
- `Trajectory`: Container for time, state, and input trajectories

### 6. `plotting` - Visualization

Flowpipe visualization showing nominal trajectory, Monte Carlo samples, and computed bounds.

**Key functions:**
- `plot_flowpipe()`: Main visualization with error bounds
- `plot_grouped()`: Group states on shared axes

### 7. `physics` - Rigid Body Dynamics

Supporting module for domain-specific applications requiring rigid body mechanics.

### 8. `development/applications` - Domain-Specific

Pre-built analysis modules for specific cyber-physical systems:

- **Satellite**: HCW (circular orbit), TH-LTV (elliptical orbit), SE(2,3) Lie group dynamics
- **Quadrotor**: Log-linearized attitude dynamics on SE(2,3)
- **Rover**: Ground vehicle with EMI disturbance analysis

## Data Flow

### Linear Systems (Modelica)

```
1. Load Model
   ir_load("model.json") → ModelicaIRModel
                           ├── .ir (DaeIR)
                           ├── .symbolic (SymbolicStateSpace)
                           ├── .states, .inputs, .parameters

2. Extract Matrices
   model.symbolic.E() → Error dynamics matrix
   model.symbolic.F() → Disturbance input matrix

3. Solve LMI
   solve_disturbance_LMI(A_list=[E], B=F, w_max=bound)
   → {P, mu, alpha, bounds_upper, bounds_lower}

4. Validate & Visualize
   simulate_dist() → Monte Carlo trajectories
   plot_flowpipe() → Visualization with bounds
```

### Nonlinear Systems (Polytopic)

```
1. Define Polytope Vertices
   polytopic_jacobians(model, x_bounds, u_bounds)
   → [A_1, A_2, ..., A_k] (Jacobians at vertices)

2. Solve Common Lyapunov LMI
   solve_time_varying_polytopic_lmi(A_list, B, ...)
   → {P(t), mu(t), bounds(t)}

3. Evaluate Time-Varying Bounds
   eval_polynomial_metric(coeffs, t) → P(t)
   compute_state_bounds(P, mu, dist_bound) → per-state bounds
```

## Key Design Decisions

### Lazy Loading
The package uses `__getattr__` for lazy loading of submodules to avoid circular imports and reduce startup time.

### Dual Backend Support
Both SymPy (symbolic) and CasADi (numeric) backends are supported:
- SymPy: Exact symbolic Jacobians, readable expressions
- CasADi: Fast numeric evaluation, automatic differentiation

### Polytopic Uncertainty
Nonlinear systems are handled via polytopic over-approximation. The system Jacobian is evaluated at vertices of a bounding box, and a common Lyapunov function is found that works for all vertices.

### Separation of Concerns
- **IR module**: Handles Modelica parsing (can be replaced with other frontends)
- **Dynamics module**: Pure state space operations (backend-agnostic)
- **Reachability module**: Mathematical analysis (input is matrices, not models)

## Extension Points

### Adding a New Application Domain

1. Create `development/applications/your_domain/`
2. Implement domain-specific dynamics in `dynamics.py`
3. Add invariant set computation in `invariant.py`
4. Optionally add visualization in `plotting.py`

### Adding a New LMI Formulation

1. Add the solver function to `reachability/lmi.py`
2. Follow the pattern of `_solve_multi_channel_LMI()`
3. Export via `reachability/__init__.py`

### Adding a New Model Frontend

1. Create a loader in `ir/` that produces `DaeIR` or `SymbolicStateSpace`
2. Wrap in a model class similar to `ModelicaIRModel`
3. The reachability module works with any `SymbolicStateSpace`

## Dependencies

```
numpy          - Numerical operations
casadi         - Symbolic/numeric computation, autodiff
cvxpy          - Convex optimization (LMI solving)
sympy          - Symbolic mathematics
scipy          - Scientific computing (optimization)
matplotlib     - Visualization
control        - Control theory utilities
cyecca         - Lie group operations (optional, for satellite/quadrotor)
```

## Performance Considerations

- LMI solving scales with state dimension O(n³) and number of polytope vertices
- Monte Carlo validation is embarrassingly parallel (not yet parallelized)
- CasADi backend is significantly faster than SymPy for large systems
- Lazy loading reduces import time from ~2s to ~0.1s
