# General Reachability Analysis Examples

This directory demonstrates CP_Reach reachability analysis for both **linear** and **nonlinear** systems using the structured workflow:

```
Modelica Models -> Rumoca (compile) -> DAE JSON -> CP_Reach (analyze)
```

## Examples

### 1. Linear System: Mass-Spring-Damper (`structured_workflow.ipynb`)

A closed-loop mass-spring-damper with PD control and feedforward. Error dynamics are **linear**, enabling standard LMI-based reachability analysis.

**Model:** `models/closed_loop.mo`
- Plant: mass-spring-damper (`m=1.0`, `k=1.0`, `c=0.2`)
- Controller: PD + feedforward
- Disturbance: bounded force `d` with `|d| <= 1.0`

**Workflow:**
1. Compile with Rumoca: `rumoca --json -m ClosedLoop models/closed_loop.mo`
2. Load IR and extract symbolic state space
3. Verify error dynamics are linear (constant Jacobian)
4. Solve LMI for invariant ellipsoid
5. Validate with Monte Carlo simulation (200 square-wave disturbances)

**Output:** `output/`
- `closedloop.json` - Compiled DAE IR
- `state_flowpipe.png` - State trajectories with flowpipe bounds
- `error_trajectories.png` - Error trajectories with reachable set bounds
- `error_ellipsoid.png` - 2D invariant ellipsoid in error space

**Results:** Bounds `[e, ev] = [0.843, 1.430]` for unit disturbance.

---

### 2. Nonlinear System: Pendulum with PID (`structured_nonlinear.ipynb`)

A controlled pendulum with gravity nonlinearity (`sin(theta)`). Error dynamics are **nonlinear**, requiring time-varying polytopic LMI bounds.

**Model:** `models/pendulum_closed_loop.mo`
- Plant: pendulum with gravity (`m=1.0`, `l=1.0`, `g=9.81`, `c=0.1`)
- Controller: PID (`Kp=5.0`, `Ki=1.0`, `Kd=0.5`) + feedforward
- Disturbance: bounded torque `d` with `|d| <= 0.1 Nm`

**Workflow:**
1. Compile with Rumoca: `rumoca --json -m PendulumClosedLoop models/pendulum_closed_loop.mo`
2. Load IR and verify error dynamics are nonlinear (Jacobian contains `cos(theta)`)
3. Generate reference trajectory: `0 -> pi/4 -> 0` over 4 seconds
4. Sample trajectory at 20 points; create polytopic Jacobian bounds at each point
5. Solve time-varying polytopic LMI for polynomial Lyapunov function `M(t)`
6. Validate with continuous-time certification (Algorithm 2)
7. Validate with Monte Carlo simulation (90 square-wave disturbances)

**Output:** `output_nonlinear/`
- `pendulumclosedloop.json` - Compiled DAE IR
- `state_flowpipe.png` - State trajectories with flowpipe bounds
- `error_trajectories.png` - Error trajectories with time-varying bounds
- `error_phase.png` - 2D phase portrait in error space
- `error_ellipse_times.png` - Time-varying invariant ellipse at multiple times
- `metric_space.png` - Trajectories in time-varying metric space (unit ball = invariant set)

**Results:** Bounds `[theta, omega, xi] = [0.045, 0.161, 0.161]` rad for 0.1 Nm disturbance.

---

## Files

### Modelica Models (`models/`)

| File | Description |
|------|-------------|
| `plant.mo` | Mass-spring-damper dynamics |
| `controller.mo` | PD controller with feedforward |
| `closed_loop.mo` | Linear closed-loop system |
| `closed_loop_composed.mo` | Alternative composition structure |
| `pendulum_closed_loop.mo` | Nonlinear pendulum with PID control |

### Configuration Files

| File | Description |
|------|-------------|
| `uncertainty.yaml` | Disturbance bounds, parameter uncertainty |
| `reach_query.yaml` | Analysis query specification (flowpipe, trajectory, alpha search) |

### Scripts

| File | Description |
|------|-------------|
| `run_analysis.py` | CLI-based end-to-end example |
| `structured_workflow.ipynb` | Linear system analysis notebook |
| `structured_nonlinear.ipynb` | Nonlinear system analysis notebook |

---

## Quick Start

### Using the CLI

```bash
# 1. Compile model with Rumoca
rumoca --json -m ClosedLoop models/closed_loop.mo > output/closed_loop.json

# 2. Run analysis
python -m cp_reach analyze \
    --ir output/closed_loop.json \
    --uncertainty uncertainty.yaml \
    --query reach_query.yaml \
    --output output/

# Or use the Python script
python run_analysis.py
```

### Using the Notebooks

1. Open `structured_workflow.ipynb` for linear systems
2. Open `structured_nonlinear.ipynb` for nonlinear systems
3. Run all cells to reproduce the analysis

---

## YAML Configuration Reference

### uncertainty.yaml

Specifies disturbance bounds for reachability analysis.

```yaml
disturbances:
  <input_name>:
    type: bounded          # "bounded" or "gaussian"
    bound: 1.0             # Maximum disturbance magnitude
    norm: inf              # Norm type: "inf", "2", or "1"

  # Example: Gaussian disturbance (for stochastic analysis)
  # w:
  #   type: gaussian
  #   covariance: [[0.1, 0], [0, 0.1]]
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `disturbances.<name>.type` | string | `"bounded"` for worst-case, `"gaussian"` for stochastic |
| `disturbances.<name>.bound` | float | Maximum magnitude `\|d(t)\| <= bound` |
| `disturbances.<name>.norm` | string | Which norm: `"inf"` (default), `"2"`, or `"1"` |
| `disturbances.<name>.covariance` | matrix | Covariance matrix (for gaussian type) |

> **Note:** Parameter uncertainty and initial conditions are parsed but not yet used in the LMI solver.

---

### reach_query.yaml

Specifies what reachability analysis to perform.

```yaml
query:
  type: flowpipe           # Analysis type
  dynamics: error          # Which dynamics to analyze
  outputs: [e, ev]         # Output variables for bounds

  # Disturbance inputs to consider
  dist_inputs: [d]

  # Reference trajectory (for flowpipe analysis)
  trajectory:
    source: polynomial     # "polynomial" or "file"

    # Polynomial planner options
    waypoints:
      - [0.0, 0.0]
      - [1.0, 0.2]
      - [2.0, 0.0]
    velocities:
      - [0.0, 0.0]
      - null               # null = free (optimized)
      - [0.0, 0.0]
    min_deriv: 4           # Minimize: 3=jerk, 4=snap
    poly_deg: 5            # Polynomial degree per segment
    duration: auto         # "auto" or fixed float
    n_points: 100          # Output samples

    # File source options (alternative to polynomial)
    # source: file
    # path: "trajectory.csv"

  # Alpha (decay rate) search for LMI
  alpha_search:
    min: 1.0e-4
    max: 10.0
    num_points: 40
    scale: log             # "log" or "linear"

  # Output options
  output_format:
    certificate: true      # Save Lyapunov P matrix
    bounds: true           # Save state bounds
    plots: true            # Generate plots
    json: true             # Save results.json
```

**Query Types:**
| Type | Description |
|------|-------------|
| `flowpipe` | Reachable set tube along a trajectory |
| `invariant_set` | Time-invariant reachable set (no trajectory) |
| `bounded_reachable_set` | Finite-horizon reachable set |

**Dynamics Options:**
| Option | Description |
|--------|-------------|
| `error` | Error dynamics: `e = x - x_ref` |
| `full_state` | Full state dynamics |
| `estimation_error` | Observer error dynamics |

**Trajectory Sources:**
| Source | Description |
|--------|-------------|
| `polynomial` | Generate minimum-derivative trajectory through waypoints |
| `file` | Load from CSV (columns: `t, x1, x2, ..., u1, u2, ...`) or NPY |
