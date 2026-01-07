# CP_Reach Examples

This directory contains examples demonstrating the capabilities of cp_reach for reachability analysis of cyber-physical systems.

## Directory Structure

```
examples/
├── general/            # Structured workflow examples (linear & nonlinear)
├── satellite/          # Satellite rendezvous reachability
├── quadrotor/          # Quadrotor invariant set computation
└── rover/              # Rover rollover analysis
```

## Getting Started

### General Examples (Recommended Starting Point)

The `general/` folder demonstrates the end-to-end structured workflow for both linear and nonlinear systems:

**Linear System: [general/structured_workflow.ipynb](general/structured_workflow.ipynb)**
- Mass-spring-damper with PD control
- Standard LMI-based reachability analysis
- Output: flowpipe bounds, invariant ellipsoid

**Nonlinear System: [general/structured_nonlinear.ipynb](general/structured_nonlinear.ipynb)**
- Pendulum with PID control and gravity nonlinearity
- Time-varying polytopic LMI bounds
- Continuous-time certification (Algorithm 2)

**CLI Script: [general/run_analysis.py](general/run_analysis.py)**
- End-to-end analysis via command line
- Compiles Modelica → runs LMI → saves results

**Configuration:**
- `general/uncertainty.yaml` - Disturbance bounds
- `general/reach_query.yaml` - Analysis parameters

See [general/README.md](general/README.md) for detailed documentation.

### Application Examples

**[satellite/satellite_error_bounds.ipynb](satellite/satellite_error_bounds.ipynb)**
- Satellite rendezvous on SE(2,3) Lie group
- Log-linear error dynamics
- Invariant set computation for orbital maneuvers

**[quadrotor/quadrotor_flowpipe.ipynb](quadrotor/quadrotor_flowpipe.ipynb)**
- Quadrotor SE(2,3) kinematics
- Nested invariant sets (angular dynamics + full state)
- Ellipsoidal reachable set bounds

**[rover/rover_plots.ipynb](rover/rover_plots.ipynb)**
- Ground vehicle rollover analysis
- Terrain angle vs velocity bounds
- Safety envelope visualization

## Running Examples

### Prerequisites

Install cp_reach:
```bash
pip install -e .
```

For satellite/quadrotor examples with Lie group dynamics:
```bash
pip install cyecca
```

### Running Notebooks

```bash
jupyter notebook examples/general/structured_workflow.ipynb
```

### Running CLI

```bash
python examples/general/run_analysis.py
```

Or using the module:
```bash
python -m cp_reach analyze \
    --ir examples/general/output/closedloop.json \
    --uncertainty examples/general/uncertainty.yaml \
    --query examples/general/reach_query.yaml
```

## Workflow Overview

The recommended cp_reach workflow:

1. **Author models** - Write plant/controller as Modelica `.mo` files
2. **Compile** - `rumoca --json -m ModelName model.mo > model.json`
3. **Specify uncertainty** - Define disturbance bounds in `uncertainty.yaml`
4. **Specify query** - Define analysis parameters in `reach_query.yaml`
5. **Analyze** - Run via notebook or CLI
6. **Validate** - Monte Carlo simulation to verify bounds
