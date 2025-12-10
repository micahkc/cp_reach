# CP_Reach Examples

This directory contains examples demonstrating the capabilities of cp_reach for reachability analysis of cyber-physical systems.

## Directory Structure

```
examples/
├── basic/              # Simple introductory examples
├── satellite/          # Satellite reachability examples
├── quadrotor/          # Quadrotor examples
├── rover/              # Rover examples
└── modelica/           # Modelica model integration examples
```

## Getting Started

### Basic Examples

**[basic/mass_spring_pd.ipynb](basic/mass_spring_pd.ipynb)**
- Simple mass-spring system with PD control
- Demonstrates the core workflow:
  1. Load model from Modelica
  2. Classify dynamics type
  3. Run nominal and Monte Carlo simulations
  4. Compute reachable set bounds via LMI
  5. Plot flowpipe visualization
- **Best starting point for new users**

### Modelica Integration

**[modelica/rumoca_flowpipe_demo.py](modelica/rumoca_flowpipe_demo.py)**
- Shows how to load models exported from RuMoCA (Modelica exporter)
- Demonstrates both SymPy and CasADi backends
- Example models: `MassSpringPD.mo`, `MassSpringPID.mo`

**[modelica/mock_rumoca_export.py](modelica/mock_rumoca_export.py)**
- Utility for creating mock RuMoCA exports for testing
- Useful for understanding the export format

### Satellite Examples

**[satellite/log_linear_tracking.ipynb](satellite/log_linear_tracking.ipynb)**
- Log-linearized satellite tracking on SE₂(3)
- Demonstrates gravity feedforward compensation
- Shows error dynamics propagation

**[satellite/outer_loop_lmi.ipynb](satellite/outer_loop_lmi.ipynb)**
- LMI-based reachable set computation for satellite outer loop
- Uses simplified log-linearization approach
- Shows ellipsoidal bound computation

### Quadrotor Examples

**[quadrotor/trajectory_tracking.ipynb](quadrotor/trajectory_tracking.ipynb)**
- Comprehensive quadrotor trajectory tracking example
- Demonstrates reachable set computation for quadrotor dynamics

**[quadrotor/quadrotor_inv_points.ipynb](quadrotor/quadrotor_inv_points.ipynb)**
- Invariant set computation for quadrotor

### Rover Examples

**[rover/simple_turn.ipynb](rover/simple_turn.ipynb)**
- Simple rover turning maneuver
- Demonstrates ground vehicle reachability analysis

## Running Examples

### Prerequisites

Install cp_reach with dependencies:

```bash
pip install -e .
```

For satellite examples, you may need additional dependencies:
```bash
pip install cyecca
```

### Running a Notebook

```bash
jupyter notebook examples/basic/mass_spring_pd.ipynb
```

### Running a Python Script

```bash
python examples/modelica/rumoca_flowpipe_demo.py
```

## Example Workflow

A typical cp_reach workflow:

1. **Define or Load Model**
   ```python
   from cp_reach import reach
   model = reach.casadi_load("model.mo")
   ```

2. **Simulate Nominal + Monte Carlo**
   ```python
   nom_res, mc_res = reach.simulate_dist(
       model, x0=x0, dist_bound=1.0, num_sims=100
   )
   ```

3. **Compute Reachable Set**
   ```python
   sol = reach.compute_reachable_set(
       model_sympy, method="lmi", dist_bound=1.0
   )
   ```

4. **Visualize Results**
   ```python
   reach.plot_flowpipe(nom_res, mc_res, error_fn=bounds_fn)
   ```

## Research Notebooks

Historical research notebooks have been moved to `/notebooks/archive/` for reference but are not maintained as examples.

## Contributing Examples

When adding new examples:
- Place in appropriate subdirectory
- Include clear docstrings/markdown explaining the example
- Keep examples focused and concise
- Update this README with a brief description
