# Simplified Log-Linear Dynamics API

## Summary

The `log_linear_dynamics.py` file has been drastically simplified from **558 lines to 191 lines** (66% reduction).

## New Simple API

Two main functions:

### 1. `simulate_nonlinear(spacecraft, t_span, state_0, controls)`

Simulates nonlinear spacecraft dynamics.

**Returns:** `array (n, 10)` - State trajectory `[p, v, q]`

**Example:**
```python
spacecraft = SE23Spacecraft()
t_span = np.linspace(0, 1000, 500)
state_0 = geostationary_initial_conditions()
controls = lambda t: np.zeros(6)  # Coast

state_traj = simulate_nonlinear(spacecraft, t_span, state_0, controls)
```

### 2. `simulate_error(spacecraft, t_span, state_ref_0, state_actual_0, controls_ref, controls_actual=None)`

Simulates log-linear error dynamics in Lie algebra coordinates.

**Returns:** `array (n, 9)` - Error trajectory `[ξ_p, ξ_v, ξ_R]`

**Example:**
```python
state_ref_0 = geostationary_initial_conditions()
state_actual_0 = state_ref_0.copy()
state_actual_0[0:3] += [100, 50, 20]  # Add position error

xi_traj = simulate_error(spacecraft, t_span, state_ref_0, state_actual_0, controls)
```

## File Structure (191 lines)

```
log_linear_dynamics.py
├── SE23Spacecraft (51 lines)
│   ├── __init__()
│   ├── gravity(p)
│   └── dynamics(t, state, controls)
│
├── LogLinearErrorDynamics (60 lines)
│   ├── __init__()
│   └── dynamics(t, xi, controls_ref)
│
├── simulate_nonlinear() (25 lines)
├── simulate_error() (41 lines)
└── geostationary_initial_conditions() (5 lines)
```

## Key Simplifications

1. **Removed verbose helper methods** - Inlined SE23 conversions
2. **Removed separate matrix builders** - Build A matrix directly
3. **Removed comparison class** - Replaced with two standalone functions
4. **Removed excessive docstrings** - Kept only essential documentation
5. **Commented out gravity feedforward** - Per user request
6. **Simplified control flow** - Fewer intermediate variables

## Jupyter Notebook

Updated `examples/log_linear_spacecraft.ipynb` to use the new API:

```python
# Old API (complex)
sim = SimulationComparison(spacecraft)
results = sim.simulate_both(t_span, state_ref_0, state_actual_0, controls)

# New API (simple)
xi_traj = simulate_error(spacecraft, t_span, state_ref_0, state_actual_0, controls)
```

The notebook now has clear examples showing:
- Constant thrust scenario
- Coast scenario (zero thrust)
- Time-varying thrust scenario
- 3D trajectory visualization
- Error component plots

## Usage

```python
from cp_reach.satellite.log_linear_dynamics import (
    SE23Spacecraft,
    simulate_nonlinear,
    simulate_error,
    geostationary_initial_conditions
)

# Setup
spacecraft = SE23Spacecraft()
t_span = np.linspace(0, 3600, 500)
state_0 = geostationary_initial_conditions()
controls = lambda t: np.array([0.01, 0, 0, 0, 0, 0.001])  # Thrust + spin

# Simulate
state_traj = simulate_nonlinear(spacecraft, t_span, state_0, controls)
xi_traj = simulate_error(spacecraft, t_span, state_0, state_0, controls)

# Plot
import matplotlib.pyplot as plt
plt.plot(t_span, xi_traj[:, 0:3])  # Position error
plt.show()
```

## Testing

All functionality tested and working:
- ✓ Nonlinear simulation
- ✓ Error dynamics simulation
- ✓ Multiple control scenarios
- ✓ Jupyter notebook examples
