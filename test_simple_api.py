"""Test the simplified API."""
import numpy as np
import sys
import importlib.util

spec = importlib.util.spec_from_file_location(
    "log_linear_dynamics",
    "/home/micah/Research/development/cp_reach/cp_reach/satellite/log_linear_dynamics.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Setup
spacecraft = mod.SE23Spacecraft()
t_span = np.linspace(0, 3600, 100)
state_0 = mod.geostationary_initial_conditions()
controls = lambda t: np.zeros(6)

print("Test 1: simulate_nonlinear")
state_traj = mod.simulate_nonlinear(spacecraft, t_span, state_0, controls)
print(f"  Output shape: {state_traj.shape}")
print(f"  Final position: {state_traj[-1, 0:3]}")
print()

print("Test 2: simulate_error")
state_actual_0 = state_0.copy()
state_actual_0[0:3] += [100, 50, 20]  # Add position error
xi_traj = mod.simulate_error(spacecraft, t_span, state_0, state_actual_0, controls)
print(f"  Output shape: {xi_traj.shape}")
print(f"  Initial error: {xi_traj[0]}")
print(f"  Final error: {xi_traj[-1]}")
print()
print("✓ Both functions work!")
