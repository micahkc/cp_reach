"""
Test script for gravity feedforward compensation in log-linear dynamics.
Directly imports the module to bypass package initialization issues.
"""
import numpy as np
import sys
import importlib.util

# Direct import of the log_linear_dynamics module
spec = importlib.util.spec_from_file_location(
    "log_linear_dynamics",
    "/home/micah/Research/development/cp_reach/cp_reach/satellite/log_linear_dynamics.py"
)
log_linear_dynamics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_linear_dynamics)

# Extract what we need
SE23Spacecraft = log_linear_dynamics.SE23Spacecraft
LogLinearErrorDynamics = log_linear_dynamics.LogLinearErrorDynamics
SimulationComparison = log_linear_dynamics.SimulationComparison
geostationary_initial_conditions = log_linear_dynamics.geostationary_initial_conditions

# Setup
spacecraft = SE23Spacecraft()
t_span = np.linspace(0, 3600, 500)  # 1 hour, 500 points

# Initial conditions - geostationary orbit with small perturbations
state_ref_0 = geostationary_initial_conditions()
state_actual_0 = state_ref_0.copy()

# Add small perturbation to deputy
state_actual_0[0:3] += np.array([100.0, 50.0, 20.0])  # 100m position error
state_actual_0[3:6] += np.array([0.5, 0.2, 0.1])      # 0.5 m/s velocity error

# Zero controls (coast)
def controls_ref(t):
    return np.zeros(6)

print("=" * 80)
print("Testing Log-Linear Dynamics with Gravity Feedforward")
print("=" * 80)
print(f"\nSimulation time: {t_span[-1]} seconds ({t_span[-1]/3600:.2f} hours)")
print(f"Number of time steps: {len(t_span)}")
print(f"\nInitial position error: {np.linalg.norm(state_actual_0[0:3] - state_ref_0[0:3]):.2f} m")
print(f"Initial velocity error: {np.linalg.norm(state_actual_0[3:6] - state_ref_0[3:6]):.2f} m/s")

# Run simulation
print("\nRunning simulation...")
sim = SimulationComparison(spacecraft)
results = sim.simulate_both(
    t_span=t_span,
    state_ref_0=state_ref_0,
    state_actual_0=state_actual_0,
    controls_ref=controls_ref,
    rtol=1e-9,
    atol=1e-12
)

print("Simulation complete!")

# Analysis
t = results['t']
xi_nonlinear = results['xi_nonlinear']
xi_loglinear = results['xi_loglinear']
error_diff = results['error_diff']

print("\n" + "=" * 80)
print("Results Summary")
print("=" * 80)

# Final errors
print(f"\nFinal time: {t[-1]:.2f} s")
print(f"\nNonlinear error (final):")
print(f"  Position (log-coords): {np.linalg.norm(xi_nonlinear[-1, 0:3]):.6e}")
print(f"  Velocity (log-coords): {np.linalg.norm(xi_nonlinear[-1, 3:6]):.6e}")
print(f"  Attitude (log-coords): {np.linalg.norm(xi_nonlinear[-1, 6:9]):.6e}")

print(f"\nLog-linear error (final):")
print(f"  Position (log-coords): {np.linalg.norm(xi_loglinear[-1, 0:3]):.6e}")
print(f"  Velocity (log-coords): {np.linalg.norm(xi_loglinear[-1, 3:6]):.6e}")
print(f"  Attitude (log-coords): {np.linalg.norm(xi_loglinear[-1, 6:9]):.6e}")

# Error difference statistics
print(f"\nError difference ||ξ_nonlinear - ξ_loglinear||:")
print(f"  Initial: {error_diff[0]:.6e}")
print(f"  Final:   {error_diff[-1]:.6e}")
print(f"  Maximum: {np.max(error_diff):.6e}")
print(f"  Mean:    {np.mean(error_diff):.6e}")

# Check if gravity feedforward is working
if np.max(error_diff) < 1e-3:
    print("\n✓ SUCCESS: Error difference is small (< 1e-3)")
    print("  Gravity feedforward compensation is working correctly!")
else:
    print(f"\n✗ WARNING: Error difference is large: {np.max(error_diff):.6e}")
    print("  Gravity feedforward may need adjustment.")

# Print some intermediate values to track behavior
print(f"\nError difference at key times:")
n_samples = 10
indices = np.linspace(0, len(t)-1, n_samples, dtype=int)
for idx in indices:
    print(f"  t={t[idx]:7.1f}s: {error_diff[idx]:.6e}")

print("\n" + "=" * 80)
