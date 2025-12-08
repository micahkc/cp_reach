"""
Test gravity feedforward with a rotated reference frame.
"""
import numpy as np
import sys
import importlib.util

# Direct import
spec = importlib.util.spec_from_file_location(
    "log_linear_dynamics",
    "/home/micah/Research/development/cp_reach/cp_reach/satellite/log_linear_dynamics.py"
)
log_linear_dynamics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_linear_dynamics)

SE23Spacecraft = log_linear_dynamics.SE23Spacecraft
SimulationComparison = log_linear_dynamics.SimulationComparison
geostationary_initial_conditions = log_linear_dynamics.geostationary_initial_conditions

# Setup
spacecraft = SE23Spacecraft()
t_span = np.linspace(0, 3600, 500)  # 1 hour

# Initial conditions with ROTATION
state_ref_0 = geostationary_initial_conditions()
# Add 30° rotation about z-axis to reference
angle = np.pi/6  # 30 degrees
state_ref_0[6:10] = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])

state_actual_0 = state_ref_0.copy()
state_actual_0[0:3] += np.array([100.0, 50.0, 20.0])  # 100m position error
state_actual_0[3:6] += np.array([0.5, 0.2, 0.1])      # 0.5 m/s velocity error

# Zero controls
def controls_ref(t):
    return np.zeros(6)

print("=" * 80)
print("Testing with ROTATED Reference Frame (30° about z-axis)")
print("=" * 80)
print(f"\nReference quaternion: {state_ref_0[6:10]}")
print(f"Rotation angle: {angle*180/np.pi:.1f}°")
print(f"\nSimulation time: {t_span[-1]} seconds")
print(f"Initial position error: {np.linalg.norm(state_actual_0[0:3] - state_ref_0[0:3]):.2f} m")

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

# Analysis
error_diff = results['error_diff']

print("\n" + "=" * 80)
print("Results with Rotated Reference")
print("=" * 80)
print(f"\nError difference ||ξ_nonlinear - ξ_loglinear||:")
print(f"  Initial: {error_diff[0]:.6e}")
print(f"  Final:   {error_diff[-1]:.6e}")
print(f"  Maximum: {np.max(error_diff):.6e}")
print(f"  Mean:    {np.mean(error_diff):.6e}")

if np.max(error_diff) < 1e-3:
    print("\n✓ SUCCESS: Gravity feedforward works with rotated reference frames!")
    print("  The transformation p_actual = p_ref + R_ref @ ξ_p is correct.")
else:
    print(f"\n✗ FAILURE: Error difference too large: {np.max(error_diff):.6e}")

print("\n" + "=" * 80)
