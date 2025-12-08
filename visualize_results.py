"""
Visualize the gravity feedforward compensation results.
"""
import numpy as np
import matplotlib.pyplot as plt
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
SimulationComparison = log_linear_dynamics.SimulationComparison
geostationary_initial_conditions = log_linear_dynamics.geostationary_initial_conditions

# Setup
spacecraft = SE23Spacecraft()
t_span = np.linspace(0, 3600, 500)  # 1 hour, 500 points

# Initial conditions
state_ref_0 = geostationary_initial_conditions()
state_actual_0 = state_ref_0.copy()
state_actual_0[0:3] += np.array([100.0, 50.0, 20.0])  # 100m position error
state_actual_0[3:6] += np.array([0.5, 0.2, 0.1])      # 0.5 m/s velocity error

# Zero controls
def controls_ref(t):
    return np.zeros(6)

# Run simulation
print("Running simulation...")
sim = SimulationComparison(spacecraft)
results = sim.simulate_both(
    t_span=t_span,
    state_ref_0=state_ref_0,
    state_actual_0=state_actual_0,
    controls_ref=controls_ref,
    rtol=1e-9,
    atol=1e-12
)
print("Done!")

# Extract results
t = results['t']
xi_nonlinear = results['xi_nonlinear']
xi_loglinear = results['xi_loglinear']
error_diff = results['error_diff']

# Create figure
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Position error
ax = axes[0]
ax.plot(t, xi_nonlinear[:, 0], 'b-', linewidth=2, label='Nonlinear (ξₚₓ)')
ax.plot(t, xi_loglinear[:, 0], 'r--', linewidth=1.5, label='Log-linear (ξₚₓ)')
ax.plot(t, xi_nonlinear[:, 1], 'g-', linewidth=2, alpha=0.7, label='Nonlinear (ξₚᵧ)')
ax.plot(t, xi_loglinear[:, 1], 'm--', linewidth=1.5, alpha=0.7, label='Log-linear (ξₚᵧ)')
ax.set_ylabel('Position Error ξₚ (m)', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Log-Linear Dynamics with Gravity Feedforward: Exact Match',
             fontsize=13, fontweight='bold')

# Velocity error
ax = axes[1]
ax.plot(t, xi_nonlinear[:, 3], 'b-', linewidth=2, label='Nonlinear (ξᵥₓ)')
ax.plot(t, xi_loglinear[:, 3], 'r--', linewidth=1.5, label='Log-linear (ξᵥₓ)')
ax.plot(t, xi_nonlinear[:, 4], 'g-', linewidth=2, alpha=0.7, label='Nonlinear (ξᵥᵧ)')
ax.plot(t, xi_loglinear[:, 4], 'm--', linewidth=1.5, alpha=0.7, label='Log-linear (ξᵥᵧ)')
ax.set_ylabel('Velocity Error ξᵥ (m/s)', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Attitude error
ax = axes[2]
ax.plot(t, xi_nonlinear[:, 6], 'b-', linewidth=2, label='Nonlinear (ξᴿₓ)')
ax.plot(t, xi_loglinear[:, 6], 'r--', linewidth=1.5, label='Log-linear (ξᴿₓ)')
ax.plot(t, xi_nonlinear[:, 7], 'g-', linewidth=2, alpha=0.7, label='Nonlinear (ξᴿᵧ)')
ax.plot(t, xi_loglinear[:, 7], 'm--', linewidth=1.5, alpha=0.7, label='Log-linear (ξᴿᵧ)')
ax.set_ylabel('Attitude Error ξᴿ (rad)', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Error difference (log scale)
ax = axes[3]
ax.semilogy(t, error_diff, 'k-', linewidth=2)
ax.set_ylabel('||ξ_nonlinear - ξ_loglinear||', fontsize=11)
ax.set_xlabel('Time (s)', fontsize=11)
ax.grid(True, alpha=0.3, which='both')
ax.set_title('Difference (Numerical Integration Error Only)', fontsize=10, style='italic')
ax.axhline(y=1e-3, color='r', linestyle='--', alpha=0.5, label='1e-3 threshold')
ax.legend()

plt.tight_layout()
plt.savefig('gravity_feedforward_results.png', dpi=300, bbox_inches='tight')
print("\nFigure saved to: gravity_feedforward_results.png")
print(f"Maximum error difference: {np.max(error_diff):.6e}")
print(f"Mean error difference: {np.mean(error_diff):.6e}")
