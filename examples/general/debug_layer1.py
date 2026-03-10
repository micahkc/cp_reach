"""
Debug script: try different configurations for Layer 1 polytopic LMI
until it converges.

Sweeps over:
  - polynomial_degree (complexity of M(t))
  - NUM_SAMPLES (number of time points)
  - OMEGA_ERROR_BOUND (polytopic box size)
  - alpha_bounds range
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# 1. Build the model and reference trajectory (same as notebook)
# ---------------------------------------------------------------------------
import rumoca
from cp_reach.ir import DaeIR, ir_to_symbolic_statespace
from cp_reach.reachability import (
    load_lie_spec,
    partition_states,
    classify_remaining_states,
    euler_equation_jacobians,
    solve_time_varying_polytopic_lmi,
    compute_state_bounds,
    quadrotor_flatness,
    compute_omega_from_R,
)
from cp_reach.planning import plan_minimum_derivative_trajectory

MODELICA_FILE = "models/quadrotor_closed_loop.mo"
MODEL_NAME = "QuadrotorClosedLoop"
D_GYRO = 0.1

print("Loading model...")
with open(MODELICA_FILE) as f:
    source = f.read()
json_str = rumoca.compile(source, MODEL_NAME)
ir = DaeIR.from_json_str(json_str)
ss = ir_to_symbolic_statespace(ir)
params = ir.get_param_defaults()

lie_spec = load_lie_spec("models/quadrotor_closed_loop_lie.json")
partition = partition_states(ss.get_state_names(), lie_spec)
classification = classify_remaining_states(ss, partition, params)

mass = params['mass']
g = params['g']
Ixx, Iyy, Izz = params['Ixx'], params['Iyy'], params['Izz']

controller_gains = {
    'Kw_x': params['Kw_x'], 'Kw_y': params['Kw_y'], 'Kw_z': params['Kw_z'],
    'Kr_x': params['Kr_x'], 'Kr_y': params['Kr_y'], 'Kr_z': params['Kr_z'],
}

# --- Trajectory ---
print("Planning trajectory...")
waypoint_xyz = np.array([
    [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [3.0, 1.0, 1.0],
    [2.0, 4.0, 1.0], [-1.0, 3.0, 1.0], [0.0, 0.0, 1.0],
])
waypoint_ned = waypoint_xyz.copy()
waypoint_ned[:, 2] *= -1
T_segments = np.array([1.5, 3.0, 3.0, 3.0, 3.0])

trajectories = {}
for d, axis in enumerate(['x', 'y', 'z']):
    bc = np.zeros((3, len(waypoint_ned), 1))
    bc[0, :, 0] = waypoint_ned[:, d]
    traj = plan_minimum_derivative_trajectory(
        bc=bc, min_deriv=3, poly_deg=5, T_guess=T_segments, bc_deriv=3,
    )
    acc = traj.metadata['derivatives']['acc'][:, 0]
    traj.metadata['derivatives']['jerk'] = np.gradient(acc, traj.t).reshape(-1, 1)
    trajectories[axis] = {
        't': traj.t, 'pos': traj.x[:, 0],
        'vel': traj.metadata['derivatives']['vel'][:, 0],
        'acc': traj.metadata['derivatives']['acc'][:, 0],
        'jerk': traj.metadata['derivatives']['jerk'][:, 0],
    }

t_eval = trajectories['x']['t']
p_ref = np.column_stack([trajectories[a]['pos'] for a in ['x', 'y', 'z']])
v_ref = np.column_stack([trajectories[a]['vel'] for a in ['x', 'y', 'z']])
a_ref = np.column_stack([trajectories[a]['acc'] for a in ['x', 'y', 'z']])
jerk_ref = np.column_stack([trajectories[a]['jerk'] for a in ['x', 'y', 'z']])

flat_ref = quadrotor_flatness(p_ref, v_ref, a_ref, jerk_ref, mass, g)
omega_ref = compute_omega_from_R(flat_ref['R_ref'], t_eval)

print(f"Trajectory: {len(t_eval)} points, T={t_eval[-1]:.1f}s")
print(f"Max omega_ref: {np.max(np.abs(omega_ref), axis=0)}")

# ---------------------------------------------------------------------------
# 2. Sweep configurations
# ---------------------------------------------------------------------------
configs = [
    # (num_samples, poly_degree, omega_bound, alpha_bounds, label)
    (5,  5,  0.5,  (0.01, 1.0),  "5 pts, deg 5, box 0.5"),
    (5,  3,  0.5,  (0.01, 1.0),  "5 pts, deg 3, box 0.5"),
    (5,  2,  0.5,  (0.01, 1.0),  "5 pts, deg 2, box 0.5"),
    (5,  1,  0.5,  (0.01, 1.0),  "5 pts, deg 1 (const M), box 0.5"),
    (10, 3,  0.5,  (0.01, 1.0),  "10 pts, deg 3, box 0.5"),
    (10, 5,  0.5,  (0.01, 1.0),  "10 pts, deg 5, box 0.5"),
    (5,  3,  0.2,  (0.01, 1.0),  "5 pts, deg 3, box 0.2"),
    (5,  3,  0.1,  (0.01, 1.0),  "5 pts, deg 3, box 0.1"),
    (5,  3,  0.05, (0.01, 1.0),  "5 pts, deg 3, box 0.05"),
    (5,  3,  0.5,  (0.1, 5.0),   "5 pts, deg 3, box 0.5, wide alpha"),
    (5,  3,  0.5,  (0.5, 10.0),  "5 pts, deg 3, box 0.5, high alpha"),
    (3,  2,  0.5,  (0.01, 1.0),  "3 pts, deg 2, box 0.5"),
    (3,  1,  0.5,  (0.01, 1.0),  "3 pts, deg 1 (const M), box 0.5"),
    (20, 3,  0.5,  (0.01, 1.0),  "20 pts, deg 3, box 0.5"),
    (20, 20, 0.5,  (0.01, 1.0),  "20 pts, deg 20, box 0.5 (notebook default)"),
]

B_omega = np.eye(3)

print("\n" + "=" * 90)
print(f"{'Config':<45} {'Status':<14} {'alpha':>8} {'mu_phys':>12} {'bounds':>25} {'Time':>6}")
print("=" * 90)

for num_samples, poly_deg, omega_bound, alpha_bounds, label in configs:
    idx = np.round(np.linspace(0, len(t_eval) - 1, num_samples)).astype(int)
    times_sample = t_eval[idx]
    omega_refs_sample = omega_ref[idx]

    J_vertices = euler_equation_jacobians(
        omega_refs_sample, omega_bound, Ixx, Iyy, Izz, controller_gains,
    )

    n_constraints = sum(len(v) for v in J_vertices) * 2 + len(J_vertices) * 2
    t0 = time.time()
    result = solve_time_varying_polytopic_lmi(
        J_vertices, B_omega, times_sample.tolist(),
        polynomial_degree=poly_deg,
        alpha_bounds=alpha_bounds,
        verbose=False,
    )
    elapsed = time.time() - t0

    status = result['status'] or 'None'
    alpha = result['alpha']
    mu_phys = result['mu_physical']

    if result['M_coefs'] is not None:
        bounds_result = compute_state_bounds(
            result['M_coefs'], mu_phys, D_GYRO, times_sample.tolist(),
        )
        bounds = bounds_result['bounds_max']
        bounds_str = f"[{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}]"
        valid = "OK" if np.max(bounds) <= omega_bound else "VIOLATED"
        bounds_str += f" {valid}"
    else:
        bounds_str = "---"
        alpha = float('nan')
        mu_phys = float('nan')

    print(f"{label:<45} {status:<14} {alpha:>8.4f} {mu_phys:>12.4e} {bounds_str:>25} {elapsed:>5.1f}s")

print("=" * 90)
