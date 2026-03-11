"""
Quadrotor path planning and closed-loop simulation.

Plans a minimum-jerk trajectory through waypoints at constant altitude,
then simulates the closed-loop Modelica quadrotor model with and without
gyro bias disturbances.

Usage:
    python quadrotor_sim.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor
import sympy as sp
import rumoca

from cp_reach.ir import DaeIR, ir_to_symbolic_statespace
from cp_reach.planning import plan_minimum_derivative_trajectory
from cp_reach.reachability import quadrotor_flatness, compute_omega_from_R

# =============================================================================
# Configuration
# =============================================================================
MODELICA_FILE = "models/quadrotor_closed_loop.mo"
MODEL_NAME = "QuadrotorClosedLoop"
OUTPUT_DIR = "output_quadrotor"
D_GYRO = 0.1  # rad/s gyro bias disturbance amplitude

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Compile Modelica model
# =============================================================================
print(f"Compiling {MODELICA_FILE}...")
with open(MODELICA_FILE) as f:
    source = f.read()
json_str = rumoca.compile(source, MODEL_NAME)
ir = DaeIR.from_json_str(json_str)
ss = ir_to_symbolic_statespace(ir)
params = ir.get_param_defaults()
state_names = ss.get_state_names()
input_names = ss.get_input_names()
mass = params["mass"]
g = params["g"]
print(f"Compiled: {len(state_names)} states, {len(input_names)} inputs")

# =============================================================================
# Plan trajectory
# =============================================================================
ALTITUDE = 1.0

waypoint_xyz = np.array([
    [0.0, 0.0, ALTITUDE],
    [7.04, -0.76, ALTITUDE],
    [10.04, 1.7, ALTITUDE],
    [10.22, 6.6, ALTITUDE],
    [13.33, 8.65, ALTITUDE],
    [20.15, 8.14, ALTITUDE],
    [19.6, -1.92, ALTITUDE],
])

waypoint_vel = np.array([
    [0.0, 0.0, 0.0],
    [2.37, 0.0, 0.0],
    [0.15, 2.67, 0.0],
    [0.49, 2.28, 0.0],
    [2.85, -0.23, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

n_waypoints = len(waypoint_xyz)
n_segments = n_waypoints - 1
T_segments = np.array([4.67, 2.17, 1.84, 1.92, 5.5, 6.46])

# Convert to NED (negate z)
waypoint_ned = waypoint_xyz.copy()
waypoint_ned[:, 2] *= -1
vel_ned = waypoint_vel.copy()
vel_ned[:, 2] *= -1

print("Planning trajectory...")
trajectories = {}
for d, axis in enumerate(["x", "y", "z"]):
    bc = np.zeros((3, n_waypoints, 1))
    bc[0, :, 0] = waypoint_ned[:, d]
    bc[1, :, 0] = vel_ned[:, d]

    traj = plan_minimum_derivative_trajectory(
        bc=bc, min_deriv=3, poly_deg=5, T_guess=T_segments, bc_deriv=3
    )

    acc = traj.metadata["derivatives"]["acc"][:, 0]
    traj.metadata["derivatives"]["jerk"] = np.gradient(acc, traj.t).reshape(-1, 1)

    trajectories[axis] = {
        "t": traj.t,
        "pos": traj.x[:, 0],
        "vel": traj.metadata["derivatives"]["vel"][:, 0],
        "acc": traj.metadata["derivatives"]["acc"][:, 0],
        "jerk": traj.metadata["derivatives"]["jerk"][:, 0],
    }

t_eval = trajectories["x"]["t"]
T_total = t_eval[-1]

p_ref = np.column_stack([trajectories[a]["pos"] for a in ["x", "y", "z"]])
v_ref = np.column_stack([trajectories[a]["vel"] for a in ["x", "y", "z"]])
a_ref = np.column_stack([trajectories[a]["acc"] for a in ["x", "y", "z"]])
jerk_ref = np.column_stack([trajectories[a]["jerk"] for a in ["x", "y", "z"]])

# Remove duplicate time points at segment boundaries
unique_mask = np.concatenate([[True], np.diff(t_eval) > 0])
t_eval = t_eval[unique_mask]
p_ref = p_ref[unique_mask]
v_ref = v_ref[unique_mask]
a_ref = a_ref[unique_mask]
jerk_ref = jerk_ref[unique_mask]

print(f"Trajectory: t=[0, {T_total:.1f}]s, {len(t_eval)} points, {n_segments} segments")

# =============================================================================
# Attitude reference via differential flatness
# =============================================================================
N = len(t_eval)
flat_ref = quadrotor_flatness(p_ref, v_ref, a_ref, jerk_ref, mass, g)
omega_ref = compute_omega_from_R(flat_ref["R_ref"], t_eval)
a_body_ref = flat_ref["a_body_ref"]
R_ref = flat_ref["R_ref"]
T_ref_val = flat_ref["T_ref"]
print(f"Flatness: thrust=[{T_ref_val.min():.2f}, {T_ref_val.max():.2f}] N, "
      f"max omega={np.max(np.abs(omega_ref), axis=0)} rad/s")

# =============================================================================
# Build numerical dynamics
# =============================================================================
print("Building dynamics function...")
full_dynamics = ss.f + ss.Bu
param_subs = {sp.Symbol(k): v for k, v in params.items()}
dynamics_with_params = full_dynamics.subs(param_subs)
dynamics_func = sp.lambdify(
    [ss.state_symbols, ss.input_symbols], dynamics_with_params, modules="numpy"
)

# Precompute input indices to avoid repeated .index() in the hot loop
_idx_p = [input_names.index(n) for n in ["px_ref", "py_ref", "pz_ref"]]
_idx_v = [input_names.index(n) for n in ["vx_ref", "vy_ref", "vz_ref"]]
_idx_R = [input_names.index(f"R{i+1}{j+1}_ref") for i in range(3) for j in range(3)]
_idx_w = [input_names.index(n) for n in ["omega_x_ref", "omega_y_ref", "omega_z_ref"]]
_idx_T = input_names.index("T_ff")
_idx_d = [input_names.index(n) for n in ["d_p", "d_q", "d_r"]]

# Build a single interpolator for all reference inputs (faster than 20+ separate ones)
# Columns: px,py,pz, vx,vy,vz, R11..R33, wx,wy,wz, T
_ref_data = np.zeros((N, 19))
_ref_data[:, 0:3] = p_ref
_ref_data[:, 3:6] = v_ref
_ref_data[:, 6:15] = R_ref.reshape(N, 9)
_ref_data[:, 15:18] = omega_ref
_ref_data[:, 18] = T_ref_val
_ref_interp = interp1d(t_eval, _ref_data, axis=0, fill_value="extrapolate")

# Map from ref_data columns to input vector indices
_ref_to_input = np.array(_idx_p + _idx_v + _idx_R + _idx_w + [_idx_T], dtype=int)


def get_inputs(t, d_p=0.0, d_q=0.0, d_r=0.0):
    u = np.zeros(len(input_names))
    ref = _ref_interp(t)
    u[_ref_to_input] = ref
    u[_idx_d[0]] = d_p
    u[_idx_d[1]] = d_q
    u[_idx_d[2]] = d_r
    return u


# =============================================================================
# Initial condition
# =============================================================================
x0 = np.zeros(len(state_names))
x0[state_names.index("px")] = p_ref[0, 0]
x0[state_names.index("py")] = p_ref[0, 1]
x0[state_names.index("pz")] = p_ref[0, 2]
x0[state_names.index("R11")] = 1.0
x0[state_names.index("R22")] = 1.0
x0[state_names.index("R33")] = 1.0

t_span = (t_eval[0], t_eval[-1])

# =============================================================================
# Nominal simulation
# =============================================================================
print("Running nominal simulation...")
nom_sol = solve_ivp(
    lambda t, s: np.array(dynamics_func(s, get_inputs(t))).flatten(),
    t_span, x0, t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
)
print(f"  Status: {nom_sol.message}")
print(
    f"  Final pos: [{nom_sol.y[state_names.index('px'), -1]:.4f}, "
    f"{nom_sol.y[state_names.index('py'), -1]:.4f}, "
    f"{nom_sol.y[state_names.index('pz'), -1]:.4f}]"
)

# =============================================================================
# Disturbance sweep (parallelized)
# =============================================================================
frequencies = np.arange(0.1, 1.0, 0.2)
n_phases = 6
n_axes = 3

# Build list of (freq, phase, axis_config) jobs
sim_jobs = []
for freq in frequencies:
    period = 1.0 / freq
    phases = np.linspace(0, period, n_phases, endpoint=False)
    for phase in phases:
        for axis_config in range(n_axes + 1):
            sim_jobs.append((freq, phase, axis_config))

print(f"Running {len(sim_jobs)} disturbed simulations...")


def run_sim(job):
    freq, phase, axis_config = job
    period = 1.0 / freq

    def make_square(t):
        return D_GYRO if (((t + phase) % period) / period) < 0.5 else -D_GYRO

    def disturbed_ode(t, state):
        d_val = make_square(t)
        if axis_config == 0:
            d = [d_val, 0, 0]
        elif axis_config == 1:
            d = [0, d_val, 0]
        elif axis_config == 2:
            d = [0, 0, d_val]
        else:
            d = [d_val, d_val, d_val]
        return np.array(dynamics_func(state, get_inputs(t, *d))).flatten()

    sol = solve_ivp(
        disturbed_ode, t_span, x0, t_eval=t_eval,
        method="RK45", rtol=1e-8, atol=1e-10,
    )
    return sol.y if sol.status == 0 else None


with ThreadPoolExecutor() as pool:
    results = pool.map(run_sim, sim_jobs)

disturbed_trajectories = [r for r in results if r is not None]
print(f"Completed {len(disturbed_trajectories)} simulations")

# =============================================================================
# Plotting
# =============================================================================
idx_px = state_names.index("px")
idx_py = state_names.index("py")
idx_pz = state_names.index("pz")
idx_vx = state_names.index("vx")
idx_vy = state_names.index("vy")
idx_vz = state_names.index("vz")
idx_ox = state_names.index("omega_x")
idx_oy = state_names.index("omega_y")
idx_oz = state_names.index("omega_z")
z_sign = [1, 1, -1]

# --- Reference trajectory ---
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
for i, (plabel, vlabel) in enumerate(
    zip(["x [m]", "y [m]", "z [m]"], ["vx [m/s]", "vy [m/s]", "vz [m/s]"])
):
    axes[i, 0].plot(t_eval, z_sign[i] * p_ref[:, i], "b-", linewidth=2)
    axes[i, 0].set_ylabel(plabel)
    axes[i, 0].grid(True)
    axes[i, 1].plot(t_eval, z_sign[i] * v_ref[:, i], "r-", linewidth=2)
    axes[i, 1].set_ylabel(vlabel)
    axes[i, 1].grid(True)
axes[-1, 0].set_xlabel("Time [s]")
axes[-1, 1].set_xlabel("Time [s]")
fig.suptitle("Reference Position Trajectory")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ref_trajectory.png", dpi=150)

# --- 3D trajectory ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for traj in disturbed_trajectories[::4]:
    ax.plot(traj[idx_px], traj[idx_py], -traj[idx_pz], "b-", alpha=0.1, linewidth=0.5)

ax.plot(p_ref[:, 0], p_ref[:, 1], -p_ref[:, 2], "k--", linewidth=2, label="Reference")
ax.plot(
    nom_sol.y[idx_px], nom_sol.y[idx_py], -nom_sol.y[idx_pz],
    "g-", linewidth=2, label="Nominal",
)
ax.scatter(*waypoint_xyz[0, :], color="g", s=80, zorder=5, label="Start")
for wp in waypoint_xyz[1:]:
    ax.scatter(*wp, color="r", s=50, zorder=5)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("Quadrotor 3D Trajectory")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trajectory_3d.png", dpi=150)

# --- Position tracking ---
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
pos_labels = ["x [m]", "y [m]", "z [m]"]
pos_indices = [idx_px, idx_py, idx_pz]

for i, (ax, label, idx_s) in enumerate(zip(axes, pos_labels, pos_indices)):
    for traj in disturbed_trajectories:
        ax.plot(t_eval, z_sign[i] * traj[idx_s], "b-", alpha=0.05, linewidth=0.5)
    ax.plot(t_eval, z_sign[i] * nom_sol.y[idx_s], "g-", linewidth=2, label="Nominal")
    ax.plot(t_eval, z_sign[i] * p_ref[:, i], "k--", linewidth=2, label="Reference")
    ax.set_ylabel(label)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
fig.suptitle("Position Tracking (Nominal + Disturbed)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/position_tracking.png", dpi=150)

# --- Angular velocity error ---
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
omega_labels = ["omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]"]
omega_indices = [idx_ox, idx_oy, idx_oz]

for i, (ax, label, idx_s) in enumerate(zip(axes, omega_labels, omega_indices)):
    omega_ref_i = interp1d(t_eval, omega_ref[:, i], fill_value="extrapolate")
    for traj in disturbed_trajectories:
        ax.plot(t_eval, traj[idx_s] - omega_ref_i(t_eval), "b-", alpha=0.05, linewidth=0.5)
    ax.plot(
        t_eval, nom_sol.y[idx_s] - omega_ref_i(t_eval),
        "g-", linewidth=2, label="Nominal",
    )
    ax.set_ylabel(f"e_{label}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
fig.suptitle("Angular Velocity Error")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/omega_error.png", dpi=150)

# --- Velocity error ---
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
vel_labels = ["vx [m/s]", "vy [m/s]", "vz [m/s]"]
vel_indices = [idx_vx, idx_vy, idx_vz]

for i, (ax, label, idx_s) in enumerate(zip(axes, vel_labels, vel_indices)):
    vel_ref_i = interp1d(t_eval, v_ref[:, i], fill_value="extrapolate")
    for traj in disturbed_trajectories:
        ax.plot(t_eval, traj[idx_s] - vel_ref_i(t_eval), "b-", alpha=0.05, linewidth=0.5)
    ax.plot(
        t_eval, nom_sol.y[idx_s] - vel_ref_i(t_eval),
        "g-", linewidth=2, label="Nominal",
    )
    ax.set_ylabel(f"e_{label}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
fig.suptitle("Velocity Error")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/velocity_error.png", dpi=150)

# --- Summary ---
max_errors = {"position": np.zeros(3), "velocity": np.zeros(3), "omega": np.zeros(3)}
for traj in disturbed_trajectories:
    for i, idx_s in enumerate([idx_px, idx_py, idx_pz]):
        ref_i = interp1d(t_eval, p_ref[:, i], fill_value="extrapolate")
        max_errors["position"][i] = max(
            max_errors["position"][i], np.max(np.abs(traj[idx_s] - ref_i(t_eval)))
        )
    for i, idx_s in enumerate([idx_vx, idx_vy, idx_vz]):
        ref_i = interp1d(t_eval, v_ref[:, i], fill_value="extrapolate")
        max_errors["velocity"][i] = max(
            max_errors["velocity"][i], np.max(np.abs(traj[idx_s] - ref_i(t_eval)))
        )
    for i, idx_s in enumerate([idx_ox, idx_oy, idx_oz]):
        ref_i = interp1d(t_eval, omega_ref[:, i], fill_value="extrapolate")
        max_errors["omega"][i] = max(
            max_errors["omega"][i], np.max(np.abs(traj[idx_s] - ref_i(t_eval)))
        )

print("=" * 60)
print("SIMULATION SUMMARY")
print("=" * 60)
print(f"Trajectory: {T_total:.1f}s, {len(t_eval)} points, {n_segments} segments")
print(f"Disturbance: gyro bias = {D_GYRO} rad/s")
print(f"Simulations: {len(disturbed_trajectories)}")
print(f"\nMax tracking errors across all disturbed simulations:")
for i, name in enumerate(["x", "y", "z"]):
    print(f"  position_{name}: {max_errors['position'][i]:.6f} m")
for i, name in enumerate(["vx", "vy", "vz"]):
    print(f"  velocity_{name}: {max_errors['velocity'][i]:.6f} m/s")
for i, name in enumerate(["omega_x", "omega_y", "omega_z"]):
    print(f"  omega_{name}:    {max_errors['omega'][i]:.6f} rad/s")

plt.show()
