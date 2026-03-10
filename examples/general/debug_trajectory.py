"""
Debug script: get the reference trajectory generation working for the quadrotor notebook.

Issues found:
1. ir.get_param_defaults() only returns {'g': 9.80665}, missing mass/inertia/gains
2. bc_deriv=3 only gives pos/vel/acc — need bc_deriv=4 for jerk (needed by flatness)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Step 1: Check what params the IR actually gives us ──────────────────────

MODELICA_FILE = "models/quadrotor_closed_loop.mo"
MODEL_NAME = "QuadrotorClosedLoop"

try:
    import rumoca

    with open(MODELICA_FILE) as f:
        source = f.read()
    json_str = rumoca.compile(source, MODEL_NAME)

    from cp_reach.ir import DaeIR, ir_to_symbolic_statespace

    ir = DaeIR.from_json_str(json_str)
    ss = ir_to_symbolic_statespace(ir)
    params_from_ir = ir.get_param_defaults()
    print(f"Params from IR: {params_from_ir}")
    print(f"States from IR: {ss.get_state_names()}")
    print(f"Inputs from IR: {ss.get_input_names()}")
except Exception as e:
    print(f"IR loading failed: {e}")
    params_from_ir = {}

# ── Step 2: Hardcode the params from the Modelica source ────────────────────

params = {
    # From RigidBody
    "g": 9.80665,
    # From Quadrotor
    "mass": 2.496,
    "Ixx": 0.0233,
    "Iyy": 0.0344,
    "Izz": 0.0527,
    # From QuadrotorClosedLoop - position gains
    "Kp_px": 0.20,
    "Kp_py": 0.20,
    "Kp_pz": 0.40,
    # Velocity gains
    "Kp_vx": 0.45,
    "Ki_vx": 0.10,
    "Kp_vy": 0.45,
    "Ki_vy": 0.10,
    "Kp_vz": 0.60,
    "Ki_vz": 0.10,
    # Attitude gains
    "Kr_x": 1.0,
    "Kr_y": 1.0,
    "Kr_z": 0.8,
    # Rate gains
    "Kw_x": 1.60,
    "Kw_y": 1.60,
    "Kw_z": 2.00,
}

mass = params["mass"]
g = params["g"]
print(f"\nUsing mass={mass}, g={g}")

# ── Step 3: Generate trajectory with correct bc_deriv ───────────────────────

from cp_reach.planning import plan_minimum_derivative_trajectory

T_seg = 3.0
waypoints = {
    "x": [0.0, 2.0, 0.0],
    "y": [0.0, 1.0, 0.0],
    "z": [0.0, -1.0, 0.0],
}

trajectories = {}
for axis, wp in waypoints.items():
    bc1 = np.array([
        [[wp[0]], [wp[1]]],
        [[0.0], [0.0]],
        [[0.0], [0.0]],
    ])
    bc2 = np.array([
        [[wp[1]], [wp[2]]],
        [[0.0], [0.0]],
        [[0.0], [0.0]],
    ])

    # bc_deriv=4 to get jerk, poly_deg=7 to have enough DOF for 4th order constraints
    try:
        traj1 = plan_minimum_derivative_trajectory(
            bc=bc1, min_deriv=3, poly_deg=5,
            T_guess=np.array([T_seg]), bc_deriv=4,
        )
        print(f"  {axis} seg1: derivatives available = {list(traj1.metadata['derivatives'].keys())}")
    except Exception as e:
        print(f"  {axis} seg1 with bc_deriv=4 failed: {e}")
        print("  Trying bc_deriv=3 and computing jerk manually...")
        traj1 = plan_minimum_derivative_trajectory(
            bc=bc1, min_deriv=3, poly_deg=5,
            T_guess=np.array([T_seg]), bc_deriv=3,
        )
        print(f"  {axis} seg1: derivatives available = {list(traj1.metadata['derivatives'].keys())}")

    try:
        traj2 = plan_minimum_derivative_trajectory(
            bc=bc2, min_deriv=3, poly_deg=5,
            T_guess=np.array([T_seg]), bc_deriv=4,
        )
    except Exception as e:
        print(f"  {axis} seg2 with bc_deriv=4 failed: {e}")
        traj2 = plan_minimum_derivative_trajectory(
            bc=bc2, min_deriv=3, poly_deg=5,
            T_guess=np.array([T_seg]), bc_deriv=3,
        )

    derivs1 = traj1.metadata["derivatives"]
    derivs2 = traj2.metadata["derivatives"]

    # If jerk is missing, compute it via finite differences from acceleration
    if "jerk" not in derivs1:
        print(f"  Computing jerk for {axis} via finite differences")
        dt1 = np.diff(traj1.t)
        acc1 = derivs1["acc"][:, 0]
        jerk1 = np.gradient(acc1, traj1.t)
        derivs1["jerk"] = jerk1.reshape(-1, 1)

        dt2 = np.diff(traj2.t)
        acc2 = derivs2["acc"][:, 0]
        jerk2 = np.gradient(acc2, traj2.t)
        derivs2["jerk"] = jerk2.reshape(-1, 1)

    trajectories[axis] = {
        "t": np.concatenate([traj1.t, traj1.t[-1] + traj2.t[1:]]),
        "pos": np.concatenate([derivs1["pos"][:, 0], derivs2["pos"][1:, 0]]),
        "vel": np.concatenate([derivs1["vel"][:, 0], derivs2["vel"][1:, 0]]),
        "acc": np.concatenate([derivs1["acc"][:, 0], derivs2["acc"][1:, 0]]),
        "jerk": np.concatenate([derivs1["jerk"][:, 0], derivs2["jerk"][1:, 0]]),
    }

t_eval = trajectories["x"]["t"]
T_total = t_eval[-1]

p_ref = np.column_stack([trajectories[a]["pos"] for a in ["x", "y", "z"]])
v_ref = np.column_stack([trajectories[a]["vel"] for a in ["x", "y", "z"]])
a_ref = np.column_stack([trajectories[a]["acc"] for a in ["x", "y", "z"]])
jerk_ref = np.column_stack([trajectories[a]["jerk"] for a in ["x", "y", "z"]])

print(f"\nTrajectory: t=[0, {T_total:.1f}]s, {len(t_eval)} points")
print(f"  p: {p_ref[0]} -> {p_ref[len(t_eval)//2]} -> {p_ref[-1]}")
print(f"  Max velocity: {np.max(np.abs(v_ref), axis=0)}")
print(f"  Max acceleration: {np.max(np.abs(a_ref), axis=0)}")
print(f"  Max jerk: {np.max(np.abs(jerk_ref), axis=0)}")

# ── Step 4: Compute flatness outputs ────────────────────────────────────────

from cp_reach.reachability import quadrotor_flatness, compute_omega_from_R

try:
    flat_ref = quadrotor_flatness(p_ref, v_ref, a_ref, jerk_ref, mass, g)
    omega_ref = compute_omega_from_R(flat_ref["R_ref"], t_eval)
    a_body_ref = flat_ref["a_body_ref"]

    print(f"\nFlatness outputs:")
    print(f"  Thrust range: [{flat_ref['T_ref'].min():.2f}, {flat_ref['T_ref'].max():.2f}] N")
    print(f"  Max omega_ref: {np.max(np.abs(omega_ref), axis=0)} rad/s")
    print(f"  Body accel at hover: a_body = {a_body_ref[0]} m/s^2")
    print("\nSUCCESS: Reference trajectory generation complete!")
except Exception as e:
    print(f"\nFlatness computation failed: {e}")
    import traceback
    traceback.print_exc()
