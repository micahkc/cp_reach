"""
Public entry points for the quadrotor Pipeline 1 toolchain.

These two functions are the surface the MATLAB/Simulink driver and any other
downstream consumer is expected to call:

    waypoints_to_trajectory(waypoints, ...)
        -> time-indexed reference trajectory (x(t), y(t), z(t), ..., omega(t))

    reachable_bounds(trajectory, ...)
        -> twelve scalar bounds: min/max deviation along each of the six
           rigid-body DOF (x, y, z, roll, pitch, yaw), in the reference frame.

The signatures and return shapes are kept MATLAB-friendly: inputs accept plain
Python lists or numpy arrays, returns are dicts of numpy arrays / floats that
MATLAB's CPython bridge maps cleanly to ``py.dict`` / ``py.numpy.ndarray``.
"""

import numpy as np

from cp_reach.development.applications.quadrotor.invariant import solve_nested
from cp_reach.development.applications.quadrotor.trajectory import (
    find_cost_function,
    plan_trajectory,
)


_DOF_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")


def waypoints_to_trajectory(
    waypoints,
    segment_times=None,
    velocities=None,
    poly_deg=7,
    min_deriv=4,
    bc_deriv=4,
    k_time=1e5,
):
    """
    Lower a waypoint sequence into a time-indexed reference trajectory.

    Internally this builds a minimum-derivative (default: minimum-snap)
    piecewise polynomial through the waypoints, then samples position,
    velocity, acceleration, and the reference angular velocity required by the
    differentially-flat quadrotor model.

    Parameters
    ----------
    waypoints : array-like, shape (N, 3)
        Sequence of N waypoints in (x, y, z) [m].
    segment_times : array-like, shape (N - 1,), optional
        Duration [s] of each segment. If omitted, durations are optimized to
        minimize the trajectory cost (minimum-snap + time penalty).
    velocities : array-like, shape (N, 3), optional
        Boundary velocity [m/s] at each waypoint. Defaults to zero at start and
        end and unconstrained-style zero at intermediate waypoints (i.e. a
        stop-and-go reference); pass explicit values for a flythrough.
    poly_deg : int
        Polynomial degree per segment. Default 7.
    min_deriv : int
        Derivative order to minimize. 4 = snap. Default 4.
    bc_deriv : int
        Number of derivative orders constrained at boundaries (pos / vel / acc
        / jerk = 4). Default 4.
    k_time : float
        Weight on total trajectory time in the cost. Default 1e5.

    Returns
    -------
    dict
        Time-indexed reference trajectory:
            t                       (M,) time [s]
            x,  y,  z               (M,) position [m]
            vx, vy, vz              (M,) velocity [m/s]
            ax, ay, az              (M,) acceleration [m/s^2]
            omega1, omega2, omega3  (M,) reference body-rate [rad/s]
            T                       (N - 1,) segment durations [s]
            poly_x, poly_y, poly_z  flattened polynomial coefficients per axis
            poly_deg                int, polynomial degree
    """
    wp = np.asarray(waypoints, dtype=float)
    if wp.ndim != 2 or wp.shape[1] != 3:
        raise ValueError(
            f"waypoints must have shape (N, 3); got {wp.shape}"
        )
    n_wp = wp.shape[0]
    if n_wp < 2:
        raise ValueError("Need at least 2 waypoints")
    n_legs = n_wp - 1

    if velocities is None:
        vel = np.zeros_like(wp)
    else:
        vel = np.asarray(velocities, dtype=float)
        if vel.shape != wp.shape:
            raise ValueError(
                f"velocities must match waypoints shape {wp.shape}; got {vel.shape}"
            )

    acc = np.zeros_like(wp)
    jerk = np.zeros_like(wp)

    # bc shape required by plan_trajectory: (n_deriv, n_waypoints, 3)
    bc = np.stack((wp, vel, acc, jerk))[:bc_deriv]

    T_opt = None if segment_times is None else np.asarray(segment_times, dtype=float)
    if T_opt is not None and T_opt.shape != (n_legs,):
        raise ValueError(
            f"segment_times must have shape ({n_legs},); got {T_opt.shape}"
        )

    cost = find_cost_function(
        poly_deg=poly_deg,
        min_deriv=min_deriv,
        rows_free=[],
        n_legs=n_legs,
        bc_deriv=bc_deriv,
    )

    return plan_trajectory(bc, cost, n_legs, poly_deg, k_time, T_opt)


def reachable_bounds(
    trajectory,
    vel_dist=0.0,
    accel_dist=0.0,
    ang_accel_dist=1.0,
    gravity=9.8,
):
    """
    Compute the per-DOF reachable-set envelope for the closed-loop quadrotor
    along a reference trajectory.

    Runs the cascaded LMI in the underlying ``invariant.solve_nested`` machinery
    -- inner-loop angular-velocity LMI first, its bound propagated as a
    disturbance into the outer-loop SE_2(3) log-linear LMI -- and exposes the
    resulting invariant set as twelve scalar bounds in the reference frame:
    a minimum and maximum deviation along each of the six rigid-body DOF
    (x, y, z, roll, pitch, yaw).

    Roll/pitch/yaw are reported from the SE_2(3) rotation channels of the
    invariant set. For the small deviations the LMI certifies these are
    interchangeable with axis-angle components to first order.

    Parameters
    ----------
    trajectory : dict
        Reference trajectory as returned by :func:`waypoints_to_trajectory`.
        Must provide keys ``ax, ay, az, omega1, omega2, omega3``.
    vel_dist : float
        Sup-norm bound on velocity disturbance [m/s]. Default 0.
    accel_dist : float
        Sup-norm bound on translational-acceleration disturbance [m/s^2].
        Default 0.
    ang_accel_dist : float
        Sup-norm bound on angular-acceleration disturbance [rad/s^2].
        Default 1.0.
    gravity : float
        Gravitational acceleration used for thrust-deviation bookkeeping.
        Default 9.8.

    Returns
    -------
    dict
        {
          "labels": ("x", "y", "z", "roll", "pitch", "yaw"),
          "units":  ("m", "m", "m", "rad", "rad", "rad"),
          "min":    (6,) np.ndarray   -- minimum deviation per DOF
          "max":    (6,) np.ndarray   -- maximum deviation per DOF
          "bounds": (6, 2) np.ndarray -- column 0 = min, column 1 = max
          # Flat scalar fields, convenient for MATLAB:
          "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
          "roll_min", "roll_max", "pitch_min", "pitch_max",
          "yaw_min", "yaw_max" :  float
          # Diagnostics:
          "omega_dist" : float  -- inner-loop angular-velocity bound used as
                                   disturbance into the outer-loop LMI
          "raw"        : dict   -- full solve_nested return (bounds_group is
                                   the (9, 2) array in the Lie group; useful
                                   for plotting / debugging)
        }
    """
    required = ("ax", "ay", "az", "omega1", "omega2", "omega3")
    missing = [k for k in required if k not in trajectory]
    if missing:
        raise ValueError(
            f"trajectory is missing required keys: {missing}. "
            "Call waypoints_to_trajectory first."
        )

    result = solve_nested(
        vel_dist=float(vel_dist),
        accel_dist=float(accel_dist),
        ang_accel_dist=float(ang_accel_dist),
        ref=trajectory,
        gravity=float(gravity),
    )

    # bounds_group is the (9, 2) axis-aligned box in the SE_2(3) Lie group
    # coordinates: rows 0..2 = position, 3..5 = velocity, 6..8 = rotation.
    # The six rigid-body DOF the report exposes are position + rotation.
    eta_bounds = np.asarray(result["se23"]["bounds_group"], dtype=float)
    pos_bounds = eta_bounds[0:3, :]   # x, y, z
    rot_bounds = eta_bounds[6:9, :]   # roll, pitch, yaw
    dof_bounds = np.vstack([pos_bounds, rot_bounds])   # (6, 2)

    out = {
        "labels": _DOF_LABELS,
        "units": ("m", "m", "m", "rad", "rad", "rad"),
        "min": dof_bounds[:, 0].copy(),
        "max": dof_bounds[:, 1].copy(),
        "bounds": dof_bounds,
        "omega_dist": float(result["angular"]["omega_dist"]),
        "raw": result,
    }
    for i, label in enumerate(_DOF_LABELS):
        out[f"{label}_min"] = float(dof_bounds[i, 0])
        out[f"{label}_max"] = float(dof_bounds[i, 1])

    return out
