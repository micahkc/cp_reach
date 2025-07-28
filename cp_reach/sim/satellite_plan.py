import numpy as np

def orbit_trajectory(config=None):
    """
    Generate satellite trajectory using configurable parameters.

    Parameters
    ----------
    config : dict, optional
        A dictionary of physical and maneuver parameters. Keys:
            - total_time (s): simulation time horizon
            - dt (s): time step
            - burn_start (s): time when burn begins
            - burn_duration (s): duration of burn
            - thrust (N): thrust magnitude
            - mass (kg): satellite mass
            - thrust_direction (3-vector): unit vector of thrust direction
            - max_slew_rate (rad/s): peak angular rate during maneuver
            - omega_amplitude (list): amplitude scaling per axis [x, y, z]
            - omega_frequency (list): frequency scaling per axis [x, y, z]
            - slew_start (float): time when attitude slew begins (s)
            - slew_duration (float): duration of attitude maneuver (s)

    Returns
    -------
    dict with keys: t, ax, ay, az, omega1, omega2, omega3, dv
    """

    # Default config values (e.g. small satellite)
    if config is None:
        config = {}

    # --- Config Parameters ---
    total_time      = config.get("total_time", 600)
    dt              = config.get("dt", 1.0)
    burn_start      = config.get("burn_start", 100)
    burn_duration   = config.get("burn_duration", 30)
    thrust          = config.get("thrust", 0.2) # Newtons
    mass            = config.get("mass", 200.0) # kg
    direction       = np.array(config.get("thrust_direction", [1, 0, -0.5]))
    direction       = direction / np.linalg.norm(direction)
    max_slew_rate   = config.get("max_slew_rate", 0.02) # rad/s

    # Angular Velocity Profile Configuration
    omega_amp = config.get("omega_amplitude", [1.0, 0.8, 0.5]) # x, y, z axis scale
    omega_freq = config.get("omega_frequency", [1.0, 1.0, 2.0]) # relative frequency

    # Slew Timing (for windowed sine profile)
    slew_start = config.get("slew_start", 200)
    slew_duration = config.get("slew_duration", 100)

    # Time Vector
    t = np.arange(0, total_time + dt, dt)


    # Translational Acceleration (km/s^2)
    acc_mag = thrust / (mass * 1000.0)
    ax = np.zeros_like(t)
    ay = np.zeros_like(t)
    az = np.zeros_like(t)

    burn_mask = (t >= burn_start) & (t <= burn_start + burn_duration)
    ax[burn_mask] = acc_mag * direction[0]
    ay[burn_mask] = acc_mag * direction[1]
    ax[burn_mask] = acc_mag * direction[2]

    # Angular Velocity Profile (rad/s)
    omega1 = np.zeros_like(t)
    omega2 = np.zeros_like(t)
    omega3 = np.zeros_like(t)

    # Slew Mask for Smooth Sinusoidal Rotation Profile
    slew_mask = (t >= slew_start) & (t <= slew_start + slew_duration)
    tau = (t[slew_mask] - slew_start) / slew_duration # normalized time in slew window

    omega1[slew_mask] = max_slew_rate * omega_amp[0] * np.sin(omega_freq[0] * np.pi * tau)
    omega2[slew_mask] = max_slew_rate * omega_amp[1] * np.sin(omega_freq[1] * np.pi * tau)
    omega3[slew_mask] = max_slew_rate * omega_amp[2] * np.sin(omega_freq[2] * np.pi * tau)

    # Accumulated delta-v
    dv = np.cumsum(np.sqrt(ax**2 + ay**2 + az**2)) * dt # km/s

    return {
        "t": t,
        "ax": ax,
        "ay": ay,
        "az": az,
        "omega1": omega1,
        "omega2": omega2,
        "omega3": omega3,
        "dv": dv
    }