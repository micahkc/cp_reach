import scipy.optimize
import numpy as np
import casadi as ca
import sympy

import cp_reach.quadrotor.dynamics as dynamics

def euler_from_dcm(R):
    """Return (yaw, pitch, roll) for a ZYX (yaw–pitch–roll) convention."""
    R = np.asarray(R)
    sy = -R[2, 0]
    sy_clamped = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy_clamped)
    if abs(sy_clamped) < 1.0 - 1e-8:  # not gimbal locked
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:  # gimbal lock: pitch ≈ ±90°
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return yaw, pitch, roll



def find_Q(deriv_order, poly_deg, n_legs):
    """
    Symbolically compute the block-diagonal cost matrix Q for minimizing
    the squared m-th derivative of a piecewise polynomial trajectory.

    This gives the integral cost:
        J = ∑ ∫ [d^m x_i(t)/dt^m]^2 dt  ≈  p^T Q p

    Parameters
    ----------
    deriv_order : int
        Derivative order to minimize in the cost function (e.g. 4 = snap).
    poly_deg : int
        Degree of the polynomial used in each trajectory segment.
    n_legs : int
        Number of trajectory segments (legs).

    Returns
    -------
    Q_block_diag : sympy.Matrix
        Symbolic block-diagonal matrix Q, shape ((poly_deg+1)*n_legs, (poly_deg+1)*n_legs),
        such that cost = p^T Q p for the entire trajectory.
    """
    k, m, n = sympy.symbols("k m n", integer=True)
    beta = sympy.symbols("beta")
    T = sympy.symbols("T")
    c = sympy.MatrixSymbol("c", poly_deg + 1, 1)

    expr = sympy.summation(
        c[k, 0] * sympy.factorial(k) / sympy.factorial(k - m) * beta**(k - m) / T**m,
        (k, m, n),
    )
    expr = expr.subs({m: deriv_order, n: poly_deg}).doit()
    J_expr = sympy.integrate(expr**2, (beta, 0, 1)).doit()

    p = sympy.Matrix([c[i, 0] for i in range(poly_deg + 1)])
    Q = sympy.Matrix([J_expr]).jacobian(p).jacobian(p) / 2

    assert (p.T @ Q @ p)[0, 0].expand() == J_expr

    Ti = sympy.MatrixSymbol("T", n_legs, 1)
    Q_blocks = [Q.subs(T, Ti[i]) for i in range(n_legs)]
    return sympy.diag(*Q_blocks)


def find_A(deriv_order, poly_deg, beta, n_legs, leg_idx, value):
    """
    Construct a constraint row enforcing a specific derivative value at a waypoint
    (start or end of a segment) in the polynomial trajectory.

    This function symbolically constructs a row of the constraint matrix A and the
    corresponding right-hand side b, such that:
        A_row * p = b_row
    enforces:
        d^m x_leg(beta) / dt^m = value

    Parameters
    ----------
    deriv_order : int
        Order of the derivative to constrain (e.g. 0 = position, 1 = velocity, etc.)
    poly_deg : int
        Degree of the polynomial used in each trajectory segment.
    beta : int or float
        Normalized time within the segment (0 = start, 1 = end).
    n_legs : int
        Total number of trajectory segments (i.e., waypoints - 1).
    leg_idx : int
        Index of the segment (0-based).
    value : sympy expression or float
        Value of the desired derivative to enforce at the given time.

    Returns
    -------
    A_row : sympy.Matrix
        1×(n_coeff * n_legs) matrix that is a row of the constraint matrix A.
    b_row : sympy.Matrix
        1×1 matrix containing the right-hand side value.
    """
    # Define symbolic variables for general expression
    k, m, n, n_c, n_l = sympy.symbols("k m n n_c n_l", integer=True)

    # Symbolic coefficient matrix and time vector
    c = sympy.MatrixSymbol("c", n_c, n_l)  # Coefficients per segment
    T = sympy.MatrixSymbol("T", n_l, 1)    # Duration per segment

    # Flattened coefficient vector p for all segments (column-wise by segment)
    p = sympy.Matrix([c[i, l] for l in range(n_legs) for i in range(poly_deg + 1)])

    # Symbolic expression for the m-th derivative of the polynomial at time beta
    expr = sympy.summation(
        c[k, leg_idx]
        * sympy.factorial(k)
        / sympy.factorial(k - m)
        * beta ** (k - m)
        / T[leg_idx]**m,
        (k, m, n)
    )

    # Substitute actual derivative order and polynomial degree
    expr = expr.subs({m: deriv_order, n: poly_deg}).doit()

    # Compute Jacobian of the expression with respect to p (row in A)
    A_row = sympy.Matrix([expr]).jacobian(p)

    # Right-hand side value
    b_row = sympy.Matrix([value])

    return A_row, b_row


def find_cost_function(poly_deg=5, min_deriv=3, rows_free=None, n_legs=2, bc_deriv=3):
    """
    Build symbolic minimum snap (or jerk, etc.) cost function for trajectory optimization.

    Parameters
    ----------
    poly_deg : int
        Degree of the polynomial for each trajectory segment.
    min_deriv : int
        Derivative order to minimize (e.g., 3 = jerk, 4 = snap).
    rows_free : list of int
        Indices of boundary condition rows to treat as free variables.
    n_legs : int
        Number of trajectory segments (legs).
    bc_deriv : int
        Number of derivatives with enforced boundary conditions (e.g. 4 for pos/vel/acc/jerk).

    Returns
    -------
    dict
        {
            "T": Symbolic vector of time variables [T_0, ..., T_{n-1}],
            "f_J": Callable cost function: J(T, bc, k_time)
            "f_p": Callable that returns flattened coefficient vector p
        }
    """
    if rows_free is None:
        rows_free = []

    # Step 1: Build the Q matrix (symbolic Hessian of cost)
    Q = find_Q(deriv_order=min_deriv, poly_deg=poly_deg, n_legs=n_legs)  # symbolic block-diag Q(T)

    # Define symbolic placeholders for boundary conditions and segment times
    n_l, n_d = sympy.symbols("n_l, n_d", integer=True)
    x = sympy.MatrixSymbol("x", n_d, n_l + 1)  # bc matrix: shape (num_derivatives, num_waypoints)
    T = sympy.MatrixSymbol("T", n_l, 1)        # symbolic segment times

    # Step 2: Build symbolic constraint matrix A and rhs vector b
    A_rows = []
    b_rows = []
    for i in range(n_legs):
        for m in range(bc_deriv):
            # Constraint at start of segment
            A0, b0 = find_A(deriv_order=m, poly_deg=poly_deg, beta=0, n_legs=n_legs, leg_idx=i, value=x[m, i])
            A_rows.append(A0)
            b_rows.append(b0)

            # Constraint at end of segment
            A1, b1 = find_A(deriv_order=m, poly_deg=poly_deg, beta=1, n_legs=n_legs, leg_idx=i, value=x[m, i + 1])
            A_rows.append(A1)
            b_rows.append(b1)

    A = sympy.Matrix.vstack(*A_rows)
    b = sympy.Matrix.vstack(*b_rows)
    I = sympy.eye(A.shape[0])

    # Step 3: Handle fixed and free rows
    rows_fixed = list(range(A.shape[0]))
    for row in rows_free:
        rows_fixed.remove(row)
    rows = rows_fixed + rows_free  # reorder rows
    C = sympy.Matrix.vstack(*[I[i, :] for i in rows])  # permutation matrix

    # Step 4: Solve constrained QP: minimize p^T Q p s.t. A p = b
    A_inv = A.inv()
    R = C @ A_inv.T @ Q @ A_inv @ C.T
    R.simplify()

    n_f = len(rows_fixed)
    n_p = len(rows_free)

    Rpp = R[n_f:, n_f:]
    Rfp = R[:n_f, n_f:]
    df = (C @ b)[:n_f, 0]
    dp = -Rpp.inv() @ Rfp.T @ df
    d = sympy.Matrix.vstack(df, dp)
    p = A_inv @ C.T @ d

    # Step 5: Construct symbolic cost and lambdify
    Ti = sympy.symbols(f"T_0:{n_legs}")
    T_vec = sympy.Matrix(Ti)
    k = sympy.symbols("k")  # time cost weight

    J = (p.T @ Q @ p)[0, 0] + k * sum(Ti)
    J = J.subs(T, T_vec)
    p = p.subs(T, T_vec)

    return {
        "T": T_vec,
        "f_J": sympy.lambdify([T_vec, x, k], J, "numpy"),
        "f_p": sympy.lambdify([T_vec, x, k], list(p), "numpy")
    }



def compute_trajectory(p, T, poly_deg, deriv=0, num_points=100):
    """
    Evaluate a piecewise polynomial trajectory and its derivatives.

    Parameters
    ----------
    p : list or np.ndarray
        Flattened list of polynomial coefficients for all segments, where each segment
        has (poly_deg + 1) coefficients. The coefficients are ordered from highest
        to lowest degree (i.e., compatible with np.polyval).
    T : list or np.ndarray
        List of durations for each trajectory segment.
    poly_deg : int
        Degree of each polynomial segment.
    deriv : int, optional
        Order of derivative to evaluate:
            0 = position,
            1 = velocity,
            2 = acceleration,
            ...
    num_points : int, optional
        Number of points per segment to evaluate.

    Returns
    -------
    dict
        Dictionary with:
        - 't': global time vector (cumulative across all segments)
        - 'x': evaluated trajectory values at each time step
    """

    p = np.array(p)
    T = np.array(T)
    n_legs = len(T)
    n_coeff = poly_deg + 1

    assert p.shape[0] == n_legs * n_coeff, (
        f"Expected {n_legs} segments × {n_coeff} coefficients = {n_legs * n_coeff}, "
        f"but got {p.shape[0]}"
    )

    t_all = []
    x_all = []

    t_offset = 0
    for i in range(n_legs):
        # Extract and flip coefficients to match np.polyval convention (highest to lowest)
        coeffs = p[i * n_coeff : (i + 1) * n_coeff][::-1]

        # Take derivative if needed
        if deriv > 0:
            coeffs = np.polyder(coeffs, m=deriv)

        # Generate normalized beta in [0, 1], then scale to time in segment
        beta = np.linspace(0, 1, num_points)
        t_leg = T[i] * beta + t_offset

        # Evaluate trajectory: apply chain rule scaling
        x_leg = np.polyval(coeffs, beta) / T[i]**deriv

        t_all.append(t_leg)
        x_all.append(x_leg)

        t_offset += T[i]

    t = np.concatenate(t_all)
    x = np.concatenate(x_all)

    return {
        "t": t,
        "x": x,
    }


def plan_trajectory(bc, cost, n_legs, poly_deg, k_time, T_opt=None):
    """
    Plans a minimum snap trajectory based on boundary conditions and optionally 
    optimizes segment durations.

    Parameters
    ----------
    bc : np.ndarray of shape (num_derivatives, num_waypoints, 3)
        Boundary conditions for the trajectory. The first axis corresponds to:
        - 0: position
        - 1: velocity
        - 2: acceleration
        - 3: jerk
    cost : dict
        Dictionary returned by `find_cost_function`, containing symbolic cost 
        and polynomial functions:
        - "f_J": cost function
        - "f_p": function to compute polynomial coefficients
    n_legs : int
        Number of trajectory segments (equal to number of waypoints - 1).
    poly_deg : int
        Degree of the polynomial for each segment.
    k_time : float
        Weight on total time in the cost function.
    T_opt : list or np.ndarray of floats, optional
        If provided, the durations for each trajectory segment. If not provided,
        the function will optimize the segment durations.

    Returns
    -------
    dict
        Dictionary containing:
        - Polynomial coefficients for x, y, z: 'poly_x', 'poly_y', 'poly_z'
        - Optimized segment durations: 'T'
        - Time vector: 't'
        - Position trajectory: 'x', 'y', 'z'
        - Velocity trajectory: 'vx', 'vy', 'vz'
        - Acceleration trajectory: 'ax', 'ay', 'az'
        - Angular velocity components: 'omega1', 'omega2', 'omega3'
        - Polynomial degree: 'poly_deg'
    """
    n_dim = 3  # x, y, z

    if T_opt is None:
        # Optimize time durations to minimize cost
        f_cost = lambda T: sum([cost["f_J"](T, bc[:, :, d], k_time) for d in range(n_dim)])
        sol = scipy.optimize.minimize(
            fun=f_cost,
            x0=[10] * n_legs,
            bounds=[(0.1, 100)] * n_legs
        )
        T_opt = sol["x"]

    # Compute polynomial coefficients
    opt_x = cost["f_p"](T_opt, bc[:, :, 0], k_time)
    opt_y = cost["f_p"](T_opt, bc[:, :, 1], k_time)
    opt_z = cost["f_p"](T_opt, bc[:, :, 2], k_time)

    # Compute trajectories for each derivative
    ref_x = compute_trajectory(opt_x, T_opt, poly_deg=poly_deg, deriv=0)
    t = ref_x['t']
    x = ref_x['x']
    y = compute_trajectory(opt_y, T_opt, poly_deg=poly_deg, deriv=0)['x']
    z = compute_trajectory(opt_z, T_opt, poly_deg=poly_deg, deriv=0)['x']
    vx = compute_trajectory(opt_x, T_opt, poly_deg=poly_deg, deriv=1)['x']
    vy = compute_trajectory(opt_y, T_opt, poly_deg=poly_deg, deriv=1)['x']
    vz = compute_trajectory(opt_z, T_opt, poly_deg=poly_deg, deriv=1)['x']
    ax = compute_trajectory(opt_x, T_opt, poly_deg=poly_deg, deriv=2)['x']
    ay = compute_trajectory(opt_y, T_opt, poly_deg=poly_deg, deriv=2)['x']
    az = compute_trajectory(opt_z, T_opt, poly_deg=poly_deg, deriv=2)['x']
    jx = compute_trajectory(opt_x, T_opt, poly_deg=poly_deg, deriv=3)['x']
    jy = compute_trajectory(opt_y, T_opt, poly_deg=poly_deg, deriv=3)['x']
    jz = compute_trajectory(opt_z, T_opt, poly_deg=poly_deg, deriv=3)['x']
    sx = compute_trajectory(opt_x, T_opt, poly_deg=poly_deg, deriv=4)['x']
    sy = compute_trajectory(opt_y, T_opt, poly_deg=poly_deg, deriv=4)['x']
    sz = compute_trajectory(opt_z, T_opt, poly_deg=poly_deg, deriv=4)['x']

    # Compute angular velocities
    romega1, romega2, romega3 = [], [], []
    for j in range(x.shape[0]):
        f_ref = dynamics.build_f_ref()
        ref_v = f_ref(
            0, 0, 0,
            [vx[j], vy[j], vz[j]],
            [ax[j], ay[j], az[j]],
            [jx[j], jy[j], jz[j]],
            [sx[j], sy[j], sz[j]],
            1, 9.8, 1, 1, 1, 0
        )
        R = ref_v[1]
        theta = np.array(ca.DM(euler_from_dcm(R))).reshape(3,)
        omega = np.array(ref_v[2]).reshape(3,)
        romega1.append(omega[0])
        romega2.append(omega[1])
        romega3.append(omega[2])

    return {
        "poly_x": opt_x,
        "poly_y": opt_y,
        "poly_z": opt_z,
        "T": T_opt,
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "omega1": romega1,
        "omega2": romega2,
        "omega3": romega3,
        "poly_deg": poly_deg,
    }
