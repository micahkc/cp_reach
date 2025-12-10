from __future__ import annotations

import numpy as np
import scipy.optimize
import sympy

from .trajectory import Trajectory


def find_Q(deriv_order: int, poly_deg: int, n_legs: int, T_mat: sympy.MatrixSymbol | None = None) -> sympy.Matrix:
    """
    Symbolically build the block-diagonal Q matrix for minimizing the squared
    deriv_order-th derivative of a piecewise polynomial trajectory.
    """
    k, m, n = sympy.symbols("k m n", integer=True)
    beta = sympy.symbols("beta")
    T = sympy.symbols("T")
    c = sympy.MatrixSymbol("c", poly_deg + 1, 1)

    expr = sympy.summation(
        c[k, 0] * sympy.factorial(k) / sympy.factorial(k - m) * beta ** (k - m) / T**m,
        (k, m, n),
    )
    expr = expr.subs({m: deriv_order, n: poly_deg}).doit()
    J_expr = sympy.integrate(expr**2, (beta, 0, 1)).doit()

    p = sympy.Matrix([c[i, 0] for i in range(poly_deg + 1)])
    Q = sympy.Matrix([J_expr]).jacobian(p).jacobian(p) / 2
    assert (p.T @ Q @ p)[0, 0].expand() == J_expr

    Ti = T_mat if T_mat is not None else sympy.MatrixSymbol("T", n_legs, 1)
    Q_blocks = [Q.subs(T, Ti[i]) for i in range(n_legs)]
    return sympy.diag(*Q_blocks)


def find_A(
    deriv_order: int,
    poly_deg: int,
    beta: float,
    n_legs: int,
    leg_idx: int,
    value,
    T_mat: sympy.MatrixSymbol | None = None,
) -> tuple[sympy.Matrix, sympy.Matrix]:
    """
    Construct a constraint row enforcing a derivative value at a waypoint
    (start or end of a segment) in the polynomial trajectory.
    """
    k, m, n, n_c, n_l = sympy.symbols("k m n n_c n_l", integer=True)
    c = sympy.MatrixSymbol("c", n_c, n_l)
    T = T_mat if T_mat is not None else sympy.MatrixSymbol("T", n_l, 1)
    p = sympy.Matrix([c[i, l] for l in range(n_legs) for i in range(poly_deg + 1)])

    expr = sympy.summation(
        c[k, leg_idx] * sympy.factorial(k) / sympy.factorial(k - m) * beta ** (k - m) / T[leg_idx] ** m,
        (k, m, n),
    )
    expr = expr.subs({m: deriv_order, n: poly_deg}).doit()

    A_row = sympy.Matrix([expr]).jacobian(p)
    b_row = sympy.Matrix([value])
    return A_row, b_row


def find_cost_function(
    poly_deg: int = 5,
    min_deriv: int = 3,
    rows_free: list[int] | None = None,
    n_legs: int = 2,
    bc_deriv: int = 3,
):
    """
    Build symbolic minimum-derivative cost and constraint solvers.
    """
    if rows_free is None:
        rows_free = []

    T_placeholder = sympy.MatrixSymbol("T", n_legs, 1)
    Q = find_Q(deriv_order=min_deriv, poly_deg=poly_deg, n_legs=n_legs, T_mat=T_placeholder)
    # Use concrete symbols to avoid stray MatrixSymbols in lambdify
    x = sympy.MatrixSymbol("x", min_deriv + 1, n_legs + 1)  # bc matrix: (num_derivatives, num_waypoints)
    T_syms = sympy.symbols(f"T_0:{n_legs}")
    T_vec = sympy.Matrix(T_syms)

    A_rows = []
    b_rows = []
    for i in range(n_legs):
        for m in range(bc_deriv):
            A0, b0 = find_A(
                deriv_order=m,
                poly_deg=poly_deg,
                beta=0,
                n_legs=n_legs,
                leg_idx=i,
                value=x[m, i],
                T_mat=T_placeholder,
            )
            A_rows.append(A0)
            b_rows.append(b0)
            A1, b1 = find_A(
                deriv_order=m,
                poly_deg=poly_deg,
                beta=1,
                n_legs=n_legs,
                leg_idx=i,
                value=x[m, i + 1],
                T_mat=T_placeholder,
            )
            A_rows.append(A1)
            b_rows.append(b1)

    A = sympy.Matrix.vstack(*A_rows)
    b = sympy.Matrix.vstack(*b_rows)
    I = sympy.eye(A.shape[0])

    rows_fixed = list(range(A.shape[0]))
    for row in rows_free:
        rows_fixed.remove(row)
    rows = rows_fixed + rows_free
    C = sympy.Matrix.vstack(*[I[i, :] for i in rows])

    A_inv = A.inv()
    R = C @ A_inv.T @ Q @ A_inv @ C.T
    R.simplify()

    n_f = len(rows_fixed)
    Rpp = R[n_f:, n_f:]
    Rfp = R[:n_f, n_f:]
    df = (C @ b)[:n_f, 0]
    dp = -Rpp.inv() @ Rfp.T @ df
    d = sympy.Matrix.vstack(df, dp)
    p = A_inv @ C.T @ d

    k_time = sympy.symbols("k")

    # Substitute matrix elements T[i] with scalar symbols T_syms[i]
    subs_T = {T_placeholder[i, 0]: T_syms[i] for i in range(n_legs)}
    J = ((p.T @ Q @ p)[0, 0]).subs(subs_T) + k_time * sum(T_syms)
    p = p.subs(subs_T)

    return {
        "T": T_vec,
        "f_J": sympy.lambdify([T_vec, x, k_time], J, "numpy"),
        "f_p": sympy.lambdify([T_vec, x, k_time], list(p), "numpy"),
    }


def compute_trajectory(p, T, poly_deg: int, deriv: int = 0, num_points: int = 100):
    """
    Evaluate a piecewise polynomial trajectory and its derivatives.
    """
    p = np.array(p)
    T = np.array(T)
    n_legs = len(T)
    n_coeff = poly_deg + 1

    if p.shape[0] != n_legs * n_coeff:
        raise ValueError(
            f"Expected {n_legs} segments × {n_coeff} coefficients = {n_legs * n_coeff}, got {p.shape[0]}"
        )

    t_all = []
    x_all = []
    t_offset = 0.0
    for i in range(n_legs):
        coeffs = p[i * n_coeff : (i + 1) * n_coeff][::-1]  # highest degree first for polyval
        if deriv > 0:
            coeffs = np.polyder(coeffs, m=deriv)

        beta = np.linspace(0.0, 1.0, num_points)
        t_leg = T[i] * beta + t_offset
        x_leg = np.polyval(coeffs, beta) / (T[i] ** deriv)

        t_all.append(t_leg)
        x_all.append(x_leg)
        t_offset += T[i]

    t = np.concatenate(t_all)
    x = np.concatenate(x_all)
    return {"t": t, "x": x}


def plan_minimum_derivative_trajectory(
    bc: np.ndarray,
    min_deriv: int = 3,
    poly_deg: int = 5,
    k_time: float = 0.0,
    T_guess: np.ndarray | None = None,
    rows_free: list[int] | None = None,
    bc_deriv: int = 3,
    control_dim: int = 0,
) -> Trajectory:
    """
    Minimum-derivative polynomial planner (e.g., minimum snap for min_deriv=4).

    Parameters
    ----------
    bc : array (n_deriv, n_waypoints, n_dim)
        Boundary conditions for derivatives at waypoints.
    min_deriv : int
        Derivative order to minimize (3=jerk, 4=snap).
    poly_deg : int
        Polynomial degree per segment.
    k_time : float
        Weight on total time.
    T_guess : array, optional
        Initial guess for segment durations; optimized if provided None.
    rows_free : list[int], optional
        Boundary rows to leave free.
    bc_deriv : int
        Number of derivatives to constrain (e.g., 4 for pos/vel/acc/jerk).
    control_dim : int
        Control dimension for the output trajectory (filled with zeros).
    """
    n_deriv, n_waypoints, n_dim = bc.shape
    n_legs = n_waypoints - 1
    cost = find_cost_function(
        poly_deg=poly_deg, min_deriv=min_deriv, rows_free=rows_free, n_legs=n_legs, bc_deriv=bc_deriv
    )

    def objective(T_vec):
        return sum(cost["f_J"](T_vec, bc[:, :, d], k_time) for d in range(n_dim))

    if T_guess is None:
        T_guess = np.ones(n_legs)
        sol = scipy.optimize.minimize(fun=objective, x0=T_guess, bounds=[(0.05, 1e3)] * n_legs)
        T_opt = sol["x"]
    else:
        T_opt = np.asarray(T_guess)

    coeffs = []
    for d in range(n_dim):
        coeffs.append(cost["f_p"](T_opt, bc[:, :, d], k_time))

    # Evaluate derivatives up to bc_deriv for convenience
    derivatives: dict[str, np.ndarray] = {}
    for deriv_order in range(bc_deriv):
        comps = [compute_trajectory(coeffs[d], T_opt, poly_deg=poly_deg, deriv=deriv_order)["x"] for d in range(n_dim)]
        derivatives[{0: "pos", 1: "vel", 2: "acc", 3: "jerk", 4: "snap"}.get(deriv_order, f"deriv{deriv_order}")] = np.vstack(comps).T

    t = compute_trajectory(coeffs[0], T_opt, poly_deg=poly_deg, deriv=0)["t"]
    pos = derivatives["pos"]

    # Populate feedforward controls when acceleration is available and matches control dims.
    u = np.zeros((len(t), control_dim))
    acc = derivatives.get("acc")
    if acc is not None and control_dim > 0:
        acc = np.atleast_2d(acc)
        if control_dim == acc.shape[1]:
            u[:, :] = acc
        elif control_dim == 1 and acc.shape[1] == 1:
            u[:, 0] = acc[:, 0]
        # otherwise leave zeros and rely on caller to map acc -> controls

    metadata = {
        "segment_times": T_opt,
        "poly_deg": poly_deg,
        "min_deriv": min_deriv,
        "coeffs": coeffs,
        "derivatives": derivatives,
        "planner": "minimum_derivative",
    }

    return Trajectory(t=t, x=pos, u=u, metadata=metadata)
