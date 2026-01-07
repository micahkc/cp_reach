"""
Convert Rumoca DAE IR to SymbolicStateSpace.

This module provides the bridge between the loaded DAE IR and CP_Reach's
existing SymbolicStateSpace abstraction, enabling reachability analysis
without the cyecca dependency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import sympy as sp

from cp_reach.ir.loader import DaeIR
from cp_reach.ir.ast_parser import ast_to_sympy, parse_all_equations

if TYPE_CHECKING:
    from cp_reach.dynamics.state_space import SymbolicStateSpace


def ir_to_symbolic_statespace(
    ir: DaeIR,
    output_names: Optional[List[str]] = None,
    simplify: bool = True,
) -> "SymbolicStateSpace":
    """
    Convert a Rumoca DAE IR to a SymbolicStateSpace.

    This function parses the equations from the DAE IR and constructs
    a SymbolicStateSpace compatible with CP_Reach's existing workflows.

    The resulting state-space has the form:
        ẋ = f(x, p, t) + Bu(t)
        y = h(x, p, t) + Du(t)

    where:
        - f is the drift dynamics (terms not involving inputs)
        - Bu is the control input vector (terms linear in inputs)
        - h is the output map (if output_names specified)
        - Du is the output feedthrough

    Parameters
    ----------
    ir : DaeIR
        Loaded DAE IR from Rumoca
    output_names : list[str], optional
        List of algebraic variable names to use as outputs.
        If None, no outputs are defined (h = None).
    simplify : bool, default=True
        Whether to simplify the resulting symbolic expressions

    Returns
    -------
    SymbolicStateSpace
        State-space representation suitable for reachability analysis

    Examples
    --------
    >>> from cp_reach.ir import DaeIR, ir_to_symbolic_statespace
    >>> ir = DaeIR.from_json("model.json")
    >>> ss = ir_to_symbolic_statespace(ir, output_names=["e", "ev"])
    >>> A = ss.A()  # State Jacobian
    >>> B = ss.B()  # Input Jacobian
    """
    # Import here to avoid circular dependency
    from cp_reach.dynamics.state_space import SymbolicStateSpace

    # Create symbol tables
    state_symbols = {name: sp.Symbol(name) for name in ir.get_state_names()}
    input_symbols = {name: sp.Symbol(name) for name in ir.get_input_names()}
    param_symbols = {name: sp.Symbol(name) for name in ir.get_parameter_names()}
    alg_symbols = {name: sp.Symbol(name) for name in ir.get_algebraic_names()}

    # Combined symbol table for parsing
    all_symbols = {**state_symbols, **input_symbols, **param_symbols, **alg_symbols}

    # Add time symbol
    t_sym = sp.Symbol("t")
    all_symbols["t"] = t_sym
    all_symbols["time"] = t_sym

    # Parse equations
    derivatives, algebraic_eqs = parse_all_equations(
        ir.equations, all_symbols, create_missing=True
    )

    # Build algebraic substitutions
    # Solve algebraic equations for the algebraic variables
    alg_subs = _solve_algebraic_equations(algebraic_eqs, alg_symbols)

    # Also parse algebraic equations from fz section if present
    if ir.algebraic_equations:
        _, more_alg_eqs = parse_all_equations(
            ir.algebraic_equations, all_symbols, create_missing=True
        )
        more_subs = _solve_algebraic_equations(more_alg_eqs, alg_symbols)
        alg_subs.update(more_subs)

    # Recursively expand algebraic substitutions
    alg_subs = _expand_substitutions(alg_subs)

    # Build state derivative expressions
    state_names = ir.get_state_names()
    input_syms = list(input_symbols.values())

    f_exprs = []
    Bu_exprs = []

    for state_name in state_names:
        if state_name in derivatives:
            # Apply algebraic substitutions
            full_expr = derivatives[state_name].subs(alg_subs)

            # Separate drift (f) from control (Bu)
            f_part, Bu_part = _separate_drift_and_control(full_expr, input_syms)

            if simplify:
                f_part = sp.simplify(f_part)
                Bu_part = sp.simplify(Bu_part)

            f_exprs.append(f_part)
            Bu_exprs.append(Bu_part)
        else:
            # State has no derivative equation (shouldn't happen in valid models)
            f_exprs.append(sp.S.Zero)
            Bu_exprs.append(sp.S.Zero)

    f = sp.Matrix(f_exprs)
    Bu = sp.Matrix(Bu_exprs)

    # Build output matrices if requested
    h = None
    Du = None
    output_syms = None

    if output_names:
        h_exprs = []
        Du_exprs = []
        output_syms = []
        missing_outputs = []

        for output_name in output_names:
            if output_name in alg_subs:
                # Get the expression for this output
                out_expr = alg_subs[output_name]
                output_syms.append(sp.Symbol(output_name))

                # Separate drift from control
                h_part, Du_part = _separate_drift_and_control(out_expr, input_syms)

                if simplify:
                    h_part = sp.simplify(h_part)
                    Du_part = sp.simplify(Du_part)

                h_exprs.append(h_part)
                Du_exprs.append(Du_part)

            elif output_name in alg_symbols:
                # Output is an algebraic variable without explicit equation
                # Try to find it in the symbol table
                output_syms.append(alg_symbols[output_name])
                h_exprs.append(alg_symbols[output_name])
                Du_exprs.append(sp.S.Zero)
            else:
                missing_outputs.append(output_name)

        if missing_outputs:
            available = list(alg_subs.keys()) + list(alg_symbols.keys())
            raise ValueError(
                f"Output names {missing_outputs} not found. "
                f"Available algebraic variables: {available}"
            )

        if h_exprs:
            h = sp.Matrix(h_exprs)
            Du = sp.Matrix(Du_exprs)

    # Get parameter defaults
    param_defaults = ir.get_param_defaults()

    return SymbolicStateSpace(
        f=f,
        Bu=Bu,
        h=h,
        Du=Du,
        state_symbols=list(state_symbols.values()),
        input_symbols=list(input_symbols.values()),
        param_symbols=list(param_symbols.values()),
        output_symbols=output_syms,
        param_defaults=param_defaults,
    )


def _separate_drift_and_control(
    expr: sp.Expr,
    input_syms: List[sp.Symbol],
) -> tuple:
    """
    Separate an expression into drift (f) and control (Bu) components.

    Given expr = f(x,p) + g(x,p,u), returns (f, g) where g contains
    all terms involving input symbols.

    Parameters
    ----------
    expr : sympy.Expr
        Full expression possibly containing input terms
    input_syms : list[Symbol]
        List of input symbols

    Returns
    -------
    tuple[Expr, Expr]
        (drift_part, control_part)
    """
    if not input_syms:
        return (expr, sp.S.Zero)

    # Expand the expression to get all terms
    expr_expanded = sp.expand(expr)

    # Use as_coefficients_dict to separate terms
    if hasattr(expr_expanded, "as_coefficients_dict"):
        coeff_dict = expr_expanded.as_coefficients_dict()
    else:
        # For non-Add expressions
        coeff_dict = {expr_expanded: sp.S.One}

    drift_part = sp.S.Zero
    control_part = sp.S.Zero

    input_sym_set = set(input_syms)

    for term, coeff in coeff_dict.items():
        # Check if this term contains any input symbols
        term_symbols = term.free_symbols
        if term_symbols & input_sym_set:
            control_part += coeff * term
        else:
            drift_part += coeff * term

    return (drift_part, control_part)


def _solve_algebraic_equations(
    algebraic_eqs: List[sp.Eq],
    alg_symbols: Dict[str, sp.Symbol],
) -> Dict[sp.Symbol, sp.Expr]:
    """
    Solve algebraic equations for the algebraic variables.

    Attempts to express each algebraic variable in terms of
    states, inputs, and parameters.

    Parameters
    ----------
    algebraic_eqs : list[Eq]
        List of algebraic equations
    alg_symbols : dict[str, Symbol]
        Algebraic variable symbols

    Returns
    -------
    dict[Symbol, Expr]
        Substitution dictionary mapping alg vars to expressions
    """
    subs = {}

    for eq in algebraic_eqs:
        if not isinstance(eq, sp.Eq):
            continue

        lhs, rhs = eq.lhs, eq.rhs

        # Check if lhs is a single algebraic variable
        if lhs in alg_symbols.values():
            subs[lhs] = rhs
        elif rhs in alg_symbols.values():
            subs[rhs] = lhs
        else:
            # Try to solve for any algebraic variable
            for alg_sym in alg_symbols.values():
                if alg_sym in eq.free_symbols:
                    try:
                        solutions = sp.solve(eq, alg_sym)
                        if solutions:
                            subs[alg_sym] = solutions[0]
                            break
                    except Exception:
                        continue

    return subs


def _expand_substitutions(
    subs: Dict[sp.Symbol, sp.Expr],
    max_iterations: int = 10,
) -> Dict[sp.Symbol, sp.Expr]:
    """
    Recursively expand substitutions until fixed point.

    For example, if we have:
        e = x - x_ref
        u_fb = -kp*e - kd*ev

    This expands u_fb to:
        u_fb = -kp*(x - x_ref) - kd*ev

    Parameters
    ----------
    subs : dict[Symbol, Expr]
        Initial substitution dictionary
    max_iterations : int
        Maximum number of expansion iterations

    Returns
    -------
    dict[Symbol, Expr]
        Fully expanded substitution dictionary
    """
    for _ in range(max_iterations):
        changed = False
        new_subs = {}

        for sym, expr in subs.items():
            new_expr = expr.subs(subs)
            new_subs[sym] = new_expr
            if new_expr != expr:
                changed = True

        subs = new_subs

        if not changed:
            break

    return subs


def ir_load(
    path: str,
    output_names: Optional[List[str]] = None,
    simplify: bool = True,
) -> "SymbolicStateSpace":
    """
    Load a Rumoca DAE IR JSON file and convert to SymbolicStateSpace.

    This is the main entry point for the IR-based workflow,
    analogous to sympy_load() for the cyecca-based workflow.

    Parameters
    ----------
    path : str
        Path to the Rumoca DAE IR JSON file
    output_names : list[str], optional
        Algebraic variables to use as outputs
    simplify : bool, default=True
        Whether to simplify expressions

    Returns
    -------
    SymbolicStateSpace
        State-space representation

    Examples
    --------
    >>> from cp_reach.ir import ir_load
    >>> ss = ir_load("closed_loop.json", output_names=["e", "ev"])
    >>> A = ss.A()
    >>> print(f"System has {A.shape[0]} states")
    """
    ir = DaeIR.from_json(path)
    return ir_to_symbolic_statespace(ir, output_names=output_names, simplify=simplify)
