"""
Rumoca AST to SymPy Expression Converter.

This module converts Rumoca's AST (Abstract Syntax Tree) representation
of expressions and equations into SymPy symbolic expressions.

The Rumoca AST uses a tagged union format where each node type is
represented as a dictionary with a single key indicating the node type.
For example:
    - {"Terminal": {"token": {"text": "1.0", ...}}}
    - {"Binary": {"op": {"Mul": {...}}, "lhs": {...}, "rhs": {...}}}
    - {"FunctionCall": {"comp": {...}, "args": [...]}}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

logger = logging.getLogger(__name__)


def ast_to_sympy(
    expr_ast: Any,
    symbols: Dict[str, sp.Symbol],
    create_missing: bool = True,
) -> sp.Expr:
    """
    Recursively convert a Rumoca AST expression to a SymPy expression.

    Parameters
    ----------
    expr_ast : dict
        Rumoca AST node representing an expression
    symbols : dict[str, Symbol]
        Mapping from variable names to SymPy symbols
    create_missing : bool, default=True
        If True, create new symbols for unknown variable references.
        If False, raise KeyError for unknown variables.

    Returns
    -------
    sympy.Expr
        Equivalent SymPy expression

    Raises
    ------
    ValueError
        If the AST contains an unsupported node type
    KeyError
        If create_missing is False and an unknown variable is referenced

    Examples
    --------
    >>> symbols = {"x": sp.Symbol("x"), "k": sp.Symbol("k")}
    >>> ast = {"Binary": {"op": {"Mul": {}}, "lhs": {...}, "rhs": {...}}}
    >>> expr = ast_to_sympy(ast, symbols)
    """
    if expr_ast is None or expr_ast == "Empty":
        return sp.S.Zero

    if not isinstance(expr_ast, dict):
        # Might be a raw number or string
        try:
            return sp.sympify(expr_ast)
        except Exception:
            return sp.S.Zero

    # Handle Terminal nodes (literals and identifiers)
    if "Terminal" in expr_ast:
        return _parse_terminal(expr_ast["Terminal"], symbols, create_missing)

    # Handle ComponentReference nodes (variable references)
    if "ComponentReference" in expr_ast:
        return _parse_component_reference(
            expr_ast["ComponentReference"], symbols, create_missing
        )

    # Handle Binary operations
    if "Binary" in expr_ast:
        return _parse_binary(expr_ast["Binary"], symbols, create_missing)

    # Handle Unary operations
    if "Unary" in expr_ast:
        return _parse_unary(expr_ast["Unary"], symbols, create_missing)

    # Handle Function calls
    if "FunctionCall" in expr_ast:
        return _parse_function_call(expr_ast["FunctionCall"], symbols, create_missing)

    # Handle If expressions
    if "If" in expr_ast:
        return _parse_if_expression(expr_ast["If"], symbols, create_missing)

    # Handle Array expressions
    if "Array" in expr_ast:
        return _parse_array(expr_ast["Array"], symbols, create_missing)

    # Handle Range expressions
    if "Range" in expr_ast:
        return _parse_range(expr_ast["Range"], symbols, create_missing)

    # Handle Parenthesized expressions (just unwrap)
    if "Parenthesized" in expr_ast:
        inner = expr_ast["Parenthesized"]
        # The inner content might be under "inner" or "expr" key
        if isinstance(inner, dict):
            if "inner" in inner:
                return ast_to_sympy(inner["inner"], symbols, create_missing)
            if "expr" in inner:
                return ast_to_sympy(inner["expr"], symbols, create_missing)
        return ast_to_sympy(inner, symbols, create_missing)

    raise ValueError(f"Unsupported AST node type: {list(expr_ast.keys())}")


def _parse_terminal(
    terminal: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a Terminal AST node."""
    token = terminal.get("token", {})
    text = token.get("text", "")
    terminal_type = terminal.get("terminal_type", "")

    # Check if it's a numeric literal
    if terminal_type in ("UnsignedInteger", "UnsignedReal", "UnsignedNumber"):
        try:
            if "." in text or "e" in text.lower():
                return sp.Float(text)
            return sp.Integer(text)
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse numeric literal '{text}': {e}")
            # Fall through to treat as symbol

    # Check for boolean literals
    if terminal_type == "Boolean" or text.lower() in ("true", "false"):
        return sp.S.true if text.lower() == "true" else sp.S.false

    # Check for string literals
    if terminal_type == "String":
        # Return as a SymPy symbol (strings not directly supported)
        return sp.Symbol(f"'{text}'")

    # Must be a variable reference
    if text in symbols:
        return symbols[text]

    if create_missing:
        sym = sp.Symbol(text)
        symbols[text] = sym
        return sym

    raise KeyError(f"Unknown symbol: {text}")


def _parse_component_reference(
    comp_ref: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a ComponentReference AST node."""
    parts = comp_ref.get("parts", [])
    if not parts:
        raise ValueError("ComponentReference has no parts")

    # Build the full qualified name
    name_parts = []
    subscripts_list = []

    for part in parts:
        ident = part.get("ident", {})
        name_parts.append(ident.get("text", ""))

        # Handle array subscripts
        subs = part.get("subs")
        if subs:
            subscripts_list.append(subs)

    name = ".".join(name_parts)

    # Look up or create symbol
    if name in symbols:
        base_sym = symbols[name]
    elif create_missing:
        base_sym = sp.Symbol(name)
        symbols[name] = base_sym
    else:
        raise KeyError(f"Unknown symbol: {name}")

    # Handle subscripting (for array access)
    if subscripts_list:
        # For now, represent as indexed symbol
        # Full array support would require more complex handling
        for subs in subscripts_list:
            indices = []
            for sub in subs:
                if isinstance(sub, dict):
                    idx = ast_to_sympy(sub, symbols, create_missing)
                    indices.append(idx)
            if indices:
                # Use Indexed for array access
                base_sym = sp.Indexed(base_sym, *indices)

    return base_sym


def _parse_binary(
    binary: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a Binary operation AST node."""
    op_dict = binary.get("op", {})
    lhs_ast = binary.get("lhs")
    rhs_ast = binary.get("rhs")

    lhs = ast_to_sympy(lhs_ast, symbols, create_missing)
    rhs = ast_to_sympy(rhs_ast, symbols, create_missing)

    # Determine the operation type
    op_name = list(op_dict.keys())[0] if op_dict else ""

    # Arithmetic operations
    if op_name == "Add":
        return lhs + rhs
    if op_name == "Sub":
        return lhs - rhs
    if op_name == "Mul":
        return lhs * rhs
    if op_name == "Div":
        return lhs / rhs
    if op_name == "Pow":
        return lhs**rhs
    if op_name == "Exp":
        return lhs**rhs
    if op_name == "ElementMul":
        return lhs * rhs  # Element-wise = regular for scalars
    if op_name == "ElementDiv":
        return lhs / rhs
    if op_name == "ElementPow":
        return lhs**rhs

    # Comparison operations
    if op_name == "Eq":
        return sp.Eq(lhs, rhs)
    if op_name == "Neq":
        return sp.Ne(lhs, rhs)
    if op_name == "Lt":
        return sp.Lt(lhs, rhs)
    if op_name == "Le":
        return sp.Le(lhs, rhs)
    if op_name == "Gt":
        return sp.Gt(lhs, rhs)
    if op_name == "Ge":
        return sp.Ge(lhs, rhs)

    # Logical operations
    if op_name == "And":
        return sp.And(lhs, rhs)
    if op_name == "Or":
        return sp.Or(lhs, rhs)

    raise ValueError(f"Unknown binary operator: {op_name}")


def _parse_unary(
    unary: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a Unary operation AST node."""
    op_dict = unary.get("op", {})
    # Rumoca uses "rhs" for the operand, but some formats use "operand"
    operand_ast = unary.get("rhs") or unary.get("operand")

    operand = ast_to_sympy(operand_ast, symbols, create_missing)

    op_name = list(op_dict.keys())[0] if op_dict else ""

    if op_name in ("Neg", "Minus"):
        return -operand
    if op_name in ("Not", "LogicalNot"):
        return sp.Not(operand)
    if op_name == "Plus":
        return operand

    raise ValueError(f"Unknown unary operator: {op_name}")


def _parse_function_call(
    func_call: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a FunctionCall AST node."""
    comp = func_call.get("comp", {})
    args_ast = func_call.get("args", [])

    # Get function name
    parts = comp.get("parts", [])
    if not parts:
        raise ValueError("FunctionCall has no function name")

    func_name_parts = [p.get("ident", {}).get("text", "") for p in parts]
    func_name = ".".join(func_name_parts)

    # Parse arguments
    args = [ast_to_sympy(arg, symbols, create_missing) for arg in args_ast]

    # Handle built-in Modelica functions
    return _apply_builtin_function(func_name, args, symbols)


def _apply_builtin_function(
    func_name: str,
    args: List[sp.Expr],
    symbols: Dict[str, sp.Symbol],
) -> sp.Expr:
    """
    Apply a built-in Modelica function.

    Maps Modelica standard functions to SymPy equivalents.
    """
    # Derivative operator (special handling)
    if func_name == "der":
        if len(args) != 1:
            raise ValueError(f"der() expects 1 argument, got {len(args)}")
        # Return a custom derivative marker
        return sp.Function("der")(*args)

    # Trigonometric functions
    if func_name == "sin":
        return sp.sin(args[0])
    if func_name == "cos":
        return sp.cos(args[0])
    if func_name == "tan":
        return sp.tan(args[0])
    if func_name == "asin":
        return sp.asin(args[0])
    if func_name == "acos":
        return sp.acos(args[0])
    if func_name == "atan":
        return sp.atan(args[0])
    if func_name == "atan2":
        return sp.atan2(args[0], args[1])
    if func_name == "sinh":
        return sp.sinh(args[0])
    if func_name == "cosh":
        return sp.cosh(args[0])
    if func_name == "tanh":
        return sp.tanh(args[0])

    # Exponential and logarithmic
    if func_name == "exp":
        return sp.exp(args[0])
    if func_name == "log":
        return sp.log(args[0])
    if func_name == "log10":
        return sp.log(args[0], 10)

    # Power and root
    if func_name == "sqrt":
        return sp.sqrt(args[0])
    if func_name == "abs":
        return sp.Abs(args[0])
    if func_name == "sign":
        return sp.sign(args[0])

    # Rounding
    if func_name == "floor":
        return sp.floor(args[0])
    if func_name == "ceil":
        return sp.ceiling(args[0])

    # Min/max
    if func_name == "min":
        return sp.Min(*args)
    if func_name == "max":
        return sp.Max(*args)

    # Modelica-specific
    if func_name == "noEvent":
        # noEvent just passes through the argument for symbolic analysis
        return args[0] if args else sp.S.Zero
    if func_name == "smooth":
        # smooth(order, expr) - return the expression for symbolic analysis
        return args[1] if len(args) > 1 else args[0]
    if func_name == "pre":
        # pre(x) - previous value, represent as a function
        return sp.Function("pre")(*args)

    # Linear algebra (for arrays)
    if func_name == "transpose":
        return sp.Function("transpose")(*args)
    if func_name == "identity":
        return sp.Function("identity")(*args)
    if func_name == "zeros":
        return sp.Function("zeros")(*args)
    if func_name == "ones":
        return sp.Function("ones")(*args)
    if func_name == "diagonal":
        return sp.Function("diagonal")(*args)
    if func_name == "cross":
        return sp.Function("cross")(*args)

    # Unknown function - create a generic SymPy function
    return sp.Function(func_name)(*args)


def _parse_if_expression(
    if_expr: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse an If expression AST node."""
    cond_blocks = if_expr.get("cond_blocks", [])
    else_block = if_expr.get("else_block", [])

    # Build a Piecewise expression
    pieces = []

    for block in cond_blocks:
        cond_ast = block.get("cond")
        expr_ast = block.get("equations", [])

        cond = ast_to_sympy(cond_ast, symbols, create_missing)

        # For if-expressions (not equations), there's usually a single expression
        if expr_ast:
            expr = ast_to_sympy(expr_ast[0], symbols, create_missing)
        else:
            expr = sp.S.Zero

        pieces.append((expr, cond))

    # Add else clause
    if else_block:
        else_expr = ast_to_sympy(else_block[0], symbols, create_missing)
        pieces.append((else_expr, True))

    if not pieces:
        return sp.S.Zero

    return sp.Piecewise(*pieces)


def _parse_array(
    array: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse an Array expression AST node."""
    elements = array.get("elements", [])
    parsed = [ast_to_sympy(el, symbols, create_missing) for el in elements]

    # Return as a SymPy Matrix for vectors/matrices
    if parsed:
        return sp.Matrix(parsed)
    return sp.Matrix([])


def _parse_range(
    range_expr: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a Range expression AST node (start:step:stop)."""
    start_ast = range_expr.get("start")
    step_ast = range_expr.get("step")
    stop_ast = range_expr.get("stop")

    start = ast_to_sympy(start_ast, symbols, create_missing)
    stop = ast_to_sympy(stop_ast, symbols, create_missing)

    if step_ast:
        step = ast_to_sympy(step_ast, symbols, create_missing)
        return sp.Function("range")(start, step, stop)

    return sp.Function("range")(start, stop)


def parse_equation(
    eq_ast: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool = True,
) -> Tuple[Optional[str], sp.Expr]:
    """
    Parse an equation AST node.

    For differential equations (der(x) = rhs), returns (state_name, rhs_expr).
    For algebraic equations (lhs = rhs), returns (None, Eq(lhs, rhs)).

    Parameters
    ----------
    eq_ast : dict
        Equation AST node
    symbols : dict[str, Symbol]
        Symbol table
    create_missing : bool
        Whether to create symbols for unknown references

    Returns
    -------
    tuple[str or None, Expr]
        (state_name, rhs) for differential equations
        (None, Eq(lhs, rhs)) for algebraic equations

    Examples
    --------
    >>> eq_ast = {"Simple": {"lhs": der(x), "rhs": -k*x}}
    >>> state, rhs = parse_equation(eq_ast, symbols)
    >>> state
    'x'
    >>> rhs
    -k*x
    """
    if "Simple" not in eq_ast:
        # Other equation types (For, If, When, Connect)
        # Return as None for now - they need special handling
        return (None, sp.S.Zero)

    simple = eq_ast["Simple"]
    lhs_ast = simple.get("lhs")
    rhs_ast = simple.get("rhs")

    lhs = ast_to_sympy(lhs_ast, symbols, create_missing)
    rhs = ast_to_sympy(rhs_ast, symbols, create_missing)

    # Check if lhs is der(something)
    if hasattr(lhs, "func") and str(lhs.func) == "der" and lhs.args:
        state_sym = lhs.args[0]
        return (str(state_sym), rhs)

    # Algebraic equation
    return (None, sp.Eq(lhs, rhs))


def parse_all_equations(
    equations: List[Dict[str, Any]],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool = True,
) -> Tuple[Dict[str, sp.Expr], List[sp.Eq]]:
    """
    Parse all equations from a DAE IR.

    Separates differential equations (der(x) = f) from algebraic equations.

    Parameters
    ----------
    equations : list[dict]
        List of equation AST nodes
    symbols : dict[str, Symbol]
        Symbol table
    create_missing : bool
        Whether to create symbols for unknown references

    Returns
    -------
    tuple[dict[str, Expr], list[Eq]]
        (derivatives, algebraic_eqs) where:
        - derivatives: mapping from state name to its derivative expression
        - algebraic_eqs: list of algebraic equations as SymPy Eq objects
    """
    derivatives = {}
    algebraic_eqs = []

    for eq_ast in equations:
        state_name, expr = parse_equation(eq_ast, symbols, create_missing)

        if state_name is not None:
            derivatives[state_name] = expr
        elif isinstance(expr, sp.Eq):
            algebraic_eqs.append(expr)

    return derivatives, algebraic_eqs
