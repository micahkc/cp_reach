"""Convert RuMoCA 0.9.20 DAE schema-7 expressions to SymPy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import sympy as sp


def ast_to_sympy(
    expr_ast: Any,
    symbols: Dict[str, sp.Symbol],
    create_missing: bool = True,
) -> sp.Expr:
    """Convert one RuMoCA schema-7 expression node to a SymPy expression."""
    if expr_ast is None:
        return sp.S.Zero
    if not isinstance(expr_ast, dict):
        raise ValueError(f"Expected a RuMoCA expression object, got {expr_ast!r}")

    if "Literal" in expr_ast:
        return _parse_literal(expr_ast["Literal"])
    if "VarRef" in expr_ast:
        return _parse_varref(expr_ast["VarRef"], symbols, create_missing)
    if "BuiltinCall" in expr_ast:
        return _parse_builtin_call(expr_ast["BuiltinCall"], symbols, create_missing)
    if "Binary" in expr_ast:
        return _parse_binary(expr_ast["Binary"], symbols, create_missing)
    if "Unary" in expr_ast:
        return _parse_unary(expr_ast["Unary"], symbols, create_missing)
    if "If" in expr_ast:
        return _parse_if_expression(expr_ast["If"], symbols, create_missing)
    if "Array" in expr_ast:
        return _parse_array(expr_ast["Array"], symbols, create_missing)
    if "Range" in expr_ast:
        return _parse_range(expr_ast["Range"], symbols, create_missing)
    if "Parenthesized" in expr_ast:
        return ast_to_sympy(expr_ast["Parenthesized"]["expr"], symbols, create_missing)

    raise ValueError(f"Unsupported RuMoCA schema-7 node: {list(expr_ast)}")


def _parse_literal(literal: Dict[str, Any]) -> sp.Expr:
    """Parse a schema-7 literal and ignore its source-span metadata."""
    value = literal["value"]
    if "Real" in value:
        return sp.Float(value["Real"])
    if "Integer" in value:
        return sp.Integer(value["Integer"])
    if "Boolean" in value:
        return sp.S.true if value["Boolean"] else sp.S.false
    if "String" in value:
        return sp.Symbol(f"'{value['String']}'")
    raise ValueError(f"Unsupported RuMoCA literal: {value!r}")


def _parse_varref(
    varref: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a schema-7 resolved variable reference."""
    name = varref["name"]["name"]
    if name in symbols:
        symbol: sp.Expr = symbols[name]
    elif create_missing:
        symbol = sp.Symbol(name)
        symbols[name] = symbol
    else:
        raise KeyError(f"Unknown symbol: {name}")

    subscripts = varref.get("subscripts", [])
    if subscripts:
        indices = [ast_to_sympy(item, symbols, create_missing) for item in subscripts]
        symbol = sp.Indexed(symbol, *indices)
    return symbol


def _parse_builtin_call(
    builtin: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a schema-7 Modelica built-in call."""
    function = builtin["function"]
    args = [ast_to_sympy(arg, symbols, create_missing) for arg in builtin.get("args", [])]

    if function == "Der":
        if len(args) != 1:
            raise ValueError(f"der() expects 1 argument, got {len(args)}")
        return sp.Function("der")(*args)

    unary_functions = {
        "Abs": sp.Abs,
        "Acos": sp.acos,
        "Asin": sp.asin,
        "Atan": sp.atan,
        "Ceil": sp.ceiling,
        "Cos": sp.cos,
        "Cosh": sp.cosh,
        "Exp": sp.exp,
        "Floor": sp.floor,
        "Log": sp.log,
        "Sign": sp.sign,
        "Sin": sp.sin,
        "Sinh": sp.sinh,
        "Sqrt": sp.sqrt,
        "Tan": sp.tan,
        "Tanh": sp.tanh,
    }
    if function in unary_functions:
        return unary_functions[function](args[0])
    if function == "Atan2":
        return sp.atan2(args[0], args[1])
    if function == "Min":
        return sp.Min(*args)
    if function == "Max":
        return sp.Max(*args)
    if function == "NoEvent":
        return args[0]
    if function == "Smooth":
        return args[1]

    return sp.Function(function)(*args)


def _parse_binary(
    binary: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a schema-7 binary expression."""
    lhs = ast_to_sympy(binary["lhs"], symbols, create_missing)
    rhs = ast_to_sympy(binary["rhs"], symbols, create_missing)
    operator = binary["op"]

    operations = {
        "Add": lambda: lhs + rhs,
        "Sub": lambda: lhs - rhs,
        "Mul": lambda: lhs * rhs,
        "Div": lambda: lhs / rhs,
        "Pow": lambda: lhs**rhs,
        "Exp": lambda: lhs**rhs,
        "ElementMul": lambda: lhs * rhs,
        "ElementDiv": lambda: lhs / rhs,
        "ElementPow": lambda: lhs**rhs,
        "Eq": lambda: sp.Eq(lhs, rhs),
        "Neq": lambda: sp.Ne(lhs, rhs),
        "Lt": lambda: sp.Lt(lhs, rhs),
        "Le": lambda: sp.Le(lhs, rhs),
        "Gt": lambda: sp.Gt(lhs, rhs),
        "Ge": lambda: sp.Ge(lhs, rhs),
        "And": lambda: sp.And(lhs, rhs),
        "Or": lambda: sp.Or(lhs, rhs),
    }
    if operator not in operations:
        raise ValueError(f"Unsupported RuMoCA binary operator: {operator}")
    return operations[operator]()


def _parse_unary(
    unary: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a schema-7 unary expression."""
    operand = ast_to_sympy(unary["rhs"], symbols, create_missing)
    operator = unary["op"]
    if operator == "Minus":
        return -operand
    if operator == "Plus":
        return operand
    if operator == "Not":
        return sp.Not(operand)
    raise ValueError(f"Unsupported RuMoCA unary operator: {operator}")


def _parse_if_expression(
    if_expr: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a conditional expression."""
    pieces = []
    for block in if_expr.get("cond_blocks", []):
        condition = ast_to_sympy(block["cond"], symbols, create_missing)
        expression = ast_to_sympy(block["equations"][0], symbols, create_missing)
        pieces.append((expression, condition))
    else_block = if_expr.get("else_block", [])
    if else_block:
        pieces.append((ast_to_sympy(else_block[0], symbols, create_missing), True))
    return sp.Piecewise(*pieces)


def _parse_array(
    array: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse an array expression."""
    return sp.Matrix(
        [ast_to_sympy(element, symbols, create_missing) for element in array["elements"]]
    )


def _parse_range(
    range_expr: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool,
) -> sp.Expr:
    """Parse a start[:step]:stop range expression."""
    start = ast_to_sympy(range_expr["start"], symbols, create_missing)
    stop = ast_to_sympy(range_expr["stop"], symbols, create_missing)
    if range_expr.get("step") is not None:
        step = ast_to_sympy(range_expr["step"], symbols, create_missing)
        return sp.Function("range")(start, step, stop)
    return sp.Function("range")(start, stop)


def parse_equation(
    equation: Dict[str, Any],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool = True,
) -> Tuple[Optional[str], sp.Expr]:
    """Parse a schema-7 residual equation into differential or algebraic form."""
    if "rhs" not in equation:
        raise ValueError(f"Unsupported RuMoCA schema-7 equation: {list(equation)}")
    residual = ast_to_sympy(equation["rhs"], symbols, create_missing)
    return _parse_residual_equation(residual)


def _parse_residual_equation(residual: sp.Expr) -> Tuple[Optional[str], sp.Expr]:
    """Convert ``der(x) - expression = 0`` to ``(x, expression)``."""
    expanded = sp.expand(residual)
    derivative_terms: Dict[str, sp.Expr] = {}
    remaining = sp.S.Zero

    terms = expanded.args if isinstance(expanded, sp.Add) else (expanded,)
    for term in terms:
        derivative = _find_der_in_term(term)
        if derivative is None:
            remaining += term
            continue
        state_name, coefficient = derivative
        derivative_terms[state_name] = derivative_terms.get(state_name, sp.S.Zero) + coefficient

    if len(derivative_terms) == 1:
        state_name, coefficient = next(iter(derivative_terms.items()))
        return state_name, sp.simplify(-remaining / coefficient)

    positive = sp.S.Zero
    negative = sp.S.Zero
    for term in terms:
        if term.could_extract_minus_sign():
            negative += -term
        else:
            positive += term
    return None, sp.Eq(positive, negative)


def _find_der_in_term(term: sp.Expr) -> Optional[Tuple[str, sp.Expr]]:
    """Find a derivative marker and return its state name and coefficient."""
    if hasattr(term, "func") and str(term.func) == "der" and term.args:
        return str(term.args[0]), sp.S.One
    if isinstance(term, sp.Mul):
        for factor in term.args:
            if hasattr(factor, "func") and str(factor.func) == "der" and factor.args:
                return str(factor.args[0]), term / factor
    return None


def parse_all_equations(
    equations: List[Dict[str, Any]],
    symbols: Dict[str, sp.Symbol],
    create_missing: bool = True,
) -> Tuple[Dict[str, sp.Expr], List[sp.Eq]]:
    """Separate schema-7 equations into derivatives and algebraic equations."""
    derivatives: Dict[str, sp.Expr] = {}
    algebraic_equations = []
    for equation in equations:
        state_name, expression = parse_equation(equation, symbols, create_missing)
        if state_name is not None:
            derivatives[state_name] = expression
        elif isinstance(expression, sp.Eq):
            algebraic_equations.append(expression)
    return derivatives, algebraic_equations
