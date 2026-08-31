"""Tests for supported RuMoCA expression schemas."""

import sympy as sp

from cp_reach.ir.ast_parser import ast_to_sympy


def _schema_7_varref(name):
    return {
        "VarRef": {
            "name": {
                "name": name,
                "component_ref": {"parts": [{"ident": name, "subs": []}]},
            },
            "subscripts": [],
        }
    }


def test_schema_7_varref_and_string_binary_operator():
    """Schema 7 uses resolved-name objects and string operator tags."""
    x = sp.Symbol("x")
    ast = {
        "Binary": {
            "op": "Add",
            "lhs": _schema_7_varref("x"),
            "rhs": {"Literal": {"value": {"Real": 1.5}}},
        }
    }

    assert ast_to_sympy(ast, {"x": x}) == x + sp.Float(1.5)


def test_schema_7_string_unary_operator():
    """Schema 7 represents unary operators as strings."""
    ast = {
        "Unary": {
            "op": "Minus",
            "rhs": {"Literal": {"value": {"Integer": 2}}},
        }
    }

    assert ast_to_sympy(ast, {}) == -2
