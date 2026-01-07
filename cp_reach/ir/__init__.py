"""
CP_Reach IR Module - Load and convert Rumoca DAE JSON to SymbolicStateSpace.

This module provides a direct path from Rumoca's DAE IR JSON format to
CP_Reach's symbolic representations, bypassing cyecca for simpler deployments.

Example usage:
    from cp_reach.ir import DaeIR, ir_to_symbolic_statespace

    # Load IR from JSON file
    ir = DaeIR.from_json("closed_loop.json")

    # Convert to SymbolicStateSpace for reachability analysis
    ss = ir_to_symbolic_statespace(ir)

    # Use with existing cp_reach workflows
    result = compute_reachable_set(ss, method="lmi", dynamics="error")
"""

from cp_reach.ir.loader import DaeIR, Component
from cp_reach.ir.ast_parser import ast_to_sympy, parse_equation
from cp_reach.ir.state_space import ir_to_symbolic_statespace, ir_load

__all__ = [
    "DaeIR",
    "Component",
    "ast_to_sympy",
    "parse_equation",
    "ir_to_symbolic_statespace",
    "ir_load",
]
