"""
CP_REACH IR module: load RuMoCA 0.9.20 DAE schema 7.

This module converts current RuMoCA DAE JSON to CP_REACH symbolic state-space
representations.

Example usage:
    import tempfile
    from pathlib import Path

    import rumoca
    from cp_reach.ir import DaeIR, ir_to_symbolic_statespace

    # Compile Modelica and load IR
    source_path = Path("closed_loop.mo").resolve()
    with tempfile.TemporaryDirectory() as workspace:
        model = rumoca.Session(roots=[], workspace=workspace).loads(
            source_path.read_text(), model="ClosedLoop", filename=source_path.name
        )
        dae_json = model.to_json("dae")
    ir = DaeIR.from_json_str(dae_json, model_name="ClosedLoop")

    # Convert to SymbolicStateSpace for reachability analysis
    ss = ir_to_symbolic_statespace(ir)

    # Or load from a pre-compiled JSON file
    ir = DaeIR.from_json("closed_loop.json")
    ss = ir_to_symbolic_statespace(ir)
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
