"""
CP_Reach Command Line Interface.

This module provides a CLI for running reachability analysis using the
structured IR + YAML workflow.

Usage:
    cp_reach analyze --ir model.json --uncertainty uncertainty.yaml --query query.yaml
    cp_reach validate --ir model.json
    cp_reach info --ir model.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cp_reach",
        description="CP_Reach: Certified reachability analysis for control systems",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run reachability analysis",
    )
    analyze_parser.add_argument(
        "--ir",
        required=True,
        help="Path to Rumoca DAE IR JSON file",
    )
    analyze_parser.add_argument(
        "--uncertainty",
        help="Path to uncertainty YAML file",
    )
    analyze_parser.add_argument(
        "--query",
        help="Path to query YAML file",
    )
    analyze_parser.add_argument(
        "--output", "-o",
        default="results/",
        help="Output directory for results (default: results/)",
    )
    analyze_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate IR and configuration files",
    )
    validate_parser.add_argument(
        "--ir",
        required=True,
        help="Path to Rumoca DAE IR JSON file",
    )
    validate_parser.add_argument(
        "--uncertainty",
        help="Path to uncertainty YAML file",
    )
    validate_parser.add_argument(
        "--query",
        help="Path to query YAML file",
    )

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display model information",
    )
    info_parser.add_argument(
        "--ir",
        required=True,
        help="Path to Rumoca DAE IR JSON file",
    )

    return parser


def cmd_analyze(args) -> int:
    """Run reachability analysis."""
    from cp_reach.reachability.workflows import analyze

    try:
        if args.verbose:
            print(f"Loading IR from: {args.ir}")
            if args.uncertainty:
                print(f"Loading uncertainty from: {args.uncertainty}")
            if args.query:
                print(f"Loading query from: {args.query}")
            print(f"Output directory: {args.output}")

        result = analyze(
            ir_path=args.ir,
            uncertainty_path=args.uncertainty,
            query_path=args.query,
            output_dir=args.output,
        )

        # Print summary
        print(f"\nAnalysis complete for model: {result.get('model_name', 'Unknown')}")
        print(f"  Status: {result.get('status', 'unknown')}")

        if "alpha" in result:
            print(f"  Alpha (decay rate): {result['alpha']:.4f}")

        if "mu" in result:
            mu = result["mu"]
            if hasattr(mu, "__iter__"):
                print(f"  Mu (magnification): {mu[0]:.4f}")
            else:
                print(f"  Mu (magnification): {mu:.4f}")

        if "bounds_upper" in result:
            bounds = result["bounds_upper"]
            print(f"  Upper bounds: {bounds}")

        if "radius_inf" in result:
            print(f"  Radius (inf-norm): {result['radius_inf']:.4f}")

        print(f"\nResults saved to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_validate(args) -> int:
    """Validate IR and configuration files."""
    from cp_reach.ir.loader import DaeIR
    from cp_reach.config.uncertainty import UncertaintySpec
    from cp_reach.config.query import ReachQuery

    errors = []
    warnings = []

    # Validate IR
    try:
        ir = DaeIR.from_json(args.ir)
        print(f"IR file: {args.ir}")
        print(f"  Model: {ir.model_name}")
        print(f"  States: {ir.n_states()}")
        print(f"  Inputs: {ir.n_inputs()}")
        print(f"  Parameters: {ir.n_parameters()}")
        print(f"  Equations: {len(ir.equations)}")
    except Exception as e:
        errors.append(f"Failed to load IR: {e}")
        ir = None

    # Validate uncertainty
    if args.uncertainty and ir:
        try:
            unc = UncertaintySpec.from_yaml(args.uncertainty)
            print(f"\nUncertainty file: {args.uncertainty}")
            print(f"  Disturbances: {list(unc.disturbances.keys())}")
            print(f"  Parameters: {list(unc.parameters.keys())}")

            unc_warnings = unc.validate_against_ir(ir)
            warnings.extend(unc_warnings)
        except Exception as e:
            errors.append(f"Failed to load uncertainty: {e}")

    # Validate query
    if args.query and ir:
        try:
            query = ReachQuery.from_yaml(args.query)
            print(f"\nQuery file: {args.query}")
            print(f"  Type: {query.type}")
            print(f"  Dynamics: {query.dynamics}")
            print(f"  Outputs: {query.outputs}")

            query_warnings = query.validate_against_ir(ir)
            warnings.extend(query_warnings)
        except Exception as e:
            errors.append(f"Failed to load query: {e}")

    # Report results
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nValidation passed!")
    return 0


def cmd_info(args) -> int:
    """Display model information."""
    from cp_reach.ir.loader import DaeIR

    try:
        ir = DaeIR.from_json(args.ir)

        print(f"Model: {ir.model_name}")
        print(f"Rumoca version: {ir.rumoca_version}")
        print()

        print("States:")
        for name, comp in ir.states.items():
            start_str = f" = {comp.start}" if comp.start is not None else ""
            print(f"  {name}: {comp.type_name}{start_str}")

        print("\nInputs:")
        for name, comp in ir.inputs.items():
            print(f"  {name}: {comp.type_name}")

        print("\nParameters:")
        for name, comp in ir.parameters.items():
            start_str = f" = {comp.start}" if comp.start is not None else ""
            print(f"  {name}: {comp.type_name}{start_str}")

        if ir.algebraics:
            print("\nAlgebraic variables:")
            for name, comp in ir.algebraics.items():
                print(f"  {name}: {comp.type_name}")

        print(f"\nEquations: {len(ir.equations)}")

        # Infer roles
        roles = ir.infer_roles()
        if roles:
            unique_roles = set(roles.values())
            print(f"\nInferred component roles: {unique_roles}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading IR: {e}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
