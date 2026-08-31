"""Command-line interface for Rumoca 0.10 Modelica analysis."""

from __future__ import annotations

import argparse
import sys


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modelica",
        required=True,
        help="Path to the Modelica source file",
    )
    parser.add_argument(
        "--model",
        help="Qualified model name when the source contains multiple models",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Additional Modelica package root (repeatable)",
    )
    parser.add_argument(
        "--workspace",
        help="Rumoca workspace directory",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cp_reach",
        description="CP Reach: certified reachability analysis for Modelica systems",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    analyze_parser = subparsers.add_parser("analyze", help="Run reachability analysis")
    _add_model_arguments(analyze_parser)
    analyze_parser.add_argument("--uncertainty", help="Path to uncertainty YAML")
    analyze_parser.add_argument("--query", help="Path to reachability query YAML")
    analyze_parser.add_argument(
        "--output",
        "-o",
        default="results/",
        help="Output directory (default: results/)",
    )
    analyze_parser.add_argument("--verbose", "-v", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate", help="Compile the model and validate its configuration"
    )
    _add_model_arguments(validate_parser)
    validate_parser.add_argument("--uncertainty", help="Path to uncertainty YAML")
    validate_parser.add_argument("--query", help="Path to reachability query YAML")

    info_parser = subparsers.add_parser("info", help="Display compiled model information")
    _add_model_arguments(info_parser)
    return parser


def _load_model(args, output_names=None):
    from cp_reach.ir import modelica_load

    return modelica_load(
        args.modelica,
        model_name=args.model,
        roots=args.root or None,
        workspace=args.workspace,
        output_names=output_names,
    )


def _compiled_name(model, fallback: str) -> str:
    return getattr(model.rumoca, "name", None) or fallback


def cmd_analyze(args) -> int:
    """Run reachability analysis."""
    from cp_reach.reachability.workflows import analyze

    try:
        if args.verbose:
            print(f"Compiling Modelica source: {args.modelica}")
            if args.model:
                print(f"Model: {args.model}")
            if args.uncertainty:
                print(f"Uncertainty: {args.uncertainty}")
            if args.query:
                print(f"Query: {args.query}")

        result = analyze(
            modelica_path=args.modelica,
            model_name=args.model,
            roots=args.root or None,
            workspace=args.workspace,
            uncertainty_path=args.uncertainty,
            query_path=args.query,
            output_dir=args.output,
        )
        print(f"\nAnalysis complete for model: {result.get('model_name', 'Unknown')}")
        print(f"  Status: {result.get('status', 'unknown')}")
        if "alpha" in result:
            print(f"  Alpha (decay rate): {result['alpha']:.4f}")
        if "mu" in result:
            mu = result["mu"]
            value = mu[0] if hasattr(mu, "__iter__") else mu
            print(f"  Mu (magnification): {value:.4f}")
        if "bounds_upper" in result:
            print(f"  Upper bounds: {result['bounds_upper']}")
        print(f"\nResults saved to: {args.output}")
        return 0
    except Exception as exc:
        print(f"Error during analysis: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_validate(args) -> int:
    """Compile a Modelica model and validate configuration names."""
    from cp_reach.config.query import ReachQuery
    from cp_reach.config.uncertainty import UncertaintySpec

    errors = []
    notices = []
    try:
        uncertainty = (
            UncertaintySpec.from_yaml(args.uncertainty) if args.uncertainty else UncertaintySpec()
        )
        query = ReachQuery.from_yaml(args.query) if args.query else ReachQuery()
        model = _load_model(args, output_names=query.outputs or None)
        notices.extend(uncertainty.validate_against_model(model))
        notices.extend(query.validate_against_model(model))

        print(f"Modelica source: {args.modelica}")
        print(f"  Model: {_compiled_name(model, args.model or args.modelica)}")
        print(f"  States: {len(model.states)}")
        print(f"  Inputs: {len(model.inputs)}")
        print(f"  Parameters: {len(model.parameters)}")
        print(f"  Algebraics: {len(model.export.algebraic_names)}")
    except Exception as exc:
        errors.append(str(exc))

    if notices:
        print("\nWarnings:")
        for message in notices:
            print(f"  - {message}")
    if errors:
        print("\nErrors:")
        for message in errors:
            print(f"  - {message}")
        return 1

    print("\nValidation passed!")
    return 0


def cmd_info(args) -> int:
    """Display information from Rumoca's checked Solve export."""
    try:
        model = _load_model(args)
        export = model.export
        print(f"Model: {_compiled_name(model, args.model or args.modelica)}")
        print("Rumoca: 0.10.x")

        print("\nStates:")
        for name, default in zip(export.state_names, export.default_states):
            print(f"  {name} = {default}")

        print("\nInputs:")
        for name in export.input_names:
            print(f"  {name}")

        print("\nParameters:")
        for name, default in model.parameters.items():
            print(f"  {name} = {default}")

        if export.algebraic_names:
            print("\nAlgebraic variables:")
            for name in export.algebraic_names:
                print(f"  {name}")

        if export.output_names:
            print("\nOutputs:")
            for name in export.output_names:
                print(f"  {name}")

        print(f"\nExplicit ODEs: {export.rhs.rows}")
        return 0
    except Exception as exc:
        print(f"Error compiling model: {exc}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    """Run the requested command."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "info":
        return cmd_info(args)
    parser.print_help()
    return 1
