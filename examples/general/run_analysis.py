#!/usr/bin/env python3
"""
Example: Running reachability analysis with the structured workflow.

This script demonstrates the end-to-end workflow:
1. Compile Modelica model with Rumoca to get DAE IR JSON
2. Load uncertainty and query specifications from YAML
3. Run reachability analysis
4. Display and save results

Prerequisites:
- rumoca installed and in PATH (or use Python bindings)
- cp_reach installed

Usage:
    python run_analysis.py
"""

import subprocess
import sys
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "results"


def compile_model_with_rumoca():
    """Compile the closed-loop Modelica model with Rumoca."""
    model_file = MODELS_DIR / "closed_loop.mo"
    ir_file = SCRIPT_DIR / "closed_loop.json"

    print(f"Compiling {model_file}...")

    try:
        # Try using rumoca CLI
        result = subprocess.run(
            ["rumoca", "--json", "-m", "ClosedLoop", str(model_file)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Save JSON output
        with open(ir_file, "w") as f:
            f.write(result.stdout)

        print(f"  -> Saved IR to {ir_file}")
        return ir_file

    except FileNotFoundError:
        print("  rumoca CLI not found, trying Python bindings...")

        try:
            import rumoca

            result = rumoca.compile(str(model_file))
            json_str = result.to_base_modelica_json()

            with open(ir_file, "w") as f:
                f.write(json_str)

            print(f"  -> Saved IR to {ir_file}")
            return ir_file

        except ImportError:
            print("ERROR: Neither rumoca CLI nor Python bindings available.")
            print("Install rumoca with: pip install rumoca")
            print("Or build from source: cargo install rumoca")
            sys.exit(1)


def run_analysis():
    """Run reachability analysis."""
    from cp_reach.reachability.workflows import analyze

    ir_file = SCRIPT_DIR / "closed_loop.json"
    uncertainty_file = SCRIPT_DIR / "uncertainty.yaml"
    query_file = SCRIPT_DIR / "reach_query.yaml"

    print("\nRunning reachability analysis...")
    print(f"  IR: {ir_file}")
    print(f"  Uncertainty: {uncertainty_file}")
    print(f"  Query: {query_file}")

    result = analyze(
        ir_path=str(ir_file),
        uncertainty_path=str(uncertainty_file),
        query_path=str(query_file),
        output_dir=str(OUTPUT_DIR),
    )

    return result


def display_results(result):
    """Display analysis results."""
    print("\n" + "=" * 60)
    print("REACHABILITY ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nModel: {result.get('model_name', 'Unknown')}")
    print(f"Query type: {result.get('query_type', 'Unknown')}")
    print(f"Dynamics: {result.get('dynamics', 'Unknown')}")
    print(f"Status: {result.get('status', 'Unknown')}")

    if "alpha" in result:
        print(f"\nOptimal alpha (decay rate): {result['alpha']:.6f}")

    if "mu" in result:
        mu = result["mu"]
        if hasattr(mu, "__iter__"):
            print(f"Mu (disturbance magnification): {mu[0]:.6f}")
        else:
            print(f"Mu (disturbance magnification): {mu:.6f}")

    if "bounds_upper" in result:
        print("\nState bounds (for unit disturbance):")
        bounds = result["bounds_upper"]
        for i, b in enumerate(bounds):
            print(f"  State {i}: +/- {b:.6f}")

    if "radius_inf" in result:
        print(f"\nInf-norm radius: {result['radius_inf']:.6f}")

    print(f"\nResults saved to: {OUTPUT_DIR}/")


def main():
    """Main entry point."""
    print("CP_Reach Structured Workflow Example")
    print("-" * 40)

    # Step 1: Compile model (if IR doesn't exist)
    ir_file = SCRIPT_DIR / "closed_loop.json"
    if not ir_file.exists():
        compile_model_with_rumoca()
    else:
        print(f"Using existing IR: {ir_file}")

    # Step 2: Run analysis
    try:
        result = run_analysis()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure the IR file exists. Run compile_model_with_rumoca() first.")
        return 1

    # Step 3: Display results
    display_results(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
