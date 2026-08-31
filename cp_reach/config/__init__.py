"""
CP_Reach Configuration Module - YAML-based uncertainty and query specifications.

This module provides parsers for the user-facing configuration files that
specify uncertainty models and reachability queries.

Example usage:
    from cp_reach.config import UncertaintySpec, ReachQuery

    # Load uncertainty specification
    unc = UncertaintySpec.from_yaml("uncertainty.yaml")

    # Load reachability query
    query = ReachQuery.from_yaml("reach_query.yaml")

    # Pass these YAML paths to cp_reach.reachability.analyze().
"""

from cp_reach.config.uncertainty import (
    UncertaintySpec,
    BoundedDisturbance,
    GaussianDisturbance,
    ParameterUncertainty,
    InitialCondition,
)
from cp_reach.config.query import ReachQuery, AlphaSearch, OutputFormat

__all__ = [
    # Uncertainty types
    "UncertaintySpec",
    "BoundedDisturbance",
    "GaussianDisturbance",
    "ParameterUncertainty",
    "InitialCondition",
    # Query types
    "ReachQuery",
    "AlphaSearch",
    "OutputFormat",
]
