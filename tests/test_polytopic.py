"""Tests for the explicit multi-state polytopic Jacobian contract."""

import numpy as np
import pytest
import sympy as sp

from cp_reach.reachability.polytopic import polytopic_jacobians


def test_polytopic_jacobians_requires_named_state_mappings():
    theta = sp.Symbol("theta")
    jacobian = sp.Matrix([[0, 1], [-sp.cos(theta), -1]])

    vertices = polytopic_jacobians(
        jacobian,
        {theta: np.array([0.0, 0.5])},
        {theta: 0.1},
    )

    assert len(vertices) == 2
    assert all(len(polytope) == 2 for polytope in vertices)


def test_polytopic_jacobians_rejects_mismatched_state_mappings():
    theta, omega = sp.symbols("theta omega")

    with pytest.raises(ValueError, match="same state symbols"):
        polytopic_jacobians(
            sp.eye(2),
            {theta: np.array([0.0])},
            {omega: 0.1},
        )
