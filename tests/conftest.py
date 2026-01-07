"""
Pytest configuration and shared fixtures for cp_reach tests.
"""

import numpy as np
import pytest


@pytest.fixture
def stable_2x2_system():
    """A simple stable 2x2 linear system for testing LMI solvers."""
    # Mass-spring-damper: m=1, c=0.5, k=2
    # State: [position, velocity]
    # ẋ = [0, 1; -2, -0.5] x + [0; 1] d
    A = np.array([[0.0, 1.0], [-2.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    return A, B


@pytest.fixture
def stable_4x4_system():
    """A stable 4x4 system representing plant + controller for error dynamics testing."""
    # Simple stable 4x4 diagonal-dominant system
    # All eigenvalues are negative (stable)
    A = np.array([
        [-2.0, 0.5, 0.0, 0.0],
        [0.0, -1.5, 0.3, 0.0],
        [0.0, 0.0, -1.0, 0.2],
        [0.0, 0.0, 0.0, -0.8],
    ])
    B = np.array([[1.0], [0.0], [0.0], [0.0]])
    C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    D = np.zeros((2, 1))
    return A, B, C, D


@pytest.fixture
def polytopic_vertices():
    """A set of polytopic system vertices for time-varying LMI tests."""
    # Two vertices representing parameter uncertainty
    A1 = np.array([[0.0, 1.0], [-1.8, -0.4]])
    A2 = np.array([[0.0, 1.0], [-2.2, -0.6]])
    B = np.array([[0.0], [1.0]])
    return [A1, A2], B


@pytest.fixture
def simple_ir_dict():
    """A minimal DAE IR dictionary for testing the loader."""
    return {
        "model_name": "TestModel",
        "rumoca_version": "0.7.0",
        "x": {
            "x1": {
                "name": "x1",
                "type_name": {"name": [{"text": "Real"}]},
                "variability": "Empty",
                "causality": "Empty",
                "shape": [],
                "start": {"Terminal": {"token": {"text": "0.0"}}},
            },
            "x2": {
                "name": "x2",
                "type_name": {"name": [{"text": "Real"}]},
                "variability": "Empty",
                "causality": "Empty",
                "shape": [],
                "start": {"Terminal": {"token": {"text": "0.0"}}},
            },
        },
        "y": {
            "y1": {
                "name": "y1",
                "type_name": {"name": [{"text": "Real"}]},
                "variability": "Empty",
                "causality": "Empty",
                "shape": [],
                "start": "Empty",
            },
        },
        "u": {
            "d": {
                "name": "d",
                "type_name": {"name": [{"text": "Real"}]},
                "variability": "Empty",
                "causality": {"Input": {}},
                "shape": [],
                "start": "Empty",
            },
        },
        "p": {
            "k": {
                "name": "k",
                "type_name": {"name": [{"text": "Real"}]},
                "variability": {"Parameter": {}},
                "causality": "Empty",
                "shape": [],
                "start": {"Terminal": {"token": {"text": "1.0"}}},
            },
        },
        "cp": {},
        "fx": [],
        "fx_init": [],
        "fz": [],
    }
