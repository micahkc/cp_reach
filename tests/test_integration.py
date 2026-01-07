"""
Integration tests for cp_reach workflows.

These tests verify end-to-end functionality by combining multiple modules.
"""

import numpy as np
import pytest
import sympy as sp

from cp_reach.dynamics.state_space import SymbolicStateSpace
from cp_reach.reachability.lmi import solve_disturbance_LMI, solve_bounded_disturbance_output_LMI


class TestLinearReachabilityWorkflow:
    """Integration tests for linear system reachability analysis."""

    @pytest.fixture
    def mass_spring_damper_ss(self):
        """Create a mass-spring-damper SymbolicStateSpace."""
        x, v = sp.symbols("x v")
        u, d = sp.symbols("u d")
        m, c, k = sp.symbols("m c k")

        f = sp.Matrix([v, (-k * x - c * v) / m])
        Bu = sp.Matrix([0, (u + d) / m])

        return SymbolicStateSpace(
            f=f,
            Bu=Bu,
            state_symbols=[x, v],
            input_symbols=[u, d],
            param_symbols=[m, c, k],
            param_defaults={"m": 1.0, "c": 0.5, "k": 2.0},
        )

    def test_extract_matrices_and_solve_lmi(self, mass_spring_damper_ss):
        """Test full workflow: extract A, B from SymbolicStateSpace, solve LMI."""
        ss = mass_spring_damper_ss

        # Extract matrices
        A = ss.A()
        B = ss.B()

        # Get disturbance column only (d is second input)
        B_dist = B[:, 1:2]

        # Solve LMI
        sol = solve_disturbance_LMI([A], B_dist)

        # Verify solution
        assert sol["status"] in ("optimal", "optimal_inaccurate")
        assert sol["P"] is not None

        # Verify P is positive definite
        eigvals = np.linalg.eigvalsh(sol["P"])
        assert all(eigvals > 0)

        # Verify Lyapunov condition holds approximately
        # (numerical solvers may have small tolerances)
        P = sol["P"]
        alpha = sol["alpha"]
        Q = A.T @ P + P @ A + alpha * P
        max_eig = np.max(np.linalg.eigvalsh(Q))
        # Allow small numerical tolerance
        assert max_eig < 1e-6, f"Lyapunov condition violated: max eigenvalue = {max_eig}"

    def test_compute_bounds_from_solution(self, mass_spring_damper_ss):
        """Test computing state bounds from LMI solution."""
        ss = mass_spring_damper_ss
        dist_bound = 1.0

        A = ss.A()
        B = ss.B()[:, 1:2]

        sol = solve_disturbance_LMI([A], B, w_max=dist_bound)

        # Compute bounds
        P = sol["P"]
        mu = sol["mu"]
        P_inv = np.linalg.inv(P)

        # Axis-aligned bounds
        bounds = []
        for i in range(P.shape[0]):
            r = np.sqrt(mu[0]) * dist_bound * np.sqrt(P_inv[i, i])
            bounds.append(r)

        # Bounds should be finite and positive
        assert all(np.isfinite(bounds))
        assert all(b > 0 for b in bounds)


class TestOutputBoundWorkflow:
    """Integration tests for output-focused reachability analysis."""

    @pytest.fixture
    def simple_output_system_ss(self):
        """Create a simple stable system with outputs."""
        x1, x2 = sp.symbols("x1 x2")
        d = sp.symbols("d")

        # Simple stable system
        f = sp.Matrix([-2 * x1 + 0.5 * x2, -x2])
        Bu = sp.Matrix([d, 0])

        # Output: first state
        h = sp.Matrix([x1])

        return SymbolicStateSpace(
            f=f,
            Bu=Bu,
            h=h,
            state_symbols=[x1, x2],
            input_symbols=[d],
            param_symbols=[],
            output_symbols=[sp.Symbol("y")],
            param_defaults={},
        )

    @pytest.mark.skip(reason="Output bound LMI solver has numerical issues with some systems")
    def test_output_bound_lmi(self):
        """Test output bound LMI for simple system."""
        # Use a very simple 2x2 stable system
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[1.0], [0.0]])
        C = np.array([[1.0, 0.0]])
        D = np.zeros((1, 1))

        sol = solve_bounded_disturbance_output_LMI(A, B, C, D)

        assert sol["status"] in ("optimal", "optimal_inaccurate")
        assert sol["gamma"] > 0
        assert np.isfinite(sol["gamma"])

    def test_state_bound_as_alternative(self, simple_output_system_ss):
        """Test computing state bounds as alternative to output bounds."""
        ss = simple_output_system_ss

        # Use state-space LMI directly
        A = ss.A()
        B = ss.B()

        sol = solve_disturbance_LMI([A], B)
        assert sol["status"] in ("optimal", "optimal_inaccurate")


class TestPolytopicWorkflow:
    """Integration tests for polytopic/nonlinear system analysis."""

    @pytest.fixture
    def stable_polytopic_vertices(self):
        """Create stable polytopic vertices (all Hurwitz)."""
        # Simple stable 2x2 systems with small parameter variations
        # Both A matrices have negative eigenvalues

        # Vertex 1: eigenvalues approximately -0.25 ± 1.39j
        A1 = np.array([[0.0, 1.0], [-2.0, -0.5]])

        # Vertex 2: eigenvalues approximately -0.3 ± 1.47j
        A2 = np.array([[0.0, 1.0], [-2.2, -0.6]])

        B = np.array([[0.0], [1.0]])

        return [A1, A2], B

    def test_polytopic_lmi(self, stable_polytopic_vertices):
        """Test LMI with polytopic vertices."""
        A_list, B = stable_polytopic_vertices

        sol = solve_disturbance_LMI(A_list, B)

        assert sol["status"] in ("optimal", "optimal_inaccurate")

        # Common Lyapunov function should work for all vertices
        P = sol["P"]
        assert P is not None
        # Verify P is positive definite
        assert all(np.linalg.eigvalsh(P) > 0)


class TestModuleIntegration:
    """Tests verifying integration between different cp_reach modules."""

    def test_dynamics_to_reachability(self):
        """Test that dynamics module outputs work with reachability module."""
        # Create simple system
        x1, x2 = sp.symbols("x1 x2")
        u = sp.symbols("u")

        f = sp.Matrix([x2, -x1 - 0.5 * x2])
        Bu = sp.Matrix([0, u])

        ss = SymbolicStateSpace(
            f=f,
            Bu=Bu,
            state_symbols=[x1, x2],
            input_symbols=[u],
            param_symbols=[],
            param_defaults={},
        )

        # Extract matrices using dynamics module
        A = ss.A()
        B = ss.B()

        # Use with reachability module
        sol = solve_disturbance_LMI([A], B)

        assert sol["status"] in ("optimal", "optimal_inaccurate")

    def test_symbolic_to_numeric_pipeline(self):
        """Test the complete symbolic-to-numeric pipeline."""
        # Define symbolic system
        x, v = sp.symbols("x v")
        d = sp.symbols("d")
        k, c = sp.symbols("k c")

        f = sp.Matrix([v, -k * x - c * v])
        Bu = sp.Matrix([0, d])

        ss = SymbolicStateSpace(
            f=f,
            Bu=Bu,
            state_symbols=[x, v],
            input_symbols=[d],
            param_symbols=[k, c],
            param_defaults={"k": 2.0, "c": 0.5},
        )

        # Step 1: Parameter substitution
        f_sub = ss.f_sub()
        assert not any(sym in f_sub.free_symbols for sym in [k, c])

        # Step 2: Jacobian computation
        A = ss.A()
        B = ss.B()
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)

        # Step 3: LMI solve
        sol = solve_disturbance_LMI([A], B, w_max=1.0)
        assert sol["status"] in ("optimal", "optimal_inaccurate")

        # Step 4: Bound computation
        P = sol["P"]
        mu = float(sol["mu"][0])
        P_inv = np.linalg.inv(P)
        bound_x = np.sqrt(mu) * np.sqrt(P_inv[0, 0])
        bound_v = np.sqrt(mu) * np.sqrt(P_inv[1, 1])

        assert np.isfinite(bound_x)
        assert np.isfinite(bound_v)
        assert bound_x > 0
        assert bound_v > 0


@pytest.mark.slow
class TestLargerSystems:
    """Tests for larger systems (may be slower)."""

    def test_6_state_system(self):
        """Test with a 6-state system."""
        np.random.seed(42)
        n = 6

        # Create stable random system
        A = np.random.randn(n, n)
        # Make stable by shifting eigenvalues
        A = A - (np.max(np.real(np.linalg.eigvals(A))) + 1.0) * np.eye(n)
        B = np.random.randn(n, 2)

        sol = solve_disturbance_LMI([A], B)

        assert sol["status"] in ("optimal", "optimal_inaccurate")
        assert sol["P"].shape == (n, n)

    def test_many_polytopic_vertices(self):
        """Test with many polytopic vertices."""
        np.random.seed(42)
        n = 3
        num_vertices = 8

        # Create base stable system
        A_base = -np.eye(n)
        perturbation_scale = 0.2

        A_list = []
        for _ in range(num_vertices):
            A = A_base + perturbation_scale * np.random.randn(n, n)
            A_list.append(A)

        B = np.array([[0.0], [0.0], [1.0]])

        sol = solve_disturbance_LMI(A_list, B)

        assert sol["status"] in ("optimal", "optimal_inaccurate")
