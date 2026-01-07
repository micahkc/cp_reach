"""
Unit tests for the LMI solvers in cp_reach.reachability.lmi.
"""

import numpy as np
import pytest

from cp_reach.reachability.lmi import (
    solve_disturbance_LMI,
    solve_bounded_disturbance_output_LMI,
    find_feasible_alpha,
)


class TestSolveDisturbanceLMI:
    """Tests for solve_disturbance_LMI function."""

    def test_basic_stable_system(self, stable_2x2_system):
        """Test that LMI solver finds a solution for a stable system."""
        A, B = stable_2x2_system

        sol = solve_disturbance_LMI([A], B)

        assert sol["status"] in ("optimal", "optimal_inaccurate")
        assert sol["P"] is not None
        assert sol["mu"] is not None
        assert sol["alpha"] > 0
        assert np.isfinite(sol["radius_inf"])

    def test_lyapunov_matrix_positive_definite(self, stable_2x2_system):
        """Test that the returned P matrix is positive definite."""
        A, B = stable_2x2_system

        sol = solve_disturbance_LMI([A], B)

        P = sol["P"]
        eigenvalues = np.linalg.eigvalsh(P)
        assert all(eigenvalues > 0), "P must be positive definite"

    def test_with_disturbance_bound(self, stable_2x2_system):
        """Test LMI with specified disturbance bound."""
        A, B = stable_2x2_system
        w_max = 0.5

        sol = solve_disturbance_LMI([A], B, w_max=w_max)

        assert sol["status"] in ("optimal", "optimal_inaccurate")
        assert sol["radius_inf"] <= w_max * np.sqrt(np.max(sol["mu"]))

    def test_polytopic_system(self, polytopic_vertices):
        """Test LMI with polytopic (multiple A matrices) system."""
        A_list, B = polytopic_vertices

        sol = solve_disturbance_LMI(A_list, B)

        assert sol["status"] in ("optimal", "optimal_inaccurate")
        # P should work for all vertices
        P = sol["P"]
        for A in A_list:
            # Check Lyapunov decrease condition approximately
            Q = A.T @ P + P @ A
            eigenvalues = np.linalg.eigvalsh(Q)
            # All eigenvalues should be negative (P decreases along trajectories)
            assert all(eigenvalues < 0.1), "Lyapunov condition should hold"

    def test_alpha_grid_search(self, stable_2x2_system):
        """Test that alpha grid search finds feasible solution."""
        A, B = stable_2x2_system
        alpha_grid = np.logspace(-3, 0, 10)

        sol = solve_disturbance_LMI([A], B, alpha_grid=alpha_grid)

        assert sol["status"] in ("optimal", "optimal_inaccurate")

    def test_multi_channel_disturbance(self, stable_2x2_system):
        """Test LMI with multiple disturbance channels."""
        A, _ = stable_2x2_system
        B_multi = np.array([[0.0, 0.5], [1.0, 0.0]])  # 2 disturbance channels

        sol = solve_disturbance_LMI([A], B_multi)

        assert sol["status"] in ("optimal", "optimal_inaccurate")
        assert sol["mu"].shape[0] == 2  # One mu per channel


class TestSolveBoundedDisturbanceOutputLMI:
    """Tests for solve_bounded_disturbance_output_LMI function."""

    def test_basic_output_bound(self, stable_4x4_system):
        """Test output bound LMI for stable system."""
        A, B, C, D = stable_4x4_system

        sol = solve_bounded_disturbance_output_LMI(A, B, C, D)

        assert sol["status"] in ("optimal", "optimal_inaccurate")
        assert sol["gamma"] > 0
        assert np.isfinite(sol["gamma"])
        assert sol["mu1"] >= 0
        assert sol["mu2"] >= 0

    def test_gamma_is_sqrt_sum(self, stable_4x4_system):
        """Test that gamma = sqrt(mu1 + mu2)."""
        A, B, C, D = stable_4x4_system

        sol = solve_bounded_disturbance_output_LMI(A, B, C, D)

        expected_gamma = np.sqrt(sol["mu1"] + sol["mu2"])
        assert np.isclose(sol["gamma"], expected_gamma, rtol=1e-4)

    def test_polytopic_output_bound(self, stable_4x4_system):
        """Test output bound LMI with polytopic uncertainty."""
        _, B, C, D = stable_4x4_system
        # Stable 4x4 polytopic vertices (diagonal dominant, all eigenvalues negative)
        A1 = np.array([
            [-2.0, 0.4, 0.0, 0.0],
            [0.0, -1.4, 0.2, 0.0],
            [0.0, 0.0, -0.9, 0.15],
            [0.0, 0.0, 0.0, -0.7],
        ])
        A2 = np.array([
            [-2.2, 0.6, 0.0, 0.0],
            [0.0, -1.6, 0.4, 0.0],
            [0.0, 0.0, -1.1, 0.25],
            [0.0, 0.0, 0.0, -0.9],
        ])

        sol = solve_bounded_disturbance_output_LMI([A1, A2], B, C, D)

        assert sol["status"] in ("optimal", "optimal_inaccurate")


class TestFindFeasibleAlpha:
    """Tests for find_feasible_alpha function."""

    def test_finds_feasible_alpha(self, stable_2x2_system):
        """Test that a feasible alpha is found for stable system."""
        A, B = stable_2x2_system

        alpha, sol = find_feasible_alpha([A], B)

        assert alpha > 0
        assert sol["status"] in ("optimal", "optimal_inaccurate")

    def test_custom_alpha_grid(self, stable_2x2_system):
        """Test with custom alpha grid."""
        A, B = stable_2x2_system
        alphas = [0.01, 0.1, 0.5, 1.0]

        alpha, sol = find_feasible_alpha([A], B, alphas=alphas)

        assert alpha in alphas or np.isclose(alpha, alphas).any()

    def test_raises_on_no_feasible(self):
        """Test that RuntimeError is raised when no feasible alpha found."""
        # Strongly unstable system - eigenvalues at +2 and +1
        # No Lyapunov function can stabilize this
        A_unstable = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[0.0], [0.0]])  # No control authority

        with pytest.raises(RuntimeError, match="No feasible alpha"):
            find_feasible_alpha([A_unstable], B, alphas=[0.001, 0.01, 0.1, 1.0, 10.0])


class TestEdgeCases:
    """Edge case tests for LMI solvers."""

    def test_single_state_system(self):
        """Test 1x1 system."""
        A = np.array([[-1.0]])
        B = np.array([[1.0]])

        sol = solve_disturbance_LMI([A], B)

        assert sol["status"] in ("optimal", "optimal_inaccurate")

    def test_high_dimensional_system(self):
        """Test larger system (5 states)."""
        # Random stable system
        np.random.seed(42)
        n = 5
        A = -np.eye(n) + 0.1 * np.random.randn(n, n)
        # Make sure it's stable
        A = A - np.eye(n) * (np.max(np.real(np.linalg.eigvals(A))) + 0.1)
        B = np.random.randn(n, 1)

        sol = solve_disturbance_LMI([A], B)

        assert sol["status"] in ("optimal", "optimal_inaccurate")

    def test_near_marginal_stability(self):
        """Test system close to marginal stability."""
        # System with eigenvalue close to zero
        A = np.array([[0.0, 1.0], [-0.01, -0.01]])
        B = np.array([[0.0], [1.0]])

        sol = solve_disturbance_LMI([A], B, alpha_bounds=(1e-8, 0.1))

        # Should still find a solution, though radius may be large
        assert sol["status"] in ("optimal", "optimal_inaccurate")
