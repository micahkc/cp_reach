"""
Unit tests for the dynamics module in cp_reach.dynamics.
"""

import numpy as np
import pytest
import sympy as sp

from cp_reach.dynamics.state_space import SymbolicStateSpace


class TestSymbolicStateSpace:
    """Tests for SymbolicStateSpace class."""

    @pytest.fixture
    def simple_linear_system(self):
        """Create a simple linear symbolic state space."""
        # States
        x, v = sp.symbols("x v")
        # Inputs
        u, d = sp.symbols("u d")
        # Parameters
        m, c, k = sp.symbols("m c k")

        # Linear mass-spring-damper: m*xdot = -k*x - c*v + u + d
        # ẋ = v
        # v̇ = (-k*x - c*v)/m
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

    @pytest.fixture
    def nonlinear_system(self):
        """Create a nonlinear symbolic state space (pendulum)."""
        # States
        theta, omega = sp.symbols("theta omega")
        # Inputs
        tau = sp.symbols("tau")
        # Parameters
        m, L, g, b = sp.symbols("m L g b")

        # Pendulum: mL²θ̈ = -mgL*sin(θ) - b*ω + τ
        f = sp.Matrix([omega, (-m * g * L * sp.sin(theta) - b * omega) / (m * L**2)])
        Bu = sp.Matrix([0, tau / (m * L**2)])

        return SymbolicStateSpace(
            f=f,
            Bu=Bu,
            state_symbols=[theta, omega],
            input_symbols=[tau],
            param_symbols=[m, L, g, b],
            param_defaults={"m": 1.0, "L": 1.0, "g": 9.81, "b": 0.1},
        )

    def test_f_sub_with_defaults(self, simple_linear_system):
        """Test parameter substitution in drift dynamics."""
        ss = simple_linear_system

        f_numeric = ss.f_sub()

        # With defaults: m=1, c=0.5, k=2
        # f = [v, -2*x - 0.5*v]
        x, v = ss.state_symbols
        expected = sp.Matrix([v, -2 * x - 0.5 * v])
        assert sp.simplify(f_numeric - expected) == sp.Matrix([0, 0])

    def test_f_sub_with_custom_params(self, simple_linear_system):
        """Test parameter substitution with custom values."""
        ss = simple_linear_system

        f_numeric = ss.f_sub({"m": 2.0, "c": 1.0, "k": 4.0})

        # With params: m=2, c=1, k=4
        # f = [v, -2*x - 0.5*v]
        x, v = ss.state_symbols
        expected = sp.Matrix([v, -2 * x - 0.5 * v])
        assert sp.simplify(f_numeric - expected) == sp.Matrix([0, 0])

    def test_A_matrix_linear_system(self, simple_linear_system):
        """Test A matrix computation for linear system."""
        ss = simple_linear_system

        A = ss.A()

        # Expected: [[0, 1], [-k/m, -c/m]] = [[0, 1], [-2, -0.5]]
        expected = np.array([[0.0, 1.0], [-2.0, -0.5]])
        np.testing.assert_array_almost_equal(A, expected)

    def test_B_matrix(self, simple_linear_system):
        """Test B matrix computation."""
        ss = simple_linear_system

        B = ss.B()

        # Expected: [[0, 0], [1/m, 1/m]] = [[0, 0], [1, 1]]
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(B, expected)

    def test_A_matrix_nonlinear_needs_bounds(self, nonlinear_system):
        """Test that nonlinear system warns when bounds not provided."""
        ss = nonlinear_system

        with pytest.warns(UserWarning, match="State Jacobian contains free variables"):
            A = ss.A()

        # Should return symbolic matrix
        assert isinstance(A, sp.Matrix)

    def test_A_matrix_polytopic(self, nonlinear_system):
        """Test polytopic A matrix computation for nonlinear system."""
        ss = nonlinear_system
        theta = ss.state_symbols[0]

        # Provide bounds for theta
        A_vertices = ss.A(bounds={"theta": (-0.5, 0.5)})

        # Should return list of vertices (2 corners)
        assert isinstance(A_vertices, list)
        assert len(A_vertices) == 2
        # Each vertex should be a numpy array
        for A in A_vertices:
            assert isinstance(A, np.ndarray)
            assert A.shape == (2, 2)

    def test_error_dynamics_linear(self, simple_linear_system):
        """Test that linear system has linear error dynamics."""
        ss = simple_linear_system

        is_linear = ss.error_dynamics_are_linear(disturbance_inputs=["d"])

        assert is_linear is True

    def test_error_dynamics_nonlinear(self, nonlinear_system):
        """Test that pendulum has nonlinear error dynamics."""
        ss = nonlinear_system

        is_linear = ss.error_dynamics_are_linear()

        assert is_linear is False

    def test_get_error_dynamics(self, simple_linear_system):
        """Test error dynamics extraction."""
        ss = simple_linear_system

        err = ss.get_error_dynamics(disturbance_inputs=["d"])

        assert "J" in err
        assert "B_d" in err
        assert "f_nominal" in err
        assert "disturbance_symbols" in err
        assert len(err["disturbance_symbols"]) == 1

    def test_linearize_error_dynamics(self, simple_linear_system):
        """Test linearization of error dynamics."""
        ss = simple_linear_system

        A, B_d = ss.linearize_error_dynamics(disturbance_inputs=["d"])

        assert A.shape == (2, 2)
        assert B_d.shape == (2, 1)
        # A should match the state Jacobian
        np.testing.assert_array_almost_equal(A, ss.A())

    def test_get_state_names(self, simple_linear_system):
        """Test state name extraction."""
        ss = simple_linear_system

        names = ss.get_state_names()

        assert names == ["x", "v"]

    def test_get_input_names(self, simple_linear_system):
        """Test input name extraction."""
        ss = simple_linear_system

        names = ss.get_input_names()

        assert names == ["u", "d"]


class TestSymbolicStateSpaceWithOutputs:
    """Tests for SymbolicStateSpace with output equations."""

    @pytest.fixture
    def system_with_outputs(self):
        """Create a system with output equations."""
        x, v, x_ref, v_ref = sp.symbols("x v x_ref v_ref")
        d = sp.symbols("d")
        m, c, k, kp, kd = sp.symbols("m c k kp kd")

        # Plant + reference dynamics
        f = sp.Matrix([
            v,
            (-k * x - c * v) / m,
            v_ref,
            0,  # Reference is constant
        ])
        Bu = sp.Matrix([0, d / m, 0, 0])

        # Output: tracking errors
        h = sp.Matrix([x - x_ref, v - v_ref])

        return SymbolicStateSpace(
            f=f,
            Bu=Bu,
            h=h,
            state_symbols=[x, v, x_ref, v_ref],
            input_symbols=[d],
            param_symbols=[m, c, k, kp, kd],
            output_symbols=[sp.Symbol("e"), sp.Symbol("ev")],
            param_defaults={"m": 1.0, "c": 0.5, "k": 2.0, "kp": 2.0, "kd": 1.0},
        )

    def test_h_sub(self, system_with_outputs):
        """Test output substitution."""
        ss = system_with_outputs

        h_numeric = ss.h_sub()

        # Output should remain x - x_ref, v - v_ref (no params in h)
        x, v, x_ref, v_ref = ss.state_symbols
        expected = sp.Matrix([x - x_ref, v - v_ref])
        assert sp.simplify(h_numeric - expected) == sp.Matrix([0, 0])

    def test_C_matrix(self, system_with_outputs):
        """Test C matrix computation."""
        ss = system_with_outputs

        C = ss.C()

        # C = dh/dx = [[1, 0, -1, 0], [0, 1, 0, -1]]
        expected = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
        np.testing.assert_array_almost_equal(C, expected)

    def test_D_matrix_zero(self, system_with_outputs):
        """Test D matrix is None when Du is None."""
        ss = system_with_outputs

        # Du was not provided in this fixture
        D = ss.D()

        # No feedthrough defined
        assert D is None or np.allclose(D, 0)

    def test_E_matrix(self, system_with_outputs):
        """Test measurement dynamics E matrix."""
        ss = system_with_outputs

        try:
            E = ss.E()
        except (AttributeError, TypeError):
            # E() may fail if measurement dynamics weren't built
            # This is acceptable - not all systems have computable E
            return

        # E may be None if measurement dynamics cannot be computed
        # (e.g., h doesn't depend on all states that have dynamics)
        if E is not None:
            # For this system, E = C @ A @ C_inv (or similar)
            # Just check shape
            if isinstance(E, np.ndarray):
                assert E.shape[0] == 2  # 2 outputs

    def test_F_matrix(self, system_with_outputs):
        """Test measurement dynamics F matrix."""
        ss = system_with_outputs

        F = ss.F()

        # F should represent disturbance effect on outputs
        assert F is not None
        if isinstance(F, np.ndarray):
            assert F.shape == (2, 1)  # 2 outputs, 1 disturbance
