"""
Unit tests for the IR loader module in cp_reach.ir.
"""

import json
import tempfile
from pathlib import Path

import pytest

from cp_reach.ir.loader import DaeIR, Component, _parse_start_value


class TestComponent:
    """Tests for the Component dataclass."""

    def test_is_parameter(self):
        """Test parameter detection."""
        param = Component(name="k", variability="parameter")
        non_param = Component(name="x", variability="")

        assert param.is_parameter() is True
        assert non_param.is_parameter() is False

    def test_is_input(self):
        """Test input detection."""
        inp = Component(name="d", causality="input")
        non_inp = Component(name="x", causality="")

        assert inp.is_input() is True
        assert non_inp.is_input() is False

    def test_is_output(self):
        """Test output detection."""
        out = Component(name="y", causality="output")
        non_out = Component(name="x", causality="")

        assert out.is_output() is True
        assert non_out.is_output() is False

    def test_is_scalar(self):
        """Test scalar vs array detection."""
        scalar = Component(name="x", shape=[])
        array = Component(name="xs", shape=[3])

        assert scalar.is_scalar() is True
        assert array.is_scalar() is False


class TestDaeIR:
    """Tests for the DaeIR dataclass."""

    def test_from_dict_basic(self, simple_ir_dict):
        """Test loading DaeIR from dictionary."""
        ir = DaeIR.from_dict(simple_ir_dict)

        assert ir.model_name == "TestModel"
        assert ir.rumoca_version == "0.7.0"
        assert len(ir.states) == 2
        assert len(ir.inputs) == 1
        assert len(ir.parameters) == 1

    def test_get_state_names(self, simple_ir_dict):
        """Test state name extraction."""
        ir = DaeIR.from_dict(simple_ir_dict)

        names = ir.get_state_names()

        assert "x1" in names
        assert "x2" in names
        assert len(names) == 2

    def test_get_input_names(self, simple_ir_dict):
        """Test input name extraction."""
        ir = DaeIR.from_dict(simple_ir_dict)

        names = ir.get_input_names()

        assert names == ["d"]

    def test_get_parameter_names(self, simple_ir_dict):
        """Test parameter name extraction."""
        ir = DaeIR.from_dict(simple_ir_dict)

        names = ir.get_parameter_names()

        assert names == ["k"]

    def test_get_param_defaults(self, simple_ir_dict):
        """Test parameter default extraction."""
        ir = DaeIR.from_dict(simple_ir_dict)

        defaults = ir.get_param_defaults()

        assert defaults["k"] == 1.0

    def test_n_states(self, simple_ir_dict):
        """Test state count."""
        ir = DaeIR.from_dict(simple_ir_dict)

        assert ir.n_states() == 2

    def test_n_inputs(self, simple_ir_dict):
        """Test input count."""
        ir = DaeIR.from_dict(simple_ir_dict)

        assert ir.n_inputs() == 1

    def test_from_json_file(self, simple_ir_dict, tmp_path):
        """Test loading from JSON file."""
        json_path = tmp_path / "test_model.json"
        with open(json_path, "w") as f:
            json.dump(simple_ir_dict, f)

        ir = DaeIR.from_json(json_path)

        assert ir.model_name == "TestModel"
        assert ir.n_states() == 2

    def test_from_json_str(self, simple_ir_dict):
        """Test loading from JSON string."""
        json_str = json.dumps(simple_ir_dict)

        ir = DaeIR.from_json_str(json_str)

        assert ir.model_name == "TestModel"

    def test_infer_roles(self):
        """Test role inference from variable names."""
        ir_dict = {
            "model_name": "RoleTest",
            "rumoca_version": "",
            "x": {
                "plant.x": {"name": "plant.x", "shape": []},
                "controller.y": {"name": "controller.y", "shape": []},
            },
            "y": {},
            "u": {},
            "p": {},
            "cp": {},
            "fx": [],
            "fx_init": [],
            "fz": [],
        }
        ir = DaeIR.from_dict(ir_dict)

        roles = ir.infer_roles()

        assert roles["plant.x"] == "plant"
        assert roles["controller.y"] == "controller"

    def test_get_algebraic_names(self, simple_ir_dict):
        """Test algebraic variable name extraction."""
        ir = DaeIR.from_dict(simple_ir_dict)

        names = ir.get_algebraic_names()

        assert "y1" in names


class TestParseStartValue:
    """Tests for the _parse_start_value helper function."""

    def test_parse_terminal_float(self):
        """Test parsing a terminal float value."""
        ast = {"Terminal": {"token": {"text": "1.5"}}}

        value = _parse_start_value(ast)

        assert value == 1.5

    def test_parse_terminal_integer(self):
        """Test parsing a terminal integer value."""
        ast = {"Terminal": {"token": {"text": "42"}}}

        value = _parse_start_value(ast)

        assert value == 42.0

    def test_parse_negative_value(self):
        """Test parsing a negative value."""
        ast = {
            "Unary": {
                "op": {"Neg": {}},
                "operand": {"Terminal": {"token": {"text": "3.14"}}},
            }
        }

        value = _parse_start_value(ast)

        assert value == -3.14

    def test_parse_empty_returns_none(self):
        """Test that Empty returns None."""
        value = _parse_start_value("Empty")

        assert value is None

    def test_parse_non_dict_returns_none(self):
        """Test that non-dict input returns None."""
        value = _parse_start_value(123)

        assert value is None

    def test_parse_invalid_text_returns_none(self):
        """Test that non-numeric text returns None."""
        ast = {"Terminal": {"token": {"text": "not_a_number"}}}

        value = _parse_start_value(ast)

        assert value is None


class TestDaeIREdgeCases:
    """Edge case tests for DaeIR."""

    def test_empty_model(self):
        """Test handling of model with no variables."""
        ir_dict = {
            "model_name": "Empty",
            "rumoca_version": "",
            "x": {},
            "y": {},
            "u": {},
            "p": {},
            "cp": {},
            "fx": [],
            "fx_init": [],
            "fz": [],
        }

        ir = DaeIR.from_dict(ir_dict)

        assert ir.n_states() == 0
        assert ir.n_inputs() == 0

    def test_missing_optional_fields(self):
        """Test handling of missing optional fields."""
        ir_dict = {
            "model_name": "Minimal",
            "x": {},
            # Missing: rumoca_version, y, u, p, cp, fx, fx_init, fz
        }

        ir = DaeIR.from_dict(ir_dict)

        assert ir.model_name == "Minimal"
        assert ir.rumoca_version == ""

    def test_constants_in_defaults(self):
        """Test that constants are included in param defaults."""
        ir_dict = {
            "model_name": "WithConstants",
            "rumoca_version": "",
            "x": {},
            "y": {},
            "u": {},
            "p": {},
            "cp": {
                "pi": {
                    "name": "pi",
                    "variability": {"Constant": {}},
                    "start": {"Terminal": {"token": {"text": "3.14159"}}},
                    "shape": [],
                }
            },
            "fx": [],
            "fx_init": [],
            "fz": [],
        }

        ir = DaeIR.from_dict(ir_dict)
        defaults = ir.get_param_defaults()

        assert "pi" in defaults
        assert abs(defaults["pi"] - 3.14159) < 0.0001

    def test_array_variable(self):
        """Test handling of array variables."""
        ir_dict = {
            "model_name": "WithArray",
            "rumoca_version": "",
            "x": {
                "xs": {
                    "name": "xs",
                    "shape": [3],
                    "type_name": {"name": [{"text": "Real"}]},
                    "variability": "Empty",
                    "causality": "Empty",
                    "start": "Empty",
                }
            },
            "y": {},
            "u": {},
            "p": {},
            "cp": {},
            "fx": [],
            "fx_init": [],
            "fz": [],
        }

        ir = DaeIR.from_dict(ir_dict)

        assert ir.states["xs"].shape == [3]
        assert ir.states["xs"].is_scalar() is False
