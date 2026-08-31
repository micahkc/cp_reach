"""Tests for the RuMoCA 0.9.20 DAE schema-7 loader."""

import json

import pytest

from cp_reach.ir.loader import Component, DaeIR, _parse_start_value


def test_component_classification():
    """Component helpers reflect the DAE category and causality."""
    assert Component(name="k", variability="parameter").is_parameter()
    assert Component(name="pi", variability="constant").is_constant()
    assert Component(name="d", causality="input").is_input()
    assert Component(name="y", causality="output").is_output()
    assert Component(name="x").is_scalar()
    assert not Component(name="xs", shape=[3]).is_scalar()


def test_from_dict_loads_schema_7(simple_ir_dict):
    """The loader extracts all analysis categories from schema 7."""
    ir = DaeIR.from_dict(simple_ir_dict, model_name="TestModel")

    assert ir.model_name == "TestModel"
    assert ir.schema_version == 7
    assert ir.get_state_names() == ["x1", "x2"]
    assert ir.get_input_names() == ["d"]
    assert ir.get_parameter_names() == ["k"]
    assert ir.get_algebraic_names() == ["y1"]
    assert ir.get_param_defaults() == {"k": 1.0}
    assert ir.n_states() == 2
    assert ir.n_inputs() == 1
    assert ir.n_parameters() == 1
    assert ir.n_algebraics() == 1


def test_from_json_uses_file_stem_as_model_name(simple_ir_dict, tmp_path):
    """File-based loads retain a useful model identifier outside DAE JSON."""
    path = tmp_path / "vehicle_model.json"
    path.write_text(json.dumps(simple_ir_dict))

    ir = DaeIR.from_json(path)

    assert ir.model_name == "vehicle_model"


def test_from_json_str_accepts_model_name(simple_ir_dict):
    """String-based loads accept the name selected in the RuMoCA session."""
    ir = DaeIR.from_json_str(json.dumps(simple_ir_dict), model_name="VehicleModel")

    assert ir.model_name == "VehicleModel"


def test_rejects_non_schema_7_data(simple_ir_dict):
    """Old or unknown DAE schemas fail explicitly."""
    simple_ir_dict["schema_version"] = 6

    with pytest.raises(ValueError, match="requires RuMoCA DAE schema 7"):
        DaeIR.from_dict(simple_ir_dict)


def test_infer_roles(simple_ir_dict):
    """Dotted names retain their useful plant/controller grouping."""
    simple_ir_dict["x"] = {
        "plant.x": {"causality": "local", "dims": [], "start": None},
        "controller.x": {"causality": "local", "dims": [], "start": None},
    }

    roles = DaeIR.from_dict(simple_ir_dict).infer_roles()

    assert roles == {"plant.x": "plant", "controller.x": "controller"}


def test_constants_are_included_in_defaults(simple_ir_dict):
    """Constants and parameters are both available for symbolic substitution."""
    simple_ir_dict["constants"] = {
        "pi": {
            "causality": "local",
            "dims": [],
            "start": {"Literal": {"value": {"Real": 3.14159}}},
        }
    }

    defaults = DaeIR.from_dict(simple_ir_dict).get_param_defaults()

    assert defaults == {"k": 1.0, "pi": 3.14159}


def test_parse_numeric_start_values():
    """Schema-7 real, integer, and negated values are supported."""
    assert _parse_start_value({"Literal": {"value": {"Real": 2.5}}}) == 2.5
    assert _parse_start_value({"Literal": {"value": {"Integer": 4}}}) == 4.0
    assert (
        _parse_start_value(
            {
                "Unary": {
                    "op": "Minus",
                    "rhs": {"Literal": {"value": {"Integer": 4}}},
                }
            }
        )
        == -4.0
    )
    assert _parse_start_value(None) is None
