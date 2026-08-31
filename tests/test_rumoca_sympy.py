"""Tests for the direct Rumoca checked-Solve to SymPy boundary."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import sympy as sp

from cp_reach.config.query import ReachQuery
from cp_reach.config.uncertainty import (
    BoundedDisturbance,
    InitialCondition,
    ParameterUncertainty,
    UncertaintySpec,
)
from cp_reach.ir.rumoca import (
    RumocaSymbolicModel,
    RumocaSympyExportError,
    modelica_load,
    modelica_loads,
    render_sympy_export,
    rumoca_model_to_symbolic,
)

GENERATED_EXPORT = """
import sympy as sp

STATE_NAMES = ["x", "v"]
INPUT_NAMES = ["d", "r"]
PARAMETER_NAMES = ["k", "c"]
ALGEBRAIC_NAMES = ["e"]
OUTPUT_NAMES = []
DEFAULT_STATES = [1.0, 0.0]
DEFAULT_PARAMETERS = [2.0, 0.5]
ALGEBRAIC_ASSIGNMENTS_COMPLETE = True

STATE_SYMBOLS = tuple(sp.Symbol(name, real=True) for name in STATE_NAMES)
INPUT_SYMBOLS = tuple(sp.Symbol(name, real=True) for name in INPUT_NAMES)
PARAMETER_SYMBOLS = tuple(sp.Symbol(name, real=True) for name in PARAMETER_NAMES)
x, v = STATE_SYMBOLS
d, r = INPUT_SYMBOLS
k, c = PARAMETER_SYMBOLS
RHS = sp.ImmutableDenseMatrix([v, -k*x - c*v + d])
VARIABLE_EXPRESSIONS = {"x": x, "v": v, "e": x - r}
"""


class FakeRumocaModel:
    def __init__(self, source: str = GENERATED_EXPORT):
        self.source = source
        self.target = None

    def render(self, target: str) -> str:
        self.target = Path(target)
        return self.source


class FakeSession:
    def __init__(self, model: FakeRumocaModel):
        self.model = model
        self.loads_call = None

    def loads(self, source: str, **kwargs):
        self.loads_call = (source, kwargs)
        return self.model


def test_render_export_validates_and_preserves_checked_order():
    model = FakeRumocaModel()

    export = render_sympy_export(model)

    assert model.target is not None
    assert (model.target / "target.toml").is_file()
    assert export.state_names == ("x", "v")
    assert export.input_names == ("d", "r")
    assert export.parameter_names == ("k", "c")
    assert export.parameter_defaults == {"k": 2.0, "c": 0.5}
    assert export.rhs.shape == (2, 1)


def test_rumoca_model_converts_total_rhs_and_named_outputs():
    model = rumoca_model_to_symbolic(FakeRumocaModel(), output_names=["e"])
    ss = model.symbolic
    x, v = ss.state_symbols
    d, r = ss.input_symbols
    k, c = ss.param_symbols

    assert isinstance(model, RumocaSymbolicModel)
    assert ss.f == sp.Matrix([v, -k * x - c * v])
    assert ss.Bu == sp.Matrix([0, d])
    assert ss.h == sp.Matrix([x])
    assert ss.Du == sp.Matrix([-r])
    assert model.states == ["x", "v"]
    assert model.inputs == ["d", "r"]
    assert model.parameters == {"k": 2.0, "c": 0.5}


def test_named_outputs_require_complete_algebraic_assignments():
    source = GENERATED_EXPORT.replace(
        "ALGEBRAIC_ASSIGNMENTS_COMPLETE = True",
        "ALGEBRAIC_ASSIGNMENTS_COMPLETE = False",
    )

    with pytest.raises(RumocaSympyExportError, match="complete explicit algebraic"):
        rumoca_model_to_symbolic(FakeRumocaModel(source), output_names=["e"])

    model = rumoca_model_to_symbolic(FakeRumocaModel(source))
    assert model.symbolic.f.shape == (2, 1)


def test_unknown_named_output_lists_available_variables():
    with pytest.raises(ValueError, match="Unknown outputs"):
        rumoca_model_to_symbolic(FakeRumocaModel(), output_names=["missing"])


def test_configuration_validates_against_generated_model_contract():
    model = rumoca_model_to_symbolic(FakeRumocaModel(), output_names=["e"])
    uncertainty = UncertaintySpec(
        disturbances={
            "d": BoundedDisturbance(0.1),
            "missing_input": BoundedDisturbance(0.1),
        },
        parameters={
            "k": ParameterUncertainty(2.0, (1.0, 3.0)),
            "missing_parameter": ParameterUncertainty(1.0, (0.0, 2.0)),
        },
        initial_conditions={
            "x": InitialCondition("zero"),
            "missing_state": InitialCondition("zero"),
        },
    )
    query = ReachQuery(
        outputs=["e", "missing_output"],
        dist_inputs=["d", "missing_disturbance"],
    )

    assert uncertainty.validate_against_model(model) == [
        "Disturbance 'missing_input' not found in model inputs",
        "Parameter 'missing_parameter' not found in model parameters",
        "Initial condition 'missing_state' not found in model",
    ]
    assert query.validate_against_model(model) == [
        "Output 'missing_output' not found in model",
        "Disturbance input 'missing_disturbance' not found in model inputs",
    ]


def test_generated_contract_rejects_bad_rhs_shape():
    source = GENERATED_EXPORT.replace(
        "RHS = sp.ImmutableDenseMatrix([v, -k*x - c*v + d])",
        "RHS = sp.ImmutableDenseMatrix([v])",
    )

    with pytest.raises(RumocaSympyExportError, match="Generated RHS has shape"):
        render_sympy_export(FakeRumocaModel(source))


def test_modelica_load_uses_isolated_source_session(tmp_path):
    session = FakeSession(FakeRumocaModel())
    path = tmp_path / "Plant.mo"
    path.write_text("model Plant end Plant;")

    model = modelica_load(
        path,
        model_name="Plant",
        output_names=["e"],
        session=session,
    )

    assert isinstance(model, RumocaSymbolicModel)
    assert session.loads_call == (
        "model Plant end Plant;",
        {"model": "Plant", "filename": "Plant.mo"},
    )


def test_modelica_loads_uses_existing_session():
    session = FakeSession(FakeRumocaModel())
    source = "model Plant end Plant;"

    model = modelica_loads(
        source,
        model_name="Plant",
        filename="Plant.mo",
        session=session,
    )

    assert isinstance(model, RumocaSymbolicModel)
    assert session.loads_call == (
        source,
        {"model": "Plant", "filename": "Plant.mo"},
    )


@pytest.mark.parametrize("version", ["0.9.20", "0.11.0"])
def test_modelica_load_requires_rumoca_0_10(monkeypatch, version):
    incompatible_rumoca = SimpleNamespace(
        version=lambda: version,
        Session=lambda **kwargs: FakeSession(FakeRumocaModel()),
    )
    monkeypatch.setitem(sys.modules, "rumoca", incompatible_rumoca)

    with pytest.raises(ImportError, match=f"found {version}"):
        modelica_load("Plant.mo")


def test_modelica_load_forwards_session_configuration(monkeypatch):
    session = FakeSession(FakeRumocaModel())
    calls = []
    current_rumoca = SimpleNamespace(
        version=lambda: "0.10.0",
        Session=lambda **kwargs: calls.append(kwargs) or session,
    )
    monkeypatch.setitem(sys.modules, "rumoca", current_rumoca)

    with tempfile.NamedTemporaryFile(suffix=".mo", mode="w") as source:
        source.write("model Plant end Plant;")
        source.flush()
        modelica_load(source.name, roots=["models"], workspace="work")

    assert calls == [{"roots": ["models"], "workspace": "work"}]


def test_render_requires_compiled_rumoca_model():
    with pytest.raises(TypeError, match="compiled rumoca.Model"):
        render_sympy_export(object())


@pytest.mark.integration
def test_live_rumoca_sympy_export():
    model_path = Path(__file__).parent / "models" / "direct_sympy.mo"
    with tempfile.TemporaryDirectory() as workspace:
        model = modelica_load(
            model_path,
            model_name="DirectSympy",
            roots=[],
            workspace=workspace,
            output_names=["e"],
        )
    x = model.symbolic.state_symbols[0]
    d, r = model.symbolic.input_symbols
    k = model.symbolic.param_symbols[0]

    assert sp.simplify(model.symbolic.f[0] + k * x) == 0
    assert model.symbolic.Bu == sp.Matrix([d])
    assert model.symbolic.h == sp.Matrix([x])
    assert model.symbolic.Du == sp.Matrix([-r])
    assert model.symbolic.E().shape == (1, 1)
    assert model.symbolic.F().shape == (1, 2)
