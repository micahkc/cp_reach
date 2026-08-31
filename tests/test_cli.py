"""CLI contract tests for the Rumoca 0.10-only workflow."""

import pytest

from cp_reach.cli import create_parser


def test_modelica_arguments_are_the_only_model_input():
    args = create_parser().parse_args(
        [
            "analyze",
            "--modelica",
            "Plant.mo",
            "--model",
            "Plant",
            "--root",
            "models",
            "--workspace",
            "build/rumoca",
        ]
    )

    assert args.modelica == "Plant.mo"
    assert args.model == "Plant"
    assert args.root == ["models"]
    assert args.workspace == "build/rumoca"
    assert not hasattr(args, "ir")


def test_json_ir_flag_is_rejected():
    with pytest.raises(SystemExit):
        create_parser().parse_args(["info", "--ir", "model.json"])
