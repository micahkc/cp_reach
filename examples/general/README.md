# General reachability examples

The verified general workflow is:

```text
Modelica source
  -> RuMoCA 0.9.20 Session
  -> DAE schema 7 JSON
  -> CP_REACH symbolic state space
  -> disturbance model
  -> reachable-set bound and flow-tube
  -> cyber-physical vulnerability hypothesis
```

## Nonlinear pendulum workflow

[`structured_nonlinear.ipynb`](structured_nonlinear.ipynb) analyzes a controlled
pendulum with gravity nonlinearity.

The model is
[`models/pendulum_closed_loop.mo`](models/pendulum_closed_loop.mo):

- plant parameters: `m=1.0 kg`, `l=1.0 m`, `g=9.81 m/s²`, and damping
  `c=0.1`;
- controller: PID plus feedforward;
- explicit disturbance input: torque `d`; and
- example assumption: `|d| <= 0.1 N·m`.

The notebook compiles that source through the current RuMoCA Python API:

```python
import tempfile
from pathlib import Path

import rumoca

from cp_reach.ir import DaeIR, ir_to_symbolic_statespace

model_name = "PendulumClosedLoop"
model_path = Path("models/pendulum_closed_loop.mo").resolve()

with tempfile.TemporaryDirectory() as workspace:
    model = rumoca.Session(roots=[], workspace=workspace).loads(
        model_path.read_text(),
        model=model_name,
        filename=model_path.name,
    )
    dae_json = model.to_json("dae")

ir = DaeIR.from_json_str(dae_json, model_name=model_name)
state_space = ir_to_symbolic_statespace(ir)
```

An isolated RuMoCA workspace is intentional. RuMoCA discovers Modelica classes
in its workspace, and this example directory contains multiple exploratory
models with repeated class names.

The remainder of the notebook:

1. verifies that the error dynamics are nonlinear;
2. generates the reference motion `0 -> pi/4 -> 0` over four seconds;
3. bounds the Jacobian over the assumed state-error region;
4. solves a time-varying polytopic LMI;
5. checks the continuous-time certificate;
6. samples bounded disturbance trajectories; and
7. plots the flow-tube, error bounds, and time-varying ellipses.

The figures answer a physical consequence question. For example, if the torque
channel represents an actuator command effect, an envelope that intersects a
vehicle-specific attitude limit supports a hypothesis that the cyber effect
could create an unsafe attitude. It does not prove how an adversary reaches the
actuator command channel.

## Adapting the notebook to a vehicle model

Change one assumption at a time and preserve it with the result:

- `MODELICA_FILE` and `MODEL_NAME`: the plant/controller model under analysis;
- disturbance inputs: the model channels representing the hypothesized effect;
- `DISTURBANCE_BOUNDS`: maximum magnitudes and units;
- trajectory: the maneuver or operating condition under study;
- polytopic error bounds: the state region covered by nonlinear linearization;
- time grid and solver settings: numerical resolution and convergence choices;
  and
- safety limits: the physical thresholds used to interpret the flow-tube.

When an effect is awkward to over-approximate theoretically, use fast RuMoCA
simulation to sample it. Label that output as a Monte Carlo approximation, keep
the sample strategy and random seed, and do not treat it as a formal bound.

## Other files

- `models/`: Modelica sources and exploratory compiled models
- `structured_nonlinear.ipynb`: current CI-gated Modelica workflow
- `uncertainty.yaml` and `reach_query.yaml`: exploratory configuration inputs
- `run_analysis.py` and other general notebooks: exploratory, not part of the
  current CI-gated workflow

Use the four notebooks listed in [`../README.md`](../README.md) as the verified
integration surface.
