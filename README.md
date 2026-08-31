# CP_REACH

CP_REACH is an interactive Python and Jupyter package for developing
cyber-physical vulnerability hypotheses. It shows how bounded disturbances can
change the reachable set, or flow-tube, of a vehicle system modeled in
Modelica.

The central question is:

> If a cyber effect can induce this physical disturbance, which vehicle states
> could become reachable, and could that envelope cross an unsafe boundary?

CP_REACH compiles a Modelica plant/controller model with RuMoCA, converts its
DAE representation to symbolic state-space dynamics, computes reachable-set
bounds, and plots those bounds alongside simulated trajectories. The result is
an analysis artifact that helps an engineer prioritize hypotheses for
evaluation with more detailed model variants; it is not, by itself, proof that
a cyber exploit exists.

## Analysis workflow

1. Start with a Modelica model of the vehicle plant and controller.
2. Add explicit input channels for hypothesized cyber-physical effects, such as
   actuator bias, command injection, sensor-induced control error, or an
   equivalent force/torque.
3. Bound each disturbance using units from the model and select an operating
   trajectory or region.
4. Compute a conservative reachable-set over-approximation with the LMI
   workflow, or use Monte Carlo simulation for a fast empirical approximation
   when theoretical analysis is cumbersome.
5. Inspect the flow-tube against safety-relevant state limits. A potential
   intersection becomes a vulnerability hypothesis to investigate.
6. Evaluate the hypothesis with the appropriate higher-fidelity Modelica
   variant from the shared CP_GLIMPSE model family, then use the result to
   refine the model and disturbance assumptions.

Formal over-approximations and sampled simulation answer different questions.
An LMI certificate can bound all disturbances covered by its assumptions.
Monte Carlo results only show the trajectories that were sampled and must not
be presented as a formal guarantee.

## Capabilities

- Modelica model compilation with RuMoCA 0.9.20
- LMI-based ellipsoidal reachable-set over-approximations
- Polytopic uncertainty for time-varying and nonlinear systems
- Monte Carlo disturbance simulation
- Trajectory generation and reference tracking
- SymPy and CasADi computation backends
- Flow-tube, trajectory, and error-bound visualization
- Rover, quadrotor, and satellite examples

## Installation

RuMoCA 0.9.20 requires Python 3.10 or newer. Install CP_REACH with Modelica
support from a checkout:

```bash
git clone https://github.com/micahkc/cp_reach.git
cd cp_reach
python -m pip install -e ".[modelica]"
```

For the tested notebook environment, including development, documentation, and
Lie-group dependencies:

```bash
python -m pip install -e ".[all]"
python -m pip install jupyterlab
jupyter lab
```

The prebuilt GHCR JupyterLab image is currently private. Authorized users can
run it locally after authenticating to GHCR; access can be granted to STR
evaluators.

## Minimal Python example

RuMoCA 0.9.20 uses its `Session` API and emits DAE schema 7. The isolated
workspace below prevents unrelated Modelica files from being discovered during
compilation.

```python
import tempfile
from pathlib import Path

import numpy as np
import rumoca

from cp_reach.ir import DaeIR, ir_to_symbolic_statespace
from cp_reach.reachability import solve_disturbance_LMI

model_name = "PendulumClosedLoop"
model_path = Path(
    "examples/general/models/pendulum_closed_loop.mo"
).resolve()

with tempfile.TemporaryDirectory() as workspace:
    session = rumoca.Session(roots=[], workspace=workspace)
    model = session.loads(
        model_path.read_text(),
        model=model_name,
        filename=model_path.name,
    )
    dae_json = model.to_json("dae")

ir = DaeIR.from_json_str(dae_json, model_name=model_name)
state_space = ir_to_symbolic_statespace(ir)

# The Modelica input named d represents the hypothesized disturbance.
disturbance_inputs = ["d"]
A, B_d = state_space.linearize_error_dynamics(
    disturbance_inputs=disturbance_inputs
)
solution = solve_disturbance_LMI(
    A_list=[A],
    B=B_d,
    w_max=0.1,
)

P_inverse = np.linalg.inv(np.asarray(solution["P"]))
mu = float(np.max(solution["mu"]))
state_bounds = np.sqrt(mu) * np.sqrt(np.diag(P_inverse))
print(state_bounds)
```

The LMI workflow assumes that the selected model, disturbance channels and
bounds, operating region, and solver result are valid. Record those assumptions
with every result so the resulting vulnerability hypothesis remains testable.

## Recommended notebooks

The notebooks currently executed by CI are:

- `examples/general/structured_nonlinear.ipynb`: nonlinear Modelica pendulum,
  polytopic LMI bounds, continuous-time checks, and Monte Carlo comparison
- `examples/quadrotor/quadrotor_flowpipe.ipynb`: nested quadrotor flow-tubes
- `examples/rover/rover_plots.ipynb`: rover safety-envelope visualization
- `examples/satellite/satellite_error_bounds.ipynb`: rendezvous error bounds

Open a notebook in JupyterLab, read its configuration cell first, then run all
cells. Treat the configuration, dependency versions, plots, solver status, and
saved numeric outputs as one analysis record.

## Package structure

- `cp_reach.ir`: RuMoCA DAE schema-7 loading and symbolic conversion
- `cp_reach.dynamics`: state-space representations and dynamics classification
- `cp_reach.reachability`: LMI solvers and simulation workflows
- `cp_reach.plotting`: flow-tube, trajectory, and error-bound visualization
- `cp_reach.planning`: reference-trajectory generation
- `cp_reach.physics`: rigid-body and SE(2,3) dynamics
- `cp_reach.applications`: rover, quadrotor, and satellite analyses

## Verification

Run the Python test suite:

```bash
pytest tests/ -v
```

Run the same four notebooks gated in CI:

```bash
pytest --nbmake -p no:cacheprovider --nbmake-timeout=600 -v \
  examples/quadrotor/quadrotor_flowpipe.ipynb \
  examples/rover/rover_plots.ipynb \
  examples/satellite/satellite_error_bounds.ipynb \
  examples/general/structured_nonlinear.ipynb
```

See [examples/](examples/) for additional implementation details and analyses.
