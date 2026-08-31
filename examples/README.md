# CP_REACH examples

These notebooks demonstrate how to turn a physical disturbance assumption into
a reachable-set analysis and a cyber-physical vulnerability hypothesis.

## Recommended starting point

Start with
[`general/structured_nonlinear.ipynb`](general/structured_nonlinear.ipynb). It
shows the full Modelica-to-flow-tube path for a controlled nonlinear pendulum:

1. compile the Modelica source through a RuMoCA 0.9.20 `Session`;
2. load DAE schema 7 into CP_REACH;
3. identify the bounded disturbance input;
4. construct time-varying polytopic dynamics;
5. compute and continuously check LMI bounds;
6. compare the certified envelope with sampled disturbance trajectories; and
7. plot the state flow-tube and error ellipses.

Read the configuration cell before running it. In a new analysis, replace the
model, model name, disturbance limits, operating trajectory, and state-region
bounds with values justified for the target vehicle and hypothesized effect.

## CI-gated notebooks

The following notebooks are executed end to end in CI:

- [`general/structured_nonlinear.ipynb`](general/structured_nonlinear.ipynb)
- [`quadrotor/quadrotor_flowpipe.ipynb`](quadrotor/quadrotor_flowpipe.ipynb)
- [`rover/rover_plots.ipynb`](rover/rover_plots.ipynb)
- [`satellite/satellite_error_bounds.ipynb`](satellite/satellite_error_bounds.ipynb)

Other notebooks are exploratory and are not part of the current verified
workflow.

## Run interactively

From the repository root:

```bash
python -m pip install -e ".[all]"
python -m pip install jupyterlab
jupyter lab
```

Open a recommended notebook and run all cells. This project is used as a Python
package from notebooks; a command-line interface is not required for the STR
workflow.

## Interpreting results

- A certified flow-tube crossing a safety limit supports a hypothesis that the
  modeled disturbance can produce an unsafe physical state under the stated
  assumptions.
- A certified flow-tube remaining inside a limit is meaningful only over the
  model, bounds, and operating region covered by the certificate.
- A Monte Carlo trajectory crossing a limit is a useful counterexample or test
  case.
- Monte Carlo trajectories remaining inside a limit are not a proof of safety.
- Reachability does not establish a cyber access path or exploit. Evaluate the
  hypothesis with the appropriate higher-fidelity Modelica variant from the
  shared CP_GLIMPSE model family.
