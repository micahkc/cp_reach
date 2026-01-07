# CP_Reach: Inputs, Outputs, and Assumptions

## What CP_Reach Does

CP_Reach computes **guaranteed bounds** on how far a controlled system can deviate from its intended behavior when subject to disturbances.

**Use case for cyber attack analysis:** If an attacker can inject signals bounded by some magnitude, CP_Reach tells you the maximum damage they can cause to system state. This shrinks your search space from "anything could happen" to "the effect must be within these bounds."

---

## Inputs

### 1. System Model (JSON file)

**What it is:** A mathematical description of your system's dynamics, compiled from Modelica.

**Format:** JSON file produced by Rumoca compiler
```bash
rumoca --json -m ModelName model.mo > model.json
```

**Content:**
- State variables (position, velocity, etc.)
- How states evolve over time (differential equations)
- Control inputs and disturbance inputs
- Physical parameters (mass, damping, etc.)

**Example snippet:**
```json
{
  "model_name": "ClosedLoop",
  "states": ["x", "v"],
  "inputs": ["u", "d"],
  "parameters": {"m": 1.0, "k": 2.0}
}
```

---

### 2. Disturbance Bound (YAML file)

**What it is:** The maximum magnitude of the disturbance/attack signal.

**Format:** YAML file

**Content:**
```yaml
disturbances:
  d:                    # Name must match an input in your model
    type: bounded
    bound: 1.0          # |d(t)| ≤ 1.0 for all time
```

**This is the key parameter:** The bound you specify here directly determines the output bounds. If you say the attacker can inject at most 1.0, CP_Reach tells you the maximum effect of a 1.0-bounded attack.

---

### 3. Analysis Configuration (YAML file)

**What it is:** Settings for the analysis.

**Format:** YAML file

**Content:**
```yaml
query:
  type: flowpipe
  dynamics: error
  dist_inputs: [d]      # Which inputs are disturbances
```

---

## Outputs

### 1. State Bounds (the main result)

**What it is:** Maximum deviation in each state variable.

**Format:** Numerical values or JSON

**Example:**
```
Position error:  |e_x| ≤ 0.84 meters
Velocity error:  |e_v| ≤ 1.43 m/s
```

**What it means:** If an attacker injects a disturbance bounded by your specified limit, the system state will deviate by **at most** these amounts from the intended trajectory.

---

### 2. Certificate (optional)

**What it is:** Mathematical proof that the bounds are valid.

**Format:** Matrix P (n×n numpy array)

**What it means:** This is the Lyapunov function that proves the bounds. You don't need to understand it, but it can be used to verify the result independently.

---

## Assumptions

For the output bounds to be valid, these must be true:

| Assumption | What it means | If violated |
|------------|---------------|-------------|
| **Model is accurate** | The JSON model correctly describes the real system | Bounds don't apply to real system |
| **Disturbance bound is correct** | Attacker signal never exceeds specified bound | Actual deviation can exceed computed bounds |
| **System starts on trajectory** | Initial error is zero | Transient may exceed bounds initially |
| **Linear error dynamics** | Error grows/shrinks linearly with error magnitude | Use nonlinear analysis instead |

**The most important assumption:** The disturbance bound must be a true upper bound on what the attacker can inject. If the attacker can inject more than you specified, the bounds are invalid.

---

## Quick Reference

| Item | Format | Required? |
|------|--------|-----------|
| System model | `.json` (from Rumoca) | Yes |
| Disturbance bound | `.yaml` | Yes |
| Query config | `.yaml` | Optional (has defaults) |
| **Output: State bounds** | Numbers / JSON | Always produced |
| Output: Certificate | `.npy` matrix | Optional |

---

## Example Usage

```bash
# 1. Compile your Modelica model
rumoca --json -m ClosedLoop model.mo > model.json

# 2. Create uncertainty.yaml with your attack bound
# disturbances:
#   d:
#     type: bounded
#     bound: 1.0

# 3. Run analysis
python -m cp_reach analyze --ir model.json --uncertainty uncertainty.yaml

# 4. Get bounds
# Output: e_x ≤ 0.84, e_v ≤ 1.43
```

**Interpretation:** An attacker who can inject signals up to magnitude 1.0 can cause at most 0.84m position error and 1.43 m/s velocity error.
