# Structured Workflow Example

This example demonstrates the clean separation of concerns in CP_Reach:

```
Authoring (Modelica) → Compilation (Rumoca) → Analysis (cp_reach)
```

## Files

### Modelica Models (`models/`)

- **plant.mo** - Mass-spring-damper dynamics
- **controller.mo** - PD controller with feedforward
- **closed_loop.mo** - Complete closed-loop system

### Configuration Files

- **uncertainty.yaml** - Disturbance bounds, parameter uncertainty
- **reach_query.yaml** - What guarantee to compute

### Scripts

- **run_analysis.py** - End-to-end example

## Quick Start

```bash
# 1. Compile model with Rumoca
rumoca --json -m ClosedLoop models/closed_loop.mo > closed_loop.json

# 2. Run analysis
python -m cp_reach analyze \
    --ir closed_loop.json \
    --uncertainty uncertainty.yaml \
    --query reach_query.yaml \
    --output results/

# Or use the Python script
python run_analysis.py
```

## Workflow

1. **Author models** - Write plant/controller/estimator as separate `.mo` files
2. **Wire them up** - Create `closed_loop.mo` that connects components
3. **Compile** - `rumoca --json -m ClosedLoop closed_loop.mo > closed_loop.json`
4. **Specify uncertainty** - Edit `uncertainty.yaml` with bounds/covariances
5. **Specify query** - Edit `reach_query.yaml` with analysis parameters
6. **Analyze** - Run `cp_reach analyze --ir ... --uncertainty ... --query ...`

## Output

Results are saved to `results/`:
- `certificate.json` - Lyapunov certificate and bounds
- Plots (if enabled in query)
