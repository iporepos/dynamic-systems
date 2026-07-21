# dynamic_systems.py

A small, self-contained Python module for simulating stock-and-flow
("levels and flows") dynamic systems with forward-Euler integration.
Written generically — nothing here is tied to any one project — but the
built-in rate functions (`linear_decay`, `weir_francis`) lean toward
reservoir/hydrology-style use.

**The module's own header docstring is the real documentation.** It's
long, thorough, and kept in sync with the code (Sphinx/reST style, with
worked examples). This file is just a map to it, plus a few things
worth remembering after time away.

## Requirements

`pandas`, `numpy`. `matplotlib` only if you use `plot_results()`.

## Quick start

```python
from dynamic_systems import DynamicSystemModel, plot_results

model = DynamicSystemModel()
model.load_data("inflow.csv")          # a CSV; columns are declared per-flow, not here
model.load_parameters("parameters.json")
model.run()                             # setup_model() + solve() + evaluate()

model.results            # combined levels + flows, at the simulation's own dt
model.results_upscaled   # same, resampled back onto the forcing data's native cadence
fig, _ = plot_results(model)
```

Or from the command line — everything (forcing path, config, outputs)
can live in one JSON file:

```bash
python dynamic_systems.py --config parameters.json
```

## Where things live in the docstring

Read top to bottom, or jump to the section you need:

- **Lifecycle** — the `load_data → load_parameters → setup_model → solve` order
- **Nomenclature** — what a Level and a Flow are, and how flows nest under a level's `"inflows"`/`"outflows"`
- **Forcing data** — how a flow's `"name"`/`"column"`/`"aggregation"` map onto CSV columns, with no naming convention required
- **The `"forcing"` key** — when (rarely) you need it separate from `"name"`
- **Forcing values and mass conservation across dt** — why `interpolation="previous"` is exact and `"linear"` (the default) isn't, and why the simulation horizon runs slightly past the raw data's last timestamp
- **Upscaled results** — how `results_upscaled` is built (sum for flows, mean for levels/stage)
- **Operational control** — the `"operation"` key (gate/withdrawal schedules)
- **Solver** — the Euler update itself
- **Stage-volume curves** — a level's own `"stage_volume_curve"`, shared by any weirs attached to it
- **Plotting** — per-item `"plotting"` (label/color) and global per-subplot y-ranges
- **Command-line interface** — `--config`, `"inputs"`, `"outputs"`, and their priority rules
- **Full worked example** — copy-pasteable CSV + JSON + both invocation styles

Individual classes/functions (`Flow`, `Level`, `StageVolumeCurve`,
`linear_decay`, `weir_francis`, ...) all have their own detailed
docstrings too — check those before re-deriving something from scratch.

## A few things worth remembering

- **`model.results` holds delivered amounts, not rates.** Flow columns
  are already `rate * dt` — summable directly. The rate form only
  exists internally, inside `step()`/`evaluate_flows()`.
- **`"previous"` vs `"linear"` interpolation is a real tradeoff, not a
  style choice.** Only `"previous"` conserves mass exactly. Check with
  `print_mass_balance_check(model)` if unsure which you're getting —
  note it also flags when a diff is just `"operation"` gating at work,
  not a real discrepancy.
- **A rate function needing a stage** (like `weir_francis`) gets its
  curve from its *level*, not from itself, unless the flow explicitly
  overrides it. Don't duplicate `"stage_volume_curve"` across every
  weir on the same reservoir.
- **`"inputs"`/`"outputs"`/`"plotting"` (top-level JSON keys) are read
  only by the CLI block**, never by `DynamicSystemModel` itself.
  Command-line flags always take priority over their JSON counterparts
  when both are given.
- Extend the module without editing it via `register_rate_function()`,
  `register_time_constant_param()`, `register_curve_param()`, and
  `register_rate_in_seconds_function()` — see the "Rate function
  registry" comment block near the top of the file for the conventions
  each one assumes.