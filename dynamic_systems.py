"""
dynamic_systems.py

Core abstractions for simulating stock-and-flow ("levels and flows") dynamic
systems using explicit (forward) Euler integration.

Lifecycle (the pattern this module is built around):

    model = LinearReservoirModel()      # a DynamicSystemModel subclass
    model.load_data(csv_path)           # -- can come first or second
    model.load_parameters(json_path)    # levels, flows, simulation settings
    model.setup_model()                 # merges data + parameters: builds levels,
                                         # flows, the simulation time grid, and
                                         # resamples forcing onto that grid
    model.solve()                       # runs the Euler integration
    model.evaluate()                    # placeholder for goodness-of-fit, later
    model.export(path=None)             # bundles/writes parameters + data + results

    model.run()                         # orchestrator: setup_model + solve + evaluate

load_data() and load_parameters() are independent and order-doesn't-matter --
each just loads its own file. setup_model() is the one place that actually
assembles everything, which is what makes this safe to call repeatedly: load
data once, then loop over parameter files (e.g. different k values, a Monte
Carlo batch of realizations) re-calling load_parameters() -> setup_model()
-> solve() on the SAME object, since the data never needs reloading.

Nomenclature
------------
Level : a stock / storage (a state variable), e.g. reservoir volume,
        soil moisture.
Flow  : a rate either crossing the system boundary or moving between
        levels. Two kinds:
          - "forcing"       : value comes from an external time series
                               (e.g. inflow, precipitation), already
                               resampled onto the model's own time grid.
          - "rate_function" : value is computed from the *current state*
                               via a registered function (e.g. linear
                               reservoir outflow, ET as a function of soil
                               moisture).

Flows are declared NESTED inside the level they act on -- each level entry
in parameters.json has its own "inflows" and "outflows" lists, e.g.:

    {"name": "storage", "initial_state": 10.0, "min": 0.0,
     "inflows":  [{"name": "inflow", "type": "forcing", "forcing": "inflow"}],
     "outflows": [{"name": "outflow", "type": "rate_function",
                   "function": "linear_decay", "params": {"residence_time": 160.0}}]}

so a flow's source/target is never written as a field that COULD disagree
with where it's attached -- nesting under "outflows" IS what makes storage
its source; there's no separate field to typo. For a flow between two
levels (a future cascade), nest it under one side only and give the other
endpoint explicitly ("target" on an outflow entry, or "source" on an inflow
entry, naming the other level) -- see _build_flows().

Data vs. parameters
--------------------
Only two files: one CSV (the forcing data) and one parameters JSON (levels
with their nested inflows/outflows, rate function params, and dt -- what
varies across realizations).
There's no separate data-manifest file, and the simulation horizon is not
set in parameters either: start/end come from the loaded forcing data's own
extent (see _build_time_grid) -- it's the data's job to say how long a
simulation makes sense, given dt. What a given forcing series IS -- its
column names, and
whether it should be summed or averaged when upscaled to a coarser dt -- is
a fixed property of that variable (rainfall is always summed, a river stage
is always averaged), not something that changes per realization. So it's
hardcoded in Python instead of configured in JSON: as a Forcing subclass
(see AccumulatingForcing below) wired up in a DynamicSystemModel subclass's
own load_data() override (see LinearReservoirModel below). The base
DynamicSystemModel.load_data() has no sensible generic implementation for
this reason -- subclass it and override load_data() for each concrete model.

Operational control ("operation" key)
--------------------------------------
By default every flow is completely free -- an inflow always delivers what
its forcing/rate function says, an outflow always drains at its natural
rate. That is the default behavior and stays that way unless a flow's JSON
entry carries an "operation" key.

"operation" represents an externally-imposed control fraction on that flow
-- e.g. an irrigation withdrawal schedule on an inflow, or a gate opening
schedule on an outflow -- and multiplies the flow's otherwise-computed
value at every timestep:

    actual_flow(t) = raw_flow(t) * operation_factor(t)

"operation" is a dict with a required "factors" list and an optional
"linear" flag, e.g.:

    "operation": {"factors": [0, 0, 1], "linear": true}

"factors" -- the control points:
  - a single number, or a list with one number -> a constant factor
    applied for the whole run
  - a list with 2+ numbers -> control points placed at equally spaced
    points across the simulation horizon: the first value sits at the
    very start, the last value at the very end, and the remaining values
    are equally spaced in between. Pass more control points for finer
    control.

"linear" -- how the control points are turned into a per-timestep series
between the control points:
  - absent, null, or false (the default) -> NEAREST-NEIGHBOR HOLD: every
    simulation timestep takes the value of whichever control point is
    nearest to it in time, producing a step-like operational series (e.g.
    a gate that snaps between fixed positions).
  - true -> LINEAR INTERPOLATION: the series ramps smoothly between
    consecutive control points instead of stepping, useful for a gradual
    opening/closing schedule rather than a sudden one. Irrelevant (and
    silently ignored) when there's only one control point, since a
    constant has nothing to interpolate between.

A shorthand is also accepted for convenience/backward compatibility: a
bare number or list (i.e. not wrapped in a dict) is treated exactly like
{"factors": <that value>, "linear": false} -- a plain list is always a
step function unless explicitly wrapped in a dict with "linear": true.

Absent or null "operation" entirely -> operation_factor(t) = 1.0
everywhere (the default, fully free, behavior). (A future extension may
allow "factors" to instead be a path to a sidecar CSV for a fully custom
series -- not implemented yet, kept simple for now.)

Every factor must lie in [0, 1] -- it represents a physical opening/gate/
withdrawal fraction, not an arbitrary multiplier. A value outside that
range is a modeling error: it is never clipped or rescaled, it raises
ValueError instead, on the theory that silently "fixing" a value that
violates the physical meaning of the parameter would hide a real mistake
in the input. See build_operation_factor() below.

Solver
------
Forward Euler only, for now:
    S(t + dt) = S(t) + dt * dS/dt(t)
dS/dt is assembled once per step from ALL flows evaluated against the same

state snapshot at t -- nothing is updated mid-step, so a cascade of levels
gives the same result regardless of flow order in the JSON. The cost is a
built-in one-step lag in any feedback where a downstream level constrains
an upstream flow -- fine as long as dt is small relative to the system's
response time.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rate function registry
# ---------------------------------------------------------------------------
# Shared signature: rate_function(state: dict[str, float], t, params) -> float
#
# `state` is the FULL level-state dict (not just the flow's own source
# level) so a downstream level can constrain an upstream flow. `params`
# always includes "source" and "target", injected from the flow's own
# definition, alongside whatever params.* the JSON specified.
#
# TIME-CONSTANT CONVENTION -- read this before adding a new rate function
# --------------------------------------------------------------------------
# Any parameter that represents a physical time constant (a residence time,
# a response time -- later: something derived from a hydraulic conductivity,
# etc.) is, BY CONVENTION, always written in HOURS in parameters.json,
# regardless of what dt_units the model itself runs at. This is a hardcoded,
# global rule -- not a per-parameter "units" field -- specifically to avoid
# the alternative: a flexible per-parameter units field is more "correct"
# in principle, but it means every config has to get every unit right, and
# every reader has to check rather than assume. One fixed convention, well
# documented, is more robust for a personal library than generality here.
#
# Mechanically: TIME_CONSTANT_PARAMS below declares which param names, for
# which rate functions, are time constants. DynamicSystemModel._build_flows
# converts each one from hours into the model's own dt_units ONCE, at setup
# time -- so the rate function itself (e.g. linear_decay) always receives
# an already-converted value and never has to think about units at all.
# Register new ones the same way you'd register a new rate function.

TIME_CONSTANT_PARAMS = {
    "linear_decay": {"residence_time"},
}


def register_time_constant_param(function_name, param_name):
    """Declare that `param_name` on `function_name` is an hours-denominated
    time constant, so _build_flows() converts it automatically. Use this
    alongside register_rate_function() when adding a function from outside
    this module.
    """
    TIME_CONSTANT_PARAMS.setdefault(function_name, set()).add(param_name)


#: how many of each dt_units make up one hour -- used to convert an
#: hours-denominated time-constant parameter into the model's own dt_units
_DT_UNITS_PER_HOUR = {"days": 1.0 / 24.0, "hours": 1.0, "minutes": 60.0, "seconds": 3600.0}


def hours_to_dt_units(value_hours, dt_units):
    """Convert a duration given in HOURS (the fixed convention for
    time-constant rate-function parameters) into the equivalent numeric
    value expressed in dt_units.
    """
    return value_hours * _DT_UNITS_PER_HOUR[dt_units]


def linear_decay(state, t, params):
    """Linear reservoir outflow via residence time, with an optional
    orifice/activation threshold:

        O = max(S_source - activation_level, 0) / tau

    Modeled as an orifice partway up the storage's wall rather than at the
    very bottom: activation_level is the dead storage below the orifice --
    however full it gets below that point, it produces no outflow through
    THIS orifice. The "head" driving the outflow is only the storage above
    that threshold. activation_level=0 (the default) recovers the plain
    bottom-orifice case, unchanged from before.

    Required params: {"residence_time": <in HOURS by convention -- see
    TIME_CONSTANT_PARAMS above>}. By the time this function runs, tau has
    already been converted into the model's own dt_units by _build_flows()
    -- this function never sees a raw hours value, and never needs to know
    what dt_units the model is running at.

    Optional params: {"activation_level": <same units as the level's own
    state, e.g. storage volume/depth>}. This is NOT a time constant -- it's
    a storage magnitude, so it is deliberately absent from
    TIME_CONSTANT_PARAMS and never gets touched by the hours conversion.
    """
    tau = params["residence_time"]
    activation = params.get("activation_level", 0.0)
    S = state[params["source"]]
    head = max(S - activation, 0.0)
    return head / tau


RATE_FUNCTIONS = {
    "linear_decay": linear_decay,
}


def register_rate_function(name, func):
    """Extend the registry from outside this module without editing it."""
    RATE_FUNCTIONS[name] = func


# ---------------------------------------------------------------------------
# Time series resampling -- standalone, reusable outside this module
# ---------------------------------------------------------------------------

def align_series_to_grid(series, target_index, interpolation="previous", aggregation="mean"):
    """Reconcile `series` (arbitrary DatetimeIndex) onto `target_index`
    (a regular DatetimeIndex -- the model's simulation grid), choosing
    automatically between:

      - UPSCALING (aggregation): triggered when the target step is coarser
        than the series' own native step. Native samples are grouped into
        each target interval [t_i, t_i+1) and reduced with `aggregation`
        (e.g. "sum" for rainfall depths, "mean" for a river stage).

      - DOWNSCALING (interpolation): triggered when the target step is
        equal to or finer than the native step. The series is reindexed
        onto the target grid and gaps are filled via `interpolation`:
        "previous" holds the last observed value (preserves total input
        volume under Euler integration -- the right default for a mass
        balance); "linear" interpolates smoothly instead.

    Kept as a standalone function -- not tied to Forcing or
    DynamicSystemModel -- so it can be reused anywhere a time series needs
    reconciling onto a different regular grid.
    """
    series = series.sort_index()
    target_index = pd.DatetimeIndex(target_index)

    native_dt = series.index.to_series().diff().median()
    target_dt = target_index.to_series().diff().median()

    if pd.isna(native_dt) or target_dt >= native_dt:
        resampled = (
            series.resample(target_dt, origin=target_index[0], label="left", closed="left")
                  .agg(aggregation)
        )
        aligned = resampled.reindex(target_index).ffill().bfill()
    else:
        combined_index = series.index.union(target_index).sort_values()
        expanded = series.reindex(combined_index)
        if interpolation == "previous":
            expanded = expanded.ffill()
        elif interpolation == "linear":
            expanded = expanded.interpolate(method="time")
        else:
            raise ValueError(f"Unknown interpolation mode: {interpolation!r}")
        aligned = expanded.reindex(target_index).ffill().bfill()

    aligned.index.name = target_index.name
    return aligned


# ---------------------------------------------------------------------------
# Operational control -- external factor applied on top of a flow's value
# ---------------------------------------------------------------------------

def build_operation_factor(operation_spec, time_index):
    """Build a per-timestep operational factor aligned to `time_index`, from
    a flow's "operation" JSON entry. See the module docstring's
    "Operational control" section for the full rules. Returns either:

      - None, meaning "no restriction" (caller should treat this as an
        implicit factor of 1.0 everywhere and may skip the multiplication
        entirely), or
      - a pandas Series indexed by `time_index`, holding the factor at
        every simulation timestep.

    Accepted `operation_spec`:
      - None                     -> returns None
      - {"factors": ..., "linear": ...} -- the canonical form. "factors" is
        required; "linear" is optional and defaults to False. See below.
      - a bare number or list    -> shorthand for
        {"factors": <that value>, "linear": False}, kept for convenience/
        backward compatibility.

    "factors" (after unwrapping the dict, if any):
      - a single number, or a list with one number -> constant factor for
        the whole run (in this case "linear" has no effect -- there's
        nothing to interpolate between).
      - a list with 2+ numbers -> control points at the simulation borders
        (first item at time_index[0], last item at time_index[-1]),
        remaining items equally spaced between them.

    "linear" controls how those control points become a per-timestep
    series:
      - False (default) -> NEAREST-NEIGHBOR HOLD: every timestep takes the
        value of its nearest control point, producing a step-like series.
      - True -> LINEAR INTERPOLATION: the series ramps smoothly between
        consecutive control points.

    Every value must be in [0, 1]. Values outside that range raise
    ValueError -- they are never clipped or rescaled, since silently
    "fixing" a value that violates the physical meaning of an operational
    factor (a gate/valve opening fraction) would hide a real input mistake.
    """
    if operation_spec is None:
        return None

    if isinstance(operation_spec, dict):
        if "factors" not in operation_spec or operation_spec["factors"] is None:
            raise ValueError(
                "operation dict is missing a 'factors' key -- expected "
                '{"factors": <number or list>, "linear": <optional bool>}.'
            )
        factors = operation_spec["factors"]
        linear = bool(operation_spec.get("linear") or False)
    else:
        factors = operation_spec
        linear = False

    if np.isscalar(factors):
        values = [float(factors)]
    else:
        values = [float(v) for v in factors]

    if not values:
        raise ValueError(
            "operation 'factors' list is empty -- omit the key, or use "
            "null, for the default unrestricted (factor = 1.0) behavior."
        )

    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"operation factor {v} is out of bounds -- every value must "
                "lie in [0, 1] (it represents a physical gate/valve opening "
                "or withdrawal fraction). Values are never clipped or "
                "rescaled -- fix the input instead."
            )

    if len(values) == 1:
        return pd.Series(values[0], index=time_index)

    control_times = pd.date_range(start=time_index[0], end=time_index[-1], periods=len(values))
    control_series = pd.Series(values, index=control_times)

    if linear:
        combined_index = control_times.union(time_index).sort_values()
        expanded = control_series.reindex(combined_index).interpolate(method="time")
        factor = expanded.reindex(time_index)
    else:
        factor = control_series.reindex(time_index, method="nearest")

    factor.index.name = time_index.name
    return factor


# ---------------------------------------------------------------------------
# Forcing: one raw data series + its own resampling strategy
# ---------------------------------------------------------------------------

class Forcing:
    """A single external time series, as loaded from CSV -- kept at its
    native resolution/timestamps.

    `aggregation` is a CLASS attribute, not a constructor argument: it's a
    fixed property of what kind of variable this is (a total/depth that
    should be summed when upscaled, vs. a state-like quantity that should be
    averaged), so it's set by subclassing rather than passed in per instance.
    Only two flavors needed so far -- this class (mean) and
    AccumulatingForcing below (sum) -- add more the same way if a different
    reducer is ever needed.

    `interpolation`, by contrast, IS a constructor argument: how to fill
    gaps when DOWNSCALING isn't a property of the variable itself, just of
    how you want to treat missing points for this particular run, so it
    stays switchable per instance.
    """

    aggregation = "mean"

    def __init__(self, path, time_col, value_col, interpolation="previous"):
        self.interpolation = interpolation
        df = pd.read_csv(path)
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        self.series = pd.Series(
            df[value_col].to_numpy(dtype=float),
            index=pd.DatetimeIndex(df[time_col]),
            name=value_col,
        )


class AccumulatingForcing(Forcing):
    """A forcing series that represents a period total/depth (e.g. rainfall)
    rather than an instantaneous rate or state -- summed, not averaged, when
    upscaled to a coarser dt.
    """

    aggregation = "sum"


# ---------------------------------------------------------------------------
# Level and Flow
# ---------------------------------------------------------------------------

class Level:
    """A single stock/storage."""

    def __init__(self, name, initial_state, min_state=None, max_state=None):
        self.name = name
        self.initial_state = float(initial_state)
        self.min_state = min_state
        self.max_state = max_state

    def clip(self, value):
        if self.min_state is not None:
            value = max(value, self.min_state)
        if self.max_state is not None:
            value = min(value, self.max_state)
        return value


class Flow:
    """A single flow, either external forcing -> level, or a rate function
    computed from current state, level -> level (or level -> external sink).

    `operation`, if given, is a pandas Series (indexed by the model's own
    time_index, built by build_operation_factor()) that multiplies the
    flow's otherwise-computed value at every timestep -- e.g. an irrigation
    withdrawal schedule on an inflow, or a gate opening schedule on an
    outflow. None means fully free (factor of 1.0 everywhere), which is the
    default. See the module docstring's "Operational control" section.
    """

    def __init__(self, name, kind, source=None, target=None,
                 forcing=None, function=None, params=None, operation=None):
        self.name = name
        self.kind = kind
        self.source = source
        self.target = target
        self.forcing = forcing
        self.function = function
        self.params = params or {}
        self.operation = operation

    def value(self, state, t, forcing_row):
        if self.kind == "forcing":
            raw = float(forcing_row[self.forcing])
        elif self.kind == "rate_function":
            func = RATE_FUNCTIONS[self.function]
            params = dict(self.params)
            params.setdefault("source", self.source)
            params.setdefault("target", self.target)
            raw = func(state, t, params)
        else:
            raise ValueError(f"Unknown flow kind: {self.kind!r}")

        if self.operation is None:
            return raw
        return raw * float(self.operation.loc[t])


# ---------------------------------------------------------------------------
# DynamicSystemModel -- the class meant to be subclassed
# ---------------------------------------------------------------------------

class DynamicSystemModel:
    """
    A generic levels-and-flows dynamic system, integrated with forward
    Euler. See module docstring for the lifecycle this class is built
    around (load_data / load_parameters / setup_model / solve / run).

    Subclass and override `derivatives()` and/or `apply_constraints()` for
    system-specific behavior (e.g. a cascade where a downstream level's
    state constrains an upstream flow, or soil-moisture overflow that must
    be re-routed as runoff rather than clipped).
    """

    def __init__(self):
        self.parameters = None
        self.data = {}
        self.levels = {}
        self.flows = []
        self.dt = None
        self.dt_units = None
        self.n_steps = None
        self.time_index = None
        self.forcing_on_grid = None
        self.results = None

    # -- loading (order-independent, no cross-dependencies) --------------

    def load_data(self, path):
        """Load forcing data. NOT implemented on the base class -- what a
        given model's forcing is (its columns, which Forcing subclass /
        aggregation applies) is fixed per model, not something to configure
        generically. Override this in a model subclass; see
        LinearReservoirModel below for the pattern.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override load_data() to construct "
            "its own Forcing object(s) from the given path -- see "
            "LinearReservoirModel for an example."
        )

    def load_parameters(self, path):
        """Parse the parameters file (levels, flows, simulation settings).
        Does not touch data or build anything -- see setup_model().
        """
        self.parameters = json.loads(Path(path).read_text())
        return self

    # -- setup: the one place data + parameters actually get merged ------

    def setup_model(self):
        """Build the simulation time grid, levels, flows, and resample all
        loaded forcing onto the grid. Time grid is built first because
        _build_flows() needs self.dt_units already known, to convert any
        hours-denominated time-constant parameters (see TIME_CONSTANT_PARAMS)
        into the model's own dt_units, AND needs self.time_index already
        known, to build each flow's "operation" series (see
        build_operation_factor()). Safe to call repeatedly -- e.g. after
        load_parameters(new_path) for a new realization against the same
        already-loaded data.
        """
        if self.parameters is None:
            raise RuntimeError("Call load_parameters() before setup_model().")
        self._build_time_grid()
        self._build_levels()
        self._build_flows()
        self._build_forcing_on_grid()
        return self

    def _build_levels(self):
        self.levels = {
            lv["name"]: Level(lv["name"], lv["initial_state"],
                               lv.get("min"), lv.get("max"))
            for lv in self.parameters["levels"]
        }

    def _build_flows(self):
        """Build Flow objects from each level's own nested "inflows" and
        "outflows" lists, instead of a flat top-level flows list.

        Nesting a flow under a level's "outflows" implies source=<that
        level>; under "inflows" implies target=<that level>. This removes
        the class of bug where a flow's source/target field silently
        doesn't match the level it's meant to act on -- there's no way to
        write it "attached" to the wrong level, because the attachment IS
        the nesting.

        For a flow between two levels (a cascade), nest it under ONE side
        only and give the OTHER endpoint explicitly: an outflow entry may
        still carry a "target" naming the downstream level; an inflow entry
        may still carry a "source" naming the upstream one. Do not list the
        same flow under both levels -- that double-counts it, and is
        checked for below.

        Also converts any hours-denominated time-constant parameters (per
        TIME_CONSTANT_PARAMS) into self.dt_units, and builds each flow's
        "operation" series (per build_operation_factor()), same as before.
        """
        self.flows = []
        for lv in self.parameters["levels"]:
            level_name = lv["name"]
            for fl in lv.get("inflows", []):
                self.flows.append(self._make_flow(fl, source=fl.get("source"), target=level_name))
            for fl in lv.get("outflows", []):
                self.flows.append(self._make_flow(fl, source=level_name, target=fl.get("target")))

        seen = set()
        for flow in self.flows:
            if flow.name in seen:
                raise ValueError(
                    f"Duplicate flow name {flow.name!r} -- flow names must be "
                    "unique across all levels' inflows/outflows (a cross-level "
                    "flow should be listed once, under one side, with the "
                    "other endpoint given explicitly)."
                )
            seen.add(flow.name)

    def _make_flow(self, fl, source, target):
        params = dict(fl.get("params") or {})
        for key in TIME_CONSTANT_PARAMS.get(fl.get("function"), ()):
            if key in params:
                params[key] = hours_to_dt_units(params[key], self.dt_units)
        try:
            operation = build_operation_factor(fl.get("operation"), self.time_index)
        except ValueError as exc:
            raise ValueError(f"Flow {fl.get('name')!r}: {exc}") from exc
        return Flow(
            name=fl["name"],
            kind=fl["type"],
            source=source,
            target=target,
            forcing=fl.get("forcing"),
            function=fl.get("function"),
            params=params,
            operation=operation,
        )

    #: maps a human-friendly dt_units value to the unit code pandas.Timedelta expects
    _TIMEDELTA_UNITS = {"days": "D", "hours": "h", "minutes": "min", "seconds": "s"}

    def _build_time_grid(self):
        """Build the simulation time grid.

        dt/dt_units come from parameters (dt_units defaults to "days" if
        omitted; also accepts "hours", "minutes", "seconds"). The HORIZON
        does not: start and end are taken directly from the loaded forcing
        data's own extent (earliest start, latest end across all loaded
        series) -- it's the data's job to say how long a simulation makes
        sense, not the parameters'. No override for this yet (e.g. a factor
        to run a shorter/longer window) -- deliberately left out for now,
        worth adding later if/when it's actually needed.
        """
        sim = self.parameters["simulation"]
        self.dt = float(sim["dt"])
        self.dt_units = sim.get("dt_units", "days")
        if self.dt_units not in self._TIMEDELTA_UNITS:
            raise ValueError(
                f"Unknown dt_units {self.dt_units!r}; use one of {list(self._TIMEDELTA_UNITS)}"
            )
        dt_timedelta = pd.Timedelta(self.dt, unit=self._TIMEDELTA_UNITS[self.dt_units])

        if not self.data:
            raise RuntimeError(
                "No forcing loaded -- call load_data() before setup_model(); "
                "the simulation horizon is derived from the forcing data's own extent."
            )
        start = min(f.series.index.min() for f in self.data.values())
        end = max(f.series.index.max() for f in self.data.values())

        self.n_steps = int((end - start) / dt_timedelta)  # truncates any partial final step
        self.time_index = pd.date_range(start=start, periods=self.n_steps + 1, freq=dt_timedelta)

    def _build_forcing_on_grid(self):
        columns = {
            name: align_series_to_grid(
                forcing.series, self.time_index,
                interpolation=forcing.interpolation,
                aggregation=forcing.aggregation,
            )
            for name, forcing in self.data.items()
        }
        self.forcing_on_grid = (
            pd.DataFrame(columns) if columns else pd.DataFrame(index=self.time_index)
        )

    # -- core dynamics -----------------------------------------------------

    def evaluate_flows(self, state, t):
        """Value of every flow at time t, given current state. Forcing
        flows are simple lookups into the already-resampled grid -- all
        the interpolation/aggregation work happened once in setup_model().
        Any flow with an "operation" entry has its raw value multiplied by
        that flow's operational factor at t -- see Flow.value().
        """
        forcing_row = self.forcing_on_grid.loc[t] if not self.forcing_on_grid.empty else pd.Series(dtype=float)
        return {flow.name: flow.value(state, t, forcing_row) for flow in self.flows}

    def derivatives_from_flow_values(self, flow_values):
        """dS/dt for every level, given already-evaluated flow values."""
        d = {name: 0.0 for name in self.levels}
        for flow in self.flows:
            q = flow_values[flow.name]
            if flow.source is not None:
                d[flow.source] -= q
            if flow.target is not None:
                d[flow.target] += q
        return d

    def derivatives(self, state, t):
        """dS/dt for every level at time t, given the current state.

        Every flow is evaluated against the SAME state snapshot -- nothing
        is updated mid-loop. Override this in a subclass for coupling this
        generic sum-of-flows form can't express. (For the individual flow
        values too, e.g. in a subclassed step(), call evaluate_flows()
        directly.)
        """
        return self.derivatives_from_flow_values(self.evaluate_flows(state, t))

    def apply_constraints(self, state):
        """Clip each level to its configured [min, max].

        NOT mass-conserving: anything clipped off simply vanishes. Fine for
        a hard floor like an empty reservoir. For a level where the excess
        has to go somewhere physically (e.g. soil moisture overflowing into
        surface runoff), override this in a subclass and re-route the
        excess as an explicit flow instead of clipping it away.
        """
        return {name: self.levels[name].clip(value) for name, value in state.items()}

    def step(self, state, t):
        """Single forward-Euler step. Returns (new_state, flow_values)."""
        flow_values = self.evaluate_flows(state, t)
        d = self.derivatives_from_flow_values(flow_values)
        new_state = {name: state[name] + self.dt * d[name] for name in state}
        new_state = self.apply_constraints(new_state)
        return new_state, flow_values

    # -- solve / evaluate / orchestrate -----------------------------------

    def solve(self):
        """Integrate the full time grid built by setup_model(). Returns and
        stashes self.results -- ONE DataFrame with both level and flow
        columns, indexed by time. Row t holds the level state AT t together
        with the flow values computed FROM that same state (i.e. the flux
        during the step from t to t+dt) -- so a flow column and a level
        column on the same row share the same underlying state, which is
        what makes them meaningful to plot side by side. The final row has
        levels only: there's no flow beyond the last step of the horizon.
        """
        if self.time_index is None:
            raise RuntimeError("Call setup_model() before solve().")
        state = {name: lv.initial_state for name, lv in self.levels.items()}
        records = []
        for t in self.time_index[:-1]:
            new_state, flow_values = self.step(state, t)
            records.append(dict(t=t, **state, **flow_values))
            state = new_state
        records.append(dict(t=self.time_index[-1], **state))

        self.results = pd.DataFrame.from_records(records).set_index("t")
        return self.results

    def evaluate(self):
        """Placeholder for goodness-of-fit assessment against observed
        data. Returns None for now -- override in a subclass once a path
        for loading observed data exists. Kept as a separate step (rather
        than folded into solve()) so run() has one obvious place to grow.
        """
        return None

    def run(self):
        """Convenience one-shot orchestrator: setup_model -> solve ->
        evaluate. For repeated realizations against the same data (e.g. a
        Monte Carlo batch), call load_parameters() -> setup_model() ->
        solve() directly in a loop instead -- skipping run()/evaluate()
        each time is usually what you want there.
        """
        self.setup_model()
        self.solve()
        self.evaluate()
        return self.results

    # -- export -----------------------------------------------------------

    def export(self, path=None):
        """Bundle parameters, raw forcing data, and simulation results (one
        combined levels+flows DataFrame). Returns the bundle as an
        in-memory dict; if `path` is given, also writes it to disk under
        that directory.
        """
        bundle = {
            "parameters": self.parameters,
            "data": {name: forcing.series for name, forcing in self.data.items()},
            "results": self.results,
        }
        if path is not None:
            outdir = Path(path)
            (outdir / "data").mkdir(parents=True, exist_ok=True)
            (outdir / "parameters.json").write_text(json.dumps(self.parameters, indent=2))
            for name, series in bundle["data"].items():
                series.to_csv(outdir / "data" / f"{name}.csv")
            if self.results is not None:
                self.results.to_csv(outdir / "results.csv")
        return bundle


# ---------------------------------------------------------------------------
# LinearReservoirModel -- concrete example subclass
# ---------------------------------------------------------------------------
# This is the pattern for every concrete model: subclass DynamicSystemModel
# and override load_data() to hardcode how its specific forcing CSV maps to
# Forcing object(s) -- column names, and which Forcing subclass (i.e. which
# aggregation) applies. Everything else (setup_model, solve, run, export) is
# inherited unchanged.

class LinearReservoirModel(DynamicSystemModel):
    """Single linear reservoir: inflow forcing (a rate, so averaged when
    upscaled) draining via linear_decay. See example_parameters.json for the
    matching levels/flows definition.
    """

    def load_data(self, path, interpolation="previous"):
        self.data = {
            "inflow": Forcing(path, time_col="date", value_col="value",
                               interpolation=interpolation),
        }
        return self


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(model, results=None, flows_to_plot=None,
                  levels_to_plot=None, figsize=(9, 6), title=None):
    """Two stacked subplots sharing the time axis:

        top    -> flows (e.g. input forcing vs. computed output)
        bottom -> level state trajectories

    Defaults to model.results (the one combined DataFrame populated by
    solve()) -- which columns go in which subplot is worked out from
    model.levels / model.flows, not from separate DataFrames. Kept as a
    plain function, not a model method -- visualization is meant to move
    out of this file once it grows, same reasoning as the rate-function
    registry.
    """
    import matplotlib.pyplot as plt

    if results is None:
        results = model.results

    level_cols = levels_to_plot or list(model.levels.keys())
    flow_cols = flows_to_plot or [flow.name for flow in model.flows]

    fig, (ax_flows, ax_levels) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if flow_cols:
        for col in flow_cols:
            ax_flows.plot(results.index, results[col], label=col)
        ax_flows.set_ylabel("flow")
        ax_flows.set_title("Input / output flows")
        ax_flows.legend(loc="best")
    else:
        ax_flows.set_visible(False)

    for col in level_cols:
        ax_levels.plot(results.index, results[col], label=col, color="tab:blue")
    ax_levels.set_ylabel("level (storage)")
    ax_levels.set_xlabel("time")
    ax_levels.set_title("Level")
    ax_levels.legend(loc="best")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_flows, ax_levels)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# python dynamic_systems.py example_inflow.csv example_parameters.json
# python dynamic_systems.py example_inflow.csv example_parameters.json --save out.png
# python dynamic_systems.py example_inflow.csv example_parameters.json --export run01/
#
# Uses LinearReservoirModel -- swap in a different DynamicSystemModel
# subclass here as more concrete models are added.

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a LinearReservoirModel and plot the result."
    )
    parser.add_argument("data", help="Path to the forcing CSV")
    parser.add_argument("parameters", help="Path to the parameters JSON (levels/flows/simulation)")
    parser.add_argument("--save", metavar="PNG_PATH", default=None,
                         help="Save the plot to this path instead of opening a window")
    parser.add_argument("--export", metavar="DIR", default=None,
                         help="Also export parameters/data/results to this directory")
    args = parser.parse_args()

    model = LinearReservoirModel()
    model.load_data(args.data)
    model.load_parameters(args.parameters)
    model.run()

    if args.export:
        model.export(path=args.export)
        print(f"Exported to {args.export}")

    fig, _ = plot_results(model)

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved plot to {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()