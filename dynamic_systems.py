r"""
dynamic_systems.py
===================

Core abstractions for simulating stock-and-flow ("levels and flows") dynamic
systems using explicit (forward) Euler integration.

Lifecycle
---------

The pattern this module is built around:

.. code-block:: python

    model = DynamicSystemModel()        # usable directly -- see "Forcing
                                         # data" below for how columns get
                                         # linked, no subclassing required
                                         # for the common case
    model.load_data(csv_path)           # -- can come first or second
    model.load_parameters(json_path)    # levels, flows, simulation settings
    model.setup_model()                 # merges data + parameters: builds
                                         # levels, flows, the simulation time
                                         # grid, and resamples forcing onto it
    model.solve()                       # runs the Euler integration
    model.evaluate()                    # placeholder for goodness-of-fit
    model.export(path=None)             # bundles/writes parameters + data +
                                         # results

    model.run()                         # orchestrator: setup_model + solve
                                         # + evaluate

``load_data()`` and ``load_parameters()`` are independent and order doesn't
matter -- each just loads its own file. ``setup_model()`` is the one place
that actually assembles everything, which is what makes this safe to call
repeatedly: load data once, then loop over parameter files (e.g. different
k values, a Monte Carlo batch of realizations), re-calling
``load_parameters()`` -> ``setup_model()`` -> ``solve()`` on the SAME
object, since the data never needs reloading.

Nomenclature
------------

Level
    A stock / storage (a state variable), e.g. reservoir volume, soil
    moisture.

Flow
    A rate either crossing the system boundary or moving between levels.
    Two kinds:

    - ``"forcing"`` : value comes from an external time series (e.g.
      inflow, precipitation), already resampled onto the model's own
      time grid.
    - ``"rate_function"`` : value is computed from the *current state*
      via a registered function (e.g. linear reservoir outflow, ET as a
      function of soil moisture).

Flows are declared **nested** inside the level they act on -- each level
entry in ``parameters.json`` has its own ``"inflows"`` and ``"outflows"``
lists, e.g.:

.. code-block:: json

    {
      "name": "storage",
      "initial_state": 10.0,
      "min": 0.0,
      "inflows": [
        {"name": "inflow", "type": "forcing", "forcing": "inflow"}
      ],
      "outflows": [
        {
          "name": "outflow",
          "type": "rate_function",
          "function": "linear_decay",
          "params": {"residence_time": 160.0}
        }
      ]
    }

so a flow's source/target is never written as a field that COULD disagree
with where it's attached -- nesting under ``"outflows"`` IS what makes
``storage`` its source; there's no separate field to typo. For a flow
between two levels (a future cascade), nest it under one side only and
give the other endpoint explicitly (``"target"`` on an outflow entry, or
``"source"`` on an inflow entry, naming the other level) -- see
:meth:`DynamicSystemModel._build_flows`.

Forcing data: column and aggregation resolution
--------------------------------------------------

Only two files are needed: one CSV (the forcing data) and one parameters
JSON (levels with their nested inflows/outflows, rate function params,
and ``dt``). There's no separate data-manifest file, and the simulation
horizon is not set in parameters either: start/end come from the loaded
forcing data's own extent (see :meth:`DynamicSystemModel._build_time_grid`)
-- it's the data's job to say how long a simulation makes sense, given
``dt``.

:meth:`DynamicSystemModel.load_data` is generic and usable directly, with
no subclassing needed for the common case: it just reads a CSV once into
a raw table. Which columns of that table actually get used, what each
one is called internally, and how each is aggregated when upscaled to a
coarser ``dt`` are ALL declared per flow, directly in ``parameters.json``
-- there's no Python-side naming convention or column map to keep in
sync. Every forcing-type flow entry may carry:

- ``"column"`` (optional): the exact CSV column name to read. If absent
  or ``null``, the flow's own ``"name"`` is used as the column name
  instead -- i.e. by default a flow is assumed to be named after its
  column.
- ``"aggregation"`` (optional): how to reduce that column when upscaling
  to a coarser ``dt`` (e.g. ``"sum"`` for a rainfall depth, ``"mean"``
  for a river stage -- any reducer :meth:`pandas.Series.resample`
  understands). Defaults to ``"mean"``.

If the resolved column isn't found in the CSV, :meth:`DynamicSystemModel.
setup_model` raises :class:`ValueError` immediately, listing the
available columns -- not partway through :meth:`DynamicSystemModel.solve`.
If two flows declare the same ``"forcing"`` key but disagree about which
column or aggregation it should resolve to, that also raises
:class:`ValueError`: two flows may freely *share* one column (each still
gets its own independent ``"operation"`` schedule -- see below), but a
single forcing key can't mean two different things at once.

Example flow entry, reading from an arbitrarily-named gauge column
rather than one matching its own ``"name"``:

.. code-block:: json

    {
      "name": "inflow",
      "column": "runoff_Station_2003",
      "type": "forcing",
      "forcing": "inflow",
      "operation": {"factors": [0, 0, 1], "linear": true}
    }

``"forcing"`` is a separate concept from ``"name"``/``"column"``: it's
the key this flow reads from in the resolved data (``self.data`` /
``self.forcing_on_grid``), used at simulation time by
:meth:`Flow.value`. ``"name"``/``"column"``/``"aggregation"`` only
control how that key gets built in the first place, from the CSV. See
:meth:`DynamicSystemModel._resolve_data_from_flows` for the resolution
logic itself.

This column/aggregation resolution depends on ``parameters.json``, so it
can't happen inside :meth:`~DynamicSystemModel.load_data` the way a
column mapping might in a simpler design -- at that point the flows
aren't known yet. Instead ``load_data()`` just reads the raw table once,
and :meth:`~DynamicSystemModel.setup_model` resolves ``self.data`` from
the flows first, before the rest of the usual setup pipeline (time grid,
levels, flows, resampling).

Override :meth:`~DynamicSystemModel.load_data` in a subclass only for a
genuinely different data shape that this raw-table-plus-JSON-columns
pattern can't express -- e.g. several separate files, a database query,
or an in-memory array. A subclass that overrides ``load_data()`` and
populates ``self.data`` directly (instead of leaving it to
``_resolve_data_from_flows()``) is respected: resolution only runs
against a raw table if one was actually loaded.

The ``"forcing"`` key: default behavior and edge cases
----------------------------------------------------------

Most of the time ``"forcing"`` can simply be omitted. When it is, it
defaults to the flow's own ``"name"``:

.. code-block:: python

    forcing_key = fl.get("forcing", fl["name"])

so ``"name"`` ends up doing triple duty by default: it's the flow's own
label, the CSV column read (unless ``"column"`` overrides it -- see
above), AND the internal key the resolved series is stored/looked-up
under. This is the common case and needs no special handling:

.. code-block:: json

    {"name": "inflow", "type": "forcing"}

reads column ``"inflow"``, stores the resolved series under forcing key
``"inflow"``, and that's the whole story -- most flows never need to
write ``"forcing"`` at all.

``"forcing"`` only earns its keep as a SEPARATE field from ``"name"`` in
one situation: when two flows should read the exact same resolved
series, without either flow's own label having to change to make that
happen. Give both flows the SAME explicit ``"forcing"`` value:

.. code-block:: json

    {"name": "natural", "column": "inflow", "type": "forcing", "forcing": "shared"},
    {
      "name": "diverted", "column": "inflow", "type": "forcing", "forcing": "shared",
      "operation": {"factors": [0, 1]}
    }

:meth:`DynamicSystemModel._resolve_data_from_flows` then builds the
underlying :class:`Forcing` object exactly once (``if forcing_key not in
self.data:``) and both flows share it -- each still gets its own
``"name"``, its own results column, and its own independent
``"operation"`` schedule; only the raw resampled series itself is
shared, so a later change to ``"column"`` only has to be made in one
place instead of two, with no risk of the two flows silently drifting
apart from each other.

That sharing pattern relies on two separate guardrails, both enforced at
:meth:`~DynamicSystemModel.setup_model` time rather than partway through
:meth:`~DynamicSystemModel.solve`:

- **Two flows may not give the same ``"forcing"`` key two different
  meanings.** If they share a ``"forcing"`` key but specify different
  ``"column"``/``"aggregation"`` values, that's a genuine contradiction
  -- :class:`ValueError`, from :meth:`_resolve_data_from_flows`, not
  silently resolved by picking one.
- **``"name"`` must still be unique across all flows**, independently of
  ``"forcing"``. Two flows sharing a ``"forcing"`` key must still have
  distinct ``"name"``s, as in the ``"natural"``/``"diverted"`` example
  above -- this is checked separately, in
  :meth:`DynamicSystemModel._build_flows`.

If a config manages to violate both guardrails at once -- the same
``"name"`` reused AND a conflicting ``"forcing"``/``"column"`` pairing
-- the forcing-key conflict is what actually gets reported, since data
resolution runs before flow construction in
:meth:`~DynamicSystemModel.setup_model`; the duplicate-name check is
never reached. Fixing the ``"forcing"`` conflict first will then surface
the duplicate-name error underneath it, if it's still there.

Forcing values and mass conservation across dt
--------------------------------------------------

Every forcing value in this module is taken to represent a quantity
delivered over ONE of its own native timesteps -- e.g. hourly inflow
data's value is "how much arrived during that hour", not an
instantaneous rate sampled at that instant. But every flow value
:meth:`DynamicSystemModel.step`'s Euler update actually consumes must be
in state-units per ONE ``dt_unit`` (so that ``dt * flow_value`` is
dimensionally valid regardless of the numeric ``dt``). Those two things
only coincide by coincidence -- e.g. hourly data run with
``dt_units="hours"`` -- and diverge whenever they don't, most commonly
when simulating at a finer ``dt`` than the forcing data's own native
resolution (a common real case: hourly gauge records, sub-hourly
routing).

:func:`align_series_to_grid` (called from
:meth:`DynamicSystemModel._build_forcing_on_grid`) handles this
automatically, with no extra parameter needed: it derives the forcing
series' own native sampling interval directly from its timestamps, and
rescales every value into "quantity per ONE ``dt_unit``" accordingly,
before it's ever read by :meth:`Flow.value`. Concretely, downscaling
(finer ``dt`` than native) splits each native value proportionally
across however many simulation timesteps fall within it, rather than
replicating it unchanged -- so a peak value from coarse data
legitimately comes out smaller, in per-``dt_unit`` terms, at finer
resolution; that's mass conservation working as intended, not a loss of
information. See that function's docstring for the exact math, which
also covers how this composes with ``"aggregation"`` during upscaling.

This "per ONE ``dt_unit``" form is what :meth:`Flow.value`,
:meth:`~DynamicSystemModel.evaluate_flows`, and
:meth:`~DynamicSystemModel.step`'s Euler update all work with
internally -- it's what makes ``dS/dt`` well-defined independent of
``dt``. It is NOT what ends up in ``model.results``, though:
:meth:`~DynamicSystemModel.solve` converts to each flow's DELIVERED
AMOUNT for that step (``rate * dt``) at the point of reporting, purely
so the results table is directly interpretable and summable -- see that
method's docstring for the split between internal rate and reported
amount.

This bucket interpretation also means a series' LAST recorded value
still owns a full native-interval-wide bucket, even though there's no
later sample to mark where that bucket ends. Left alone, the simulation
horizon would stop exactly at that last timestamp and silently drop this
final bucket entirely -- so :meth:`DynamicSystemModel._build_time_grid`
extends ``end`` by each series' own native interval, specifically to
cover it. A practical consequence: the simulation horizon runs slightly
past the raw data's own last timestamp (by the coarsest loaded series'
native interval), which can shift where ``"operation"`` control points
land, since those are placed relative to the overall horizon -- see the
"Operational control" section below.

``interpolation="linear"`` is NOT compatible with exact mass
conservation under this bucket convention -- it smooths values across
bucket boundaries rather than respecting them, so summed totals will be
close but not exact. ``"previous"`` (the default) is the only mode that
conserves mass exactly, since it reproduces each bucket's own value
verbatim before rescaling. Use ``"linear"`` only where a smoother-looking
series matters more than exact volume conservation.

Operational control (the ``"operation"`` key)
-----------------------------------------------

By default every flow is completely free -- an inflow always delivers
what its forcing/rate function says, an outflow always drains at its
natural rate. That is the default behavior and stays that way unless a
flow's JSON entry carries an ``"operation"`` key.

``"operation"`` represents an externally-imposed control fraction on that
flow -- e.g. an irrigation withdrawal schedule on an inflow, or a gate
opening schedule on an outflow -- and multiplies the flow's
otherwise-computed value at every timestep:

.. math::

    Q_{actual}(t) = Q_{raw}(t) \cdot f_{op}(t)

``"operation"`` is a dict with a required ``"factors"`` list and an
optional ``"linear"`` flag, e.g.:

.. code-block:: json

    {"factors": [0, 0, 1], "linear": true}

``"factors"`` -- the control points:

- a single number, or a list with one number -> a constant factor
  applied for the whole run.
- a list with 2+ numbers -> control points placed at equally spaced
  points across the simulation horizon: the first value sits at the
  very start, the last value at the very end, and the remaining values
  are equally spaced in between. Pass more control points for finer
  control.

``"linear"`` -- how the control points are turned into a per-timestep
series between the control points:

- absent, ``null``, or ``false`` (the default) -- **nearest-neighbor
  hold**: every simulation timestep takes the value of whichever control
  point is nearest to it in time, producing a step-like operational
  series (e.g. a gate that snaps between fixed positions).
- ``true`` -- **linear interpolation**: the series ramps smoothly between
  consecutive control points instead of stepping, useful for a gradual
  opening/closing schedule rather than a sudden one. Irrelevant (and
  silently ignored) when there's only one control point, since a
  constant has nothing to interpolate between.

A shorthand is also accepted for convenience/backward compatibility: a
bare number or list (i.e. not wrapped in a dict) is treated exactly like
``{"factors": <that value>, "linear": false}`` -- a plain list is always
a step function unless explicitly wrapped in a dict with
``"linear": true``.

Absent or ``null`` ``"operation"`` entirely means :math:`f_{op}(t) = 1`
for all :math:`t` (the default, fully free, behavior). (A future
extension may allow ``"factors"`` to instead be a path to a sidecar CSV
for a fully custom series -- not implemented yet, kept simple for now.)

Solver
------

Forward Euler only, for now:

.. math::

    S(t + \Delta t) = S(t) + \Delta t \cdot \frac{dS}{dt}(t)

:math:`dS/dt` is assembled once per step from ALL flows evaluated against
the same state snapshot at :math:`t` -- nothing is updated mid-step, so a
cascade of levels gives the same result regardless of flow order in the
JSON. The cost is a built-in one-step lag in any feedback where a
downstream level constrains an upstream flow -- fine as long as ``dt`` is
small relative to the system's response time.

Full worked example
--------------------

A minimal, self-contained run -- copy both files as-is to try it. Note
the inflow's ``"column": "value"`` below: the CSV's column is called
``"value"``, not ``"inflow"``, so the flow spells out the override
explicitly instead of relying on the ``"name"``-as-column default.

``inflow.csv`` (the forcing data):

.. code-block:: text

    date,value
    2020-01-01,5.0
    2020-01-02,5.0
    2020-01-03,5.0
    2020-01-04,5.0
    2020-01-05,5.0
    2020-01-06,5.0
    2020-01-07,5.0
    2020-01-08,5.0
    2020-01-09,5.0
    2020-01-10,5.0

``parameters.json`` (levels/flows/simulation -- includes a gated outflow
and a ramped-up irrigation inflow, to exercise ``"operation"`` as well):

.. code-block:: json

    {
      "simulation": {"dt": 1, "dt_units": "days"},
      "levels": [
        {
          "name": "storage",
          "initial_state": 10.0,
          "min": 0.0,
          "inflows": [
            {
              "name": "inflow",
              "column": "value",
              "type": "forcing",
              "forcing": "inflow",
              "operation": {"factors": [0, 0, 1], "linear": true}
            }
          ],
          "outflows": [
            {
              "name": "outflow",
              "type": "rate_function",
              "function": "linear_decay",
              "params": {"residence_time": 48.0, "activation_level": 2.0},
              "operation": {"factors": [0, 1]}
            }
          ]
        }
      ]
    }

Running it:

.. code-block:: python

    from dynamic_systems import DynamicSystemModel, plot_results

    model = DynamicSystemModel()
    model.load_data("inflow.csv")
    model.load_parameters("parameters.json")
    model.run()

    print(model.results)          # combined levels + flows DataFrame
    fig, _ = plot_results(model)
    fig.savefig("storage.png")

Or from the command line:

.. code-block:: bash

    python dynamic_systems.py inflow.csv parameters.json --save storage.png
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
    """Declare that a rate-function parameter is an hours-denominated
    time constant, so :meth:`DynamicSystemModel._build_flows` converts
    it automatically.

    Use this alongside :func:`register_rate_function` when adding a rate
    function from outside this module.

    :param function_name: Name the rate function is registered under in
        :data:`RATE_FUNCTIONS`.
    :type function_name: str
    :param param_name: Name of the parameter, within that function's
        ``params``, that represents a time constant expressed in hours.
    :type param_name: str
    :returns: None
    """
    TIME_CONSTANT_PARAMS.setdefault(function_name, set()).add(param_name)


#: how many of each dt_units make up one hour -- used to convert an
#: hours-denominated time-constant parameter into the model's own dt_units
_DT_UNITS_PER_HOUR = {"days": 1.0 / 24.0, "hours": 1.0, "minutes": 60.0, "seconds": 3600.0}


def hours_to_dt_units(value_hours, dt_units):
    """Convert a duration given in hours into the equivalent value
    expressed in ``dt_units``.

    Hours is the fixed convention for time-constant rate-function
    parameters (see :data:`TIME_CONSTANT_PARAMS`), regardless of what
    ``dt_units`` a given model runs at.

    :param value_hours: Duration in hours.
    :type value_hours: float
    :param dt_units: Target unit; one of ``"days"``, ``"hours"``,
        ``"minutes"``, ``"seconds"``.
    :type dt_units: str
    :returns: The duration expressed in ``dt_units``.
    :rtype: float
    """
    return value_hours * _DT_UNITS_PER_HOUR[dt_units]


# CURVE-PARAM CONVENTION -- for params that must be BUILT, not just converted
# --------------------------------------------------------------------------
# Some rate-function params aren't scalars at all -- e.g. weir_francis()
# below needs a whole stage-storage relationship, built once from a JSON
# spec (a CSV path or inline stage/volume lists) into a queryable object
# (see StageVolumeCurve). That construction -- reading a CSV, fitting a
# curve -- must happen ONCE, at setup time, exactly like the hours ->
# dt_units conversion above, and for the same reason: doing it inside the
# rate function itself would mean re-reading a file or re-fitting a curve
# on every single timestep.
#
# CURVE_PARAMS below declares which param names, for which rate functions,
# hold a raw JSON spec that needs building into an object.
# DynamicSystemModel._build_flows resolves each one ONCE, at setup time --
# so the rate function itself (e.g. weir_francis) always receives an
# already-built object and never touches JSON or a file path directly.

CURVE_PARAMS = {
    "weir_francis": {"stage_volume_curve"},
}


def register_curve_param(function_name, param_name):
    """Declare that a rate-function parameter holds a raw JSON spec that
    must be built into an object once, at setup time, rather than being
    read directly by the rate function -- see :class:`StageVolumeCurve`
    for the motivating example.

    Use this alongside :func:`register_rate_function` when adding a rate
    function from outside this module.

    :param function_name: Name the rate function is registered under in
        :data:`RATE_FUNCTIONS`.
    :type function_name: str
    :param param_name: Name of the parameter, within that function's
        ``params``, that holds a spec needing to be built into an
        object.
    :type param_name: str
    :returns: None
    """
    CURVE_PARAMS.setdefault(function_name, set()).add(param_name)


# PHYSICALLY-DIMENSIONED RATE CONVENTION -- for formulas fixed to SI seconds
# --------------------------------------------------------------------------
# A hydraulic formula like the Francis weir equation is dimensionally fixed
# to SI seconds (Q comes out in cubic meters per SECOND) -- unlike
# linear_decay()'s tau, there's no per-parameter unit to convert, because
# the whole formula's output is what's pinned to seconds. But every flow
# value in this module must come out in volume-units per dt_unit (so that
# dt * flow_value is a valid volume increment in solve()) -- so a
# seconds-pinned rate function's raw output still needs converting, once,
# by the number of seconds in one dt_unit.
#
# RATE_IN_SECONDS_FUNCTIONS below declares which rate functions are
# seconds-pinned this way. DynamicSystemModel._build_flows injects a
# precomputed "_seconds_per_dt_unit" conversion factor into that function's
# params ONCE, at setup time -- so the rate function just multiplies by it,
# and never has to know dt_units itself.

RATE_IN_SECONDS_FUNCTIONS = {"weir_francis"}


def register_rate_in_seconds_function(function_name):
    """Declare that a rate function's raw output is dimensionally fixed
    to SI seconds (e.g. a hydraulic formula like the Francis weir
    equation), so :meth:`DynamicSystemModel._build_flows` injects a
    ``"_seconds_per_dt_unit"`` conversion factor into its params
    automatically -- see :func:`weir_francis` for the pattern.

    Use this alongside :func:`register_rate_function` when adding a rate
    function from outside this module.

    :param function_name: Name the rate function is registered under in
        :data:`RATE_FUNCTIONS`.
    :type function_name: str
    :returns: None
    """
    RATE_IN_SECONDS_FUNCTIONS.add(function_name)


#: how many seconds make up one dt_unit -- used to convert a seconds-pinned
#: rate function's raw SI output into the model's own dt_units
_SECONDS_PER_DT_UNIT = {"days": 86400.0, "hours": 3600.0, "minutes": 60.0, "seconds": 1.0}


def linear_decay(state, t, params):
    r"""Linear reservoir outflow via residence time, with an optional
    orifice/activation threshold:

    .. math::

        O = \frac{\max(S_{source} - S_{activation},\ 0)}{\tau}

    Modeled as an orifice partway up the storage's wall rather than at
    the very bottom: ``activation_level`` (:math:`S_{activation}`) is the
    dead storage below the orifice -- however full it gets below that
    point, it produces no outflow through *this* orifice. The "head"
    driving the outflow is only the storage above that threshold.
    ``activation_level=0`` (the default) recovers the plain
    bottom-orifice case, unchanged from before.

    :param state: Full level-state dict (not just this flow's own source
        level), so a downstream level can constrain an upstream flow.
    :type state: dict[str, float]
    :param t: Current simulation time (unused by this function, but part
        of the shared rate-function signature).
    :type t: pandas.Timestamp
    :param params: Must include ``"residence_time"`` (:math:`\tau`, in
        HOURS by convention -- see :data:`TIME_CONSTANT_PARAMS`; by the
        time this function runs, it has already been converted into the
        model's own ``dt_units`` by
        :meth:`DynamicSystemModel._build_flows`, so this function never
        sees a raw hours value). May include ``"activation_level"``
        (same units as the level's own state -- NOT a time constant, so
        deliberately absent from :data:`TIME_CONSTANT_PARAMS` and never
        touched by the hours conversion; default 0.0). Also carries
        ``"source"`` and ``"target"``, injected automatically from the
        flow's own definition.
    :type params: dict
    :returns: The outflow rate :math:`O`.
    :rtype: float
    """
    tau = params["residence_time"]
    activation = params.get("activation_level", 0.0)
    S = state[params["source"]]
    head = max(S - activation, 0.0)
    return head / tau


# ---------------------------------------------------------------------------
# Stage-volume curves -- convert a level's volume state into a stage
# (elevation), for rate functions like weir_francis() that are inherently
# defined in terms of head above a crest, not storage
# ---------------------------------------------------------------------------

class StageVolumeCurve:
    """A reservoir's stage-storage (elevation-volume) relationship,
    resolved once at setup time from a flow's ``"stage_volume_curve"``
    JSON spec and then queried at every timestep by rate functions like
    :func:`weir_francis` -- see :meth:`from_spec` for the JSON shape,
    and :data:`CURVE_PARAMS` for how/when that resolution happens.

    All units are SI: stage in meters, volume in cubic meters. The point
    ``(volume=0, stage=0)`` is ALWAYS included, in addition to whatever
    points are supplied -- an empty reservoir has, by definition, no
    head above its own bottom, and this guarantees
    :meth:`stage_from_volume` is well-defined all the way down to zero
    storage regardless of the lowest surveyed point.

    :param stage: Surveyed stage values (meters), NOT including the
        ``(0, 0)`` anchor -- it's added automatically.
    :type stage: array-like
    :param volume: Corresponding surveyed volume values (cubic meters),
        NOT including the ``(0, 0)`` anchor.
    :type volume: array-like
    :param fit_power: If ``False`` (the default), :meth:`stage_from_volume`
        linearly interpolates between the surveyed points. If ``True``,
        a single-parameter power curve :math:`h = v^{a}` is fit instead
        (via log-log least squares through the origin) and used for
        every query -- see :meth:`stage_from_volume`.
    :type fit_power: bool
    :raises ValueError: If fewer than one surveyed point is given, if
        the resulting (volume, stage) pairs aren't consistent (the same
        volume mapped to two different stages), or -- when
        ``fit_power=True`` -- if there's no positive point to fit
        against.
    """

    def __init__(self, stage, volume, fit_power=False):
        stage = np.concatenate(([0.0], np.asarray(stage, dtype=float)))
        volume = np.concatenate(([0.0], np.asarray(volume, dtype=float)))

        if len(volume) < 2:
            raise ValueError(
                "stage_volume_curve needs at least one surveyed (stage, "
                "volume) point in addition to the automatic (0, 0) anchor."
            )

        order = np.argsort(volume, kind="stable")
        volume = volume[order]
        stage = stage[order]
        _, unique_idx = np.unique(volume, return_index=True)
        if len(unique_idx) != len(volume):
            # duplicate volumes are only a contradiction if they map to
            # different stages -- dedupe cleanly, or raise if they disagree
            deduped_volume, deduped_stage = [], []
            for v in np.unique(volume):
                stages_here = stage[volume == v]
                if not np.allclose(stages_here, stages_here[0]):
                    raise ValueError(
                        f"stage_volume_curve is inconsistent: volume {v!r} "
                        f"maps to multiple different stage values "
                        f"{sorted(set(stages_here.tolist()))!r}."
                    )
                deduped_volume.append(v)
                deduped_stage.append(stages_here[0])
            volume = np.array(deduped_volume)
            stage = np.array(deduped_stage)

        self.volume = volume
        self.stage = stage
        self.fit_power = fit_power
        self._exponent = self._fit_power_exponent(stage, volume) if fit_power else None

    @staticmethod
    def _fit_power_exponent(stage, volume):
        """Fit :math:`h = v^{a}` by log-log least squares through the
        origin: :math:`a = \\frac{\\sum \\log(v)\\log(h)}{\\sum \\log(v)^2}`,
        using only the strictly-positive points (the ``(0, 0)`` anchor
        and any other zero-valued point can't contribute in log-space,
        but the fit still passes through the origin automatically since
        :math:`0^{a} = 0`).

        :param stage: Stage values, including the ``(0, 0)`` anchor.
        :type stage: numpy.ndarray
        :param volume: Volume values, including the ``(0, 0)`` anchor.
        :type volume: numpy.ndarray
        :returns: The fitted exponent :math:`a`.
        :rtype: float
        :raises ValueError: If there's no strictly-positive point to fit.
        """
        mask = (stage > 0) & (volume > 0)
        if not np.any(mask):
            raise ValueError(
                "stage_volume_curve fit_power=True needs at least one "
                "surveyed point with both stage > 0 and volume > 0 to fit "
                "h = v^a against."
            )
        log_v = np.log(volume[mask])
        log_h = np.log(stage[mask])
        return float(np.sum(log_v * log_h) / np.sum(log_v ** 2))

    def stage_from_volume(self, volume):
        """Resolve the stage (elevation, meters) corresponding to a
        given volume (cubic meters).

        :param volume: Current storage.
        :type volume: float
        :returns: Corresponding stage, via linear interpolation between
            surveyed points (clamped to the surveyed range at the
            extremes -- see note below) or via the fitted power curve,
            depending on ``fit_power``.
        :rtype: float

        .. note::
            With ``fit_power=False``, a volume beyond the surveyed range
            is clamped to the nearest end point's stage (flat
            extrapolation) rather than raising -- storage can organically
            exceed a surveyed curve's range during a simulation (e.g. an
            extreme flood), and halting the run outright would often be
            worse than a conservative flat estimate. This is standard
            :func:`numpy.interp` behavior; if you need something better
            than flat extrapolation beyond the surveyed range, extend the
            curve's own data instead, or use ``fit_power=True`` (the
            power curve extrapolates smoothly by construction, with no
            clamping).
        """
        if self.fit_power:
            return 0.0 if volume <= 0 else float(volume) ** self._exponent
        return float(np.interp(volume, self.volume, self.stage))

    @classmethod
    def from_spec(cls, spec):
        """Build a :class:`StageVolumeCurve` from a flow's
        ``"stage_volume_curve"`` JSON entry.

        .. code-block:: json

            {
              "data": "/path/to/curve.csv",
              "column_stage": "H",
              "column_volume": "V",
              "fit_power": false
            }

        ``"data"`` is either a path to a CSV (with ``"stage"`` and
        ``"volume"`` columns by default -- override via
        ``"column_stage"``/``"column_volume"`` for a CSV with different
        column names, no naming convention required, same pattern as a
        flow's own ``"column"`` override -- see the module docstring's
        "Forcing data" section) or an inline dict:

        .. code-block:: json

            {"data": {"stage": [1, 2, 3], "volume": [102, 140, 300]}}

        ``"fit_power"`` defaults to ``false`` (linear interpolation); see
        the :class:`StageVolumeCurve` constructor for what ``true``
        does.

        :param spec: The raw ``"stage_volume_curve"`` dict.
        :type spec: dict
        :returns: The built curve.
        :rtype: StageVolumeCurve
        :raises ValueError: If ``"data"`` is missing, is neither a
            string nor a dict, or (for a dict) is missing ``"stage"``/
            ``"volume"``, or (for a CSV) is missing the resolved stage/
            volume columns.
        """
        if "data" not in spec or spec["data"] is None:
            raise ValueError(
                'stage_volume_curve is missing a "data" key -- expected a '
                "CSV path (string) or an inline "
                '{"stage": [...], "volume": [...]} dict.'
            )
        data = spec["data"]
        fit_power = bool(spec.get("fit_power", False))

        if isinstance(data, str):
            df = pd.read_csv(data)
            stage_col = spec.get("column_stage") or "stage"
            volume_col = spec.get("column_volume") or "volume"
            missing = [c for c in (stage_col, volume_col) if c not in df.columns]
            if missing:
                raise ValueError(
                    f"stage_volume_curve CSV {data!r} is missing column(s) "
                    f"{missing!r} (via 'column_stage'/'column_volume' if "
                    f"given, else the defaults 'stage'/'volume'). Available "
                    f"columns: {list(df.columns)}"
                )
            stage = df[stage_col].to_numpy(dtype=float)
            volume = df[volume_col].to_numpy(dtype=float)
        elif isinstance(data, dict):
            if "stage" not in data or "volume" not in data:
                raise ValueError(
                    'inline stage_volume_curve "data" dict must have '
                    '"stage" and "volume" keys.'
                )
            stage = data["stage"]
            volume = data["volume"]
        else:
            raise ValueError(
                'stage_volume_curve "data" must be a CSV path (string) or '
                'an inline {"stage": [...], "volume": [...]} dict, got '
                f"{type(data).__name__}."
            )

        return cls(stage=stage, volume=volume, fit_power=fit_power)


#: standard Francis-formula coefficient for a rectangular, fully
#: suppressed weir, in SI units (Q in cubic meters/second, L and H in
#: meters)
FRANCIS_WEIR_COEFFICIENT = 1.84


def weir_francis(state, t, params):
    r"""Francis weir equation for a rectangular, fully suppressed weir:

    .. math::

        Q = C_w \cdot L \cdot H^{3/2}

    where :math:`C_w` is :data:`FRANCIS_WEIR_COEFFICIENT` (1.84, the
    standard SI Francis coefficient), :math:`L` is the crest length
    (``"width"``, meters), and :math:`H` is the head above the crest.

    The level's own state is a VOLUME, not an elevation, so head is
    derived via the flow's ``"stage_volume_curve"``
    (:class:`StageVolumeCurve`):

    .. math::

        H = \max(h(S_{source}) - h_{activation},\ 0)

    where :math:`h(\cdot)` is :meth:`StageVolumeCurve.stage_from_volume`
    and :math:`h_{activation}` (``"activation_level"``, meters, default
    0.0) is the crest's own elevation above the curve's zero point --
    i.e. how high the crest sits above the reservoir bottom. Note this
    is a STAGE-domain threshold (meters), unlike :func:`linear_decay`'s
    ``"activation_level"``, which is in the level's own state units
    directly (there is no stage-volume curve involved in that formula).

    The Francis formula is dimensionally fixed to SI seconds (:math:`Q`
    comes out in :math:`m^3/s`), but every flow value in this module
    must be volume-units per ``dt_unit``. That conversion is applied
    here via ``params["_seconds_per_dt_unit"]``, injected once at setup
    time by :meth:`DynamicSystemModel._build_flows` -- see
    :data:`RATE_IN_SECONDS_FUNCTIONS`.

    :param state: Full level-state dict (volume, cubic meters).
    :type state: dict[str, float]
    :param t: Current simulation time (unused by this function, but part
        of the shared rate-function signature).
    :type t: pandas.Timestamp
    :param params: Must include ``"width"`` (:math:`L`, meters --
        MANDATORY, no default) and ``"stage_volume_curve"`` (already
        resolved into a :class:`StageVolumeCurve` object by
        :meth:`DynamicSystemModel._build_flows` -- see
        :data:`CURVE_PARAMS` -- this function never sees the raw JSON
        spec). May include ``"activation_level"`` (:math:`h_{activation}`,
        meters; default 0.0). Also carries ``"source"``, ``"target"``,
        and ``"_seconds_per_dt_unit"``, all injected automatically.
    :type params: dict
    :returns: The outflow rate :math:`Q`, in cubic meters per ``dt_unit``.
    :rtype: float
    """
    width = params["width"]
    activation = params.get("activation_level", 0.0)
    curve = params["stage_volume_curve"]
    seconds_per_dt_unit = params["_seconds_per_dt_unit"]

    S = state[params["source"]]
    stage = curve.stage_from_volume(S)
    head = max(stage - activation, 0.0)

    discharge_si = FRANCIS_WEIR_COEFFICIENT * width * head ** 1.5  # m^3/s
    return discharge_si * seconds_per_dt_unit  # volume-units per dt_unit


RATE_FUNCTIONS = {
    "linear_decay": linear_decay,
    "weir_francis": weir_francis,
}


def register_rate_function(name, func):
    """Extend :data:`RATE_FUNCTIONS` from outside this module without
    editing it.

    :param name: Name the function should be registered under (this is
        the value used in a flow's ``"function"`` JSON field).
    :type name: str
    :param func: Callable with signature ``func(state, t, params) ->
        float``. See :func:`linear_decay` for the expected signature and
        conventions.
    :type func: callable
    :returns: None
    """
    RATE_FUNCTIONS[name] = func


# ---------------------------------------------------------------------------
# Time series resampling -- standalone, reusable outside this module
# ---------------------------------------------------------------------------

def _native_interval(series):
    """Median spacing between consecutive timestamps in ``series`` -- its
    own native sampling interval.

    Used both by :func:`align_series_to_grid` (to decide upscale vs.
    downscale, and for the mass-conserving rescale) and by
    :meth:`DynamicSystemModel._build_time_grid` (to extend the
    simulation horizon far enough to cover the last recorded value's own
    bucket -- see that method's docstring).

    :param series: A time series with a ``DatetimeIndex``.
    :type series: pandas.Series
    :returns: The median native interval, or ``pandas.Timedelta(0)`` for
        a series with fewer than two points (nothing to space).
    :rtype: pandas.Timedelta
    """
    diffs = series.index.to_series().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(0)
    return diffs.median()


def align_series_to_grid(series, target_index, interpolation="previous", aggregation="mean",
                          dt_unit_duration=None):
    r"""Reconcile a time series onto a different regular time grid,
    choosing automatically between upscaling and downscaling.

    Every value in ``series`` is taken to represent a quantity delivered
    over ONE of its own native timesteps (e.g. an hourly inflow series'
    value is "how much arrived during that hour") -- this is the
    convention every forcing series in this module follows (see the
    module docstring's "Forcing data" section).

    - **Upscaling** (target step coarser than native): native samples
      are grouped into each target interval :math:`[t_i, t_{i+1})` and
      reduced with ``aggregation``:

      - ``"sum"`` (e.g. rainfall depths): the reduced value is a total
        FOR THAT WHOLE TARGET INTERVAL.
      - ``"mean"`` (e.g. a river stage, or any other native-rate series):
        the reduced value is still denominated PER ONE NATIVE INTERVAL
        (just averaged across however many native samples fall in the
        target interval) -- a different basis than ``"sum"``'s.

    - **Downscaling** (target step finer than native): the series is
      reindexed onto the target grid, and gaps are filled via
      ``interpolation`` (``"previous"`` holds the last observed value;
      ``"linear"`` interpolates smoothly instead). Either way, the
      result is still denominated PER ONE NATIVE INTERVAL, same as
      ``"mean"`` above -- holding or interpolating doesn't change what a
      value means, only how it's distributed across finer timestamps.

    If ``dt_unit_duration`` is given, every value is additionally
    rescaled into "quantity per ONE ``dt_unit``" -- the convention
    :class:`DynamicSystemModel` requires so that its Euler step's
    ``dt * flow_value`` (where ``dt`` is a plain COUNT of ``dt_units``,
    not a duration) is dimensionally valid regardless of what that count
    happens to be. Because ``"sum"`` and everything else end up
    denominated differently (per TARGET interval vs. per NATIVE
    interval), the rescale factor differs correspondingly:

    .. math::

        \text{sum case:}\quad & g = \text{(bucket total)} \cdot
            \frac{\Delta t_{dt\_unit}}{\Delta t_{target}} \\
        \text{mean / downscale case:}\quad & g = \text{(value)} \cdot
            \frac{\Delta t_{dt\_unit}}{\Delta t_{native}}

    ``dt_unit_duration`` (:math:`\Delta t_{dt\_unit}`) has to be passed
    in explicitly rather than derived from ``target_index`` alone,
    because a target grid's own step spacing
    (:math:`\Delta t_{target}`) confounds the DURATION of one
    ``dt_unit`` with the model's numeric step count (``dt``) -- e.g. a
    grid spaced 5 minutes apart could mean ``dt_units="minutes"`` with
    ``dt=5``, and one ``dt_unit`` is 1 minute, not 5.

    Kept as a standalone function -- not tied to :class:`Forcing` or
    :class:`DynamicSystemModel` -- so it can be reused anywhere a time
    series needs reconciling onto a different regular grid.
    ``dt_unit_duration=None`` (the default) skips this rescale entirely,
    preserving plain time-regridding behavior for callers that don't
    need it.

    :param series: Source time series, arbitrary ``DatetimeIndex``.
    :type series: pandas.Series
    :param target_index: The model's simulation grid, a regular
        ``DatetimeIndex``.
    :type target_index: pandas.DatetimeIndex
    :param interpolation: Gap-fill mode when downscaling; ``"previous"``
        or ``"linear"``.
    :type interpolation: str
    :param aggregation: Reduction applied when upscaling, e.g.
        ``"mean"`` or ``"sum"`` (any :meth:`pandas.Series.resample`
        aggregation name).
    :type aggregation: str
    :param dt_unit_duration: Duration of exactly ONE ``dt_unit`` (e.g.
        ``pandas.Timedelta(1, "min")`` for ``dt_units="minutes"``). If
        given, rescales every value into "quantity per ONE dt_unit" as
        described above. If ``None``, no rescale is applied.
    :type dt_unit_duration: pandas.Timedelta or None
    :returns: ``series`` reindexed onto ``target_index``.
    :rtype: pandas.Series
    :raises ValueError: If ``interpolation`` is not ``"previous"`` or
        ``"linear"``.
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
        if dt_unit_duration is not None:
            basis_dt = target_dt if aggregation == "sum" else native_dt
            aligned = aligned * (dt_unit_duration / basis_dt)
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
        if dt_unit_duration is not None:
            aligned = aligned * (dt_unit_duration / native_dt)

    aligned.index.name = target_index.name
    return aligned


# ---------------------------------------------------------------------------
# Operational control -- external factor applied on top of a flow's value
# ---------------------------------------------------------------------------

def build_operation_factor(operation_spec, time_index):
    """Build a per-timestep operational factor aligned to ``time_index``,
    from a flow's ``"operation"`` JSON entry.

    See the module docstring's "Operational control" section for the
    full rules. In brief, ``operation_spec`` may be:

    - ``None`` -> returns ``None`` (no restriction).
    - ``{"factors": ..., "linear": ...}`` -- the canonical form.
      ``"factors"`` is required; ``"linear"`` is optional and defaults
      to ``False``.
    - a bare number or list -> shorthand for
      ``{"factors": <that value>, "linear": False}``, kept for
      convenience/backward compatibility.

    ``"factors"`` (after unwrapping the dict, if any):

    - a single number, or a list with one number -> constant factor for
      the whole run (``"linear"`` has no effect here -- there's nothing
      to interpolate between).
    - a list with 2+ numbers -> control points at the simulation borders
      (first item at ``time_index[0]``, last item at ``time_index[-1]``),
      remaining items equally spaced between them.

    ``"linear"`` controls how those control points become a per-timestep
    series: ``False`` (default) holds each timestep at its nearest
    control point (a step function); ``True`` linearly interpolates
    between consecutive control points instead.

    :param operation_spec: The flow's raw ``"operation"`` JSON value (a
        dict, a bare number, a bare list, or ``None``).
    :type operation_spec: dict or float or int or list or None
    :param time_index: The model's simulation time grid.
    :type time_index: pandas.DatetimeIndex
    :returns: ``None`` if there's no restriction (caller should treat
        this as an implicit factor of 1.0 everywhere and may skip the
        multiplication entirely), otherwise a series of per-timestep
        factors indexed by ``time_index``.
    :rtype: pandas.Series or None
    :raises ValueError: If an operation dict is missing ``"factors"``,
        if the resulting list of factors is empty, or if any factor lies
        outside :math:`[0, 1]` -- factors represent a physical
        gate/valve opening or withdrawal fraction and are never clipped
        or rescaled to fit that range.
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
    """A single external time series, kept at its native
    resolution/timestamps.

    Build one of these either directly from a CSV path (the constructor)
    or from a :class:`pandas.DataFrame` already loaded in memory (via
    :meth:`from_dataframe`) -- the latter matters when several forcing
    series share one source file, so the file is only ever read once.
    :meth:`DynamicSystemModel._resolve_data_from_flows` uses
    :meth:`from_dataframe` for exactly this reason.

    :param path: Path to the forcing CSV.
    :type path: str or pathlib.Path
    :param time_col: Name of the timestamp column.
    :type time_col: str
    :param value_col: Name of the value column.
    :type value_col: str
    :param interpolation: How to fill gaps when downscaling.
    :type interpolation: str
    :param aggregation: How to reduce this series when upscaling to a
        coarser ``dt`` (e.g. ``"sum"`` for a rainfall depth, ``"mean"``
        for a river stage -- any reducer
        :meth:`pandas.Series.resample` understands). This is a property
        of what the variable physically represents, so when driven by
        ``parameters.json`` it comes from a flow's own ``"aggregation"``
        key rather than being fixed on the class -- see the module
        docstring's "Forcing data" section.
    :type aggregation: str
    """

    def __init__(self, path, time_col, value_col, interpolation="previous", aggregation="mean"):
        self.interpolation = interpolation
        self.aggregation = aggregation
        df = pd.read_csv(path)
        self.series = self._series_from_dataframe(df, time_col, value_col)

    @classmethod
    def from_dataframe(cls, df, time_col, value_col, interpolation="previous", aggregation="mean"):
        """Build a :class:`Forcing` from an already-loaded DataFrame,
        without re-reading a CSV from disk.

        Useful when several forcing series share the same source file --
        read the CSV once with :func:`pandas.read_csv`, then call this
        once per value column instead of constructing several
        :class:`Forcing` objects (each of which would otherwise re-read
        and re-parse the same file). See
        :meth:`DynamicSystemModel._resolve_data_from_flows` for the
        pattern.

        :param df: Already-loaded forcing table, containing at least
            ``time_col`` and ``value_col``.
        :type df: pandas.DataFrame
        :param time_col: Name of the timestamp column.
        :type time_col: str
        :param value_col: Name of the value column to extract.
        :type value_col: str
        :param interpolation: How to fill gaps when downscaling.
        :type interpolation: str
        :param aggregation: How to reduce this series when upscaling.
        :type aggregation: str
        :returns: A new :class:`Forcing` wrapping just that one column.
        :rtype: Forcing
        """
        forcing = cls.__new__(cls)
        forcing.interpolation = interpolation
        forcing.aggregation = aggregation
        forcing.series = cls._series_from_dataframe(df, time_col, value_col)
        return forcing

    @staticmethod
    def _series_from_dataframe(df, time_col, value_col):
        """Shared parsing logic behind both the constructor and
        :meth:`from_dataframe`: parse/sort the timestamp column and
        return a single named, DatetimeIndex-ed series for ``value_col``.

        :param df: Source table.
        :type df: pandas.DataFrame
        :param time_col: Name of the timestamp column.
        :type time_col: str
        :param value_col: Name of the value column to extract.
        :type value_col: str
        :returns: The extracted series.
        :rtype: pandas.Series
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        return pd.Series(
            df[value_col].to_numpy(dtype=float),
            index=pd.DatetimeIndex(df[time_col]),
            name=value_col,
        )


# ---------------------------------------------------------------------------
# Level and Flow
# ---------------------------------------------------------------------------

class Level:
    """A single stock/storage.

    :param name: Level name, matched against the ``"name"`` field in
        ``parameters.json``.
    :type name: str
    :param initial_state: Initial value of the stock at simulation
        start.
    :type initial_state: float
    :param min_state: Optional lower bound enforced by :meth:`clip`.
    :type min_state: float or None
    :param max_state: Optional upper bound enforced by :meth:`clip`.
    :type max_state: float or None
    """

    def __init__(self, name, initial_state, min_state=None, max_state=None):
        self.name = name
        self.initial_state = float(initial_state)
        self.min_state = min_state
        self.max_state = max_state

    def clip(self, value):
        """Clamp ``value`` to ``[min_state, max_state]``.

        :param value: Candidate new state.
        :type value: float
        :returns: The clamped value.
        :rtype: float
        """
        if self.min_state is not None:
            value = max(value, self.min_state)
        if self.max_state is not None:
            value = min(value, self.max_state)
        return value


class Flow:
    """A single flow, either external forcing -> level, or a rate
    function computed from current state, level -> level (or level ->
    external sink).

    :param name: Flow name; must be unique across all levels' inflows/
        outflows.
    :type name: str
    :param kind: ``"forcing"`` or ``"rate_function"``.
    :type kind: str
    :param source: Name of the level this flow drains (``None`` for an
        external inflow with no upstream level).
    :type source: str or None
    :param target: Name of the level this flow fills (``None`` for an
        external outflow with no downstream level).
    :type target: str or None
    :param forcing: Name of the forcing column to read, if ``kind`` is
        ``"forcing"``.
    :type forcing: str or None
    :param function: Name of the registered rate function to call, if
        ``kind`` is ``"rate_function"``. See :data:`RATE_FUNCTIONS`.
    :type function: str or None
    :param params: Extra parameters passed to the rate function.
    :type params: dict or None
    :param operation: Per-timestep multiplier built by
        :func:`build_operation_factor`, applied on top of the flow's
        otherwise-computed value -- e.g. an irrigation withdrawal
        schedule on an inflow, or a gate opening schedule on an outflow.
        ``None`` means fully free (an implicit factor of 1.0
        everywhere), which is the default. See the module docstring's
        "Operational control" section.
    :type operation: pandas.Series or None
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
        """Evaluate this flow's value at time ``t``.

        :param state: Full level-state dict at time ``t``.
        :type state: dict[str, float]
        :param t: Current simulation timestep.
        :type t: pandas.Timestamp
        :param forcing_row: Row of already-resampled forcing values at
            ``t`` (used only when ``kind`` is ``"forcing"``).
        :type forcing_row: pandas.Series
        :returns: The flow's value at ``t``, after applying
            ``operation`` if set.
        :rtype: float
        :raises ValueError: If ``kind`` is neither ``"forcing"`` nor
            ``"rate_function"``.
        """
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
# DynamicSystemModel -- usable directly; subclass only for a data shape
# the generic load_data()/_resolve_data_from_flows() pattern can't express
# ---------------------------------------------------------------------------

class DynamicSystemModel:
    """A generic levels-and-flows dynamic system, integrated with
    forward Euler.

    See the module docstring for the lifecycle this class is built
    around (``load_data`` / ``load_parameters`` / ``setup_model`` /
    ``solve`` / ``run``), and its "Forcing data" section for how CSV
    columns get linked to flows -- entirely through ``parameters.json``,
    with no Python-side naming convention. This class is usable directly
    for that common case; subclass and override :meth:`load_data` only
    for a genuinely different data shape (several files, a database, an
    in-memory array), and/or override :meth:`derivatives` and/or
    :meth:`apply_constraints` for system-specific dynamics (e.g. a
    cascade where a downstream level's state constrains an upstream
    flow, or soil-moisture overflow that must be re-routed as runoff
    rather than clipped).

    :ivar parameters: Parsed contents of the parameters JSON, set by
        :meth:`load_parameters`.
    :vartype parameters: dict or None
    :ivar data: Resolved :class:`Forcing` objects, keyed by forcing key;
        built by :meth:`_resolve_data_from_flows` during
        :meth:`setup_model` (or populated directly by an overridden
        :meth:`load_data`, for a non-CSV data shape).
    :vartype data: dict[str, Forcing]
    :ivar levels: Built :class:`Level` objects, keyed by name; set by
        :meth:`setup_model`.
    :vartype levels: dict[str, Level]
    :ivar flows: Built :class:`Flow` objects; set by :meth:`setup_model`.
    :vartype flows: list[Flow]
    :ivar dt: Timestep size, in ``dt_units``.
    :vartype dt: float or None
    :ivar dt_units: One of ``"days"``, ``"hours"``, ``"minutes"``,
        ``"seconds"``.
    :vartype dt_units: str or None
    :ivar n_steps: Number of simulation steps.
    :vartype n_steps: int or None
    :ivar time_index: The simulation time grid.
    :vartype time_index: pandas.DatetimeIndex or None
    :ivar forcing_on_grid: All loaded forcing, resampled onto
        ``time_index``.
    :vartype forcing_on_grid: pandas.DataFrame or None
    :ivar results: Combined levels + flows output of :meth:`solve`.
    :vartype results: pandas.DataFrame or None
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
        self._forcing_table = None
        self._forcing_time_col = None
        self._forcing_interpolation = None

    # -- loading (order-independent, no cross-dependencies) --------------

    def load_data(self, path, time_col="date", interpolation="linear"):
        """Read the raw forcing CSV once, without yet knowing which
        columns are needed.

        Column and aggregation resolution happens later, in
        :meth:`setup_model`, once ``parameters.json`` (and therefore
        each flow's ``"name"``/``"column"``/``"aggregation"``) is
        available -- see :meth:`_resolve_data_from_flows` and the module
        docstring's "Forcing data" section. Override this method only
        for a data shape this raw-table pattern can't express (e.g.
        several separate files, a database query, an in-memory array);
        an override that populates ``self.data`` directly is respected,
        since resolution only runs against a raw table if one was
        actually loaded (see :meth:`_resolve_data_from_flows`).

        :param path: Path to the forcing CSV. May contain any number of
            value columns; which ones get used is entirely up to what
            ``parameters.json`` references.
        :type path: str or pathlib.Path
        :param time_col: Name of the timestamp column.
        :type time_col: str
        :param interpolation: Gap-fill mode passed through to every
            :class:`Forcing` built from this table.
        :type interpolation: str
        :returns: ``self``, for chaining.
        :rtype: DynamicSystemModel
        """
        self._forcing_table = pd.read_csv(path)
        self._forcing_time_col = time_col
        self._forcing_interpolation = interpolation
        self.data = {}
        return self

    def load_parameters(self, path):
        """Parse the parameters file (levels, flows, simulation
        settings). Does not touch data or build anything -- see
        :meth:`setup_model`.

        :param path: Path to the parameters JSON file.
        :type path: str or pathlib.Path
        :returns: ``self``, for chaining.
        :rtype: DynamicSystemModel
        """
        self.parameters = json.loads(Path(path).read_text())
        return self

    # -- setup: the one place data + parameters actually get merged ------

    def setup_model(self):
        """Resolve forcing data from the flows, then build the
        simulation time grid, levels, flows, and resample all forcing
        onto the grid.

        :meth:`_resolve_data_from_flows` runs first because
        :meth:`_build_time_grid` needs ``self.data`` already populated
        (the simulation horizon is derived from the forcing extent).
        :meth:`_build_flows` in turn needs ``self.dt_units`` (built by
        :meth:`_build_time_grid`) already known, to convert any
        hours-denominated time-constant parameters (see
        :data:`TIME_CONSTANT_PARAMS`) into the model's own ``dt_units``,
        AND needs ``self.time_index`` already known, to build each
        flow's ``"operation"`` series (see
        :func:`build_operation_factor`). Safe to call repeatedly -- e.g.
        after ``load_parameters(new_path)`` for a new realization
        against the same already-loaded data.

        :returns: ``self``, for chaining.
        :rtype: DynamicSystemModel
        :raises RuntimeError: If called before :meth:`load_parameters`.
        """
        if self.parameters is None:
            raise RuntimeError("Call load_parameters() before setup_model().")
        self._resolve_data_from_flows()
        self._build_time_grid()
        self._build_levels()
        self._build_flows()
        self._build_forcing_on_grid()
        return self

    def _resolve_data_from_flows(self):
        """Build ``self.data`` (one :class:`Forcing` per distinct
        ``"forcing"`` key used by a forcing-type flow) by reading each
        such flow's ``"name"``/``"column"``/``"aggregation"`` directly
        out of ``self.parameters`` -- see the module docstring's
        "Forcing data" section for the full rules.

        In brief, for every forcing-type flow: the forcing key is
        ``fl.get("forcing", fl["name"])``; the source column is
        ``fl.get("column") or fl["name"]``; the aggregation is
        ``fl.get("aggregation", "mean")``.

        A no-op if no raw table was loaded (i.e. an overridden
        :meth:`load_data` populated ``self.data`` directly instead) --
        that ``self.data`` is left untouched.

        :returns: None
        :raises ValueError: If a flow's resolved column isn't present in
            the loaded CSV, or if two flows disagree about which column
            or aggregation the same ``"forcing"`` key should come from.
        """
        if self._forcing_table is None:
            return

        table_columns = set(self._forcing_table.columns)
        resolved = {}  # forcing_key -> (column, aggregation), for conflict checking
        self.data = {}

        for lv in self.parameters["levels"]:
            flows = list(lv.get("inflows", [])) + list(lv.get("outflows", []))
            for fl in flows:
                if fl.get("type") != "forcing":
                    continue
                forcing_key = fl.get("forcing", fl["name"])
                column = fl.get("column") or fl["name"]
                aggregation = fl.get("aggregation", "mean")

                if column not in table_columns:
                    raise ValueError(
                        f"Flow {fl['name']!r} references column {column!r} "
                        f"(via 'column' if given, else its own 'name'), which "
                        f"was not found in the forcing CSV. Available columns: "
                        f"{sorted(table_columns)}"
                    )

                if forcing_key in resolved and resolved[forcing_key] != (column, aggregation):
                    raise ValueError(
                        f"forcing key {forcing_key!r} is linked to "
                        f"(column={resolved[forcing_key][0]!r}, "
                        f"aggregation={resolved[forcing_key][1]!r}) by one flow "
                        f"and to (column={column!r}, aggregation={aggregation!r}) "
                        f"by another -- each forcing key must resolve to a "
                        f"single column and aggregation."
                    )
                resolved[forcing_key] = (column, aggregation)

                if forcing_key not in self.data:
                    self.data[forcing_key] = Forcing.from_dataframe(
                        self._forcing_table, time_col=self._forcing_time_col,
                        value_col=column, interpolation=self._forcing_interpolation,
                        aggregation=aggregation,
                    )

    def _build_levels(self):
        """Build :class:`Level` objects from
        ``self.parameters["levels"]``.

        :returns: None
        """
        self.levels = {
            lv["name"]: Level(lv["name"], lv["initial_state"],
                               lv.get("min"), lv.get("max"))
            for lv in self.parameters["levels"]
        }

    def _build_flows(self):
        """Build :class:`Flow` objects from each level's own nested
        ``"inflows"`` and ``"outflows"`` lists, instead of a flat
        top-level flows list.

        Nesting a flow under a level's ``"outflows"`` implies
        ``source=<that level>``; under ``"inflows"`` implies
        ``target=<that level>``. This removes the class of bug where a
        flow's source/target field silently doesn't match the level it's
        meant to act on -- there's no way to write it "attached" to the
        wrong level, because the attachment *is* the nesting.

        For a flow between two levels (a cascade), nest it under ONE
        side only and give the OTHER endpoint explicitly: an outflow
        entry may still carry a ``"target"`` naming the downstream
        level; an inflow entry may still carry a ``"source"`` naming the
        upstream one. Do not list the same flow under both levels --
        that double-counts it, and is checked for below.

        Also converts any hours-denominated time-constant parameters
        (per :data:`TIME_CONSTANT_PARAMS`) into ``self.dt_units``, and
        builds each flow's ``"operation"`` series (per
        :func:`build_operation_factor`).

        :returns: None
        :raises ValueError: If the same flow name appears more than
            once across all levels' inflows/outflows.
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
        """Build a single :class:`Flow` from its raw JSON entry.

        :param fl: The flow's raw JSON entry (one item from a level's
            ``"inflows"`` or ``"outflows"`` list).
        :type fl: dict
        :param source: Resolved source level name.
        :type source: str or None
        :param target: Resolved target level name.
        :type target: str or None
        :returns: The constructed flow.
        :rtype: Flow
        :raises ValueError: If ``fl["operation"]`` is invalid (see
            :func:`build_operation_factor`), or if a
            :data:`CURVE_PARAMS`-registered param (e.g.
            ``"stage_volume_curve"``) is invalid (see
            :meth:`StageVolumeCurve.from_spec`).
        """
        params = dict(fl.get("params") or {})
        function_name = fl.get("function")

        for key in TIME_CONSTANT_PARAMS.get(function_name, ()):
            if key in params:
                params[key] = hours_to_dt_units(params[key], self.dt_units)

        for key in CURVE_PARAMS.get(function_name, ()):
            if key in params:
                try:
                    params[key] = StageVolumeCurve.from_spec(params[key])
                except ValueError as exc:
                    raise ValueError(f"Flow {fl.get('name')!r}, param {key!r}: {exc}") from exc

        if function_name in RATE_IN_SECONDS_FUNCTIONS:
            params["_seconds_per_dt_unit"] = _SECONDS_PER_DT_UNIT[self.dt_units]

        try:
            operation = build_operation_factor(fl.get("operation"), self.time_index)
        except ValueError as exc:
            raise ValueError(f"Flow {fl.get('name')!r}: {exc}") from exc
        return Flow(
            name=fl["name"],
            kind=fl["type"],
            source=source,
            target=target,
            forcing=fl.get("forcing", fl.get("name")),
            function=function_name,
            params=params,
            operation=operation,
        )

    #: maps a human-friendly dt_units value to the unit code pandas.Timedelta expects
    _TIMEDELTA_UNITS = {"days": "D", "hours": "h", "minutes": "min", "seconds": "s"}

    def _build_time_grid(self):
        """Build the simulation time grid.

        ``dt``/``dt_units`` come from parameters (``dt_units`` defaults
        to ``"days"`` if omitted; also accepts ``"hours"``, ``"minutes"``,
        ``"seconds"``). The horizon does not: start and end are taken
        directly from the loaded forcing data's own extent -- it's the
        data's job to say how long a simulation makes sense, not the
        parameters'.

        ``end`` is the latest point at which any series' own bucket
        still has something to contribute, NOT simply the latest
        recorded timestamp. Every forcing value represents a quantity
        for the interval STARTING at its own timestamp (see the module
        docstring's "Forcing values and mass conservation across dt"
        section) -- so a series' very last recorded value still owns a
        full bucket extending one more native interval PAST its own
        timestamp, even though no further sample exists to mark that
        bucket's own end. Stopping ``end`` at the last raw timestamp
        would silently drop that entire final bucket, since
        :meth:`solve` deliberately excludes flow evaluation at the very
        last grid point (see its docstring) -- so without this
        extension, whichever series has the latest native sample loses
        its last interval's worth of volume/mass. Each series
        contributes its OWN native interval width here (a coarser series
        needs a bigger extension than a finer one); ``end`` is the
        latest of these across all loaded series.

        No override for the overall horizon yet (e.g. a factor to run a
        shorter/longer window) -- deliberately left out for now, worth
        adding later if/when it's actually needed.

        :returns: None
        :raises ValueError: If ``dt_units`` is not one of the supported
            units.
        :raises RuntimeError: If no forcing has been loaded/resolved yet.
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
        end = max(f.series.index.max() + _native_interval(f.series) for f in self.data.values())

        self.n_steps = int((end - start) / dt_timedelta)  # truncates any partial final step
        self.time_index = pd.date_range(start=start, periods=self.n_steps + 1, freq=dt_timedelta)

    def _build_forcing_on_grid(self):
        """Resample every loaded :class:`Forcing` series onto
        ``self.time_index`` via :func:`align_series_to_grid`, and
        combine them into ``self.forcing_on_grid``.

        Every forcing value is additionally rescaled into "quantity per
        ONE ``dt_unit``" -- the basis :meth:`step`'s Euler update
        requires (``dt * flow_value``, where ``dt`` is a plain count of
        ``dt_units``) -- by passing the duration of exactly one
        ``dt_unit`` through to :func:`align_series_to_grid` as
        ``dt_unit_duration``. This is what keeps a forcing series
        mass/volume-conserving regardless of how its own native sampling
        interval relates to the model's ``dt``/``dt_units`` -- see that
        function's docstring for the underlying math.

        :returns: None
        """
        dt_unit_duration = pd.Timedelta(1, unit=self._TIMEDELTA_UNITS[self.dt_units])
        columns = {
            name: align_series_to_grid(
                forcing.series, self.time_index,
                interpolation=forcing.interpolation,
                aggregation=forcing.aggregation,
                dt_unit_duration=dt_unit_duration,
            )
            for name, forcing in self.data.items()
        }
        self.forcing_on_grid = (
            pd.DataFrame(columns) if columns else pd.DataFrame(index=self.time_index)
        )

    # -- core dynamics -----------------------------------------------------

    def evaluate_flows(self, state, t):
        """Value of every flow at time ``t``, given current state.

        Forcing flows are simple lookups into the already-resampled
        grid -- all the interpolation/aggregation work happened once in
        :meth:`setup_model`. Any flow with an ``"operation"`` entry has
        its raw value multiplied by that flow's operational factor at
        ``t`` -- see :meth:`Flow.value`.

        :param state: Full level-state dict at time ``t``.
        :type state: dict[str, float]
        :param t: Current simulation timestep.
        :type t: pandas.Timestamp
        :returns: Flow values keyed by flow name.
        :rtype: dict[str, float]
        """
        forcing_row = self.forcing_on_grid.loc[t] if not self.forcing_on_grid.empty else pd.Series(dtype=float)
        return {flow.name: flow.value(state, t, forcing_row) for flow in self.flows}

    def derivatives_from_flow_values(self, flow_values):
        """:math:`dS/dt` for every level, given already-evaluated flow
        values.

        :param flow_values: Flow values keyed by flow name, as returned
            by :meth:`evaluate_flows`.
        :type flow_values: dict[str, float]
        :returns: Derivative of each level's state, keyed by level name.
        :rtype: dict[str, float]
        """
        d = {name: 0.0 for name in self.levels}
        for flow in self.flows:
            q = flow_values[flow.name]
            if flow.source is not None:
                d[flow.source] -= q
            if flow.target is not None:
                d[flow.target] += q
        return d

    def derivatives(self, state, t):
        """:math:`dS/dt` for every level at time ``t``, given the
        current state.

        Every flow is evaluated against the SAME state snapshot --
        nothing is updated mid-loop. Override this in a subclass for
        coupling this generic sum-of-flows form can't express. (For the
        individual flow values too, e.g. in a subclassed :meth:`step`,
        call :meth:`evaluate_flows` directly.)

        :param state: Full level-state dict at time ``t``.
        :type state: dict[str, float]
        :param t: Current simulation timestep.
        :type t: pandas.Timestamp
        :returns: Derivative of each level's state, keyed by level name.
        :rtype: dict[str, float]
        """
        return self.derivatives_from_flow_values(self.evaluate_flows(state, t))

    def apply_constraints(self, state):
        """Clip each level to its configured ``[min, max]``.

        NOT mass-conserving: anything clipped off simply vanishes. Fine
        for a hard floor like an empty reservoir. For a level where the
        excess has to go somewhere physically (e.g. soil moisture
        overflowing into surface runoff), override this in a subclass
        and re-route the excess as an explicit flow instead of clipping
        it away.

        :param state: Candidate new state, before clipping.
        :type state: dict[str, float]
        :returns: The clipped state.
        :rtype: dict[str, float]
        """
        return {name: self.levels[name].clip(value) for name, value in state.items()}

    def step(self, state, t):
        """Single forward-Euler step.

        :param state: Full level-state dict at time ``t``.
        :type state: dict[str, float]
        :param t: Current simulation timestep.
        :type t: pandas.Timestamp
        :returns: ``(new_state, flow_values)``.
        :rtype: tuple[dict[str, float], dict[str, float]]
        """
        flow_values = self.evaluate_flows(state, t)
        d = self.derivatives_from_flow_values(flow_values)
        new_state = {name: state[name] + self.dt * d[name] for name in state}
        new_state = self.apply_constraints(new_state)
        return new_state, flow_values

    # -- solve / evaluate / orchestrate -----------------------------------

    def solve(self):
        """Integrate the full time grid built by :meth:`setup_model`.

        Returns and stashes ``self.results`` -- ONE DataFrame with both
        level and flow columns, indexed by time. Row ``t`` holds the
        level state AT ``t`` together with each flow's DELIVERED AMOUNT
        during the step from ``t`` to ``t + dt`` -- so a flow column and
        a level column on the same row share the same underlying state,
        which is what makes them meaningful to plot side by side, and a
        flow column is directly summable into a total (see
        :func:`print_mass_balance_check`). The final row has levels
        only: there's no flow beyond the last step of the horizon.

        Note this "delivered amount" is NOT the same thing
        :meth:`evaluate_flows`/:meth:`step` work with internally, which
        is each flow's RATE (state-units per ONE ``dt_unit`` -- see
        :meth:`Flow.value`). The rate form is what the Euler update
        itself needs (``dt * rate`` is what actually gets added to a
        level's state, inside :meth:`step`) and what makes
        ``dS/dt`` well-defined independent of ``dt``; ``solve()`` is the
        one place that converts to the "delivered amount" a human
        reading the results table would actually want (``rate * dt``),
        purely for reporting -- nothing upstream of this method ever
        sees or uses the delivered-amount form.

        :returns: The combined levels + flows DataFrame.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If called before :meth:`setup_model`.
        """
        if self.time_index is None:
            raise RuntimeError("Call setup_model() before solve().")
        state = {name: lv.initial_state for name, lv in self.levels.items()}
        records = []
        for t in self.time_index[:-1]:
            new_state, flow_values = self.step(state, t)
            delivered = {name: value * self.dt for name, value in flow_values.items()}
            records.append(dict(t=t, **state, **delivered))
            state = new_state
        records.append(dict(t=self.time_index[-1], **state))

        self.results = pd.DataFrame.from_records(records).set_index("t")
        return self.results

    def evaluate(self):
        """Placeholder for goodness-of-fit assessment against observed
        data.

        Returns ``None`` for now -- override in a subclass once a path
        for loading observed data exists. Kept as a separate step
        (rather than folded into :meth:`solve`) so :meth:`run` has one
        obvious place to grow.

        :returns: None
        """
        return None

    def run(self):
        """Convenience one-shot orchestrator: :meth:`setup_model` ->
        :meth:`solve` -> :meth:`evaluate`.

        For repeated realizations against the same data (e.g. a Monte
        Carlo batch), call ``load_parameters()`` -> ``setup_model()`` ->
        ``solve()`` directly in a loop instead -- skipping
        ``run()``/``evaluate()`` each time is usually what you want
        there.

        :returns: The combined levels + flows DataFrame, same as
            :meth:`solve`.
        :rtype: pandas.DataFrame
        """
        self.setup_model()
        self.solve()
        self.evaluate()
        return self.results

    # -- export -----------------------------------------------------------

    def export(self, path=None):
        """Bundle parameters, raw forcing data, and simulation results
        (one combined levels+flows DataFrame).

        :param path: If given, also write the bundle to disk under this
            directory (``parameters.json``, ``data/<name>.csv`` per
            forcing series, ``results.csv``).
        :type path: str or pathlib.Path or None
        :returns: The bundle, with keys ``"parameters"``, ``"data"``
            (dict of forcing series), and ``"results"``.
        :rtype: dict
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
# Plotting
# ---------------------------------------------------------------------------

def plot_results(model, results=None, flows_to_plot=None,
                  levels_to_plot=None, figsize=(9, 6), title=None):
    """Plot a model's results as two stacked subplots sharing the time
    axis: flows on top, level state trajectories on the bottom.

    Defaults to ``model.results`` (the one combined DataFrame populated
    by :meth:`DynamicSystemModel.solve`) -- which columns go in which
    subplot is worked out from ``model.levels`` / ``model.flows``, not
    from separate DataFrames. Kept as a plain function, not a model
    method -- visualization is meant to move out of this file once it
    grows, same reasoning as the rate-function registry.

    :param model: The model to plot.
    :type model: DynamicSystemModel
    :param results: Results DataFrame to plot; defaults to
        ``model.results``.
    :type results: pandas.DataFrame or None
    :param flows_to_plot: Flow column names to include; defaults to all
        of ``model.flows``.
    :type flows_to_plot: list[str] or None
    :param levels_to_plot: Level column names to include; defaults to
        all of ``model.levels``.
    :type levels_to_plot: list[str] or None
    :param figsize: Matplotlib figure size, ``(width, height)`` in
        inches.
    :type figsize: tuple[float, float]
    :param title: Optional figure suptitle.
    :type title: str or None
    :returns: ``(fig, (ax_flows, ax_levels))``.
    :rtype: tuple[matplotlib.figure.Figure, tuple]
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
# Mass balance check -- demonstrates the mass conservation described in the
# module docstring's "Forcing values and mass conservation across dt" section
# ---------------------------------------------------------------------------

def print_mass_balance_check(model):
    """Print, for every loaded forcing series, its raw native total
    against its delivered total after resampling onto the simulation
    grid -- a direct demonstration of the mass conservation described in
    the module docstring's "Forcing values and mass conservation across
    dt" section.

    - **raw total**: ``model.data[name].series.sum()`` -- the forcing
      values exactly as loaded, before any resampling. Since every
      native value is a quantity for its own bucket (see the module
      docstring), this sum is the true total delivered over the whole
      series, independent of ``dt``/``dt_units``.
    - **delivered total**: ``model.results[name].sum()`` -- since
      :meth:`DynamicSystemModel.solve` already stores each flow's
      DELIVERED AMOUNT per step (rate scaled by ``dt``, not the bare
      rate -- see that method's docstring), summing the column directly
      gives the true total, with no further scaling needed here.

    Under ``interpolation="previous"`` (the default) these two totals
    match exactly. Under ``"linear"`` they will be close but not exact
    -- see the module docstring for why.

    :param model: A model that has already been run (``model.results``
        and ``model.data`` populated -- i.e. after
        :meth:`DynamicSystemModel.run` or an equivalent
        ``setup_model()``/``solve()`` call).
    :type model: DynamicSystemModel
    :returns: None
    """
    print("Mass balance check (raw native total vs. delivered total):")
    for name, forcing in model.data.items():
        raw_total = float(forcing.series.sum())
        if name in model.results.columns:
            delivered_total = float(model.results[name].sum())
        else:
            delivered_total = float("nan")
        diff = delivered_total - raw_total
        pct = (diff / raw_total * 100.0) if raw_total else float("nan")
        print(
            f"  {name!r}: raw total = {raw_total:,.4f}   "
            f"delivered total = {delivered_total:,.4f}   "
            f"diff = {diff:,.6f} ({pct:+.4f}%)"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# python dynamic_systems.py example_inflow.csv example_parameters.json
# python dynamic_systems.py example_inflow.csv example_parameters.json --save out.png
# python dynamic_systems.py example_inflow.csv example_parameters.json --export run01/
#
# DynamicSystemModel is usable directly -- subclass it only for a data
# shape the generic load_data()/_resolve_data_from_flows() pattern can't
# express (see the module docstring's "Forcing data" section).

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a DynamicSystemModel and plot the result."
    )
    parser.add_argument("data", help="Path to the forcing CSV")
    parser.add_argument("parameters", help="Path to the parameters JSON (levels/flows/simulation)")
    parser.add_argument("--save", metavar="PNG_PATH", default=None,
                         help="Save the plot to this path instead of opening a window")
    parser.add_argument("--export", metavar="DIR", default=None,
                         help="Also export parameters/data/results to this directory")
    args = parser.parse_args()

    model = DynamicSystemModel()
    model.load_data(args.data)
    model.load_parameters(args.parameters)
    model.run()

    print()
    print_mass_balance_check(model)

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