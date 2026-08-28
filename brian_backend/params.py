"""
This file is an attemt to convert CBGTPy's origina; `popdata` into per neuron parameter arrays! 
By default CBGTPy stores network parameters in a pandas dataframe popdata (refer function `helper_popconstruct'
in file `https://github.com/CoAxLab/CBGTPy/blob/main/nchoice/popconstruct_nchoice.py`, line 24). 
This file has one row per population. values are wrapped in 
`common.tracetype.Trace` (refer class `Trace` in `https://github.com/CoAxLab/CBGTPy/blob/main/common/tracetype.py`, line 8). 


But, there is a difference between how legacy integrator deals with this information and how BRIAN2 handles it.

The legacy integrator expands the table into a list of arrays:
  agent.Taum        -> [array(75), array(750), array(750), ...]   # one per population
  agent.Taum[popid] -> array of length N[popid]


But, BRIAN2 wants a flat array over all neurons in one NeuronGroup (`https://brian2.readthedocs.io/en/stable/reference/brian2.groups.neurongroup.NeuronGroup.html`). 
  Taum -> array(4269)      # every neuron in the network, concatenated

This file is basically a module that would perform this conversion. And is fully unit-testable on its own. 

To verify against the legacy expansion, one may run: 
  `python -m brian_backend.params`


Contrast with the legacy code:########################################################################################################################################

In the file `https://github.com/CoAxLab/CBGTPy/blob/main/common/agentmatrixinit.py` we have the function `expandParamByCell` (ref lines: 50-67):

```python
def expandParamByCell(popdata,param,defaultvalue=np.nan):

    databypop = []

    if param not in popdata.columns:
        #print(param + " not found, initializing to " + str(defaultvalue))
        pass

    for idx1,row1 in popdata.iterrows():

        fillvalue = defaultvalue
        if param in popdata.columns:
            if not row1[param].is_nan():
                fillvalue = untrace(row1[param])

        array = np.ones(row1['N']) * fillvalue
        databypop.append(array)
    return databypop
```
This is the function we are actually trying to replace. This function does the following precisely:

1. Line 54 - when the column is absent -> every population gets the default values
2. Line 62 - when the column is present, but cell is NaN -> again initializing with default value
3. Line 63 - when both guardrails are satisfied it runs `untrace(row1[param])`, where `untrace is a function located at 
   `https://github.com/CoAxLab/CBGTPy/blob/main/common/tracetype.py`
   ```
   def untrace(data):
    if isinstance(data, list):
        return [x.val for x in data]
    if isinstance(data, pd.DataFrame):
        return trace(data,None).applymap(lambda x: x.val)
    try:
        return data.val
    except:
        return data
    ```
    In short untrace(row1[param]) extracts the underlying value from a wrapped/traced object, or returns the value unchanged if it's already a plain primitive.
4. Line 65 - creates the matrix, and it is important to notice that this operation is always homogenous within a population of neurons

When is it called upon:########################################################################################################################################################################################

There is this function `initializeAgent` (ref: `https://github.com/CoAxLab/CBGTPy/blob/main/common/agentmatrixinit.py`, line 101) in which this one is used at line 204-205:
```
setattr(agent, prop, expandParamByCell(popdata, prop, 0))
```
Even though the function declares the default value to be NaN, here the default values would always be zero. But this causes the following: 
  * `g_ahp` / `Vk` / `alpha_ca` are silently zero
  * `V` starts at 0, not at rest

How we converted to flat indexing:################################################################################################################################################################################

Populations are concatenated in `popdata` row order. Population `p`
occupies the contiguous slice `[starts[p], stops[p])` of every flat array:

    popdata row 0: GPi action 1  (N=75)   -> flat indices    0 to 74
    popdata row 1: GPi action 2  (N=75)   -> flat indices   75 to 149
    popdata row 2: STN action 1  (N=750)  -> flat indices  150 to 899
    ...

ow order is preserved, so `popid` means the same thing here as it does in
the legacy agent. That is deliberate, just to make the validation process easy to make life simple!
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "PopulationLayout",
    "build_layout",
    "flatten_param",
    "build_param_table",
    "NEURON_PARAMS",
    "INITIAL_STATE",
    "DEAD_PARAMS",
]

# We deliberately do not import common.tracetype here, the needed functions are handful of lines, it is better to copy than make it a dependency

def _is_missing(cell) -> bool:
    """True if a popdata cell carries no value.

    Mirrors `https://github.com/CoAxLab/CBGTPy/blob/main/common/tracetype.py:30` exactly::
    ```
        def is_nan(self):
           return pd.isnull(self.val)
    ```
    which is the test `expandParamByCell` applies at
    `https://github.com/CoAxLab/CBGTPy/blob/main/common/agentmatrixinit.py:62`.

    The `getattr` fallback to bare `pd.isnull` lets this module also accept
    a popdata frame that has already been through
    `https://github.com/CoAxLab/CBGTPy/blob/main/common/tracetype.py:550 untrace()`, where cells are plain numbers rather
    than `Trace` instances.
    """
    is_nan = getattr(cell, "is_nan", None)
    if is_nan is not None:
        return bool(is_nan())
    return bool(pd.isnull(cell))


def _raw(cell):
    """Unwrap a `Trace`, or pass a plain value straight through.

    Mirrors the scalar branch of `https://github.com/CoAxLab/CBGTPy/blob/main/common/tracetype.py:550 untrace()`:

        try:
            return data.val
        except:
            return data

    The value itself is stored at `https://github.com/CoAxLab/CBGTPy/blob/main/common/tracetype.py:27` (`self.val = val`
    in `Trace.__init__`); the `meta` provenance set alongside it is of no
    interest to the port and is discarded here.
    """
    return getattr(cell, "val", cell)


@dataclass(frozen=True, eq=False)
class PopulationLayout:
    """Where each CBGTPy population lives inside the flat neuron array.

    `eq=False` is deliberate. A frozen dataclass would otherwise synthesise
    `__eq__` as a field-by-field tuple comparison, and three of the fields are
    numpy arrays -> so `layout_a == layout_b` would raise
    `ValueError: The truth value of an array with more than one element is
    ambiguous`, and `hash(layout)` would raise `TypeError: unhashable type:
    'numpy.ndarray``. With `eq=False` the class falls back to identity
    semantics, which is what a layout actually is: a description of one
    network, not a value to be compared.

    Attributes
    ----------
    names
        Population type name per row, e.g. `"GPi"`, `"dSPN"`, `"FSI"`.
        Repeats across action channels.
    actions
        Action-channel label per row, or ``None`` for channel-agnostic
        populations (`FSI`, `CxI`). Taken from popdata's `action` column,
        which is NaN for those.
    labels
        Unique human-readable key per row: `"GPi:1"`, `"FSI"`, etc.
        Convenient for monitors and log messages.
    sizes
        `N` per population.
    starts, stops
        Half-open flat-index bounds, `stops[p] == starts[p] + sizes[p]`.
    n_total
        Total neuron count across all populations.
    pop_index
        Length-`n_total` array mapping each flat neuron index back to its
        `popid`. Used by the rolling firing-rate buffer to bin
        spikes by population without a Python loop.
    """

    names: tuple[str, ...]
    actions: tuple[object, ...]
    labels: tuple[str, ...]
    sizes: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    n_total: int
    pop_index: np.ndarray = field(repr=False)

    # -- lookups --------------------------------------------------------------

    @property
    def n_pop(self) -> int:
        """Number of populations (== number of popdata rows)."""
        return len(self.names)

    def slice(self, popid: int) -> slice:
        """Flat-array slice for one population."""
        return slice(int(self.starts[popid]), int(self.stops[popid]))

    def popids_named(self, name: str) -> list[int]:
        """All popids whose population type is `name`.

        Returns one popid per action channel, in channel order. So
        `popids_named("Th")` gives the thalamic population of every action
        channel - which is exactly what the decision rule in
        `interface_nchoice.py` reads (`out_popids`).
        """
        return [p for p, n in enumerate(self.names) if n == name]

    def indices_named(self, name: str) -> np.ndarray:
        """Flat neuron indices for every population of type `name`."""
        parts = [np.arange(self.starts[p], self.stops[p])
                 for p in self.popids_named(name)]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts).astype(np.int64)

    def popid_of_label(self, label: str) -> int:
        """popid for a unique label such as `"dSPN:2"`."""
        return self.labels.index(label)

    # The following function is purely for the sanity check of a potential user, just to be sure about what is happening underneath
    def describe(self) -> str:
        """Multi-line summary, for eyeballing that the layout is sane."""
        lines = [f"{self.n_pop} populations, {self.n_total} neurons total", ""]
        width = max(len(x) for x in self.labels)
        for p in range(self.n_pop):
            lines.append(
                f"  [{p:2d}] {self.labels[p]:<{width}}  "
                f"N={int(self.sizes[p]):>5d}  "
                f"flat[{int(self.starts[p]):>5d}:{int(self.stops[p]):>5d}]"
            )
        return "\n".join(lines)


def build_layout(popdata: pd.DataFrame) -> PopulationLayout:
    """Compute the flat-array layout from a CBGTPy `popdata` frame.

    Population order is popdata row order, preserved exactly so that `popid`
    is interchangeable between this port and the legacy agent - the legacy
    `agent.V[popid]` and our `V[layout.slice(popid)]` refer to the same
    neurons, in the same order.

    `popdata` is built by `https://github.com/CoAxLab/CBGTPy/blob/main/nchoice/popconstruct_nchoice.py:24
    helper_popconstruct`, which sets the row order at lines 36-45:

        GPi, STN, GPe, dSPN, iSPN, Cx, Th, FSI, CxI

    and then duplicates the first seven per action channel via
    `ModifyViaSelector(..., SelName([...]))` at line 50. FSI and CxI are NOT
    in that selector list, so they stay single and carry a NaN `action` -
    which is why `actions` is `None` for them here.
    """
    names, actions, labels, sizes = [], [], [], []

    for _, row in popdata.iterrows():
        name = str(_raw(row["name"]))

        # 'action' is absent for channel-agnostic populations (FSI, CxI) and
        # NaN-valued for them when the column exists.
        action = None
        if "action" in popdata.columns and not _is_missing(row["action"]):
            action = _raw(row["action"])

        names.append(name)
        actions.append(action)
        labels.append(name if action is None else f"{name}:{action}")
        sizes.append(int(_raw(row["N"])))

    sizes_arr = np.asarray(sizes, dtype=np.int64)
    stops = np.cumsum(sizes_arr)
    starts = stops - sizes_arr
    n_total = int(stops[-1]) if len(stops) else 0

    # neuron -> popid, e.g. [0]*75 + [1]*75 + [2]*750 + ...
    pop_index = np.repeat(np.arange(len(sizes_arr), dtype=np.int64), sizes_arr)

    if len(set(labels)) != len(labels):
        # Not fatal -- popid indexing still works -- but label lookups become
        # ambiguous, so the caller should know.
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        raise ValueError(
            f"popdata produced duplicate population labels: {dupes}. "
            "Expected (name, action) to be unique per row."
        )

    return PopulationLayout(
        names=tuple(names),
        actions=tuple(actions),
        labels=tuple(labels),
        sizes=sizes_arr,
        starts=starts.astype(np.int64),
        stops=stops.astype(np.int64),
        n_total=n_total,
        pop_index=pop_index,
    )



# Parameter flattening



def flatten_param(
    popdata: pd.DataFrame,
    param: str,
    layout: PopulationLayout | None = None,
    default: float = 0.0,
) -> np.ndarray:
    """Expand one popdata column into a flat per-neuron float64 array.

    Semantically identical to
    `np.concatenate(expandParamByCell(popdata, param, default))`
    (`https://github.com/CoAxLab/CBGTPy/blob/main/common/agentmatrixinit.py:50-67`), but built in one allocation instead
    of a list of per-population arrays. The `_selfcheck` at the bottom of this
    file asserts that equality element-for-element on the real default
    network, so the two are not merely intended to agree.

    Line-for-line correspondence with the legacy function:

    ===========================  ==========================================
    `expandParamByCell`        here
    ===========================  ==========================================
    line 54 column-absent test   the `param not in popdata.columns` early
                                 return below
    line 62 `is_nan()` guard   `_is_missing(cell)` -> `continue`
    line 63 `untrace(...)`     `_raw(cell)`
    line 65 `np.ones(N)*fill`  `out[layout.slice(popid)] = value`
    ===========================  ==========================================

    Parameters
    ----------
    popdata
        CBGTPy population frame.
    param
        Column name to expand. A column that does not exist is not an error -
        the whole array is filled with `default`, matching the legacy
        behaviour that produces zeros.
    layout
        Precomputed layout; built on demand if omitted.
    default
        Fallback for a missing column or a NaN cell. `initializeAgent` always
        passes 0, so leave this alone unless you are deliberately deviating.
    """
    if layout is None:
        layout = build_layout(popdata)

    out = np.full(layout.n_total, float(default), dtype=np.float64)

    if param not in popdata.columns:
        return out

    column = popdata[param]
    for popid in range(layout.n_pop):
        cell = column.iloc[popid]
        if _is_missing(cell):
            continue  # already holds `default`
        value = _raw(cell)
        try:
            out[layout.slice(popid)] = float(value)
        except (TypeError, ValueError) as exc:
            # popdata carries non-numeric columns too ('name', 'action'), and
            # a typo in a parameter list is otherwise reported as a bare
            # "could not convert string to float". 
            raise TypeError(
                f"popdata column {param!r} is not numeric: population "
                f"{layout.labels[popid]!r} (popid {popid}) holds {value!r}. "
                "flatten_param expands numeric per-neuron parameters only."
            ) from exc

    return out


def build_param_table(
    popdata: pd.DataFrame,
    params=None,
    layout: PopulationLayout | None = None,
    default: float = 0.0,
) -> dict[str, np.ndarray]:
    """Expand many columns at once.

    Returns `{param_name: flat_array}`. Missing columns yield all-`default`
    arrays rather than raising, deliberately - see `flatten_param``.

    This is the bulk equivalent of the loop at
    `https://github.com/CoAxLab/CBGTPy/blob/main/common/agentmatrixinit.py:204-205`:
    ```
        for prop in propertylist:
            setattr(agent, prop, expandParamByCell(popdata, prop, 0))
    ```

    with `propertylist` declared at `agentmatrixinit.py:104-202`. The
    difference is that we choose which names to expand (defaulting to
    `NEURON_PARAMS`) rather than expanding all 96 unconditionally, because
    Brian2 charges memory for every declared parameter.
    """
    if layout is None:
        layout = build_layout(popdata)
    if params is None:
        params = NEURON_PARAMS

    return {
        name: flatten_param(popdata, name, layout=layout, default=default)
        for name in params
    }



# The parameter sets the Brian2 model needs

# Time-invariant per-neuron constants referenced by the membrane and synaptic
# equations. These become Brian2 NeuronGroup
# parameters - declared `: 1` in the equations, assigned once at build time.
#
# Each entry is annotated with three provenance sites:
#   default=   where the value comes from, nchoice/paramfile_nchoice.py
#   agent=     where initializeAgent allocates it, https://github.com/CoAxLab/CBGTPy/blob/main/common/agentmatrixinit.py
#   used=      where the legacy integrator reads it,
#              https://github.com/CoAxLab/CBGTPy/blob/main/nchoice/agent_timestep_plasticity.pyx
#
MEMBRANE_PARAMS = (
    # capacitance (nF).  default=nchoice/paramfile_nchoice.py:5   agent=common/agentmatrixinit.py:148  nchoice/agent_timestep_plasticity.pyx:192
    "C",
    # membrane time constant (ms).  default=nchoice/paramfile_nchoice.py:6   agent=common/agentmatrixinit.py:146  nchoice/agent_timestep_plasticity.pyx:192
    "Taum",
    # leak reversal (mV).  default=nchoice/paramfile_nchoice.py:7   agent=common/agentmatrixinit.py:147  nchoice/agent_timestep_plasticity.pyx:192
    "RestPot",
    # post-spike reset target (mV).  default=nchoice/paramfile_nchoice.py:8   agent=common/agentmatrixinit.py:134  nchoice/agent_timestep_plasticity.pyx:179
    # Read by the reset, NOT by the membrane equation -- see
    # equations.CONSUMED_OUTSIDE_EQUATIONS.
    "ResetPot",
    # spike threshold (mV).  default=nchoice/paramfile_nchoice.py:9   agent=common/agentmatrixinit.py:133
    # nchoice/agent_timestep_plasticity.pyx:178 (reset gate), nchoice/agent_timestep_plasticity.nchoice/agent_timestep_plasticity.pyx:195 (Vaux clamp), nchoice/agent_timestep_plasticity.nchoice/agent_timestep_plasticity.pyx:208 (spike detect)
    "Threshold",
    # Ca decay time constant (ms).  default=nchoice/paramfile_nchoice.py:12  agent=common/agentmatrixinit.py:153  nchoice/agent_timestep_plasticity.pyx:193
    "Tau_ca",
)

# T-type Ca / post-inhibitory rebound. MODEL_SPEC.md section 3.7,
# transcribed from nchoice/agent_timestep_plasticity.nchoice/agent_timestep_plasticity.pyx:167-174.
TTYPE_PARAMS = (
    # max conductance; 0 except GPe/STN (nchoice/paramfile_nchoice.py:33-34).
    # default=nchoice/paramfile_nchoice.py:18  agent=common/agentmatrixinit.py:132  nchoice/agent_timestep_plasticity.pyx:174
    "g_T",
    # half-activation voltage (mV).  default=nchoice/paramfile_nchoice.py:16  agent=common/agentmatrixinit.py:128
    # nchoice/agent_timestep_plasticity.pyx:168 (branch select), nchoice/agent_timestep_plasticity.nchoice/agent_timestep_plasticity.pyx:174 (conduction gate)
    "V_h",
    # reversal potential (mV).  default=nchoice/paramfile_nchoice.py:17  agent=common/agentmatrixinit.py:150  nchoice/agent_timestep_plasticity.pyx:192
    "V_T",
    # de-inactivation time constant, V < V_h (ms).
    # default=nchoice/paramfile_nchoice.py:15  agent=common/agentmatrixinit.py:130  nchoice/agent_timestep_plasticity.pyx:170
    "tauhp",
    # inactivation time constant, V >= V_h (ms).
    # default=nchoice/paramfile_nchoice.py:14  agent=common/agentmatrixinit.py:131  nchoice/agent_timestep_plasticity.pyx:172
    "tauhm",
)

# Anomalous delayed rectifier + slow K. MODEL_SPEC.md section 3.8(c),
# transcribed from nchoice/agent_timestep_plasticity.nchoice/agent_timestep_plasticity.pyx:184-190.
INTRINSIC_PARAMS = (
    # ADR max conductance.  default=nchoice/paramfile_nchoice.py:19  agent=common/agentmatrixinit.py:136  nchoice/agent_timestep_plasticity.pyx:184
    "g_adr_max",
    # ADR half-activation (mV).  default=nchoice/paramfile_nchoice.py:20  agent=common/agentmatrixinit.py:137  used=nchoice/agent_timestep_plasticity.pyx:184
    "Vadr_h",
    # ADR slope (mV).  default=nchoice/paramfile_nchoice.py:21  agent=common/agentmatrixinit.py:138  used=nchoice/agent_timestep_plasticity.pyx:184
    "Vadr_s",
    # Reversal shared by BOTH g_adr and g_k. That is what nchoice/agent_timestep_plasticity.pyx:192 does -- the
    # same ADRRevPot appears in both current terms. Not a transcription slip.
    # default=nchoice/paramfile_nchoice.py:22  agent=common/agentmatrixinit.py:149  used=nchoice/agent_timestep_plasticity.pyx:192 (twice)
    "ADRRevPot",
    # Slow K max conductance.  default=nchoice/paramfile_nchoice.py:23  agent=common/agentmatrixinit.py:140  used=nchoice/agent_timestep_plasticity.pyx:190
    "g_k_max",
    # Slow K half-activation (mV).  default=nchoice/paramfile_nchoice.py:24  agent=common/agentmatrixinit.py:142  used=nchoice/agent_timestep_plasticity.pyx:188
    "Vk_h",
    # Slow K slope (mV).  default=nchoice/paramfile_nchoice.py:25  agent=common/agentmatrixinit.py:143  used=nchoice/agent_timestep_plasticity.pyx:188
    "Vk_s",
    # Slow K max time constant (ms).  default=nchoice/paramfile_nchoice.py:26  agent=common/agentmatrixinit.py:141  used=nchoice/agent_timestep_plasticity.pyx:187
    "tau_k_max",
)

# Receptor kinetics and reversals. MODEL_SPEC.md section 1.3; all defaults live
# in the receptordefaults dict at nchoice/paramfile_nchoice.py:39-46.
RECEPTOR_PARAMS = (
    # default=nchoice/paramfile_nchoice.py:39  agent=common/agentmatrixinit.py:115  used=nchoice/agent_timestep_plasticity.pyx:52,53,54,55 (drive + decay)
    "Tau_AMPA",
    # default=nchoice/paramfile_nchoice.py:41  agent=common/agentmatrixinit.py:116  used=nchoice/agent_timestep_plasticity.pyx:66,67,68,69
    "Tau_GABA",
    # default=nchoice/paramfile_nchoice.py:43  agent=common/agentmatrixinit.py:117  used=nchoice/agent_timestep_plasticity.pyx:78,79,80,81 and nchoice/agent_timestep_plasticity.pyx:90,91 (NMDA saturation)
    "Tau_NMDA",
    # default=nchoice/paramfile_nchoice.py:40  agent=common/agentmatrixinit.py:154  used=nchoice/agent_timestep_plasticity.pyx:198
    "RevPot_AMPA",
    # default=nchoice/paramfile_nchoice.py:42  agent=common/agentmatrixinit.py:155  used=nchoice/agent_timestep_plasticity.pyx:199
    "RevPot_GABA",
    # default=nchoice/paramfile_nchoice.py:44  agent=common/agentmatrixinit.py:156  used=nchoice/agent_timestep_plasticity.pyx:197
    "RevPot_NMDA",
    # Optogenetic excitation reversal.  default=nchoice/paramfile_nchoice.py:45  agent=common/agentmatrixinit.py:157
    # used=nchoice/agent_timestep_plasticity.pyx:202.  paramfile comment says "verify the exact values".
    "RevPot_ChR2",
    # Optogenetic inhibition reversal, -400. Far below any physiological
    # chloride reversal - a phenomenological knob, not a measured constant.
    # Preserve the number. default=nchoice/paramfile_nchoice.py:46  agent=common/agentmatrixinit.py:158  used=nchoice/agent_timestep_plasticity.pyx:204
    "RevPot_NpHR",
)

# External Ornstein-Uhlenbeck drive. Transcribed
# from nchoice/agent_timestep_plasticity.pyx:52-55 (AMPA), nchoice/agent_timestep_plasticity.pyx:66-69 (GABA), nchoice/agent_timestep_plasticity.pyx:78-81 (NMDA). Population-specific
# values live in the basestim dict at nchoice/paramfile_nchoice.py:48-87; the
# celldefaults dict does not define them, so any population absent from basestim
# gets 0 via the expandParamByCell fallback.
#
# NOTE: FreqExt_* is NOT a constant. The trial driver rewrites it every
# millisecond to deliver the stimulus - see nchoice/interface_nchoice.py, and
# equations.DRIVER_WRITABLE. It is loaded here as an initial value only.
EXTERNAL_PARAMS = (
    # Stimulus-carrying external AMPA rate (Hz). agent=common/agentmatrixinit.py:105  used=nchoice/agent_timestep_plasticity.pyx:52,53
    "FreqExt_AMPA",
    # agent=common/agentmatrixinit.py:107  used=nchoice/agent_timestep_plasticity.pyx:66,67
    "FreqExt_GABA",
    # agent=common/agentmatrixinit.py:108  used=nchoice/agent_timestep_plasticity.pyx:78,79
    "FreqExt_NMDA",
    # The driver's stash of the pre-stimulus baseline, copied back into
    # FreqExt_AMPA between stimuli. Never read by the integrator itself.
    # agent=common/agentmatrixinit.py:106  used= (driver only)
    "FreqExt_AMPA_basestim",
    # Efficacy of one external connection.  agent=common/agentmatrixinit.py:109  used=nchoice/agent_timestep_plasticity.pyx:52,53
    "MeanExtEff_AMPA",
    # agent=common/agentmatrixinit.py:110  used=nchoice/agent_timestep_plasticity.pyx:66,67
    "MeanExtEff_GABA",
    # agent=common/agentmatrixinit.py:111  used=nchoice/agent_timestep_plasticity.pyx:78,79
    "MeanExtEff_NMDA",
    # Number of external connections.  agent=common/agentmatrixinit.py:112  used=nchoice/agent_timestep_plasticity.pyx:52,53
    "MeanExtCon_AMPA",
    # agent=common/agentmatrixinit.py:113  used=nchoice/agent_timestep_plasticity.pyx:66,67
    "MeanExtCon_GABA",
    # agent=common/agentmatrixinit.py:114  used=nchoice/agent_timestep_plasticity.pyx:78,79
    "MeanExtCon_NMDA",
)

# Dopamine / plasticity constants. Loaded now so
# that it is available later; the fixed-weight network ignores all of these
# except dpmn_type and dpmn_cortex, which synapses.py needs in order to know
# which pathways are plasticity-eligible.
DOPAMINE_PARAMS = (
    # 1 = D1/dSPN (nchoice/paramfile_nchoice.py:121), 2 = D2/iSPN (nchoice/paramfile_nchoice.py:130), 0 elsewhere.
    # agent=common/agentmatrixinit.py:173  used=nchoice/agent_timestep_plasticity.pyx:283 (block gate), nchoice/agent_timestep_plasticity.pyx:63 (XPRE sign), nchoice/agent_timestep_plasticity.pyx:301/303
    # (which get_fDA branch), nchoice/agent_timestep_plasticity.pyx:14/23 (the branch test inside get_fDA)
    "dpmn_type",
    # 1 for cortical source populations; set on Cx at nchoice/paramfile_nchoice.py:30.
    # agent=common/agentmatrixinit.py:200  used=nchoice/agent_timestep_plasticity.pyx:63 (XPRE), nchoice/agent_timestep_plasticity.pyx:308 (which sources get plastic)
    "dpmn_cortex",
    # Phasic dopamine decay time constant. default=nchoice/paramfile_nchoice.py:89  agent=common/agentmatrixinit.py:161  used=nchoice/agent_timestep_plasticity.pyx:284
    "dpmn_tauDOP",
    # Presynaptic trace gain. default=nchoice/paramfile_nchoice.py:93  agent=common/agentmatrixinit.py:163  used=nchoice/agent_timestep_plasticity.pyx:285
    "dpmn_dPRE",
    # Presynaptic trace time constant. default=nchoice/paramfile_nchoice.py:97  agent=common/agentmatrixinit.py:164  used=nchoice/agent_timestep_plasticity.pyx:285
    "dpmn_tauPRE",
    # Postsynaptic trace gain. default=nchoice/paramfile_nchoice.py:95  agent=common/agentmatrixinit.py:166  used=nchoice/agent_timestep_plasticity.pyx:286
    "dpmn_dPOST",
    # Postsynaptic trace time constant. default=nchoice/paramfile_nchoice.py:98  agent=common/agentmatrixinit.py:167  used=nchoice/agent_timestep_plasticity.pyx:286
    "dpmn_tauPOST",
    # Eligibility trace time constant. default=nchoice/paramfile_nchoice.py:96  agent=common/agentmatrixinit.py:169  used=nchoice/agent_timestep_plasticity.pyx:288
    "dpmn_tauE",
    # Dopamine scale. default=nchoice/paramfile_nchoice.py:105  agent=common/agentmatrixinit.py:170  used=nchoice/agent_timestep_plasticity.pyx:290
    "dpmn_m",
    # Tonic dopamine level. default=nchoice/paramfile_nchoice.py:91  agent=common/agentmatrixinit.py:171  used=nchoice/agent_timestep_plasticity.pyx:290
    "dpmn_DAt",
    # Motivational decay time constant. default=nchoice/paramfile_nchoice.py:92 (1e100)  agent=common/agentmatrixinit.py:172
    # used= NOWHERE. The comment at nchoice/agent_timestep_plasticity.pyx:293 records that lines 1645-1647 of the
    # reference implementation (motivational decay) are deliberately excluded.
    # Carried for completeness only.
    "dpmn_taum",
    # Soft upper weight bound. default=nchoice/paramfile_nchoice.py:125 (dSPN) / :133 (iSPN)
    # agent=common/agentmatrixinit.py:174  used=nchoice/agent_timestep_plasticity.pyx:317
    "dpmn_wmax",
    # Learning rate; NEGATIVE for iSPN (nchoice/paramfile_nchoice.py:131, -38.2), which is what
    # flips D2 plasticity's sign. default=nchoice/paramfile_nchoice.py:122/131  agent=common/agentmatrixinit.py:175
    # used=nchoice/agent_timestep_plasticity.pyx:310
    "dpmn_alphaw",
    # fDA piecewise breakpoint. default=nchoice/paramfile_nchoice.py:116  agent=common/agentmatrixinit.py:178
    # used=nchoice/agent_timestep_plasticity.pyx:16,20 (D1 branch), nchoice/agent_timestep_plasticity.pyx:24,26 (D2 branch) inside get_fDA
    "dpmn_x_fda",
    # fDA saturation level. default=nchoice/paramfile_nchoice.py:117  agent=common/agentmatrixinit.py:179
    # used=nchoice/agent_timestep_plasticity.pyx:19,20 (D1 branch), nchoice/agent_timestep_plasticity.pyx:25,26 (D2 branch) inside get_fDA
    "dpmn_y_fda",
    # D2-only scale factor applied to fDA. default=nchoice/paramfile_nchoice.py:118  agent=common/agentmatrixinit.py:180
    # used=nchoice/agent_timestep_plasticity.pyx:25,26 (D2 branch only)
    "dpmn_d2_DA_eps",
)

# Parameters that agentmatrixinit.py requests under names the parameter file
# never defines, so they are ALWAYS zero. Kept here as an explicit,
# named list rather than an accident. 
#
#   agentmatrixinit name  ->  paramfile name   ->  effect
#   g_ahp                 ->  Eff_ca           ->  AHP current term == 0
#   Vk                    ->  RestPot_ca       ->  (irrelevant, multiplied by 0)
#   alpha_ca              ->  Alpha_ca         ->  Ca never increments; stays 0
#
# The port omits the AHP term from the membrane equation entirely, which is
# exact given these are zero. Do not "fix" the names without re-validating.
DEAD_PARAMS = ("g_ahp", "Vk", "alpha_ca")

NEURON_PARAMS = (
    MEMBRANE_PARAMS
    + TTYPE_PARAMS
    + INTRINSIC_PARAMS
    + RECEPTOR_PARAMS
    + EXTERNAL_PARAMS
    + DOPAMINE_PARAMS
)


# Initial values for the state variables, as the legacy code actually sets
# them. Where the name appears in popdata the value is read from there; where
# it does not, expandParamByCell's 0 fallback applies. The comment on each line
# records which case it is.
INITIAL_STATE = {
    # -- membrane -------------------------------------------------------------
    #
    # 'V' is NOT a popdata column - celldefaults (nchoice/paramfile_nchoice.py:4-28) never defines
    # it - so agentmatrixinit.py:205 fills it with the 0 fallback. That is
    # ABOVE Threshold (-50, nchoice/paramfile_nchoice.py:9). Tracing the first timestep:
    #
    #   nchoice/agent_timestep_plasticity.pyx:168   cond = (V < V_h)  ->  (0 < -60) = False, so h inactivates
    #   nchoice/agent_timestep_plasticity.pyx:178   cond = (V <= Threshold)  ->  (0 <= -50) = False
    #   nchoice/agent_timestep_plasticity.pyx:179   V -= (V - ResetPot) * 1  ->  V = ResetPot
    #   nchoice/agent_timestep_plasticity.pyx:181   cond = 0 * (RefrState == 0) = 0, so nothing integrates
    #
    # Net effect: the network starts at ResetPot, one step late. Reproduce the
    # 0 rather than "helpfully" starting at RestPot - the one-step offset is
    # part of what we compares against later.
    "V": 0.0,                 # not a popdata column -> 0.  agent=common/agentmatrixinit.py:127
    # Decays from 0 and its only source is `Ca += alpha_ca` at nchoice/agent_timestep_plasticity.pyx:211 with
    # alpha_ca identically 0. Flat zero for the whole run. agent=common/agentmatrixinit.py:151
    "Ca": 0.0,
    # IS a popdata column: celldefaults n_k = 0 at nchoice/paramfile_nchoice.py:27.  agent=common/agentmatrixinit.py:144
    "n_k": None,
    # IS a popdata column: celldefaults h = 1 at nchoice/paramfile_nchoice.py:28.  agent=common/agentmatrixinit.py:129
    # Note h starts at 1, i.e. T-type fully de-inactivated.
    "h": None,
    # Refractory counter, set to 10 at nchoice/agent_timestep_plasticity.pyx:212 and decremented at nchoice/agent_timestep_plasticity.pyx:182.
    # Replaced in the port by Brian2's `refractory` duration; carried here only
    # so the legacy-side comparison later has something to line up against.
    # not a popdata column -> 0.  agent=common/agentmatrixinit.py:135
    "RefrState": 0.0,

    # -- synaptic conductances ------------------------------------------------
    # None of these are popdata columns, so all start at 0.
    # LS_* are decayed at nchoice/agent_timestep_plasticity.pyx:55/69/81 and incremented at nchoice/agent_timestep_plasticity.pyx:61/75/91.
    "LS_AMPA": 0.0,           # agent=common/agentmatrixinit.py:121
    "LS_GABA": 0.0,           # agent=common/agentmatrixinit.py:122
    "LS_NMDA": 0.0,           # agent=common/agentmatrixinit.py:123
    # ExtS_* start at 0 and relax toward ExtMuS_* over Tau_*; see nchoice/agent_timestep_plasticity.pyx:54/68/80.
    # There is no burn-in shortcut in the legacy code either -- this is why
    # interface_nchoice.py:96 runs 5000 warm-up steps before the first trial.
    "ExtS_AMPA": 0.0,         # agent=common/agentmatrixinit.py:118
    "ExtS_GABA": 0.0,         # agent=common/agentmatrixinit.py:119
    "ExtS_NMDA": 0.0,         # agent=common/agentmatrixinit.py:120
    # Optogenetic drive; 0 means no stimulation. Written by the driver at
    # interface_nchoice.py:156 etc. Read at nchoice/agent_timestep_plasticity.pyx:202,204.  agent=common/agentmatrixinit.py:189
    "ExtS_Opto": 0.0,

    # -- spike bookkeeping ----------------------------------------------------
    # Incremented at nchoice/agent_timestep_plasticity.pyx:82/83, latched and cleared at nchoice/agent_timestep_plasticity.pyx:213/214. Both exist
    # ONLY to drive the NMDA saturation decay at nchoice/agent_timestep_plasticity.pyx:90. The port replaces them
    # with a Brian2 `(event-driven)` synaptic variable  and does not
    # carry them on the NeuronGroup -- they are listed here for the later
    # comparison only.
    "timesincelastspike": 0.0,   # agent=common/agentmatrixinit.py:124
    "Ptimesincelastspike": 0.0,  # agent=common/agentmatrixinit.py:125

    # -- plasticity traces ----------------------------------------------------
    # All ARE popdata columns, all defaulting to 0.0 in dpmndefaults.
    "dpmn_DAp": None,         # nchoice/paramfile_nchoice.py:107  agent=common/agentmatrixinit.py:160  nchoice/agent_timestep_plasticity.pyx:284
    "dpmn_APRE": None,        # nchoice/paramfile_nchoice.py:108  agent=common/agentmatrixinit.py:162  nchoice/agent_timestep_plasticity.pyx:285
    "dpmn_APOST": None,       # nchoice/paramfile_nchoice.py:109  agent=common/agentmatrixinit.py:165  nchoice/agent_timestep_plasticity.pyx:286
    "dpmn_E": None,           # nchoice/paramfile_nchoice.py:106  agent=common/agentmatrixinit.py:168  nchoice/agent_timestep_plasticity.pyx:288
    # XPRE/XPOST are zeroed at the top of EVERY timestep (nchoice/agent_timestep_plasticity.nchoice/agent_timestep_plasticity.pyx:41-42), so their
    # initial value is irrelevant to dynamics. Carried for completeness.
    "dpmn_XPRE": None,        # nchoice/paramfile_nchoice.py:110  agent=common/agentmatrixinit.py:198  nchoice/agent_timestep_plasticity.pyx:41,63
    "dpmn_XPOST": None,       # nchoice/paramfile_nchoice.py:111  agent=common/agentmatrixinit.py:199  nchoice/agent_timestep_plasticity.pyx:42,215
}
# A value of `None` above means "read it from popdata via flatten_param";
# a float means "popdata has no such column, so expandParamByCell's 0 fallback
# applies and we hardcode the same constant".


def build_initial_state(
    popdata: pd.DataFrame,
    layout: PopulationLayout | None = None,
) -> dict[str, np.ndarray]:
    """Flat initial values for every state variable, matching `initializeAgent`.

    Entries whose `INITIAL_STATE` value is `None` are read from popdata;
    the rest are filled with the recorded constant. Either way the result is a
    flat float64 array per variable.

    Equivalent to what `common/agentmatrixinit.py:204-205` leaves in the agent
    immediately after construction, restricted to the variables that actually
    carry state. Nothing else in CBGTPy touches these before the first timestep:
    `initializeAgent` returns at `agentmatrixinit.py:218` and the next thing
    to run is `multitimestep_mutator` at
    `nchoice/interface_nchoice.py:96`.
    """
    if layout is None:
        layout = build_layout(popdata)

    state = {}
    for name, const in INITIAL_STATE.items():
        if const is None:
            state[name] = flatten_param(popdata, name, layout=layout, default=0.0)
        else:
            state[name] = np.full(layout.n_total, float(const), dtype=np.float64)
    return state


# Self-check


def _selfcheck() -> int:
    """Build the default n-choice popdata and diff against the legacy expansion.

    Run with `python -m brian_backend.params` from the repository root.
    Returns a process exit code: 0 on success, 1 on any mismatch.
    """
    import sys

    sys.path.insert(0, ".")

    import nchoice.paramfile_nchoice as pf
    import nchoice.init_params_nchoice as par
    import nchoice.popconstruct_nchoice as pc
    from common.agentmatrixinit import expandParamByCell

    popdata = pc.helper_popconstruct(
        par.helper_actionchannels(None),
        pf.popspecific,
        par.helper_cellparams(pf.celldefaults),
        par.helper_receptor(pf.receptordefaults),
        pf.basestim,
        par.helper_dpmn(pf.dpmndefaults),
        par.helper_d1(pf.dSPNdefaults),
        par.helper_d2(pf.iSPNdefaults),
    )

    layout = build_layout(popdata)
    print(layout.describe())
    print()

    # Every name the legacy agent expands, so the check covers the dead ones
    # and the state variables too, not only NEURON_PARAMS.
    checked = tuple(NEURON_PARAMS) + DEAD_PARAMS + tuple(INITIAL_STATE)

    failures = []
    for name in checked:
        legacy = np.concatenate(expandParamByCell(popdata, name, 0))
        ours = flatten_param(popdata, name, layout=layout, default=0.0)

        if legacy.shape != ours.shape:
            failures.append(f"{name}: shape {ours.shape} != legacy {legacy.shape}")
            continue
        if not np.array_equal(legacy, ours):
            bad = int(np.count_nonzero(legacy != ours))
            first = int(np.flatnonzero(legacy != ours)[0])
            failures.append(
                f"{name}: {bad} of {legacy.size} entries differ; "
                f"first at index {first} (ours={ours[first]}, legacy={legacy[first]})"
            )

    print(f"compared {len(checked)} parameters against expandParamByCell")

    # Report the always-zero parameters explicitly rather than letting them
    # pass silently - if one ever becomes nonzero, the port's omission of the
    # AHP term stops being exact and we need to know.
    print("\ndead-parameter check (MODEL_SPEC.md 1.4):")
    for name in DEAD_PARAMS:
        arr = flatten_param(popdata, name, layout=layout)
        status = "all zero, as expected" if not arr.any() else "*** NONZERO ***"
        print(f"  {name:<10} {status}")
        if arr.any():
            failures.append(
                f"{name}: expected all-zero (MODEL_SPEC 1.4) but found nonzero "
                "entries -- the AHP term can no longer be omitted"
            )

    # Sanity: layout arithmetic must be self-consistent.
    assert layout.stops[-1] == layout.n_total
    assert np.array_equal(layout.stops - layout.starts, layout.sizes)
    assert layout.pop_index.shape == (layout.n_total,)
    print("\nlayout arithmetic consistent")

    th = layout.popids_named("Th")
    print(f"Th populations (decision readout): popids {th}, "
          f"{layout.indices_named('Th').size} neurons")

    if failures:
        print(f"\nFAIL -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nOK - flat expansion matches expandParamByCell exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())


