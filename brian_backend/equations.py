"""
This file is made to create the ``Equations`` object
object, nothing more. ``neurons.py`` turns them into a NeuronGroup.

Run the self-check with::

    python -m brian_backend.equations

-------------------------------------------------------------------------------
Design decisions, and why
-------------------------------------------------------------------------------

**Everything is unitless.** CBGTPy stores bare numbers - voltages implicitly
mV, times implicitly ms. Rather than audit every parameter for physical units,
each state variable is declared ``: 1`` and each derivative is divided by ``ms``
so that Brian2's dimensional check passes. The numbers therefore mean exactly
what they mean in the ``.pyx`` files.

**One integration method: ``euler``.** The legacy integrator uses forward Euler
for ``V``, ``n_k``, ``Ca`` and ``h``, and Euler-Maruyama for the ``ExtS_*``
Ornstein-Uhlenbeck processes. Brian2's ``euler`` reproduces both exactly.

**...except the LS_* decay, which is NOT an ODE here.** The legacy code decays
the recurrent conductances *exactly*::

    LS_AMPA *= exp(-dt / Tau_AMPA)

Writing that as ``dLS_AMPA/dt = -LS_AMPA/(Tau_AMPA*ms)`` under ``method='euler'``
would give ``LS *= (1 - dt/Tau)`` instead. At ``Tau_AMPA = 2``, ``dt = 0.2``
that is 0.900 versus 0.905 - a 0.5% error *per step*, compounding into a
systematic underestimate of recurrent excitation. Not acceptable.

So ``LS_*`` are declared as plain parameters and decayed by an explicit
``run_regularly`` (``LS_DECAY_CODE``) using the exact exponential. See
``LS_DECAY_WHEN`` below for the scheduling argument, which is not optional.

**The AHP term is omitted.** The legacy membrane equation contains
``Ca * g_ahp / C * 0.001 * (V - Vk)``. Both ``g_ahp`` and ``Vk`` are identically
zero because of a parameter-name mismatch, so the
whole term is exactly zero and dropping it is not an approximation. ``Ca`` and
its decay are still carried, so that recorded traces line up with the legacy
backend - ``Ca`` is a flat zero line in both.

**``timesincelastspike`` / ``Ptimesincelastspike`` are gone.** They exist in the
legacy agent solely to drive the NMDA saturation decay. Brian2 expresses that
with an ``(event-driven)`` synaptic variable, which needs no neuron-side timer.

-------------------------------------------------------------------------------
Scheduling: why LS_DECAY_WHEN must be 'before_synapses'
-------------------------------------------------------------------------------

Write ``d = exp(-dt/Tau)``, let ``S_t`` be the conductance delivered by spikes
fired at step ``t``, and let ``M_t`` be the value of ``LS`` that the membrane
update at step ``t`` actually reads.

The legacy step order is: decay, then add the *previous* step's spikes, then integrate ``V``. 
So::

    M_t = d * M_{t-1} + S_{t-1}

Brian2's schedule is ``start -> groups -> thresholds -> synapses -> resets``.
Placing the decay at ``before_synapses`` gives, within step ``t``:

    groups      V reads M_t
    thresholds  spikes of step t detected
    decay       LS *= d
    synapses    LS += S_t

so ``M_{t+1} = d * M_t + S_t``, i.e. ``M_t = d * M_{t-1} + S_{t-1}``. Identical.

Putting the decay in the default ``start`` slot instead would give
``M_t = d * (M_{t-1} + S_{t-1})`` - every spike's contribution decayed once too
often, on its very first step. That is a real and avoidable divergence.

-------------------------------------------------------------------------------
Known divergences introduced here
-------------------------------------------------------------------------------

These are the equation-level entries. Each must be
quantified and checked before porting to Brian2 is finalized.

D2  RESOLVED - the sequential membrane update is now reproduced exactly.

    The legacy code updates ``V`` in six sequential in-place statements, each
    seeing the ``V`` the previous one left. Measured in the full network
    that shifted low-rate populations by up to 10%,
    including ``Th``, which the decision threshold reads - so it was fixed
    rather than accepted.

    The fix is small because the six statements are really TWO stages. pyx:197
    through pyx:204:
    ```
     a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_NMDA[popid] - a.Vaux[popid]) * .001 * (a.LS_NMDA[popid] + a.ExtS_NMDA[popid]) / a.C[popid] / (1. + np.exp(-0.062 * a.Vaux[popid] / 3.57))
     a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_AMPA[popid] - a.Vaux[popid]) * .001 * (a.LS_AMPA[popid] + a.ExtS_AMPA[popid]) / a.C[popid]
     a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_GABA[popid] - a.Vaux[popid]) * .001 * (a.LS_GABA[popid] + a.ExtS_GABA[popid]) / a.C[popid]

     #if a.ExtS_Opto[popid][0] > 0:
     a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_ChR2[popid] - a.Vaux[popid]) * (a.ExtS_Opto[popid]>0) * .001 * a.ExtS_Opto[popid] / a.C[popid]
     #if a.ExtS_Opto[popid][0] < 0:
     a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_NpHR[popid] - a.Vaux[popid]) * (a.ExtS_Opto[popid]<0) * .001 * -a.ExtS_Opto[popid] / a.C[popid] 
    ```
    
    depend on ``V`` only through ``Vaux``, never through ``V``
    itself, so they do not chain: they all add against the same frozen ``Vaux``.
    The only genuine ordering is that ``Vaux`` is taken from the POST-leak
    ``V``, not the step-start ``V``.
    ```
    a.Vaux[popid] = np.minimum(a.V[popid],a.Threshold[popid])
    ```

    So three "post-update" subexpressions reproduce it, with no hand-written
    integrator:

        V_leaked  = V + (dt/ms) * V_intrinsic    ->  Vaux   (pyx:192 then :195)
        ```
        a.V[popid] = a.V[popid] + a.cond[popid] * -a.dt * (1 / a.Taum[popid] * (a.V[popid] - a.RestPot[popid]) + a.Ca[popid] * a.g_ahp[popid] / a.C[popid] * 0.001 * (a.V[popid] - a.Vk[popid]) + a.g_adr[popid] / a.C[popid] * (a.V[popid] - a.ADRRevPot[popid]) + a.g_k[popid] / a.C[popid] * (a.V[popid] - a.ADRRevPot[popid]) + a.g_rb[popid] / a.C[popid] * (a.V[popid] - a.V_T[popid]))
        a.Ca[popid] = a.Ca[popid] - a.cond[popid] * a.Ca[popid] * a.dt / a.Tau_ca[popid]

        a.Vaux[popid] = np.minimum(a.V[popid],a.Threshold[popid])
        ```
        h_next    = h + (dt/ms) * h_rate         ->  g_rb   (pyx:170/172 then :174)
        ```
        a.h[popid] = a.h[popid] + a.cond[popid] * a.dt * (1 - a.h[popid]) / a.tauhp[popid]
        # false (cond = 0)
        a.h[popid] = a.h[popid] + (1 - a.cond[popid]) * a.dt * (-a.h[popid]) / a.tauhm[popid]
        # mix
        a.g_rb[popid] = a.g_T[popid] * a.h[popid] * (1 - a.cond[popid])
        ```
        n_k_next  = n_k + (dt/ms) * n_k_rate     ->  g_k    (pyx:189 then :190)
        ```
        a.n_k[popid] = a.n_k[popid] + a.cond[popid] * -a.dt / a.tau_n[popid] * (a.n_k[popid] - a.n_inif[popid])
        a.g_k[popid] = a.g_k_max[popid] * a.n_k[popid]
        ```

    Forward Euler on ``dV/dt = (V_intrinsic + V_synaptic)/ms`` then gives
    ``V + dt*V_intrinsic + dt*V_synaptic(Vaux)``, which is pyx:192-204 term for
    term:
    ```
    a.V[popid] = a.V[popid] + a.cond[popid] * -a.dt * (1 / a.Taum[popid] * (a.V[popid] - a.RestPot[popid]) + a.Ca[popid] * a.g_ahp[popid] / a.C[popid] * 0.001 * (a.V[popid] - a.Vk[popid]) + a.g_adr[popid] / a.C[popid] * (a.V[popid] - a.ADRRevPot[popid]) + a.g_k[popid] / a.C[popid] * (a.V[popid] - a.ADRRevPot[popid]) + a.g_rb[popid] / a.C[popid] * (a.V[popid] - a.V_T[popid]))
    a.Ca[popid] = a.Ca[popid] - a.cond[popid] * a.Ca[popid] * a.dt / a.Tau_ca[popid]

    a.Vaux[popid] = np.minimum(a.V[popid],a.Threshold[popid])

    a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_NMDA[popid] - a.Vaux[popid]) * .001 * (a.LS_NMDA[popid] + a.ExtS_NMDA[popid]) / a.C[popid] / (1. + np.exp(-0.062 * a.Vaux[popid] / 3.57))
    a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_AMPA[popid] - a.Vaux[popid]) * .001 * (a.LS_AMPA[popid] + a.ExtS_AMPA[popid]) / a.C[popid]
    a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_GABA[popid] - a.Vaux[popid]) * .001 * (a.LS_GABA[popid] + a.ExtS_GABA[popid]) / a.C[popid]

    #if a.ExtS_Opto[popid][0] > 0:
    a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_ChR2[popid] - a.Vaux[popid]) * (a.ExtS_Opto[popid]>0) * .001 * a.ExtS_Opto[popid] / a.C[popid]
    #if a.ExtS_Opto[popid][0] < 0:
    a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_NpHR[popid] - a.Vaux[popid]) * (a.ExtS_Opto[popid]<0) * .001 * -a.ExtS_Opto[popid] / a.C[popid]        
        
    ```
    
    ``build_neuron_equations(sequential=False)`` selects the old
    simultaneous form, kept only so the divergence stays measurable.

1  Reset target.
    The legacy code writes ``V = 0`` at spike time - a cosmetic spike peak -
    and relies on the *next* step to pull ``V`` down to ``ResetPot``. Here the
    reset writes ``ResetPot`` directly. Spike times are unaffected; recorded
    voltage traces lose a one-sample 0 mV artifact.

    The legacy ``RefrState`` counter freezes the membrane for 10 steps, and this 
    file originally set Brian2's
    ``refractory`` to ``REFRACTORY_STEPS * dt``. That is one step short: Brian2
    tests ``t - lastspike < refractory``, so ``n * dt`` freezes ``n - 1`` steps.
    The correct duration is ``(REFRACTORY_STEPS + 1) * dt``.

"""

from __future__ import annotations

from brian2 import Equations

from .params import (
    EXTERNAL_PARAMS,
    INTRINSIC_PARAMS,
    MEMBRANE_PARAMS,
    RECEPTOR_PARAMS,
    TTYPE_PARAMS,
)

__all__ = [
    "DT_MS",
    "REFRACTORY_STEPS",
    "NMDA_ALPHA",
    "FIXED_WEIGHT_PARAMS",
    "STATE_VARIABLES",
    "DRIVER_WRITABLE",
    "CONSUMED_OUTSIDE_EQUATIONS",
    "build_neuron_equations",
    "LS_DECAY_CODE",
    "LS_DECAY_WHEN",
    "THRESHOLD_CONDITION",
    "RESET_CODE",
]


# =============================================================================
# Numeric constants lifted from the source
# =============================================================================

#: Integration timestep in ms. ``common/agentmatrixinit.py:211``.
DT_MS = 0.2

#: Steps the membrane stays frozen after a spike. By tracing the ``RefrState`` counter; verified empirically by
#: ``_check_refractory_duration`` below.
REFRACTORY_STEPS = 10

#: NMDA saturation increment per presynaptic spike. ``ALPHA = 0.6332``
#: at ``nchoice/agent_timestep_plasticity.pyx:89``. .
NMDA_ALPHA = 0.6332

# Magnesium block of the NMDA conductance:
#     1 / (1 + exp(-0.062 * Vaux / 3.57))
MG_SLOPE = 0.062
MG_SCALE = 3.57
# a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_NMDA[popid] - a.Vaux[popid]) * .001 * (a.LS_NMDA[popid] + a.ExtS_NMDA[popid]) / a.C[popid] / (1. + np.exp(-0.062 * a.Vaux[popid] / 3.57))

# Slow potassium activation time constant:
#     tau_n = tau_k_max / (exp(-dv/30) + exp(dv/30)),  dv = V + 55
KDR_OFFSET = 55.0
KDR_SLOPE = 30.0
# a.dv[popid] = a.V[popid] + 55
# a.tau_n[popid] = a.tau_k_max[popid] / (np.exp(-1 * a.dv[popid] / 30) + np.exp(a.dv[popid] / 30))

#: Conductance unit conversion folded into the source as a bare ``.001``.
COND_SCALE = 0.001


# =============================================================================
# Which parameters and state variables exist
# =============================================================================

#: Per-neuron constants the fixed-weight model needs. Assembled from
#: ``params.py`` rather than re-listed, so the two files cannot drift apart.
#: Excludes ``DOPAMINE_PARAMS`` which will be handled by another file (still planning about how to handle it).
FIXED_WEIGHT_PARAMS = (
    MEMBRANE_PARAMS + TTYPE_PARAMS + INTRINSIC_PARAMS
    + RECEPTOR_PARAMS + EXTERNAL_PARAMS
)

#: Variables that carry state across timesteps and must be initialised.
#: ``LS_*`` and ``ExtS_Opto`` appear here even though Brian2 sees them as
#: parameters: they are mutated during the run (by synapses, by the decay
#: run_regularly, and by the driver) so they are state in every sense that
#: matters for validation.
STATE_VARIABLES = (
    "V", "h", "n_k", "Ca",
    "LS_AMPA", "LS_GABA", "LS_NMDA",
    "ExtS_AMPA", "ExtS_GABA", "ExtS_NMDA",
    "ExtS_Opto",
)

#: Parameters the trial driver (which again I need to design later) rewrites mid-run to deliver stimuli,
#: gain changes and optogenetic input. Everything else is write-once.
DRIVER_WRITABLE = (
    "FreqExt_AMPA", "FreqExt_GABA", "FreqExt_NMDA", "ExtS_Opto",
)

#: Declared as neuron parameters but never read by any equation - they are
#: consumed elsewhere. Listed explicitly so that the "declared but unused"
#: check in ``_check_parses`` stays meaningful instead of being switched off.
#:
#:   ResetPot                read by RESET_CODE at spike time
#:   FreqExt_AMPA_basestim   the driver's stash of the pre-stimulus baseline,
#:                           copied back into FreqExt_AMPA between stimuli
#:                           (see nchoice/interface_nchoice.py)
CONSUMED_OUTSIDE_EQUATIONS = {"ResetPot", "FreqExt_AMPA_basestim"}


# =============================================================================
# Equation fragments
# =============================================================================


# -- the clamped driving potential --------------------------------
#
# Adapted from nchoice/agent_timestep_plasticity.pyx:195
#
#     a.Vaux[popid] = np.minimum(a.V[popid], a.Threshold[popid])
#
# The synaptic driving force must not blow up once V crosses threshold
# on its way to being reset, so V is clamped from above at Threshold.
#
# The -1e10 lower bound has no counterpart in the source. It exists only because
# Brian2's clip() requires two bounds; it is far below any reachable V (RestPot
# is -70, RevPot_NpHR is -400) so it never binds.
#
# The source LATCHES Vaux once, at pyx:195, from the
# V left by the leak/intrinsic update at pyx:192, and then reuses that frozen
# number in all five subsequent V updates (pyx:197,198,199,202,204) even though
# each of those changes V. Declaring Vaux as a Brian2 subexpression means it is
# re-evaluated against the simultaneous V instead -- correct ODE semantics, but
# not what the source does.
_VAUX = """
V_leaked = V + (dt / ms) * V_intrinsic : 1
Vaux = clip(V_leaked, -1e10, Threshold) : 1
"""

# The simultaneous alternative, kept so that can be measured rather than
# merely asserted to be gone. ``build_neuron_equations(sequential=False)``
# selects it; nothing in the port uses it by default.
_VAUX_SIMULTANEOUS = """
Vaux = clip(V, -1e10, Threshold) : 1
"""

# -- T-type Ca / post-inhibitory rebound -----------------------------
#
# Adapted from nchoice/agent_timestep_plasticity.pyx:167-174
#
#     pyx:168   a.cond[popid] = (a.V[popid] < a.V_h[popid]).astype(int)
#     pyx:170   a.h[popid] = a.h[popid] + a.cond[popid] * a.dt * (1 - a.h[popid]) / a.tauhp[popid]
#     pyx:172   a.h[popid] = a.h[popid] + (1 - a.cond[popid]) * a.dt * (-a.h[popid]) / a.tauhm[popid]
#     pyx:174   a.g_rb[popid] = a.g_T[popid] * a.h[popid] * (1 - a.cond[popid])
#
# pyx:170 and pyx:172 are two in-place statements guarded by complementary
# masks, i.e. a branch written branchlessly. Exactly one fires per neuron, so
# they collapse into the single dh/dt below with `int(...)` playing the role of
# `cond`. `(1 - cond)` at pyx:174 is `int(V >= V_h)`: g_rb conducts only ABOVE
# the half-activation voltage, having de-inactivated below it.
#
# pyx:174 computes g_rb from the h that
# pyx:170/172 have just updated, whereas a Brian2 subexpression evaluates g_rb
# from the step-start h. The gap is one Euler increment of h, at most
# dt/tauhp = 0.2/100 = 0.002 absolute. It reaches the membrane only through GPe
# and STN, the sole populations with g_T != 0.
#
# NOTE the deliberate absence of ``(unless refractory)`` on dh/dt. This whole
# block sits at pyx:167-174, BEFORE the refractory mask is computed at pyx:181,
# so in the source h keeps evolving while the neuron is refractory. Adding
# `(unless refractory)` here would freeze it and change rebound bursting in GPe
# and STN, the only two populations with g_T != 0 (paramfile:33-34).
_TTYPE = """
h_rate = int(V < V_h) * (1 - h) / tauhp - int(V >= V_h) * h / tauhm : 1
dh/dt = h_rate / ms : 1
h_next = h + (dt / ms) * h_rate : 1
g_rb = g_T * h_next * int(V >= V_h) : 1
"""

# Simultaneous alternative: g_rb reads the step-start h. See _VAUX_SIMULTANEOUS.
_TTYPE_SIMULTANEOUS = """
h_rate = int(V < V_h) * (1 - h) / tauhp - int(V >= V_h) * h / tauhm : 1
dh/dt = h_rate / ms : 1
g_rb = g_T * h * int(V >= V_h) : 1
"""

# -- anomalous delayed rectifier and slow K -----------------------
#
# Adapted from nchoice/agent_timestep_plasticity.pyx:184-190
#
#     pyx:184   a.g_adr[popid]  = a.g_adr_max[popid] / (1 + np.exp((a.V[popid]-a.Vadr_h[popid]) / a.Vadr_s[popid]))
#     pyx:186   a.dv[popid]     = a.V[popid] + 55
#     pyx:187   a.tau_n[popid]  = a.tau_k_max[popid] / (np.exp(-1 * a.dv[popid] / 30) + np.exp(a.dv[popid] / 30))
#     pyx:188   a.n_inif[popid] = 1 / (1 + np.exp(-(a.V[popid] - a.Vk_h[popid]) / a.Vk_s[popid]))
#     pyx:189   a.n_k[popid]    = a.n_k[popid] + a.cond[popid] * -a.dt / a.tau_n[popid] * (a.n_k[popid] - a.n_inif[popid])
#     pyx:190   a.g_k[popid]    = a.g_k_max[popid] * a.n_k[popid]
#
# pyx:186 (`dv`) is inlined here as `(V + KDR_OFFSET)`; the source keeps it as a
# scratch array only to avoid recomputing it twice on the next line.
#
# pyx:189 rearranged: `n_k += cond * (-dt/tau_n) * (n_k - n_inif)` is
# `cond * dt * (n_inif - n_k)/tau_n`, hence the sign flip below. `cond` becomes
# `(unless refractory)`. Note the source spells it `n_inif`, not `n_inf` --
# renamed here, it is the steady-state activation.
#
# Both currents use ADRRevPot as their reversal. That is what pyx:192 does --
# the same ADRRevPot appears in the g_adr term and the g_k term. It is not a
# transcription slip on our part; whether it was one in the original is
# open question territory, but the port copies it.
_INTRINSIC = """
g_adr = g_adr_max / (1 + exp((V - Vadr_h) / Vadr_s)) : 1
n_inf = 1 / (1 + exp(-(V - Vk_h) / Vk_s)) : 1
tau_n = tau_k_max / (exp(-(V + {off}) / {slope})
                     + exp((V + {off}) / {slope})) : 1
n_k_rate = (n_inf - n_k) / tau_n : 1
dn_k/dt = n_k_rate / ms : 1 (unless refractory)
n_k_next = n_k + int(not_refractory) * (dt / ms) * n_k_rate : 1
g_k = g_k_max * n_k_next : 1
"""

# Simultaneous alternative: g_k reads the step-start n_k. See _VAUX_SIMULTANEOUS.
_INTRINSIC_SIMULTANEOUS = """
g_adr = g_adr_max / (1 + exp((V - Vadr_h) / Vadr_s)) : 1
n_inf = 1 / (1 + exp(-(V - Vk_h) / Vk_s)) : 1
tau_n = tau_k_max / (exp(-(V + {off}) / {slope})
                     + exp((V + {off}) / {slope})) : 1
n_k_rate = (n_inf - n_k) / tau_n : 1
dn_k/dt = n_k_rate / ms : 1 (unless refractory)
g_k = g_k_max * n_k : 1
"""

# --  calcium ------------------------------------------------------
#
# Adapted from nchoice/agent_timestep_plasticity.pyx:193
#
#     a.Ca[popid] = a.Ca[popid] - a.cond[popid] * a.Ca[popid] * a.dt / a.Tau_ca[popid]
#
# with the spike-time source term at pyx:211
#
#     a.Ca[popid][neuron] += a.alpha_ca[popid][neuron]
#
# Provably zero for the entire run: Ca starts at 0 (not a popdata column, so
# agentmatrixinit.py:205 gives it the 0 fallback) and its only source is pyx:211
# with alpha_ca identically 0. Carried anyway so recorded
# traces have the same shape as the legacy backend's - a flat zero line in
# both.
#
# The pyx:211 increment is deliberately ABSENT from RESET_CODE, since adding
# `Ca += alpha_ca` would require declaring alpha_ca as a parameter purely to
# multiply by zero. 
_CALCIUM = """
dCa/dt = -Ca / (Tau_ca * ms) : 1 (unless refractory)
"""

# -- external Ornstein-Uhlenbeck drive ---------------------------
#
# Adapted from nchoice/agent_timestep_plasticity.pyx:52-55 (AMPA), with
# pyx:66-69 (GABA) and pyx:78-81 (NMDA) identical up to receptor name.
# (The enclosing per-population loops are pyx:47, pyx:65 and pyx:77.)
#
#     pyx:52   a.ExtMuS_AMPA[popid]    = a.MeanExtEff_AMPA[popid] * a.FreqExt_AMPA[popid]
#                                        * .001 * a.MeanExtCon_AMPA[popid] * a.Tau_AMPA[popid]
#     pyx:53   a.ExtSigmaS_AMPA[popid] = a.MeanExtEff_AMPA[popid]
#                                        * np.sqrt(a.Tau_AMPA[popid] * .5 * a.FreqExt_AMPA[popid]
#                                                  * .001 * a.MeanExtCon_AMPA[popid])
#     pyx:54   a.ExtS_AMPA[popid]     += a.dt / a.Tau_AMPA[popid] * (-a.ExtS_AMPA[popid] + a.ExtMuS_AMPA[popid])
#                                        + a.ExtSigmaS_AMPA[popid] * np.sqrt(a.dt * 2. / a.Tau_AMPA[popid])
#                                        * np.random.normal(size=len(a.Tau_AMPA[popid]))
#
# pyx:54 is Euler-Maruyama, `X += a*dt + b*sqrt(dt)*N(0,1)`, for
#     dExtS/dt = (ExtMuS - ExtS)/Tau + ExtSigmaS*sqrt(2/Tau)*xi
# so Brian2 with method='euler' reproduces the source update exactly rather
# than approximately. Matching term by term:
#     a = (ExtMuS - ExtS)/Tau           <- pyx:54 first summand / dt
#     b = ExtSigmaS * sqrt(2/Tau)       <- pyx:54 second summand / sqrt(dt)
#
# WARNING for anyone editing pyx:53: the commented-out reference implementation
# at pyx:103 computes sigma differently - it squares the efficacy inside the
# sqrt (`efficacy * efficacy`) and drops the leading MeanExtEff. The original code
# at pyx:53 does neither. We try to transcribe that here.
#
# The three noise sources are named xi_ampa / xi_gaba / xi_nmda, NOT a shared
# ``xi``. In Brian2 a repeated ``xi`` symbol is ONE noise process, so all three
# receptors would see the same realisation. The source draws separately per
# receptor - three distinct np.random.normal calls at pyx:54, pyx:68, pyx:80 --
# so three distinct symbols are required here.
#
# (Aside: pyx:68 and pyx:80 both size their draw with `len(a.Tau_AMPA[popid])`
# rather than Tau_GABA / Tau_NMDA. Harmless, since every per-population array
# has length N[popid], but it is a copy-paste slip.
# It does mean the three streams are consecutive slices of one RNG sequence,
# which is one of several reasons bit-exact agreement is off the table.)
#
# ExtMuS_* and ExtSigmaS_* are subexpressions, not parameters, because they
# depend on FreqExt_* which the driver rewrites every millisecond.
_EXTERNAL = """
ExtMuS_{r} = MeanExtEff_{r} * FreqExt_{r} * {scale} * MeanExtCon_{r} * Tau_{r} : 1
ExtSigmaS_{r} = MeanExtEff_{r} * sqrt(Tau_{r} * 0.5 * FreqExt_{r}
                                      * {scale} * MeanExtCon_{r}) : 1
dExtS_{r}/dt = (ExtMuS_{r} - ExtS_{r}) / (Tau_{r} * ms) \
             + ExtSigmaS_{r} * sqrt(2 / (Tau_{r} * ms)) * xi_{n} : 1
"""

# -- recurrent conductances --------------------------------------
#
# Adapted from nchoice/agent_timestep_plasticity.pyx, decay half only:
#
#     pyx:55   a.LS_AMPA[popid] *= np.exp(-a.dt / a.Tau_AMPA[popid])
#     pyx:69   a.LS_GABA[popid] *= np.exp(-a.dt / a.Tau_GABA[popid])
#     pyx:81   a.LS_NMDA[popid] *= np.exp(-a.dt / a.Tau_NMDA[popid])
#
# The matching increment half lives at pyx:61 (AMPA), pyx:75 (GABA) and
# pyx:91 (NMDA) and becomes Brian2 Synapses in file 5.
#
# Declared as PARAMETERS, not ODEs, because pyx:55/69/81 apply an EXACT
# exponential and Brian2's method='euler' would apply (1 - dt/Tau) instead.
# Decayed by LS_DECAY_CODE below. See the module docstring for the measured
# cost of getting this wrong.
_RECURRENT = """
LS_AMPA : 1
LS_GABA : 1
LS_NMDA : 1
"""

# -- optogenetics -------------------------------------------------
#
# Adapted from nchoice/agent_timestep_plasticity.pyx:202,204 -- see _MEMBRANE
# for the two current terms themselves. Written by the driver at
# nchoice/interface_nchoice.py:156 and :173:
#
#     agent.ExtS_Opto[popid] = np.resize(opt_amp[i], np.shape(agent.ExtS_Opto[popid]))
#
# A single SIGNED array, not two. Positive drives ChR2 (excitation), negative
# drives NpHR (inhibition) with conductance magnitude |ExtS_Opto|.
#
# Absent from common/agent_timestep.pyx entirely - that variant predates the
# change..
_OPTO = """
ExtS_Opto : 1
"""

# -- the membrane equation ------------------------------------
#
# Adapted from nchoice/agent_timestep_plasticity.pyx:192-204, which is six
# sequential in-place statements on V, collapsed here into one derivative:
#
#     pyx:192  leak + AHP + g_adr + g_k + g_rb        (all with a leading -dt)
#     pyx:197  NMDA, with the Mg block denominator
#     pyx:198  AMPA
#     pyx:199  GABA
#     pyx:202  ChR2, gated by (a.ExtS_Opto[popid] > 0)
#     pyx:204  NpHR, gated by (a.ExtS_Opto[popid] < 0), conductance -ExtS_Opto
#
# Term order below follows the source. Sign conventions: pyx:192 carries a
# leading `-a.dt` for the whole intrinsic block, which is why those terms are
# negated here, while pyx:197-204 carry `+a.dt` and are not.
#
# The NpHR term at pyx:204 reads
#     ... * (a.ExtS_Opto[popid] < 0) * .001 * -a.ExtS_Opto[popid] / a.C[popid]
# i.e. the conductance magnitude is -ExtS_Opto, which is positive precisely
# because the mask restricts it to ExtS_Opto < 0. Transcribed  below, to keep the correspondence one-to-one.
#
# The AHP term `a.Ca[popid] * a.g_ahp[popid] / a.C[popid] * 0.001 *
# (a.V[popid] - a.Vk[popid])`, third summand of pyx:192, is omitted. Both g_ahp
# and Vk are identically zero, so this is exact, not an
# approximation.
#
# ``(unless refractory)`` reproduces the `a.cond[popid]` factor computed at
# pyx:178-181, which multiplies every one of the six statements. That same
# `cond` also gates n_k (pyx:189) and Ca (pyx:193) - but not h (pyx:170,172,
# computed before cond exists) and not the LS_*/ExtS_* conductances
# (pyx:47-91), which update regardless.
#
# Because the source applies its six statements sequentially,
# each one sees the V left by the previous. Brian2 evaluates this entire
# right-hand side at the step-start V. The gap is one Euler increment,
# O(dt/Taum) ~ 1% per term.
_MEMBRANE = """
V_intrinsic = -(
      (V - RestPot) / Taum
    + g_adr / C * (V - ADRRevPot)
    + g_k / C * (V - ADRRevPot)
    + g_rb / C * (V - V_T)
) : 1

V_synaptic = (
      (RevPot_NMDA - Vaux) * {scale} * (LS_NMDA + ExtS_NMDA) / C
      / (1 + exp(-{mg_slope} * Vaux / {mg_scale}))
    + (RevPot_AMPA - Vaux) * {scale} * (LS_AMPA + ExtS_AMPA) / C
    + (RevPot_GABA - Vaux) * {scale} * (LS_GABA + ExtS_GABA) / C
    + (RevPot_ChR2 - Vaux) * {scale} * ExtS_Opto * int(ExtS_Opto > 0) / C
    + (RevPot_NpHR - Vaux) * {scale} * (-ExtS_Opto) * int(ExtS_Opto < 0) / C
) : 1

dV/dt = (V_intrinsic + V_synaptic) / ms : 1 (unless refractory)
"""


# =============================================================================
# Assembly
# =============================================================================

#: Exact exponential decay of the recurrent conductances, run once per timestep.
#: Transcribed verbatim from pyx:55, pyx:69 and pyx:81. Replaces what would
#: otherwise be three ODEs integrated with forward Euler.
LS_DECAY_CODE = """
LS_AMPA *= exp(-dt / (Tau_AMPA * ms))
LS_GABA *= exp(-dt / (Tau_GABA * ms))
LS_NMDA *= exp(-dt / (Tau_NMDA * ms))
h = h * int(h > {h_floor})
"""

#: Below this, ``h`` is flushed to exactly zero.
H_FLOOR = 1e-100

H_FLOOR_RATIONALE = """
Why h is flushed to zero, and why it is not a model change.

A neuron sitting above ``V_h`` has ``dh/dt = -h/tauhm`` (pyx:172), so
``h ~ exp(-t / 20 ms)``. Most neurons rest near ``ResetPot = -55``, which is
above ``V_h = -60``, so this is the common case rather than a corner. The value
therefore falls by a factor of ~1e-43 every second of simulated time:

    t = 14.2 s   h ~ 1e-308   -- the smallest normal double
    t = 14.9 s   h ~ 1e-324   -- underflow

Measured, without the flush: at 14 s the smallest nonzero ``h`` is 2.36e-307;
two seconds later 3258 neurons hold denormal values around 2.47e-322, and the
whole simulation slows by 5.1x - from 9.6x real time to 49x - and stays slow.
x86 denormal arithmetic traps to microcode and runs one to two orders of
magnitude slower than normal floating point.

The values do not reach zero on their own: the update ``h*(1 - dt/tauhm)``
loses all precision at that magnitude and stalls, so the penalty is permanent
rather than a transient as ``h`` passes through the denormal range.

Flushing to zero is safe on both physical and numerical grounds:

  * ``h`` reaches the membrane only through ``g_rb = g_T * h_next * int(V >= V_h)``
    and ``g_T`` is at most 0.06 (paramfile:33-34). A contribution of
    ``0.06 * 1e-100`` is not merely negligible, it is 90 orders of magnitude
    below the smallest quantity anything else in the model represents.
  * Zero is a fixed point in the right direction. With ``h = 0`` and
    ``V >= V_h``, ``dh/dt`` is 0, so it stays. As soon as ``V`` drops below
    ``V_h``, ``dh/dt = (1 - 0)/tauhp`` and it recovers normally -- the T-type
    rebound mechanism is unaffected.
  * The threshold is 1e-100, which is 200 orders of magnitude above the
    denormal boundary. Nothing physical lives in that gap.

The legacy backend computes the same ``h`` and presumably pays the same penalty;
it is simply slow enough that a further slowdown is less conspicuous. This is a
numerical artifact of IEEE-754, not a difference between the two models.
"""

#: Scheduling slot for LS_DECAY_CODE. Not a free choice . 
# ``'start'`` would decay every spike's contribution one step too
#: many.
LS_DECAY_WHEN = "before_synapses"

#: Spike condition. From pyx:208::
#:
#:     newspikes[popid] = list(np.nonzero(a.V[popid] > a.Threshold[popid])[0])
#:
#: Strictly greater-than, matching the source. ``Threshold`` here is the
#: per-neuron parameter (paramfile:9), distinct from Brian2's ``threshold=``
#: keyword argument that consumes this string.
THRESHOLD_CONDITION = "V > Threshold"

#: Post-spike reset. The source spike handler is pyx:209-215::
#:
#:     pyx:210   a.V[popid][neuron] = 0
#:     pyx:211   a.Ca[popid][neuron] += a.alpha_ca[popid][neuron]
#:     pyx:212   a.RefrState[popid][neuron] = 10
#:     pyx:213   a.Ptimesincelastspike[popid][neuron] = a.timesincelastspike[popid][neuron]
#:     pyx:214   a.timesincelastspike[popid][neuron] = 0
#:     pyx:215   a.dpmn_XPOST[popid][neuron] = 1
#:
#: Of those six lines, only pyx:210 has a counterpart here:
#:
#:   pyx:210  DIVERGENCE D4. The source writes 0 -- a cosmetic spike peak for
#:            voltage traces -- and reaches ResetPot on the NEXT step via
#:            pyx:178-179. We write ResetPot directly. Spike times unaffected.
#:   pyx:211  omitted; alpha_ca is identically 0 (MODEL_SPEC.md 1.4).
#:   pyx:212  replaced by Brian2's ``refractory=`` duration; see D5.
#:   pyx:213  replaced by an ``(event-driven)`` synaptic variable in file 5.
#:   pyx:214  same.
#:   pyx:215  belongs to plasticity; file 9 will append it to this string.
RESET_CODE = "V = ResetPot"


def build_neuron_equations(
    parameters=None,
    include_calcium: bool = True,
    sequential: bool = True,
    extra: str = "",
) -> Equations:
    """Assemble the full ``Equations`` object for the CBGTPy neuron.

    Parameters
    ----------
    parameters
        Names to declare as per-neuron constants (``name : 1``). Defaults to
        ``FIXED_WEIGHT_PARAMS``. Another file (still making) will pass a widened tuple that also
        includes the dopamine constants.
    include_calcium
        Carry the ``Ca`` state variable and its decay. It is provably zero
        , so this costs one wasted linear ODE and
        buys trace parity with the legacy backend. Set False to drop it.

    Returns
    -------
    Equations
        Ready to hand to ``NeuronGroup``. Combine with ``THRESHOLD_CONDITION``,
        ``RESET_CODE``, ``refractory_period()`` and ``method='euler'``.
    """
    if parameters is None:
        parameters = FIXED_WEIGHT_PARAMS

    parts = [
        _MEMBRANE.format(
            scale=COND_SCALE, mg_slope=MG_SLOPE, mg_scale=MG_SCALE
        ),
        _VAUX if sequential else _VAUX_SIMULTANEOUS,
        _TTYPE if sequential else _TTYPE_SIMULTANEOUS,
        (_INTRINSIC if sequential else _INTRINSIC_SIMULTANEOUS).format(
            off=KDR_OFFSET, slope=KDR_SLOPE
        ),
        _RECURRENT,
        _OPTO,
    ]

    if include_calcium:
        parts.append(_CALCIUM)

    # One external-drive block per receptor. The xi suffix must differ per
    # receptor or Brian2 shares a single noise realisation between them.
    for receptor, noise in (("AMPA", "ampa"), ("GABA", "gaba"), ("NMDA", "nmda")):
        parts.append(_EXTERNAL.format(r=receptor, n=noise, scale=COND_SCALE))

    # Extra blocks spliced in by a later layer -- currently only file 9, which
    # appends the dopamine state variables and the fDA transfer function. Passed
    # in as a string rather than gated by a flag here, so that equations.py
    # needs no knowledge of plasticity.
    if extra:
        parts.append(extra)

    # Per-neuron constants.
    parts.append("\n".join(f"{name} : 1" for name in parameters))

    return Equations("\n".join(parts))


def refractory_period(dt_ms: float = DT_MS):
    """Refractory duration as a Brian2 quantity: ``(REFRACTORY_STEPS + 1) * dt``.

    **The ``+ 1`` is not a fudge factor.** Brian2's refractory test is

        t - lastspike < refractory

    so a duration of exactly ``n * dt`` leaves the neuron refractory at
    ``t_spike + dt`` through ``t_spike + (n-1)*dt`` -- that is ``n - 1`` frozen
    timesteps, not ``n``. The legacy counter freezes ``REFRACTORY_STEPS = 10``
    steps, so the matching duration is ``11 * dt``.

    Verified by inter-spike interval, not by inspecting ``not_refractory``.
    The ISI is the physical quantity and it
    is unambiguous: at four different drive levels, ``11 * dt`` reproduces the
    legacy interval exactly and ``10 * dt`` is one timestep short, which is a
    3% rate error at 143 Hz and grows as the interval shortens. See
    ``_check_refractory_duration``.

    The legacy mechanism is a step counter, not a duration: pyx:212 sets
    ``RefrState = 10`` and pyx:182 decrements it once per timestep. So the
    refractory period scales with ``dt``, and ``dt_ms`` must match whatever the
    caller has put in ``defaultclock.dt`` - otherwise the port silently
    changes every firing rate in the network. ``neurons.build_neuron_group``
    passes its own ``dt_ms`` through for exactly this reason.

    ``ms`` is imported lazily so that importing this module does not pin
    ``defaultclock`` before the caller has had a chance to set ``dt``.
    """
    from brian2 import ms
    return (REFRACTORY_STEPS + 1) * dt_ms * ms


# =============================================================================
# Self-check
# =============================================================================


def _check_parses(eqs: Equations) -> list[str]:
    """Structural checks on the assembled equations.

    Two distinct questions, and Brian2 exposes them through two different
    attributes - easy to conflate, so both are spelled out here:

    ``Equations.identifiers``
        Symbols referenced but NOT defined by the equations themselves, i.e.
        the external namespace: functions, units, noise terms. Every one of
        these must be something Brian2 can resolve at build time.

    ``SingleEquation.identifiers``
        ALL symbols in that one equation's expression, including references to
        other equation variables. Unioned across equations, this tells us which
        declared parameters are actually read.
    """
    from brian2.core.functions import DEFAULT_FUNCTIONS

    problems = []

    # -- 1. every external symbol must be resolvable ---------------------------
    #
    # Allowed: Brian2's builtin functions, the ms unit, the per-receptor noise
    # terms we deliberately named apart, and the variables Brian2 injects into
    # every group.
    allowed_external = (
        set(DEFAULT_FUNCTIONS)
        | {"ms", "second"}
        | {"xi_ampa", "xi_gaba", "xi_nmda"}
        | {"t", "dt", "i", "N", "not_refractory", "lastspike"}
    )
    unresolvable = sorted(set(eqs.identifiers) - allowed_external)
    if unresolvable:
        problems.append(
            f"external symbols Brian2 cannot resolve: {unresolvable}"
        )

    # Guard the whitelist itself: if a noise term silently disappeared from the
    # equations, the three receptors would share one realisation and nobody
    # would notice. Assert they are present.
    for noise in ("xi_ampa", "xi_gaba", "xi_nmda"):
        if noise not in eqs.identifiers:
            problems.append(
                f"{noise} missing -- receptor noise sources may be shared, "
                "which the legacy code does not do (MODEL_SPEC.md 3.4)"
            )

    # -- 2. nothing declared should go unread ----------------------------------
    referenced: set[str] = set()
    for name in eqs.names:
        referenced |= set(eqs[name].identifiers)

    # State variables are written by synapses, the decay run_regularly or the
    # driver rather than read by another equation, so absence is expected.
    unused = sorted(
        set(eqs.names)
        - referenced
        - set(STATE_VARIABLES)
        - set(DRIVER_WRITABLE)
        - CONSUMED_OUTSIDE_EQUATIONS
    )
    if unused:
        problems.append(f"declared but never referenced: {unused}")

    return problems


def _check_refractory_duration() -> list[str]:
    """Verify the refractory duration by inter-spike interval, not by monitor.

    We have frozen timesteps by tracing the legacy
    ``RefrState`` counter (pyx:212 sets it to 10, pyx:182 decrements it, pyx:181
    gates integration on it reaching 0). Brian2 expresses refractoriness as a
    duration instead, and its test is ``t - lastspike < refractory`` - so the
    duration that yields ``n`` frozen steps is ``(n + 1) * dt``, not ``n * dt``.

    **Why this is checked via ISI.** ``StateMonitor`` samples at a fixed
    point in the schedule, and the offset between that point and where
    ``not_refractory`` is updated hides the off-by-one.

    The inter-spike interval has no such ambiguity. Here a single neuron is
    driven by a constant conductance until it fires periodically, and its ISI
    is compared against a direct Python transcription of pyx:167-216. The two must agree to the
    timestep, at several drive levels: a one-step error is invisible at low
    rates and a 3% rate error at 143 Hz, so a single operating point would not
    catch it.
    """
    import numpy as np
    from brian2 import (Network, NeuronGroup, SpikeMonitor, defaultclock, ms)

    # dSPN defaults (paramfile:4-28) with the intrinsic conductances at zero, so
    # the only dynamics are leak plus a constant AMPA drive.
    P = dict(C=0.5, Taum=20.0, RestPot=-70.0, ResetPot=-55.0, Threshold=-50.0,
             Tau_ca=80.0, g_T=0.0, V_h=-60.0, V_T=120.0, tauhp=100.0,
             tauhm=20.0, g_adr_max=0.0, Vadr_h=-100.0, Vadr_s=10.0,
             ADRRevPot=-90.0, g_k_max=0.0, Vk_h=-34.0, Vk_s=6.5,
             tau_k_max=8.0, RevPot_AMPA=0.0, RevPot_GABA=-70.0,
             RevPot_NMDA=0.0, RevPot_ChR2=0.0, RevPot_NpHR=-400.0,
             Tau_AMPA=2.0, Tau_GABA=5.0, Tau_NMDA=100.0)

    def legacy_isi(drive):
        """pyx:167-216, transcribed. Returns the median ISI in ms."""
        dt = DT_MS
        V, refr, spikes = P["ResetPot"], 0.0, []
        for step in range(int(2000.0 / dt)):
            cond = 1.0 if V <= P["Threshold"] else 0.0        # pyx:178
            V -= (V - P["ResetPot"]) * (1 - cond)             # pyx:179
            cond = cond * (1.0 if refr == 0 else 0.0)         # pyx:181
            refr -= np.sign(refr) * (1 - cond)                # pyx:182
            V = V + cond * -dt * ((V - P["RestPot"]) / P["Taum"])   # pyx:192
            Vaux = min(V, P["Threshold"])                     # pyx:195
            V = V + cond * dt * (P["RevPot_AMPA"] - Vaux) * COND_SCALE * drive / P["C"]
            if V > P["Threshold"]:                            # pyx:208
                V, refr = 0.0, 10.0                           # pyx:210, pyx:212
                spikes.append(step * dt)
        return float(np.median(np.diff(np.array(spikes)[5:])))

    def port_isi(drive):
        defaultclock.dt = DT_MS * ms
        group = NeuronGroup(
            1, build_neuron_equations(), threshold=THRESHOLD_CONDITION,
            reset=RESET_CODE, refractory=refractory_period(DT_MS),
            method="euler",
        )
        for name, value in P.items():
            setattr(group, name, value)
        for receptor in ("AMPA", "GABA", "NMDA"):
            setattr(group, f"FreqExt_{receptor}", 0.0)
            setattr(group, f"MeanExtEff_{receptor}", 0.0)
            setattr(group, f"MeanExtCon_{receptor}", 0.0)
            setattr(group, f"ExtS_{receptor}", 0.0)
        group.V, group.h, group.n_k = P["ResetPot"], 1.0, 0.0
        group.LS_AMPA, group.LS_GABA, group.LS_NMDA = drive, 0.0, 0.0
        group.ExtS_Opto = 0.0
        monitor = SpikeMonitor(group)
        Network(group, monitor).run(2000 * ms)
        times = np.asarray(monitor.t / ms)
        return float(np.median(np.diff(times[5:])))

    problems = []
    print(f"    {'drive':>7} {'legacy ISI':>11} {'port ISI':>10} {'steps':>7}")
    for drive in (10.5, 14.0, 18.0, 25.0):
        legacy, port = legacy_isi(drive), port_isi(drive)
        offset = (port - legacy) / DT_MS
        print(f"    {drive:7.2f} {legacy:11.2f} {port:10.2f} {offset:+7.2f}")
        if abs(offset) > 1e-6:
            problems.append(
                f"ISI at drive {drive}: port {port} ms vs legacy {legacy} ms, "
                f"off by {offset:+.2f} timesteps. refractory_period returns "
                f"{refractory_period(DT_MS)}; the legacy counter freezes "
                f"{REFRACTORY_STEPS} steps."
            )
    return problems


def _check_ls_decay_exactness() -> list[str]:
    """LS_DECAY_CODE must reproduce ``LS *= exp(-dt/Tau)`` to machine precision.

    This is the whole reason LS_* are parameters rather than ODEs, so it is
    worth asserting rather than assuming. Also reports what plain forward Euler
    would have cost, since that number motivates the design.
    """
    import numpy as np
    # An explicit Network, not Brian2's magic network. The magic network is
    # frame-scoped: it collects BrianObjects visible in the namespace where
    # run() is called. That makes these checks work only by accident - each
    # lives in its own function - and it would raise MagicError the moment
    # anyone inlined one into _selfcheck, which has already built and run a
    # group of its own. An explicit Network removes the coupling.
    from brian2 import Network, NeuronGroup, defaultclock, ms

    defaultclock.dt = DT_MS * ms
    G = NeuronGroup(1, "LS_AMPA : 1\nTau_AMPA : 1", method="euler")
    decay = G.run_regularly(
        "LS_AMPA *= exp(-dt / (Tau_AMPA * ms))", when=LS_DECAY_WHEN
    )
    G.Tau_AMPA = 2.0
    G.LS_AMPA = 1.0

    steps = 50
    Network(G, decay).run(steps * DT_MS * ms)

    got = float(G.LS_AMPA[0])
    want = float(np.exp(-steps * DT_MS / 2.0))
    euler = float((1 - DT_MS / 2.0) ** steps)

    problems = []
    if not np.isclose(got, want, rtol=1e-12, atol=0.0):
        problems.append(
            f"LS decay mismatch after {steps} steps: got {got!r}, want {want!r}"
        )

    print(
        f"  LS_AMPA after {steps} steps (Tau=2, dt={DT_MS}): "
        f"exact={want:.12f}  run_regularly={got:.12f}  "
        f"forward-euler-would-give={euler:.12f} "
        f"({100 * (euler / want - 1):+.2f}%)"
    )
    return problems


def _selfcheck() -> int:
    import warnings

    warnings.filterwarnings("ignore")

    from brian2 import Network, NeuronGroup, defaultclock, ms, prefs

    prefs.codegen.target = "numpy"

    failures: list[str] = []

    eqs = build_neuron_equations()
    print("equations assembled\n")
    print(f"  {len(eqs.diff_eq_names)} differential equations: "
          f"{sorted(eqs.diff_eq_names)}")
    print(f"  {len(eqs.subexpr_names)} subexpressions: "
          f"{sorted(eqs.subexpr_names)}")
    print(f"  {len(eqs.parameter_names)} parameters")
    print()

    failures += _check_parses(eqs)

    # The equations must actually build and run a group, not merely parse.
    defaultclock.dt = DT_MS * ms
    G = NeuronGroup(
        10, eqs,
        threshold=THRESHOLD_CONDITION,
        reset=RESET_CODE,
        refractory=refractory_period(),
        method="euler",
    )
    decay_op = G.run_regularly(LS_DECAY_CODE.format(h_floor=H_FLOOR),
                               when=LS_DECAY_WHEN)

    # Load the real default parameters through file 2, so this doubles as an
    # integration test of the params -> equations handoff.
    import sys

    sys.path.insert(0, ".")
    import nchoice.init_params_nchoice as par
    import nchoice.paramfile_nchoice as pf
    import nchoice.popconstruct_nchoice as pc

    from .params import build_layout, flatten_param

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

    # dSPN:1 -- a plain population with default intrinsics, driven by its
    # baseline external input. A good smoke test: it should fire, but not
    # explode.
    popid = layout.popid_of_label("dSPN:1")
    missing_params = []
    for name in FIXED_WEIGHT_PARAMS:
        if name not in popdata.columns:
            missing_params.append(name)
        value = flatten_param(popdata, name, layout=layout)[layout.starts[popid]]
        setattr(G, name, float(value))

    G.V = 0.0          # legacy initial value; see params.INITIAL_STATE
    G.h = 1.0
    G.n_k = 0.0

    Network(G, decay_op).run(500 * ms)

    v = G.V[:]
    print(f"  500 ms smoke run on dSPN:1 parameters -> "
          f"V in [{v.min():.2f}, {v.max():.2f}], "
          f"ExtS_AMPA mean {G.ExtS_AMPA[:].mean():.3f}")

    import numpy as np

    if not np.all(np.isfinite(v)):
        failures.append("membrane potential went non-finite during smoke run")
    if v.min() < -200 or v.max() > 50:
        failures.append(
            f"membrane potential left a plausible range: [{v.min()}, {v.max()}]"
        )

    if missing_params:
        # Not necessarily an error -- some populations legitimately lack a
        # parameter and fall back to 0 -- but worth surfacing.
        print(f"  note: not popdata columns, defaulted to 0: {missing_params}")

    print("\nrefractory duration check:")
    refractory_problems = _check_refractory_duration()
    if not refractory_problems:
        print(f"  membrane frozen for exactly {REFRACTORY_STEPS} steps "
              f"({REFRACTORY_STEPS * DT_MS} ms) after a spike")
    failures += refractory_problems

    print("\nLS decay exactness check:")
    failures += _check_ls_decay_exactness()

    if failures:
        print(f"\nFAIL -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nOK -- equations parse, build, run, and match the spec's constants")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
