##################################
  Protecting the reference grid
##################################

In reference mode the synthesiser reads statistics from a real network. When
that network is confidential, the question is what an observer of the published
synthetic grids can infer about it. The perturbation layer
(:mod:`powergrid_synth.privacy`) exists to answer that question and to bound
the answer.

The threat model
================

Both adversaries considered hold the synthetic grids *and* the source code,
which is open; no protection depends on the method staying secret.

A **single-sample adversary** observes one grid, against which the
synthesiser's own randomness is already a poor estimator. A **pooling
adversary** observes many grids from a common reference and combines them.
Repeated sampling averages out that randomness, so any characteristic fixed
across runs becomes estimable. Deliberate perturbation sets a floor pooling
cannot cross, because the displacement is identical in every sample.

All protection claims are made against the pooling adversary, on the reasoning
that an adversary holding one grid is already poorly served, while an adversary
holding many is the realistic case once synthetic grids are published.

Three modes per parameter
=========================

Every reference-derived input is registered in
:data:`powergrid_synth.privacy.registry.REGISTRY` with the modes it accepts:

``off``
    The reference value never reaches the output. Where the quantity has its
    own extraction routine, that routine is not called at all; where it is a
    by-product of loading the reference network, it is discarded before the
    pipeline sees it. The pipeline uses a public built-in value instead.

``perturb``
    The reference value is read and displaced by calibrated noise before use.

``raw``
    The reference value is read and used unchanged.

Not every parameter offers all three. Structural inputs with no public
substitute have no ``off``: a reference-shaped topology cannot be generated
while refusing to read the reference topology. Requesting an unavailable mode
raises an error naming what is available.

**Every parameter defaults to** ``raw``. Passing no configuration reproduces
the unprotected behaviour exactly, so adding the argument to an existing call
changes nothing until you configure it.

How the noise is applied
========================

Every perturbed quantity goes through the same four steps, whatever it is. Let
:math:`\theta` be the value read from the reference, :math:`s \ge 0` the
configured strength, and :math:`k > 0` the constant declared for that
parameter.

1. **Move to a space where noise is safe.** A transform :math:`T` maps
   :math:`\theta` from its natural domain onto the whole real line. A
   probability goes through the logit, a positive quantity through the log, a
   bounded interval through an affine logit, an unbounded one through the
   identity.

2. **Draw a displacement.** :math:`\eta \sim \mathrm{Laplace}(0, b)` with
   :math:`b = s\,k`.

3. **Invert.** :math:`\theta' = T^{-1}(T(\theta) + \eta)`, which lands back in
   the original domain by construction — no clipping, and no mass piling up at
   a boundary where an observer could detect it.

4. **Clamp** where a floor or a bound is declared.

The scale is :math:`s\,k` and never a function of :math:`\theta`. Two
consequences follow:

- the same configuration behaves identically on every reference network,
  whatever its size or number of voltage levels — a scale derived from
  :math:`\theta` would leak information about :math:`\theta` through the noise
  meant to hide it;
- registering an additional parameter never weakens the existing ones, because
  the strength is applied per parameter rather than divided among them.

Setting :math:`s = 0` gives :math:`b = 0`, so zero strength is exactly
equivalent to not perturbing at all.

Guarantees and limits
=====================

**Noise is drawn once per run.** All synthetic grids produced in one run share
the same displaced parameters, so pooling within a release cannot average the
noise away.

.. warning::

   This is sound only if you publish **one batch per reference network**. If
   you publish several batches from the same reference with fresh entropy,
   independent draws average out across batches and an adversary pooling across
   them recovers what pooling within one batch cannot. Fix the seed so every
   batch carries the identical offset.

**Structural zeros are preserved.** Where the reference exhibits none of
something, that zero is kept exactly rather than perturbed; noise there would
invent behaviour the reference does not have. The consequence is a genuine
disclosure — preserved zeros identify precisely which bins are empty.

**Topology perturbation can fall back.** A perturbed connection-count pattern
must still describe a network that can physically exist. If fifty attempts fail
to produce one, the unperturbed pattern is used for that voltage level and the
fallback is recorded in the run report.

**A displaced target is not a displaced output.** Some parameters are targets
handed to the topology generator rather than values written to the output, and
the generator is free to miss them. The network span is the clearest case: its
target is displaced as configured, but the realised span is governed more by
the degree sequence and connectivity filtering than by the requested diameter.
Measure the parameter you care about rather than assuming the mode setting is
sufficient.

**A single draw is not a guarantee.** Perturbation displaces the expectation
across hypothetical releases. It does not promise that the one release you
actually publish sits far from the truth: a draw can land toward the reference
as easily as away from it, and at low strength it often does.

**No formal privacy guarantee is claimed.** Protection is evidenced by
measurement, not by a theorem.

**Every run produces an audit record** listing, per parameter, the mode, the
strength, the noise scale, the resolved seed, and whether any fallback
occurred. It is written next to the exported grid as
``<output_name>_perturbation_report.json`` and attached to the graph as
``graph.graph["perturbation_report"]``. This is the artefact to retain
alongside a release.

Measuring what is actually protected
====================================

``scripts/measure_topology_recovery.py`` simulates a pooling adversary against
the public IEEE-118 case and reports, per parameter and per strength, how close
the adversary gets. ``scripts/make_recovery_figure.py`` draws the result. Run
both against your own reference before relying on a configuration:

.. code-block:: bash

   python scripts/measure_topology_recovery.py --reference-case case118 --n-grids 30
   python scripts/make_recovery_figure.py

The estimators themselves live in :mod:`powergrid_synth.privacy.recovery` and
can be called directly on any pool of generated graphs.

What the measurements establish, on IEEE-118 with thirty grids per
configuration: buses per voltage level are displaced well beyond a pooling
adversary's uncertainty, but only on levels holding few buses --- the noise is
drawn in buses, so the same strength is proportionally negligible on a level
carrying most of the network. The connection-count pattern is displaced only
from strength 1.0 upward; at 0.5 the pooled degree distribution landed *closer*
to the reference than an unperturbed one. The network span is not displaced at
any tested strength, for the reason given above. Re-run the study on your own
reference rather than carrying these numbers over: they describe this
generator, on this case, for these seeds.

Registered parameters
=====================

**Transmission**

.. list-table::
   :widths: 30 22 48
   :header-rows: 1

   * - Parameter
     - Modes
     - What it is
   * - ``degrees_by_level``
     - perturb / raw
     - Connection-count pattern per voltage level, used as an empirical
       distribution from which a fresh sequence is drawn.
   * - ``node_count``
     - perturb / raw
     - Buses per voltage level. Noise is in buses, so the same strength is
       proportionally weaker on a level that holds many of them.
   * - ``diameters_by_level``
     - perturb / raw
     - Network span per level, in hops. A target, not an output value.
   * - ``transformer_degrees``
     - raw
     - Transformer connection pattern between levels. No perturbation yet.
   * - ``base_kv_map``
     - off / raw
     - Nominal voltage levels in kV. No ``perturb``: arbitrary noise would
       produce non-standard voltages.

**Distribution**

All nine fitted feeder parameters — ``dist.hop_dist``, ``dist.degree_dist``,
``dist.degree_clip``, ``dist.intermediate_frac``, ``dist.injection_frac``,
``dist.load_deviation``, ``dist.cable_length``, ``dist.length_clip`` — accept
all three modes. ``dist.input_model`` is restricted to ``off``: no allocator
reads it.

.. note::

   No adversary estimator exists for the distribution pipeline yet, so its
   protection is asserted rather than measured.

Configuration
=============

Pass ``perturbation_config`` to :func:`powergrid_synth.synthesize` or
:func:`powergrid_synth.synthesize_distribution`, as a path to a TOML file, a
mapping, or a :class:`~powergrid_synth.privacy.settings.PerturbationSettings`.

.. code-block:: toml

   strength = 0.4          # global default for every parameter
   seed = 42               # omit for fresh entropy each run

   [parameters.degrees_by_level]
   mode = "perturb"
   strength = 0.3          # per-parameter override

   [parameters.base_kv_map]
   mode = "off"

   # Names containing a dot must be quoted.
   [parameters."dist.cable_length"]
   mode = "perturb"
   strength = 0.6

``perturbation.example.toml`` in the repository root is a worked file covering
every registered parameter. The strengths in it are illustrative, not
recommendations: the right setting depends on the grid and on what the data
owner considers disclosable.

An unknown parameter name, a mode a parameter does not support, or a strength
outside ``[0, 10]`` is an error rather than a warning — a silently ignored key
would be a silent loss of protection.
