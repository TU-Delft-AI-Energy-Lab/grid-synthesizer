"""
powergrid_synth.privacy
=======================

Modular perturbation of every value PowerGridSynth derives from a reference
network, across both the transmission and distribution pipelines.

The operator sets one strength between 0.0 (maximum realism, nothing
perturbed) and higher values (progressively harder to reverse-engineer), plus
per-parameter overrides, in a single TOML file.  Each parameter can be:

``off``
    Never read from the reference at all; the pipeline uses its public
    built-in value instead.  The strongest protection available, because a
    value that is never read cannot leak.
``perturb``
    Read from the reference and displaced by Laplace noise.
``raw``
    Read from the reference and used unchanged.

Sub-modules
-----------
registry  — declarative descriptor for every reference-derived parameter
engine    — shape-dispatched perturbation, with data-independent noise scaling
settings  — TOML loading, validation and mode/strength resolution
report    — per-run record of what was protected and what was not
noise     — transform pairs and Laplace draws
rng       — stream-keyed generators with a once-per-run guarantee
recovery  — adversarial recovery estimators used by the evaluation notebook

Note on terminology
-------------------
This package does not claim a formal differential-privacy guarantee.  Noise
scale is ``strength * k`` with ``k`` declared per parameter and independent of
the data; protection is demonstrated by measured recovery floors rather than
by a proof.  See the modular perturbation layer spec for the reasoning.
"""

from .noise import (
    Transform,
    identity,
    log,
    logit,
    affine_logit,
    perturb,
    perturb_integer,
)
from .rng import make_rng, resolve_seed
from .registry import (
    REGISTRY,
    Domain,
    FieldSpec,
    Mode,
    ParameterDescriptor,
    Shape,
    descriptors_for_domain,
    get_descriptor,
)
from .settings import (
    PerturbationConfigError,
    PerturbationSettings,
    ParameterSetting,
    coerce,
    load,
    resolve,
)
from .engine import (
    measure_inflation_bias,
    noise_scale_for,
    perturb_by_shape,
    perturb_degree_sequence,
    perturb_graph_node_count,
    perturb_integer_value,
    perturb_probability_vector,
    perturb_scalar_family,
)
from .report import ParameterRecord, PerturbationReport, build_report
from .recovery import (
    recover_node_count,
    recover_degree_dist,
    recover_degree_dist_full,
)

__all__ = [
    # noise primitives
    "Transform",
    "identity",
    "log",
    "logit",
    "affine_logit",
    "perturb",
    "perturb_integer",
    # rng
    "make_rng",
    "resolve_seed",
    # registry
    "REGISTRY",
    "Domain",
    "FieldSpec",
    "Mode",
    "ParameterDescriptor",
    "Shape",
    "descriptors_for_domain",
    "get_descriptor",
    # settings
    "PerturbationConfigError",
    "PerturbationSettings",
    "ParameterSetting",
    "coerce",
    "load",
    "resolve",
    # engine
    "measure_inflation_bias",
    "noise_scale_for",
    "perturb_by_shape",
    "perturb_degree_sequence",
    "perturb_graph_node_count",
    "perturb_integer_value",
    "perturb_probability_vector",
    "perturb_scalar_family",
    # report
    "ParameterRecord",
    "PerturbationReport",
    "build_report",
    # recovery
    "recover_node_count",
    "recover_degree_dist",
    "recover_degree_dist_full",
]
