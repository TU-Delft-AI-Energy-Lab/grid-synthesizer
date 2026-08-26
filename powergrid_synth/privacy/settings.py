"""
settings — load, validate and resolve the perturbation configuration.

All perturbation behaviour is controlled from one TOML file.  This module
turns that file into a :class:`PerturbationSettings` object that the synthesis
pipelines consult, and rejects anything it cannot make sense of *before*
synthesis starts.

Failing loudly matters here more than usual.  A mistyped parameter name that
was silently ignored would leave the operator believing a value is protected
when it is not, so every unknown key, unsupported mode and out-of-range
strength is an error rather than a warning.

Configuration format::

    strength = 0.4          # global default for every parameter
    seed = 42               # omit for fresh entropy each run

    [parameters.degrees_by_level]
    mode = "perturb"
    strength = 1.0          # per-parameter override

    [parameters.base_kv_map]
    mode = "off"            # never read from the reference

Spec coverage: FR-02, FR-03, FR-06, FR-07, FR-08, FR-09, FR-22
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .registry import REGISTRY, Domain, Mode, ParameterDescriptor

# Strength is unbounded in principle, but a value this large is far more
# likely to be a typo than an intention.
MAX_STRENGTH = 10.0

# Used when neither the parameter nor the file names a strength.  Zero means a
# config that sets only modes perturbs nothing until a strength is chosen.
DEFAULT_STRENGTH = 0.0


class PerturbationConfigError(ValueError):
    """Raised when a perturbation configuration cannot be used as written."""


@dataclass(frozen=True)
class ParameterSetting:
    """Resolved mode and strength for one parameter.

    Args:
        mode: The resolved mode.
        strength: The resolved strength.  Always zero when *mode* is not
            ``PERTURB``, so downstream code can rely on it alone.
    """

    mode: Mode
    strength: float

    @property
    def should_extract(self) -> bool:
        """Whether the reference value should be read at all.

        ``False`` only for ``off``, where the pipeline skips extraction
        entirely and the downstream allocator uses its public fallback.
        """
        return self.mode is not Mode.OFF

    @property
    def should_perturb(self) -> bool:
        """Whether noise should be applied to the extracted value."""
        return self.mode is Mode.PERTURB and self.strength > 0.0


@dataclass
class PerturbationSettings:
    """Resolved configuration for every registered parameter.

    Args:
        global_strength: Strength applied to parameters with no override.
        seed: Top-level RNG seed, or ``None`` to draw fresh entropy per run.
        per_parameter: Resolved setting for every registered parameter name.
        source: Where the configuration came from, for the run report.
    """

    global_strength: float = DEFAULT_STRENGTH
    seed: Optional[int] = None
    per_parameter: Dict[str, ParameterSetting] = field(default_factory=dict)
    source: str = "defaults"

    def for_parameter(self, name: str) -> ParameterSetting:
        """Return the resolved setting for *name*.

        Args:
            name: Registered parameter name.

        Returns:
            The resolved :class:`ParameterSetting`.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self.per_parameter:
            raise KeyError(
                f"Unknown parameter {name!r}. Registered parameters: "
                f"{', '.join(sorted(REGISTRY))}."
            )
        return self.per_parameter[name]

    def mode_of(self, name: str) -> Mode:
        """Return the resolved mode for *name*."""
        return self.for_parameter(name).mode

    def strength_of(self, name: str) -> float:
        """Return the resolved strength for *name*."""
        return self.for_parameter(name).strength

    def should_extract(self, name: str) -> bool:
        """Whether *name* should be read from the reference network."""
        return self.for_parameter(name).should_extract

    def should_perturb(self, name: str) -> bool:
        """Whether *name* should have noise applied after extraction."""
        return self.for_parameter(name).should_perturb

    def active_for_domain(self, domain: Domain) -> Dict[str, ParameterSetting]:
        """Return resolved settings for one pipeline's parameters only (FR-13a).

        Args:
            domain: The pipeline whose parameters are wanted.

        Returns:
            Mapping of parameter name to setting, for that domain alone.
        """
        return {
            name: setting
            for name, setting in self.per_parameter.items()
            if REGISTRY[name].domain is domain
        }


def _validate_strength(value, context: str) -> float:
    """Return *value* as a float, rejecting non-numeric and out-of-range input.

    Args:
        value: The raw value from the configuration.
        context: Human-readable location, used in the error message.

    Returns:
        The validated strength.

    Raises:
        PerturbationConfigError: If not numeric or outside ``[0, MAX_STRENGTH]``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerturbationConfigError(
            f"{context}: strength must be a number, got {value!r}."
        )
    strength = float(value)
    if not 0.0 <= strength <= MAX_STRENGTH:
        raise PerturbationConfigError(
            f"{context}: strength must be between 0.0 and {MAX_STRENGTH}, "
            f"got {strength}."
        )
    return strength


def _validate_mode(raw_mode, descriptor: ParameterDescriptor) -> Mode:
    """Return the mode named by *raw_mode*, if the parameter supports it.

    Args:
        raw_mode: The raw value from the configuration.
        descriptor: Descriptor for the parameter being configured.

    Returns:
        The validated mode.

    Raises:
        PerturbationConfigError: If the name is not a mode, or the parameter
            does not support it (FR-08).
    """
    supported = sorted(mode.value for mode in descriptor.supported_modes)
    if not isinstance(raw_mode, str):
        raise PerturbationConfigError(
            f"Parameter {descriptor.name!r}: mode must be a string, got {raw_mode!r}. "
            f"Supported modes: {supported}."
        )
    try:
        mode = Mode(raw_mode)
    except ValueError:
        raise PerturbationConfigError(
            f"Parameter {descriptor.name!r}: {raw_mode!r} is not a mode. "
            f"Supported modes for this parameter: {supported}."
        ) from None
    if mode not in descriptor.supported_modes:
        raise PerturbationConfigError(
            f"Parameter {descriptor.name!r} does not support mode {mode.value!r}. "
            f"Supported modes: {supported}. "
            f"When off: {descriptor.fallback_note or 'no public fallback available'}."
        )
    return mode


def resolve(raw_config: Optional[dict] = None, *, source: str = "defaults") -> PerturbationSettings:
    """Resolve a parsed configuration into settings for every parameter.

    Resolution order for each parameter is: the value given for that
    parameter, else the file-level default, else the descriptor default.
    Every registered parameter appears in the result, whether or not the
    configuration mentions it, so the run report is always complete.

    Args:
        raw_config: Parsed TOML mapping, or ``None`` for pure defaults.
        source: Where the configuration came from, recorded for reporting.

    Returns:
        Fully resolved :class:`PerturbationSettings`.

    Raises:
        PerturbationConfigError: On any unknown key, unsupported mode or
            out-of-range strength.
    """
    config = raw_config or {}

    unknown_top_level = set(config) - {"strength", "seed", "parameters"}
    if unknown_top_level:
        raise PerturbationConfigError(
            f"Unknown top-level key(s) {sorted(unknown_top_level)}. "
            "Valid keys: 'strength', 'seed', 'parameters'."
        )

    global_strength = (
        _validate_strength(config["strength"], "Global")
        if "strength" in config
        else DEFAULT_STRENGTH
    )

    seed = config.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise PerturbationConfigError(f"seed must be an integer, got {seed!r}.")

    parameter_section = config.get("parameters", {})
    if not isinstance(parameter_section, dict):
        raise PerturbationConfigError(
            f"[parameters] must be a table, got {type(parameter_section).__name__}."
        )

    unknown_parameters = set(parameter_section) - set(REGISTRY)
    if unknown_parameters:
        raise PerturbationConfigError(
            f"Unknown parameter(s) {sorted(unknown_parameters)} in [parameters]. "
            f"Registered parameters: {', '.join(sorted(REGISTRY))}."
        )

    resolved: Dict[str, ParameterSetting] = {}
    for name, descriptor in REGISTRY.items():
        entry = parameter_section.get(name, {})
        if not isinstance(entry, dict):
            raise PerturbationConfigError(
                f"[parameters.{name}] must be a table, got {type(entry).__name__}."
            )

        unknown_entry_keys = set(entry) - {"mode", "strength"}
        if unknown_entry_keys:
            raise PerturbationConfigError(
                f"[parameters.{name}]: unknown key(s) {sorted(unknown_entry_keys)}. "
                "Valid keys: 'mode', 'strength'."
            )

        mode = (
            _validate_mode(entry["mode"], descriptor)
            if "mode" in entry
            else descriptor.default_mode
        )
        strength = (
            _validate_strength(entry["strength"], f"[parameters.{name}]")
            if "strength" in entry
            else global_strength
        )

        # Strength is meaningless outside perturb mode; zeroing it here means
        # downstream code never has to check the mode and the strength.
        if mode is not Mode.PERTURB:
            strength = 0.0

        resolved[name] = ParameterSetting(mode=mode, strength=strength)

    return PerturbationSettings(
        global_strength=global_strength,
        seed=seed,
        per_parameter=resolved,
        source=source,
    )


def load(path: Union[str, Path]) -> PerturbationSettings:
    """Load and resolve a perturbation configuration from a TOML file (FR-06).

    Args:
        path: Path to the TOML file.

    Returns:
        Fully resolved :class:`PerturbationSettings`.

    Raises:
        FileNotFoundError: If the file does not exist.
        PerturbationConfigError: If the file is not valid TOML, or the
            configuration is invalid.

    Example:
        >>> settings = load("perturbation.toml")
        >>> settings.should_perturb("degrees_by_level")
        True
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Perturbation config not found: {config_path}")

    try:
        with config_path.open("rb") as handle:
            raw_config = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise PerturbationConfigError(
            f"{config_path} is not valid TOML: {error}"
        ) from error

    return resolve(raw_config, source=str(config_path))


def coerce(
    config: Union[str, Path, PerturbationSettings, dict, None],
) -> PerturbationSettings:
    """Accept any supported configuration form and return resolved settings.

    Lets the synthesis entry points take a path, an already-resolved settings
    object, a raw mapping, or nothing at all, without each having to branch.

    Args:
        config: A path to a TOML file, a :class:`PerturbationSettings`, a raw
            mapping, or ``None`` for defaults.

    Returns:
        Fully resolved :class:`PerturbationSettings`.

    Raises:
        PerturbationConfigError: If *config* is of an unsupported type, or the
            configuration is invalid.
    """
    if config is None:
        return resolve(None, source="defaults")
    if isinstance(config, PerturbationSettings):
        return config
    if isinstance(config, (str, Path)):
        return load(config)
    if isinstance(config, dict):
        return resolve(config, source="mapping")
    raise PerturbationConfigError(
        f"Unsupported perturbation config type {type(config).__name__}. "
        "Expected a path, a PerturbationSettings, a mapping, or None."
    )
