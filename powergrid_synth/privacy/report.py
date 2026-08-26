"""
report — per-run record of what was protected and what was not.

Every synthesis run produces a :class:`PerturbationReport` listing each
registered parameter of the running pipeline with its resolved mode, strength
and noise scale, plus the resolved seed.  Two purposes:

* **Reproducibility.**  The resolved seed is recorded even when the
  configuration asked for fresh entropy, so a run can be repeated exactly.
* **Audit.**  The report states plainly which reference-derived values were
  read unchanged, which were perturbed, and which were never read.  An
  operator should not have to read the source to find that out.

Fallback events are recorded too.  A degree sequence that could not be
perturbed into a realisable network falls back to the unperturbed sequence,
and a report that omitted that would overstate the protection actually
achieved.

Spec coverage: FR-10
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .registry import REGISTRY, Domain
from .settings import PerturbationSettings


@dataclass
class ParameterRecord:
    """What happened to one parameter during a run.

    Args:
        name: Registered parameter name.
        mode: Resolved mode, as a string.
        strength: Resolved strength.
        noise_scale: Laplace scale actually used, or ``None`` when not
            perturbed.  For scalar families this is the scale of a field with
            the default noise constant; per-field constants may differ.
        description: Plain-language summary from the descriptor.
        fallback_note: What was used instead, when the mode was ``off``.
        fallback_triggered: Whether a perturbation attempt gave up and used
            the unperturbed value.
        detail: Extra per-parameter information, such as per-level bus deltas.
    """

    name: str
    mode: str
    strength: float
    noise_scale: Optional[float]
    description: str
    fallback_note: str = ""
    fallback_triggered: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationReport:
    """Complete record of one synthesis run's perturbation behaviour.

    Args:
        domain: Which pipeline produced this report.
        resolved_seed: The seed actually used, even when drawn fresh.
        config_source: Where the configuration came from.
        global_strength: The file-level strength default.
        parameters: One record per registered parameter of *domain*.
    """

    domain: str
    resolved_seed: int
    config_source: str
    global_strength: float
    parameters: List[ParameterRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a plain dictionary suitable for JSON."""
        return {
            "domain": self.domain,
            "resolved_seed": self.resolved_seed,
            "config_source": self.config_source,
            "global_strength": self.global_strength,
            "parameters": [asdict(record) for record in self.parameters],
        }

    def write_json(self, path: Union[str, Path]) -> Path:
        """Write the report to *path* as JSON.

        Args:
            path: Destination file path.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False), encoding="utf-8"
        )
        return destination

    def summary_lines(self) -> List[str]:
        """Return a compact human-readable summary, one line per group.

        Returns:
            Lines suitable for printing to stdout.
        """
        # Group by *effective* behaviour, not the configured mode.  A
        # parameter in perturb mode at strength zero is not protected, and a
        # report that filed it under "perturb" would overstate what the run
        # actually did (spec Risk 5).
        by_mode: Dict[str, List[str]] = {}
        for record in self.parameters:
            effective_mode = record.mode
            if record.mode == "perturb" and not record.strength:
                effective_mode = "raw (strength 0)"
            by_mode.setdefault(effective_mode, []).append(record.name)

        lines = [
            # Kept ASCII: this line is printed to stdout, and a Windows
            # console using cp1252 cannot encode typographic dashes.
            f"Perturbation report ({self.domain}) - "
            f"seed {self.resolved_seed}, config: {self.config_source}"
        ]
        for mode in ("perturb", "raw (strength 0)", "raw", "off"):
            names = sorted(by_mode.get(mode, []))
            if not names:
                continue
            lines.append(f"  {mode:<16} ({len(names):>2}): {', '.join(names)}")

        if not by_mode.get("perturb"):
            lines.append(
                "  NOTE: nothing was perturbed in this run. Reference-derived "
                "values marked 'raw' reach the output unchanged."
            )

        fallbacks = [r.name for r in self.parameters if r.fallback_triggered]
        if fallbacks:
            lines.append(
                f"  fallback : {', '.join(sorted(fallbacks))} "
                "(perturbation gave up; unperturbed value used)"
            )
        return lines


def build_report(
    settings: PerturbationSettings,
    domain: Domain,
    resolved_seed: int,
    *,
    details: Optional[Dict[str, Dict[str, Any]]] = None,
    fallbacks: Optional[Dict[str, bool]] = None,
) -> PerturbationReport:
    """Assemble a report for one run of one pipeline.

    Args:
        settings: The resolved settings used for the run.
        domain: Which pipeline is reporting.
        resolved_seed: The seed actually used.
        details: Optional extra information keyed by parameter name.
        fallbacks: Optional fallback flags keyed by parameter name.

    Returns:
        A populated :class:`PerturbationReport` covering every parameter of
        *domain*, whether or not the configuration mentioned it.
    """
    details = details or {}
    fallbacks = fallbacks or {}

    records: List[ParameterRecord] = []
    for name, setting in sorted(settings.active_for_domain(domain).items()):
        descriptor = REGISTRY[name]
        records.append(
            ParameterRecord(
                name=name,
                mode=setting.mode.value,
                strength=setting.strength,
                noise_scale=(
                    setting.strength * descriptor.noise_k
                    if setting.should_perturb
                    else None
                ),
                description=descriptor.description,
                fallback_note=descriptor.fallback_note,
                fallback_triggered=bool(fallbacks.get(name, False)),
                detail=details.get(name, {}),
            )
        )

    return PerturbationReport(
        domain=domain.value,
        resolved_seed=resolved_seed,
        config_source=settings.source,
        global_strength=settings.global_strength,
        parameters=records,
    )
