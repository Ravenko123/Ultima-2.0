"""Experiment result containers for the ML backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

from .parameter_space import ParameterSample

__all__ = [
    "ExperimentArtifact",
    "TrialScorecard",
    "ExperimentResult",
]


@dataclass(slots=True, frozen=True)
class ExperimentArtifact:
    """References to files generated during an experiment run."""

    run_id: str
    telemetry_file: Path
    config_dump: Path | None = None
    notes: str | None = None


@dataclass(slots=True)
class TrialScorecard:
    """Holds optimization metrics and constraint status for a trial."""

    metrics: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, bool] = field(default_factory=dict)

    def passes_all_constraints(self) -> bool:
        return all(self.constraints.values()) if self.constraints else True

    def summary(self) -> Mapping[str, float | bool]:
        summary: MutableMapping[str, float | bool] = {}
        summary.update(self.metrics)
        summary.update({f"constraint::{key}": status for key, status in self.constraints.items()})
        return dict(summary)


@dataclass(slots=True)
class ExperimentResult:
    """Full record of an experiment trial."""

    sample: ParameterSample
    scorecard: TrialScorecard
    artifacts: Sequence[ExperimentArtifact] = field(default_factory=tuple)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    generation: int | None = None

    def mark_finished(self) -> None:
        self.finished_at = datetime.now(timezone.utc)

    def metric(self, key: str, default: float | None = None) -> float | None:
        return self.scorecard.metrics.get(key, default)
