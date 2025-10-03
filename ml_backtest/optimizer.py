"""Simple random-search optimizer wrapping the experiment runner."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence

from .experiment_runner import ExperimentRunner
from .parameter_space import ParameterSpace
from .results import ExperimentResult

__all__ = [
	"OptimizationSettings",
	"OptimizationSummary",
	"ExperimentOptimizer",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OptimizationSettings:
	trials: int = 20
	primary_metric: str = "net_profit"
	maximize: bool = True
	random_seed: int | None = None
	stop_on_plateau: int | None = None


@dataclass(slots=True)
class OptimizationSummary:
	best_result: ExperimentResult | None
	history: Sequence[ExperimentResult] = field(default_factory=tuple)

	def metric_series(self, metric: str) -> List[float | None]:
		return [result.metric(metric) for result in self.history]


class ExperimentOptimizer:
	"""Coordinates parameter sampling and experiment execution."""

	def __init__(
		self,
		runner: ExperimentRunner,
		parameter_space: ParameterSpace,
		settings: OptimizationSettings | None = None,
	) -> None:
		self.runner = runner
		self.parameter_space = parameter_space
		self.settings = settings or OptimizationSettings()
		self._rng = random.Random(self.settings.random_seed)

	def optimize(self) -> OptimizationSummary:
		history: list[ExperimentResult] = []
		best: ExperimentResult | None = None
		best_score = -math.inf if self.settings.maximize else math.inf
		plateau_counter = 0

		for trial_index in range(self.settings.trials):
			sample = self.parameter_space.sample(rng=self._rng)
			logger.info("Starting trial %d with sample %s", trial_index + 1, sample.values)
			result = self.runner.run(sample)
			history.append(result)

			score = self._extract_score(result)
			if score is None:
				logger.debug(
					"Trial %d produced no score for metric %s",
					trial_index + 1,
					self.settings.primary_metric,
				)
			else:
				is_better = (score > best_score) if self.settings.maximize else (score < best_score)
				if is_better:
					best = result
					best_score = score
					plateau_counter = 0
					logger.info("New best score %.4f on trial %d", score, trial_index + 1)
				else:
					plateau_counter += 1

			if self.settings.stop_on_plateau is not None and plateau_counter >= self.settings.stop_on_plateau:
				logger.info("Stopping optimization after %d plateau trials", plateau_counter)
				break

		return OptimizationSummary(best_result=best, history=tuple(history))

	def _extract_score(self, result: ExperimentResult) -> float | None:
		metric = result.metric(self.settings.primary_metric)
		if metric is None:
			return None
		if not result.scorecard.passes_all_constraints():
			return None
		return float(metric)