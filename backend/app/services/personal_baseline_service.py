from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import median
from typing import Any, Dict, List, Optional


@dataclass
class BaselineUpdate:
    observed_value: Optional[float]
    accepted: bool
    reason: str
    valid_history_count: int
    personal_reference: float
    personal_weight: float
    blended_baseline: float
    phase: str
    baseline_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersonalBaselineService:
    """Explainable group-to-personal baseline migration prototype.

    Values are assumed to already be normalized upstream. This module does not
    define medical thresholds and must not be interpreted as a clinical model.
    """

    def __init__(
        self,
        group_reference: float = 0.0,
        transition_k: float = 8.0,
        max_personal_weight: float = 0.8,
        outlier_check_after: int = 5,
        outlier_floor: float = 0.8,
        mad_multiplier: float = 4.0,
    ) -> None:
        self.group_reference = float(group_reference)
        self.transition_k = float(transition_k)
        self.max_personal_weight = float(max_personal_weight)
        self.outlier_check_after = int(outlier_check_after)
        self.outlier_floor = float(outlier_floor)
        self.mad_multiplier = float(mad_multiplier)
        self._accepted: List[float] = []
        self._baseline = self.group_reference

    @property
    def accepted_values(self) -> List[float]:
        return list(self._accepted)

    def _personal_reference(self) -> float:
        if not self._accepted:
            return self.group_reference
        return float(median(self._accepted))

    def _weight(self) -> float:
        n = len(self._accepted)
        if n <= 0:
            return 0.0
        raw = n / (n + self.transition_k)
        return min(self.max_personal_weight, raw)

    def _phase(self, weight: float) -> str:
        n = len(self._accepted)
        if n < 3:
            return "population_reference"
        if weight < 0.60:
            return "mixed_baseline"
        return "personal_dominant"

    def _is_outlier(self, value: float) -> bool:
        if len(self._accepted) < self.outlier_check_after:
            return False

        center = median(self._accepted)
        deviations = [abs(v - center) for v in self._accepted]
        mad = median(deviations) if deviations else 0.0

        # Robust threshold for normalized engineering signals.
        threshold = max(self.outlier_floor, self.mad_multiplier * 1.4826 * mad)
        return abs(value - center) > threshold

    def update(self, value: Optional[float]) -> BaselineUpdate:
        previous_baseline = self._baseline

        if value is None:
            personal = self._personal_reference()
            weight = self._weight()
            return BaselineUpdate(
                observed_value=None,
                accepted=False,
                reason="missing_ignored",
                valid_history_count=len(self._accepted),
                personal_reference=personal,
                personal_weight=round(weight, 4),
                blended_baseline=round(self._baseline, 4),
                phase=self._phase(weight),
                baseline_delta=0.0,
            )

        value = float(value)
        if not isfinite(value):
            raise ValueError("observation must be finite or None")

        if self._is_outlier(value):
            personal = self._personal_reference()
            weight = self._weight()
            return BaselineUpdate(
                observed_value=value,
                accepted=False,
                reason="outlier_rejected",
                valid_history_count=len(self._accepted),
                personal_reference=round(personal, 4),
                personal_weight=round(weight, 4),
                blended_baseline=round(self._baseline, 4),
                phase=self._phase(weight),
                baseline_delta=0.0,
            )

        self._accepted.append(value)
        personal = self._personal_reference()
        weight = self._weight()
        self._baseline = (1.0 - weight) * self.group_reference + weight * personal

        return BaselineUpdate(
            observed_value=value,
            accepted=True,
            reason="accepted",
            valid_history_count=len(self._accepted),
            personal_reference=round(personal, 4),
            personal_weight=round(weight, 4),
            blended_baseline=round(self._baseline, 4),
            phase=self._phase(weight),
            baseline_delta=round(abs(self._baseline - previous_baseline), 4),
        )
