from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


NORMAL_OPTIONS = {
    "no_intervention",
    "breathing",
    "music",
    "diary",
    "knowledge",
    "emotional_support",
}
SAFETY_OPTION = "safety_support"


@dataclass(frozen=True)
class JITAIState:
    risk_level: str = "low"
    distress_level: int = 0
    available: bool = True
    minutes_since_last_intervention: int | None = None
    reminders_last_24h: int = 0
    preferred_options: tuple[str, ...] = ()
    need_type: str = "support"
    user_initiated: bool = False
    data_sufficient: bool = True
    physiological_deviation: bool = False
    cycle_phase: str = "unknown"


@dataclass(frozen=True)
class JITAIDecision:
    action: str
    reason: str
    rule_stage: str
    blocked: bool
    trace: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class JITAIPolicyService:
    """Explainable rule-first JITAI policy.

    Decision order:
    Safety -> Need -> Availability -> Burden -> Preference
    """

    MAX_PROACTIVE_REMINDERS_24H = 3
    MIN_PROACTIVE_INTERVAL_MINUTES = 120
    NEED_DISTRESS_THRESHOLD = 4

    def decide(self, state: JITAIState) -> JITAIDecision:
        trace: list[str] = []

        # 1. Safety: high/crisis risk blocks all normal recommendation paths.
        risk = (state.risk_level or "low").lower()
        if risk in {"high", "crisis"}:
            trace.append(f"safety:block_normal_path:risk={risk}")
            return JITAIDecision(
                action=SAFETY_OPTION,
                reason="high_or_crisis_risk",
                rule_stage="safety",
                blocked=False,
                trace=tuple(trace),
            )
        trace.append("safety:pass")

        # 2. Need: avoid turning every state fluctuation into a push.
        has_need = (
            state.user_initiated
            or state.distress_level >= self.NEED_DISTRESS_THRESHOLD
            or state.physiological_deviation
        )
        if not has_need:
            trace.append("need:no_meaningful_need")
            return self._no_intervention(
                "need",
                "low_need",
                trace,
            )

        if not state.data_sufficient and not state.user_initiated:
            trace.append("need:insufficient_data_for_proactive_intervention")
            return self._no_intervention(
                "need",
                "insufficient_data",
                trace,
            )
        trace.append("need:pass")

        # 3. Availability: do not interrupt when the user says they are busy/unavailable.
        if not state.available:
            trace.append("availability:user_unavailable")
            return self._no_intervention(
                "availability",
                "user_unavailable",
                trace,
            )
        trace.append("availability:pass")

        # 4. Burden: frequency control applies to proactive pushes.
        if not state.user_initiated:
            if (
                state.minutes_since_last_intervention is not None
                and state.minutes_since_last_intervention < self.MIN_PROACTIVE_INTERVAL_MINUTES
            ):
                trace.append("burden:cooldown_active")
                return self._no_intervention(
                    "burden",
                    "recent_intervention",
                    trace,
                )

            if state.reminders_last_24h >= self.MAX_PROACTIVE_REMINDERS_24H:
                trace.append("burden:daily_frequency_cap")
                return self._no_intervention(
                    "burden",
                    "frequency_cap",
                    trace,
                )
        trace.append("burden:pass")

        # 5. Preference: choose a low-burden option aligned with need and preferences.
        candidates = self._candidate_options(state.need_type)
        preferred = self._first_supported_preference(state.preferred_options, candidates)
        if preferred:
            trace.append(f"preference:selected:{preferred}")
            return JITAIDecision(
                action=preferred,
                reason="preferred_option",
                rule_stage="preference",
                blocked=False,
                trace=tuple(trace),
            )

        default_action = candidates[0]
        trace.append(f"preference:default:{default_action}")
        return JITAIDecision(
            action=default_action,
            reason="default_for_need_type",
            rule_stage="preference",
            blocked=False,
            trace=tuple(trace),
        )

    def _candidate_options(self, need_type: str) -> tuple[str, ...]:
        normalized = (need_type or "support").lower()
        mapping = {
            "calm": ("breathing", "music", "emotional_support"),
            "reflect": ("diary", "emotional_support"),
            "knowledge": ("knowledge", "emotional_support"),
            "support": ("emotional_support", "breathing", "music"),
        }
        return mapping.get(normalized, mapping["support"])

    def _first_supported_preference(
        self,
        preferences: Iterable[str],
        candidates: tuple[str, ...],
    ) -> str | None:
        for option in preferences:
            if option in NORMAL_OPTIONS and option in candidates and option != "no_intervention":
                return option
        return None

    def _no_intervention(
        self,
        stage: str,
        reason: str,
        trace: list[str],
    ) -> JITAIDecision:
        return JITAIDecision(
            action="no_intervention",
            reason=reason,
            rule_stage=stage,
            blocked=True,
            trace=tuple(trace),
        )
