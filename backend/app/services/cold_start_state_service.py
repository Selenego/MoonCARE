from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ColdStartInput:
    """Minimal cold-start inputs.

    This prototype deliberately avoids hard-coded clinical normal ranges.
    `physiological_deviation` is expected to be produced by an upstream,
    separately validated detector or reference process.
    """

    ema_distress: Optional[int] = None  # 0-5, optional
    physiological_deviation: Optional[bool] = None
    cycle_phase: Optional[str] = None
    history_count: int = 0


@dataclass
class ColdStartResult:
    state_label: str
    confidence: float
    data_completeness: float
    data_sufficient: bool
    degraded: bool
    uncertainty_note: str
    explanation: str
    personalization_level: str = "population_reference_only"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ColdStartStateService:
    """Conservative cold-start state estimation for engineering validation.

    Scope:
    - No diagnosis.
    - No claim of a learned personal baseline when history_count == 0.
    - No raw physiological threshold is embedded here.
    - Confidence follows signal completeness, not output severity.
    """

    def evaluate(self, data: ColdStartInput) -> ColdStartResult:
        available = sum(
            value is not None
            for value in (
                data.ema_distress,
                data.physiological_deviation,
                data.cycle_phase,
            )
        )
        completeness = available / 3.0

        # Conservative confidence ceiling during cold start.
        # 0 signals -> 0.20; 1 -> 0.35; 2 -> 0.50; 3 -> 0.65.
        confidence = round(0.20 + 0.15 * available, 2)
        data_sufficient = available >= 2
        degraded = not data_sufficient

        if data.ema_distress is not None and not 0 <= data.ema_distress <= 5:
            raise ValueError("ema_distress must be in [0, 5]")

        if data.ema_distress is not None and data.ema_distress >= 4:
            label = "elevated_distress"
            explanation = (
                "当前 EMA 主观困扰较高；冷启动阶段仅做保守状态提示，"
                "不据此形成诊断或精准个体结论。"
            )
        elif data.physiological_deviation is True:
            label = "physiological_deviation"
            explanation = (
                "上游输入提示存在生理偏离信号；由于缺少充分个人历史，"
                "当前仅标记为需要关注的相对状态。"
            )
        elif available == 0:
            label = "uncertain"
            explanation = "当前缺少可用 EMA、周期和生理状态信息，暂不做具体状态判断。"
        elif not data_sufficient:
            label = "uncertain"
            explanation = (
                "当前有效信息较少，仅保留低置信度参考，不形成精准个体结论。"
            )
        else:
            label = "reference_stable"
            explanation = (
                "现有信号未显示明显高困扰或已知生理偏离；"
                "该结论仅是冷启动参考状态，不代表个人长期基线。"
            )

        if available == 0:
            uncertainty = "数据不足：需补充至少一类当日状态信息。"
        elif not data_sufficient:
            uncertainty = "数据较少：已降低置信度，建议结合后续 EMA/周期/生理数据继续观察。"
        else:
            uncertainty = "冷启动阶段：尚无稳定个人历史，结论仅作群体参考层面的保守描述。"

        personalization_level = (
            "population_reference_only"
            if data.history_count <= 0
            else "limited_history_not_personal_baseline"
        )

        return ColdStartResult(
            state_label=label,
            confidence=confidence,
            data_completeness=round(completeness, 3),
            data_sufficient=data_sufficient,
            degraded=degraded,
            uncertainty_note=uncertainty,
            explanation=explanation,
            personalization_level=personalization_level,
        )
