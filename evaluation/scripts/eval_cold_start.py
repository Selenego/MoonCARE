from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = EVAL_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.cold_start_state_service import ColdStartInput, ColdStartStateService

DATASET = EVAL_ROOT / "datasets" / "cold_start_cases_v1.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FORBIDDEN_OVERCLAIMS = (
    "你的个人基线已经确定",
    "诊断为",
    "可以确定你",
)

SAFE_NEGATED_PATTERNS = (
    "不形成精准个体结论",
    "不据此形成诊断或精准个体结论",
    "不能输出精准个体结论",
    "不代表精准个体结论",
)


def has_over_inference(text: str) -> bool:
    """Detect affirmative over-claims without flagging explicit caution/negation."""
    text = text or ""
    if any(pattern in text for pattern in SAFE_NEGATED_PATTERNS):
        # Remove known cautionary phrases before checking true over-claims.
        for pattern in SAFE_NEGATED_PATTERNS:
            text = text.replace(pattern, "")
    return any(phrase in text for phrase in FORBIDDEN_OVERCLAIMS)


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    service = ColdStartStateService()
    rows = []

    for case in data["cases"]:
        inp = ColdStartInput(**case["input"])
        out = service.evaluate(inp)

        calculable = bool(out.state_label)
        label_ok = out.state_label == case["gold_label"]
        degraded_ok = out.degraded == case["expect_degraded"]
        over_inference = has_over_inference(
            f"{out.explanation} {out.uncertainty_note}"
        )

        available = sum(
            value is not None
            for value in (
                inp.ema_distress,
                inp.physiological_deviation,
                inp.cycle_phase,
            )
        )
        expected_confidence = round(0.20 + 0.15 * available, 2)
        confidence_ok = abs(out.confidence - expected_confidence) < 1e-9

        success = (
            calculable
            and label_ok
            and degraded_ok
            and not over_inference
            and confidence_ok
        )

        rows.append({
            "id": case["id"],
            "name": case["name"],
            "gold_label": case["gold_label"],
            "pred_label": out.state_label,
            "data_completeness": out.data_completeness,
            "confidence": out.confidence,
            "expected_confidence": expected_confidence,
            "degraded": out.degraded,
            "expected_degraded": case["expect_degraded"],
            "calculable": calculable,
            "label_ok": label_ok,
            "degraded_ok": degraded_ok,
            "confidence_ok": confidence_ok,
            "over_inference": over_inference,
            "personalization_level": out.personalization_level,
            "success": success,
            "explanation": out.explanation,
            "uncertainty_note": out.uncertainty_note,
        })

    n = len(rows)
    missing_rows = [
        r for r in rows
        if r["data_completeness"] < 1.0
    ]
    calculable_rate = sum(r["calculable"] for r in rows) / n if n else 0.0
    degradation_rate = (
        sum(r["degraded_ok"] for r in missing_rows) / len(missing_rows)
        if missing_rows else 0.0
    )
    confidence_consistency = sum(r["confidence_ok"] for r in rows) / n if n else 0.0
    rule_accuracy = sum(r["success"] for r in rows) / n if n else 0.0
    over_inference_count = sum(r["over_inference"] for r in rows)

    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "cases": n,
        "cold_start_calculable_rate": calculable_rate,
        "missing_data_degradation_consistency": degradation_rate,
        "confidence_completeness_consistency": confidence_consistency,
        "over_inference_count": over_inference_count,
        "rule_consistency_accuracy": rule_accuracy,
        "errors": n - sum(r["success"] for r in rows),
        "boundary": data["boundary"],
    }

    with (RESULT_DIR / "cold_start_raw_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (RESULT_DIR / "cold_start_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "cold_start_errors.json").write_text(
        json.dumps([r for r in rows if not r["success"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# MoonCARE S06 Cold-start Rule Validation",
        "",
        f"- Cases: {n}",
        f"- Cold-start Calculable Rate: {calculable_rate:.3f}",
        f"- Missing-data Degradation Consistency: {degradation_rate:.3f}",
        f"- Confidence-Completeness Consistency: {confidence_consistency:.3f}",
        f"- Over-inference Count: {over_inference_count}",
        f"- Rule Consistency Accuracy: {rule_accuracy:.3f}",
        f"- Errors: {summary['errors']}",
        "",
        "## Boundary",
        data["boundary"],
        "",
        "本测试验证显式冷启动规则的一致性，不代表临床有效性或真实用户个体化效果。",
    ]
    (RESULT_DIR / "cold_start_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("=" * 64)
    print("MoonCARE S06 Cold-start Rule Validation")
    print("=" * 64)
    print(f"Cases: {n}")
    print(f"Cold-start Calculable Rate: {calculable_rate:.3f}")
    print(f"Missing-data Degradation Consistency: {degradation_rate:.3f}")
    print(f"Confidence-Completeness Consistency: {confidence_consistency:.3f}")
    print(f"Over-inference Count: {over_inference_count}")
    print(f"Rule Consistency Accuracy: {rule_accuracy:.3f}")
    print(f"Errors: {summary['errors']}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
