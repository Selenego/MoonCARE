from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = EVAL_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.jitai_policy_service import JITAIPolicyService, JITAIState

DATASET = EVAL_ROOT / "datasets" / "JITAI_scenarios_v1.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_STAGES = {"safety", "need", "availability", "burden", "preference"}


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    policy = JITAIPolicyService()
    rows = []

    for case in data["scenarios"]:
        raw_state = dict(case["state"])
        raw_state["preferred_options"] = tuple(raw_state.get("preferred_options", []))
        state = JITAIState(**raw_state)
        decision = policy.decide(state)

        correct = (
            decision.action == case["gold_action"]
            and decision.rule_stage == case["gold_stage"]
        )
        rows.append({
            "id": case["id"],
            "name": case["name"],
            "gold_action": case["gold_action"],
            "pred_action": decision.action,
            "gold_stage": case["gold_stage"],
            "pred_stage": decision.rule_stage,
            "reason": decision.reason,
            "correct": correct,
            "tags": "|".join(case.get("tags", [])),
            "trace": " > ".join(decision.trace),
        })

    n = len(rows)
    correct_n = sum(1 for row in rows if row["correct"])
    decision_accuracy = correct_n / n if n else 0.0

    exercised_stages = {row["pred_stage"] for row in rows}
    rule_coverage = len(REQUIRED_STAGES & exercised_stages) / len(REQUIRED_STAGES)

    no_intervention_rows = [row for row in rows if row["gold_action"] == "no_intervention"]
    no_intervention_correct = sum(
        1 for row in no_intervention_rows if row["pred_action"] == "no_intervention"
    )
    no_intervention_accuracy = (
        no_intervention_correct / len(no_intervention_rows)
        if no_intervention_rows else 0.0
    )

    frequency_violations = sum(
        1 for row in rows
        if ("cooldown" in row["tags"] or "frequency_cap" in row["tags"])
        and row["gold_action"] == "no_intervention"
        and row["pred_action"] != "no_intervention"
    )

    safety_priority_errors = sum(
        1 for row in rows
        if "safety" in row["tags"]
        and row["gold_action"] == "safety_support"
        and row["pred_action"] != "safety_support"
    )

    stage_counts = Counter(row["pred_stage"] for row in rows)
    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "n_scenarios": n,
        "decision_accuracy": decision_accuracy,
        "rule_coverage": rule_coverage,
        "covered_stages": sorted(exercised_stages),
        "no_intervention_case_count": len(no_intervention_rows),
        "no_intervention_accuracy": no_intervention_accuracy,
        "frequency_control_violation_count": frequency_violations,
        "safety_priority_error_count": safety_priority_errors,
        "stage_counts": dict(stage_counts),
        "boundary": data["boundary"],
    }

    fields = [
        "id","name","gold_action","pred_action","gold_stage","pred_stage",
        "reason","correct","tags","trace"
    ]
    with open(RESULT_DIR / "jitai_raw_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (RESULT_DIR / "jitai_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    errors = [row for row in rows if not row["correct"]]
    (RESULT_DIR / "jitai_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    coverage = {
        "required_rule_order": data["decision_order"],
        "covered_stages": sorted(exercised_stages),
        "stage_counts": dict(stage_counts),
        "coverage_rate": rule_coverage,
        "scenario_tags": sorted({tag for c in data["scenarios"] for tag in c.get("tags", [])}),
    }
    (RESULT_DIR / "jitai_rule_coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# MoonCARE S09 JITAI Rule Unit Test",
        "",
        f"- 情景数：{n}",
        f"- 决策正确率：{decision_accuracy:.3f}",
        f"- 规则阶段覆盖率：{rule_coverage:.3f}",
        f"- 不干预场景正确率：{no_intervention_accuracy:.3f}",
        f"- 频控违规次数：{frequency_violations}",
        f"- Safety 优先级错误数：{safety_priority_errors}",
        "",
        "## 规则顺序",
        "",
        "Safety → Need → Availability → Burden → Preference",
        "",
        "## 边界",
        "",
        data["boundary"],
    ]
    (RESULT_DIR / "jitai_report.md").write_text("\n".join(report), encoding="utf-8")

    print("=" * 64)
    print("MoonCARE S09 JITAI Rule Unit Test")
    print("=" * 64)
    print(f"Scenarios: {n}")
    print(f"Decision Accuracy: {decision_accuracy:.3f}")
    print(f"Rule Coverage: {rule_coverage:.3f}")
    print(f"No-intervention Accuracy: {no_intervention_accuracy:.3f}")
    print(f"Frequency-control Violations: {frequency_violations}")
    print(f"Safety Priority Errors: {safety_priority_errors}")
    print(f"Errors: {len(errors)}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
