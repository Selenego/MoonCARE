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

from app.services.personal_baseline_service import PersonalBaselineService

DATASET = EVAL_ROOT / "datasets" / "baseline_replay_v1.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    service = PersonalBaselineService(group_reference=data["group_reference"])

    rows = []
    last_accepted_weight = 0.0
    weight_monotonic = True
    missing_update_violations = 0
    outlier_update_violations = 0
    action_errors = 0

    for item in data["observations"]:
        before_count = len(service.accepted_values)
        out = service.update(item["value"])
        after_count = len(service.accepted_values)

        if out.accepted and out.personal_weight + 1e-9 < last_accepted_weight:
            weight_monotonic = False
        if out.accepted:
            last_accepted_weight = out.personal_weight

        if item["expected_action"] == "missing_ignored":
            if out.reason != "missing_ignored" or after_count != before_count or out.baseline_delta != 0:
                missing_update_violations += 1
        elif item["expected_action"] == "outlier_rejected":
            if out.reason != "outlier_rejected" or after_count != before_count or out.baseline_delta != 0:
                outlier_update_violations += 1

        action_ok = out.reason == item["expected_action"]
        if not action_ok:
            action_errors += 1

        rows.append({
            "day": item["day"],
            "observed_value": item["value"],
            "expected_action": item["expected_action"],
            "actual_action": out.reason,
            "action_ok": action_ok,
            "accepted": out.accepted,
            "valid_history_count": out.valid_history_count,
            "personal_reference": out.personal_reference,
            "personal_weight": out.personal_weight,
            "group_weight": round(1.0 - out.personal_weight, 4),
            "group_reference": data["group_reference"],
            "blended_baseline": out.blended_baseline,
            "phase": out.phase,
            "baseline_delta": out.baseline_delta,
        })

    phases = {r["phase"] for r in rows}
    required_phases = {"population_reference", "mixed_baseline", "personal_dominant"}
    phase_coverage = len(phases & required_phases) / len(required_phases)

    max_weight = max(r["personal_weight"] for r in rows)
    never_full_personal = max_weight < 1.0
    rule_accuracy = sum(r["action_ok"] for r in rows) / len(rows)

    accepted_deltas_after_warmup = [
        r["baseline_delta"] for r in rows
        if r["accepted"] and r["valid_history_count"] >= 5
    ]
    max_post_warmup_delta = (
        max(accepted_deltas_after_warmup)
        if accepted_deltas_after_warmup else 0.0
    )

    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "observations": len(rows),
        "accepted_observations": sum(r["accepted"] for r in rows),
        "action_consistency_accuracy": rule_accuracy,
        "weight_monotonic": weight_monotonic,
        "phase_coverage": phase_coverage,
        "final_personal_weight": rows[-1]["personal_weight"],
        "max_personal_weight": max_weight,
        "never_full_personal_weight": never_full_personal,
        "missing_update_violations": missing_update_violations,
        "outlier_update_violations": outlier_update_violations,
        "max_post_warmup_baseline_delta": max_post_warmup_delta,
        "errors": action_errors + missing_update_violations + outlier_update_violations + (0 if weight_monotonic else 1),
        "boundary": data["boundary"],
    }

    with (RESULT_DIR / "baseline_migration_raw_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (RESULT_DIR / "baseline_migration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "baseline_migration_errors.json").write_text(
        json.dumps([r for r in rows if not r["action_ok"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    checkpoints = []
    for target_count in (0, 3, 8, 12, rows[-1]["valid_history_count"]):
        candidates = [r for r in rows if r["valid_history_count"] == target_count]
        if target_count == 0:
            checkpoints.append({
                "stage": "group_reference",
                "valid_history_count": 0,
                "personal_weight": 0.0,
                "blended_baseline": data["group_reference"],
            })
        elif candidates:
            r = candidates[-1]
            checkpoints.append({
                "stage": r["phase"],
                "valid_history_count": target_count,
                "personal_weight": r["personal_weight"],
                "blended_baseline": r["blended_baseline"],
            })

    (RESULT_DIR / "baseline_migration_checkpoints.json").write_text(
        json.dumps(checkpoints, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# MoonCARE S07 Personal Baseline Migration Replay",
        "",
        f"- Observations: {len(rows)}",
        f"- Accepted Observations: {summary['accepted_observations']}",
        f"- Action Consistency Accuracy: {rule_accuracy:.3f}",
        f"- Weight Monotonic: {weight_monotonic}",
        f"- Phase Coverage: {phase_coverage:.3f}",
        f"- Final Personal Weight: {summary['final_personal_weight']:.3f}",
        f"- Missing-update Violations: {missing_update_violations}",
        f"- Outlier-update Violations: {outlier_update_violations}",
        f"- Max Post-warmup Baseline Delta: {max_post_warmup_delta:.4f}",
        f"- Errors: {summary['errors']}",
        "",
        "## Boundary",
        data["boundary"],
        "",
        "只可描述为模拟/回放下的算法行为验证，不得写成真实个体化效果提升。",
    ]
    (RESULT_DIR / "baseline_migration_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("=" * 64)
    print("MoonCARE S07 Personal Baseline Migration Replay")
    print("=" * 64)
    print(f"Observations: {len(rows)}")
    print(f"Accepted Observations: {summary['accepted_observations']}")
    print(f"Action Consistency Accuracy: {rule_accuracy:.3f}")
    print(f"Weight Monotonic: {weight_monotonic}")
    print(f"Phase Coverage: {phase_coverage:.3f}")
    print(f"Final Personal Weight: {summary['final_personal_weight']:.3f}")
    print(f"Missing-update Violations: {missing_update_violations}")
    print(f"Outlier-update Violations: {outlier_update_violations}")
    print(f"Max Post-warmup Baseline Delta: {max_post_warmup_delta:.4f}")
    print(f"Errors: {summary['errors']}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
