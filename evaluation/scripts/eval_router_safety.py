from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = EVAL_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agents.perception_agent import PerceptionAgent
from app.agents.router import Router


class DummyAgent:
    """避免评测 Router 时真正调用外部 LLM。"""
    def __init__(self, name: str):
        self.name = name

    def respond(self, message: str, state: Dict) -> str:
        return f"[dummy:{self.name}]"




ROUTE_ALIASES = {
    "light_comfort": "support",
    "support_fallback": "support",
    "knowledge_fallback": "knowledge",
    "action_support_fallback": "action_support",
    "intervention_fallback": "intervention",
}


def canonicalize_route(route: str) -> str:
    """Normalize implementation sub-routes to the S01/S02 evaluation label space."""
    route = str(route or "")
    if route in ROUTE_ALIASES:
        return ROUTE_ALIASES[route]
    if route.startswith("knowledge"):
        return "knowledge"
    if route.startswith("action_support"):
        return "action_support"
    if route.startswith("intervention"):
        return "intervention"
    if route.startswith("support"):
        return "support"
    return route


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def precision_recall_f1(rows: List[dict], labels: List[str]) -> dict:
    out = {}
    for label in labels:
        tp = sum(1 for r in rows if r["gold_route"] == label and r["pred_route"] == label)
        fp = sum(1 for r in rows if r["gold_route"] != label and r["pred_route"] == label)
        fn = sum(1 for r in rows if r["gold_route"] == label and r["pred_route"] != label)
        p = safe_div(tp, tp + fp)
        rc = safe_div(tp, tp + fn)
        f1 = safe_div(2 * p * rc, p + rc)
        out[label] = {"precision": p, "recall": rc, "f1": f1, "support": tp + fn}
    return out


def main() -> None:
    dataset_path = EVAL_ROOT / "datasets" / "test_v1.json"
    result_dir = EVAL_ROOT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = data["cases"]

    perception = PerceptionAgent()
    router = Router()

    # 直接注入 DummyAgent，避免触发真实 Support/Knowledge/Intervention LLM。
    router._support = DummyAgent("support")
    router._knowledge = DummyAgent("knowledge")
    router._intervention = DummyAgent("intervention")

    rows = []
    for case in cases:
        state = perception.analyze(case["text"], cycle_phase=None, sensor_data={})
        _, raw_pred_route = router.route(case["text"], state, agent_mode="auto")
        pred_route = canonicalize_route(raw_pred_route)
        row = {
            **case,
            "raw_pred_route": raw_pred_route,
            "pred_route": pred_route,
            "pred_risk": state.get("risk_level", "low"),
            "route_correct": pred_route == case["gold_route"],
            "safety_gold_positive": case["gold_risk"] in {"high", "crisis"},
            "safety_pred_positive": state.get("risk_level") in {"high", "crisis"},
        }
        rows.append(row)

    labels = ["support", "knowledge", "action_support", "intervention"]
    accuracy = safe_div(sum(r["route_correct"] for r in rows), len(rows))
    per_class = precision_recall_f1(rows, labels)
    macro_f1 = sum(per_class[x]["f1"] for x in labels) / len(labels)

    mixed = [r for r in rows if r["category"] == "mixed_intent"]
    mixed_success = safe_div(sum(r["route_correct"] for r in mixed), len(mixed))

    safety_tp = sum(1 for r in rows if r["safety_gold_positive"] and r["safety_pred_positive"])
    safety_fp = sum(1 for r in rows if not r["safety_gold_positive"] and r["safety_pred_positive"])
    safety_fn = sum(1 for r in rows if r["safety_gold_positive"] and not r["safety_pred_positive"])
    safety_tn = sum(1 for r in rows if not r["safety_gold_positive"] and not r["safety_pred_positive"])
    safety_precision = safe_div(safety_tp, safety_tp + safety_fp)
    safety_recall = safe_div(safety_tp, safety_tp + safety_fn)
    safety_priority_accuracy = safe_div(
        sum(1 for r in rows if r["safety_gold_positive"] and r["pred_route"] == "intervention"),
        sum(1 for r in rows if r["safety_gold_positive"]),
    )

    confusion = {g: {p: 0 for p in labels} for g in labels}
    for r in rows:
        if r["gold_route"] in confusion and r["pred_route"] in labels:
            confusion[r["gold_route"]][r["pred_route"]] += 1

    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "n_cases": len(rows),
        "router": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "mixed_intent_success_rate": mixed_success,
            "per_class": per_class,
        },
        "safety": {
            "tp": safety_tp,
            "fp": safety_fp,
            "fn": safety_fn,
            "tn": safety_tn,
            "precision": safety_precision,
            "recall": safety_recall,
            "priority_accuracy": safety_priority_accuracy,
        },
        "error_count": sum(1 for r in rows if not r["route_correct"]),
        "route_normalization": ROUTE_ALIASES,
    }

    # raw results
    fieldnames = [
        "id","text","category","gold_route","raw_pred_route","pred_route","gold_risk","pred_risk",
        "dominant_intent","route_correct","expected_behavior","notes"
    ]
    with open(result_dir / "raw_results.csv","w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(result_dir / "confusion_matrix.csv","w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["gold\\pred"] + labels)
        for g in labels:
            w.writerow([g] + [confusion[g][p] for p in labels])

    errors = [r for r in rows if not r["route_correct"]]
    (result_dir / "top_errors.json").write_text(
        json.dumps(errors[:10], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    false_negatives = [
        r for r in rows if r["safety_gold_positive"] and not r["safety_pred_positive"]
    ]
    (result_dir / "safety_false_negatives.json").write_text(
        json.dumps(false_negatives, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = []
    report.append("# MoonCARE Router / Safety 基线评测\n")
    report.append(f"- 数据集：{data['dataset_name']} v{data['version']}")
    report.append(f"- 样本数：{len(rows)}")
    report.append(f"- Router Accuracy：{accuracy:.3f}")
    report.append(f"- Router Macro-F1：{macro_f1:.3f}")
    report.append(f"- Mixed Intent Success：{mixed_success:.3f}")
    report.append(f"- Safety Precision：{safety_precision:.3f}")
    report.append(f"- Safety Recall：{safety_recall:.3f}")
    report.append(f"- Safety Priority Accuracy：{safety_priority_accuracy:.3f}")
    report.append(f"- Router Error Count：{summary['error_count']}")
    report.append("\n## Per-class\n")
    for label, m in per_class.items():
        report.append(
            f"- {label}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}, n={m['support']}"
        )
    report.append("\n> 注意：本轮为规则层离线基线评测，使用 DummyAgent 避免外部 LLM API 对结果造成干扰。")
    report.append("> `light_comfort` 等实现子路由会归一到 S01/S02 的目标标签空间，同时在 raw_results.csv 保留 raw_pred_route。")
    (result_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print("=" * 60)
    print("MoonCARE Router / Safety baseline evaluation")
    print("=" * 60)
    print(f"Cases: {len(rows)}")
    print(f"Router Accuracy: {accuracy:.3f}")
    print(f"Router Macro-F1: {macro_f1:.3f}")
    print(f"Mixed Intent Success: {mixed_success:.3f}")
    print(f"Safety Precision: {safety_precision:.3f}")
    print(f"Safety Recall: {safety_recall:.3f}")
    print(f"Safety Priority Accuracy: {safety_priority_accuracy:.3f}")
    print(f"Errors: {summary['error_count']}")
    print(f"Results written to: {result_dir}")


if __name__ == "__main__":
    main()
