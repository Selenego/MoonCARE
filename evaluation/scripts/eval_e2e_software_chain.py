from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = EVAL_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agents.perception_agent import PerceptionAgent
from app.agents.router import Router
from app.agents.knowledge_agent import KnowledgeAgent
from app.services.agent_service import AgentService


DATASET_PATH = EVAL_ROOT / "datasets" / "e2e_software_chain_v1.json"
KNOWLEDGE_PATH = BACKEND_DIR / "app" / "data" / "knowledge_base.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


ROUTE_ALIASES = {
    "light_comfort": "support",
    "support_fallback": "support",
    "knowledge_fallback": "knowledge",
    "action_support_fallback": "action_support",
    "intervention_fallback": "intervention",
}


def canonicalize_route(route: str) -> str:
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


class OfflineSupportAgent:
    """Deterministic support stub to avoid external LLM calls."""
    def respond(self, message: str, state: Dict[str, Any]) -> str:
        intent = state.get("support_intent", "support")
        if intent == "action_support":
            return "先做一个最小动作：慢慢呼吸几次，然后告诉我现在最难受的部分。"
        return "我在听。你可以继续说说现在最明显的感受。"


class OfflineInterventionAgent:
    """Deterministic safe intervention stub for offline path verification."""
    def respond(self, message: str, state: Dict[str, Any]) -> str:
        return (
            "如果你现在有伤害自己的风险，请优先联系身边可信任的人、当地急救或危机支持资源，"
            "并尽量不要独处。"
        )


def build_offline_knowledge_agent() -> KnowledgeAgent:
    """Use the project's real local-card retrieval without external APIs."""
    agent = KnowledgeAgent.__new__(KnowledgeAgent)
    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as f:
        agent.knowledge = json.load(f)
    agent._embedding_available = False
    agent.embedder = None
    agent.llm = None
    return agent


def risk_matches(pred: str, gold: str) -> bool:
    if gold == "crisis":
        return pred in {"high", "crisis"}
    return pred == gold


def main() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]

    perception = PerceptionAgent()
    router = Router()
    router._support = OfflineSupportAgent()
    router._knowledge = build_offline_knowledge_agent()
    router._intervention = OfflineInterventionAgent()
    agent_service = AgentService()

    rows = []
    logs = []

    for case in cases:
        started = time.perf_counter()

        state = perception.analyze(case["text"], cycle_phase=None, sensor_data={})
        t_perception = (time.perf_counter() - started) * 1000

        route_started = time.perf_counter()
        reply, raw_route = router.route(case["text"], state, agent_mode="auto")
        t_route = (time.perf_counter() - route_started) * 1000

        pred_route = canonicalize_route(raw_route)
        pred_risk = state.get("risk_level", "low")

        action_started = time.perf_counter()
        actions = agent_service._generate_action_suggestions(state, pred_route)
        t_actions = (time.perf_counter() - action_started) * 1000

        total_ms = (time.perf_counter() - started) * 1000

        route_ok = pred_route == case["gold_route"]
        risk_ok = risk_matches(pred_risk, case["gold_risk"])
        reply_ok = bool(str(reply or "").strip())

        expected_actions = case.get("expect_actions")
        if expected_actions is True:
            action_ok = len(actions) > 0
        elif expected_actions is False:
            action_ok = len(actions) == 0
        else:
            action_ok = True

        success = route_ok and risk_ok and reply_ok and action_ok

        row = {
            "id": case["id"],
            "name": case["name"],
            "gold_route": case["gold_route"],
            "pred_route": pred_route,
            "raw_route": raw_route,
            "gold_risk": case["gold_risk"],
            "pred_risk": pred_risk,
            "route_ok": route_ok,
            "risk_ok": risk_ok,
            "reply_ok": reply_ok,
            "action_count": len(actions),
            "action_ok": action_ok,
            "success": success,
            "perception_ms": round(t_perception, 3),
            "route_ms": round(t_route, 3),
            "action_ms": round(t_actions, 3),
            "total_ms": round(total_ms, 3),
            "path_note": case["path_note"],
        }
        rows.append(row)

        logs.append({
            "trace_id": case["id"],
            "input": {
                "message_length": len(case["text"]),
                "cycle_phase": None,
                "sensor_data_present": False,
            },
            "perception": {
                "risk_level": pred_risk,
                "support_intent": state.get("support_intent"),
                "cycle_phase": state.get("cycle_phase"),
            },
            "router": {
                "raw_route": raw_route,
                "canonical_route": pred_route,
            },
            "output": {
                "reply_nonempty": reply_ok,
                "action_count": len(actions),
                "actions": [a.get("action") for a in actions],
            },
            "timing_ms": {
                "perception": round(t_perception, 3),
                "route": round(t_route, 3),
                "actions": round(t_actions, 3),
                "total": round(total_ms, 3),
            },
            "success": success,
        })

    n = len(rows)
    successes = sum(1 for row in rows if row["success"])
    success_rate = successes / n if n else 0.0
    route_accuracy = sum(1 for row in rows if row["route_ok"]) / n if n else 0.0
    risk_accuracy = sum(1 for row in rows if row["risk_ok"]) / n if n else 0.0
    avg_total = sum(row["total_ms"] for row in rows) / n if n else 0.0

    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "n_cases": n,
        "software_chain_success_rate": success_rate,
        "route_accuracy": route_accuracy,
        "risk_accuracy": risk_accuracy,
        "average_total_latency_ms": round(avg_total, 3),
        "error_count": n - successes,
        "scope_boundary": data["scope"],
    }

    fields = list(rows[0].keys()) if rows else []
    with (RESULT_DIR / "e2e_software_chain_raw_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (RESULT_DIR / "e2e_software_chain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "e2e_software_chain_errors.json").write_text(
        json.dumps([r for r in rows if not r["success"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (RESULT_DIR / "e2e_software_chain_traces.jsonl").open("w", encoding="utf-8") as f:
        for item in logs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = [
        "# MoonCARE S11 Software-chain E2E Baseline",
        "",
        f"- Cases: {n}",
        f"- Software-chain success rate: {success_rate:.3f}",
        f"- Route accuracy: {route_accuracy:.3f}",
        f"- Risk accuracy: {risk_accuracy:.3f}",
        f"- Average local chain latency: {avg_total:.3f} ms",
        f"- Errors: {n - successes}",
        "",
        "## Scope boundary",
        "",
        data["scope"],
        "",
        "该结果只能描述为“离线软件链路 E2E 基线”，不能替代真实硬件、生产数据库写回、",
        "真实 Tool 执行、用户完成/拒绝回流以及正式演示录像。",
    ]
    (RESULT_DIR / "e2e_software_chain_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("=" * 64)
    print("MoonCARE S11 Software-chain E2E Baseline")
    print("=" * 64)
    print(f"Cases: {n}")
    print(f"Software-chain Success Rate: {success_rate:.3f}")
    print(f"Route Accuracy: {route_accuracy:.3f}")
    print(f"Risk Accuracy: {risk_accuracy:.3f}")
    print(f"Average Local Chain Latency: {avg_total:.3f} ms")
    print(f"Errors: {n - successes}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
