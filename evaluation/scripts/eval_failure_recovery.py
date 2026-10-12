from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = EVAL_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agents.perception_agent import PerceptionAgent
from app.agents.knowledge_agent import KnowledgeAgent
from app.services.agent_service import AgentService
from app.utils.safety import SAFE_INTERVENTION_FALLBACK

DATASET = EVAL_ROOT / "datasets" / "failure_cases_v1.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FAKE_SUCCESS_MARKERS = (
    "已经为你完成",
    "已经执行成功",
    "操作已完成",
    "已帮你保存",
    "已记录完成",
)


def has_fake_success(text: str) -> bool:
    text = text or ""
    return any(marker in text for marker in FAKE_SUCCESS_MARKERS)


def build_empty_knowledge_agent() -> KnowledgeAgent:
    agent = KnowledgeAgent.__new__(KnowledgeAgent)
    agent.knowledge = []
    agent._embedding_available = False
    agent.embedder = None
    agent.llm = None
    return agent


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    service = AgentService()
    perception = PerceptionAgent()
    empty_knowledge = build_empty_knowledge_agent()

    rows = []

    for case in data["cases"]:
        started = time.perf_counter()
        fault = case["fault"]
        output = ""
        detected = True
        recovered = False
        detail = ""

        try:
            if fault == "llm_timeout_support":
                state = {"risk_level": "low", "agent_mode": "auto"}
                output = service._timeout_fallback("我今天有点难过，想找个人说说话。", state)
                recovered = bool(output.strip())
                detail = "used AgentService._timeout_fallback"

            elif fault == "llm_timeout_action_request":
                state = {"risk_level": "medium", "agent_mode": "auto"}
                output = service._timeout_fallback("我现在很烦躁，我该怎么做？", state)
                recovered = bool(output.strip()) and any(
                    marker in output for marker in ("先", "然后", "接着", "可以")
                )
                detail = "used action-request degraded reply"

            elif fault == "llm_timeout_crisis":
                state = {"risk_level": "crisis", "agent_mode": "auto"}
                output = service._timeout_fallback("我真的撑不住了，想消失。", state)
                recovered = output == SAFE_INTERVENTION_FALLBACK
                detail = "safety fallback must override normal fallback"

            elif fault == "unexpected_error_knowledge":
                state = {"risk_level": "low", "agent_mode": "knowledge"}
                output = service._soft_error_fallback("为什么经前会情绪波动？", state)
                recovered = bool(output.strip())
                detail = "used AgentService._soft_error_fallback"

            elif fault == "rag_empty_result":
                output = empty_knowledge.respond(
                    "为什么经前会情绪波动？",
                    {"risk_level": "low", "cycle_phase": "未知"},
                )
                recovered = bool(output.strip())
                detail = "KnowledgeAgent had zero local cards and no LLM"

            elif fault == "missing_cycle_info":
                state = perception.analyze(
                    "今天有点难受。",
                    cycle_phase=None,
                    sensor_data={},
                )
                output = str(state.get("cycle_phase"))
                recovered = output in {"未知", "unknown", "None", ""}
                detail = f"cycle_phase={output!r}; expected explicit unknown/missing"

            elif fault == "missing_sensor_data":
                state = perception.analyze(
                    "今天状态还可以。",
                    cycle_phase="经期",
                    sensor_data=None,
                )
                output = json.dumps(state, ensure_ascii=False, default=str)
                recovered = isinstance(state.get("sensor_data"), dict)
                detail = "sensor_data None should degrade to empty dict"

            else:
                detected = False
                detail = "unknown injected fault"

        except Exception as exc:
            output = f"{type(exc).__name__}: {exc}"
            recovered = False
            detail = "exception escaped recovery boundary"

        elapsed_ms = (time.perf_counter() - started) * 1000
        fake_success = has_fake_success(output)
        if fake_success:
            recovered = False

        rows.append({
            "id": case["id"],
            "fault": fault,
            "description": case["description"],
            "detected": detected,
            "recovered": recovered,
            "fake_success": fake_success,
            "elapsed_ms": round(elapsed_ms, 3),
            "detail": detail,
            "output_preview": output.replace("\n", " ")[:180],
        })

    n = len(rows)
    detected_n = sum(1 for r in rows if r["detected"])
    recovered_n = sum(1 for r in rows if r["recovered"])
    fake_success_n = sum(1 for r in rows if r["fake_success"])
    detection_rate = detected_n / n if n else 0.0
    recovery_rate = recovered_n / n if n else 0.0
    avg_recovery_ms = (
        sum(r["elapsed_ms"] for r in rows) / n if n else 0.0
    )

    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "n_cases": n,
        "fault_detection_rate": detection_rate,
        "recovery_success_rate": recovery_rate,
        "average_local_recovery_ms": round(avg_recovery_ms, 3),
        "fake_success_count": fake_success_n,
        "failure_count": n - recovered_n,
        "boundary": data["boundary"],
    }

    fields = list(rows[0].keys())
    with (RESULT_DIR / "failure_recovery_raw_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (RESULT_DIR / "failure_recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "failure_recovery_errors.json").write_text(
        json.dumps([r for r in rows if not r["recovered"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# MoonCARE S12 Failure Recovery Baseline",
        "",
        f"- Cases: {n}",
        f"- Fault Detection Rate: {detection_rate:.3f}",
        f"- Recovery Success Rate: {recovery_rate:.3f}",
        f"- Average Local Recovery Time: {avg_recovery_ms:.3f} ms",
        f"- Fake Success Count: {fake_success_n}",
        f"- Failures: {n - recovered_n}",
        "",
        "## Boundary",
        "",
        data["boundary"],
    ]
    (RESULT_DIR / "failure_recovery_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("=" * 64)
    print("MoonCARE S12 Failure Recovery Baseline")
    print("=" * 64)
    print(f"Cases: {n}")
    print(f"Fault Detection Rate: {detection_rate:.3f}")
    print(f"Recovery Success Rate: {recovery_rate:.3f}")
    print(f"Average Local Recovery Time: {avg_recovery_ms:.3f} ms")
    print(f"Fake Success Count: {fake_success_n}")
    print(f"Failures: {n - recovered_n}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
