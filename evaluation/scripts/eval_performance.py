from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

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

DATASET = EVAL_ROOT / "datasets" / "performance_cases_v1.json"
KNOWLEDGE_PATH = BACKEND_DIR / "app" / "data" / "knowledge_base.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_PER_CASE = 50
CONCURRENCY_LEVELS = [1, 5, 10]
CONCURRENCY_TASKS = 100


class OfflineSupportAgent:
    def respond(self, message: str, state: Dict[str, Any]) -> str:
        if state.get("support_intent") == "action_support":
            return "先做一个最小动作：慢慢呼吸几次，然后告诉我现在最难受的部分。"
        return "我在听。你可以继续说说现在最明显的感受。"


class OfflineInterventionAgent:
    def respond(self, message: str, state: Dict[str, Any]) -> str:
        return "优先保证安全，并联系可信任的人或当地紧急支持资源。"


def build_offline_knowledge_agent() -> KnowledgeAgent:
    agent = KnowledgeAgent.__new__(KnowledgeAgent)
    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as f:
        agent.knowledge = json.load(f)
    agent._embedding_available = False
    agent.embedder = None
    agent.llm = None
    return agent


def build_components():
    perception = PerceptionAgent()
    router = Router()
    router._support = OfflineSupportAgent()
    router._knowledge = build_offline_knowledge_agent()
    router._intervention = OfflineInterventionAgent()
    agent_service = AgentService()
    return perception, router, agent_service


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def canonicalize_route(route: str) -> str:
    route = str(route or "")
    aliases = {
        "light_comfort": "support",
        "support_fallback": "support",
        "knowledge_fallback": "knowledge",
        "action_support_fallback": "action_support",
        "intervention_fallback": "intervention",
    }
    if route in aliases:
        return aliases[route]
    for prefix in ("knowledge", "action_support", "intervention", "support"):
        if route.startswith(prefix):
            return prefix
    return route


def one_run(text: str, components=None) -> Dict[str, float]:
    perception, router, agent_service = components or build_components()

    t0 = time.perf_counter()
    state = perception.analyze(text, cycle_phase=None, sensor_data={})
    t1 = time.perf_counter()

    reply, raw_route = router.route(text, state, agent_mode="auto")
    t2 = time.perf_counter()

    route = canonicalize_route(raw_route)
    _ = agent_service._generate_action_suggestions(state, route)
    t3 = time.perf_counter()

    return {
        "perception_ms": (t1 - t0) * 1000,
        "route_ms": (t2 - t1) * 1000,
        "action_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
        "reply_ok": bool(str(reply or "").strip()),
    }


def summarize_metric(values: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3) if values else 0.0,
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3) if values else 0.0,
    }


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    components = build_components()

    rows = []
    for case in cases:
        for run_index in range(1, RUNS_PER_CASE + 1):
            try:
                result = one_run(case["text"], components)
                error = ""
            except Exception as exc:
                result = {
                    "perception_ms": 0.0,
                    "route_ms": 0.0,
                    "action_ms": 0.0,
                    "total_ms": 0.0,
                    "reply_ok": False,
                }
                error = f"{type(exc).__name__}: {exc}"

            rows.append({
                "case_id": case["id"],
                "run": run_index,
                **{k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()},
                "error": error,
            })

    ok_rows = [r for r in rows if not r["error"] and r["reply_ok"]]
    errors = [r for r in rows if r["error"] or not r["reply_ok"]]

    stage_summary = {}
    for metric in ("perception_ms", "route_ms", "action_ms", "total_ms"):
        stage_summary[metric] = summarize_metric([float(r[metric]) for r in ok_rows])

    concurrency_rows = []
    for workers in CONCURRENCY_LEVELS:
        started = time.perf_counter()
        success = 0
        failed = 0
        latencies = []

        def task(i: int):
            text = cases[i % len(cases)]["text"]
            # 每个并发任务使用独立组件，避免共享对象引入线程安全噪音。
            return one_run(text, build_components())

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task, i) for i in range(CONCURRENCY_TASKS)]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res["reply_ok"]:
                        success += 1
                        latencies.append(res["total_ms"])
                    else:
                        failed += 1
                except Exception:
                    failed += 1

        wall_seconds = time.perf_counter() - started
        concurrency_rows.append({
            "workers": workers,
            "tasks": CONCURRENCY_TASKS,
            "success": success,
            "failed": failed,
            "error_rate": round(failed / CONCURRENCY_TASKS, 4),
            "wall_time_s": round(wall_seconds, 3),
            "throughput_tasks_per_s": round(CONCURRENCY_TASKS / wall_seconds, 2) if wall_seconds else 0.0,
            "p50_task_ms": round(percentile(latencies, 0.50), 3),
            "p95_task_ms": round(percentile(latencies, 0.95), 3),
        })

    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "scope": data["scope"],
        "runs_per_case": RUNS_PER_CASE,
        "total_serial_runs": len(rows),
        "serial_success_rate": round(len(ok_rows) / len(rows), 4) if rows else 0.0,
        "serial_error_count": len(errors),
        "stage_latency": stage_summary,
        "concurrency": concurrency_rows,
    }

    with (RESULT_DIR / "performance_raw_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (RESULT_DIR / "performance_concurrency.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fieldnames = list(concurrency_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(concurrency_rows)

    (RESULT_DIR / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "performance_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = stage_summary["total_ms"]
    print("=" * 64)
    print("MoonCARE S13 Local Performance Baseline")
    print("=" * 64)
    print(f"Serial Runs: {len(rows)}")
    print(f"Success Rate: {summary['serial_success_rate']:.3f}")
    print(f"Local-chain P50: {total['p50_ms']:.3f} ms")
    print(f"Local-chain P95: {total['p95_ms']:.3f} ms")
    print(f"Errors: {len(errors)}")
    print("-" * 64)
    for item in concurrency_rows:
        print(
            f"Concurrency {item['workers']}: "
            f"success={item['success']}/{item['tasks']}, "
            f"error_rate={item['error_rate']:.3f}, "
            f"throughput={item['throughput_tasks_per_s']:.2f} tasks/s, "
            f"P95={item['p95_task_ms']:.3f} ms"
        )
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
