from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = EVAL_ROOT.parent
DATASET = EVAL_ROOT / "datasets" / "tool_cases_v1.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def check_case(case: dict) -> dict:
    check_type = case["check_type"]
    observed = False
    evidence = []

    if check_type == "file_markers":
        path = PROJECT_ROOT / case["source_file"]
        if path.exists():
            text = read_text(path)
            all_ok = all(marker in text for marker in case.get("all_markers", []))
            any_markers = case.get("any_markers", [])
            any_ok = True if not any_markers else any(marker in text for marker in any_markers)
            observed = all_ok and any_ok

            if observed:
                evidence.append(f"matched markers in {case['source_file']}")
            else:
                missing = [m for m in case.get("all_markers", []) if m not in text]
                if missing:
                    evidence.append("missing all_markers: " + " | ".join(missing))
                if any_markers and not any_ok:
                    evidence.append("none of any_markers matched")
        else:
            evidence.append(f"missing file: {case['source_file']}")

    elif check_type == "file_absence":
        path = PROJECT_ROOT / case["source_file"]
        observed = not path.exists()
        evidence.append(
            f"{'absent as expected' if observed else 'unexpectedly exists'}: {case['source_file']}"
        )

    elif check_type == "recursive_absence":
        root = PROJECT_ROOT / case["search_root"]
        patterns = case.get("patterns", [])
        found = []

        if root.exists():
            for path in root.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                text = read_text(path)
                for pattern in patterns:
                    if pattern in text:
                        found.append(f"{path.relative_to(PROJECT_ROOT)} :: {pattern}")

        observed = len(found) == 0
        if observed:
            evidence.append("no unified ToolExecutor/ToolRegistry markers found")
        else:
            evidence.extend(found[:10])

    else:
        evidence.append(f"unsupported check_type: {check_type}")

    return {
        **case,
        "check_pass": observed,
        "evidence": "; ".join(evidence),
    }


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = [check_case(case) for case in data["cases"]]

    fieldnames = [
        "id", "name", "feature", "expected_type", "check_pass",
        "counts_as_real_executable", "agent_direct_tool_call",
        "source_file", "evidence", "notes",
    ]
    with open(RESULT_DIR / "tool_eval.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for r in rows if r["check_pass"])
    executable = [r for r in rows if r["check_pass"] and r.get("counts_as_real_executable")]
    action_only = [r for r in rows if r["check_pass"] and r["expected_type"] == "action_suggestion"]
    direct_tools = [r for r in rows if r["check_pass"] and r.get("agent_direct_tool_call")]
    gaps = [r for r in rows if r["check_pass"] and r["expected_type"] == "gap_confirmed"]

    by_type = Counter(r["expected_type"] for r in rows if r["check_pass"])
    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "audit_scope": "source_code_static_audit",
        "n_cases": len(rows),
        "evidence_check_pass_count": passed,
        "evidence_check_pass_rate": passed / len(rows) if rows else 0.0,
        "real_executable_evidence_count": len(executable),
        "action_suggestion_only_count": len(action_only),
        "agent_direct_tool_call_evidence_count": len(direct_tools),
        "confirmed_gap_count": len(gaps),
        "passed_by_type": dict(by_type),
        "important_boundary": (
            "本结果验证的是源码中能力/接口/调用证据是否存在，不等同于真实运行时 Tool Calling 成功率。"
            "日记、音乐、周期等后端能力存在，但聊天 Agent 返回 Action Suggestion 不等于 Agent 已直接执行这些工具。"
        ),
    }

    (RESULT_DIR / "tool_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# MoonCARE S03 Tool Capability Baseline",
        "",
        f"- 数据集：{data['dataset_name']} v{data['version']}",
        f"- 检查项：{len(rows)}",
        f"- 源码证据检查通过：{passed}/{len(rows)} ({summary['evidence_check_pass_rate']:.1%})",
        f"- 存在真实可执行能力证据：{len(executable)} 项",
        f"- 仅 Action Suggestion：{len(action_only)} 项",
        f"- Agent/MCP 直接 Tool Calling 证据：{len(direct_tools)} 项",
        f"- 已确认架构缺口：{len(gaps)} 项",
        "",
        "## 关键结论",
        "",
        "1. Menstrual、Diary、Music、Report 等均存在真实后端 API 或 Service 实现。",
        "2. KnowledgeAgent 已存在本地 RAG 检索链路，但属于 Agent 内部能力，尚未证明已纳入统一 Tool Registry。",
        "3. Awareness Memory 通过 JSON-RPC `tools/call` 调用 MCP，是当前最明确的 Tool Calling 证据。",
        "4. Breathing / Diary / Music 在聊天 Agent 中主要表现为 Action Suggestion + 前端路由，不应描述成 Agent 已直接执行工具。",
        "5. 当前源码未发现统一 ToolExecutor / ToolRegistry，后续应补充 Tool 选择、参数、执行、失败恢复的运行时测试。",
        "",
        "> 边界：本轮是源码静态能力基线，不是最终 Tool Calling 正确率实验，也不是线上用户数据。",
    ]
    (RESULT_DIR / "tool_report.md").write_text("\n".join(report), encoding="utf-8")

    print("=" * 64)
    print("MoonCARE S03 Tool Capability Baseline")
    print("=" * 64)
    print(f"Cases: {len(rows)}")
    print(f"Evidence checks passed: {passed}/{len(rows)}")
    print(f"Evidence pass rate: {summary['evidence_check_pass_rate']:.3f}")
    print(f"Real executable evidence: {len(executable)}")
    print(f"Action suggestion only: {len(action_only)}")
    print(f"Direct Tool Calling evidence: {len(direct_tools)}")
    print(f"Confirmed gaps: {len(gaps)}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
