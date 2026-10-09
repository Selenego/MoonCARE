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

from app.agents.knowledge_agent import KnowledgeAgent


DATASET_PATH = EVAL_ROOT / "datasets" / "rag_cases_v1.json"
KNOWLEDGE_PATH = BACKEND_DIR / "app" / "data" / "knowledge_base.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def build_offline_agent() -> KnowledgeAgent:
    """
    构造一个不初始化 LLM / Embedding API 的 KnowledgeAgent，
    但复用项目当前 KnowledgeAgent 自身的真实关键词检索实现。
    """
    agent = KnowledgeAgent.__new__(KnowledgeAgent)
    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as f:
        agent.knowledge = json.load(f)

    # 强制使用项目自己的 keyword fallback，避免任何外部 API。
    agent._embedding_available = False
    agent.embedder = None
    agent.llm = None
    return agent


def reciprocal_rank(ids: list[str], gold_id: str) -> float:
    for rank, item_id in enumerate(ids, start=1):
        if item_id == gold_id:
            return 1.0 / rank
    return 0.0


def main() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]

    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KNOWLEDGE_PATH}")

    agent = build_offline_agent()
    kb_by_id = {str(item.get("id")): item for item in agent.knowledge}

    rows = []
    for case in cases:
        gold_id = case["gold_id"]
        if gold_id not in kb_by_id:
            raise ValueError(
                f"{case['id']} gold_id={gold_id!r} not found in current knowledge_base.json"
            )

        retrieved = agent._retrieve_top_k(case["query"], k=3)
        retrieved_ids = [str(item.get("id")) for _, item in retrieved]
        scores = [float(score) for score, _ in retrieved]

        top1_hit = bool(retrieved_ids and retrieved_ids[0] == gold_id)
        top3_hit = gold_id in retrieved_ids
        rr = reciprocal_rank(retrieved_ids, gold_id)
        retrieval_has_signal = bool(scores and scores[0] > 0)

        rows.append(
            {
                **case,
                "gold_question": kb_by_id[gold_id].get("question", ""),
                "pred_top1_id": retrieved_ids[0] if retrieved_ids else "",
                "pred_top1_question": retrieved[0][1].get("question", "") if retrieved else "",
                "top1_score": scores[0] if scores else 0.0,
                "top3_ids": "|".join(retrieved_ids),
                "top3_scores": "|".join(f"{s:.3f}" for s in scores),
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "reciprocal_rank": rr,
                "retrieval_has_signal": retrieval_has_signal,
            }
        )

    n = len(rows)
    top1 = safe_div(sum(1 for r in rows if r["top1_hit"]), n)
    top3 = safe_div(sum(1 for r in rows if r["top3_hit"]), n)
    mrr3 = safe_div(sum(r["reciprocal_rank"] for r in rows), n)
    coverage = safe_div(sum(1 for r in rows if r["retrieval_has_signal"]), n)

    summary = {
        "dataset": data["dataset_name"],
        "version": data["version"],
        "n_cases": n,
        "retrieval_mode": "KnowledgeAgent keyword fallback (embedding disabled)",
        "top1_hit_rate": top1,
        "top3_hit_rate": top3,
        "mrr_at_3": mrr3,
        "retrieval_signal_coverage": coverage,
        "top1_error_count": sum(1 for r in rows if not r["top1_hit"]),
        "top3_miss_count": sum(1 for r in rows if not r["top3_hit"]),
        "boundary": (
            "该实验仅评测本地知识卡检索，不调用外部 LLM，也不评测最终生成答案、"
            "医学正确性或真实用户泛化性能。"
        ),
    }

    fields = [
        "id", "query", "gold_id", "gold_question",
        "pred_top1_id", "pred_top1_question", "top1_score",
        "top3_ids", "top3_scores", "top1_hit", "top3_hit",
        "reciprocal_rank", "retrieval_has_signal", "source_type",
    ]
    with open(RESULT_DIR / "rag_raw_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (RESULT_DIR / "rag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top1_errors = [r for r in rows if not r["top1_hit"]]
    (RESULT_DIR / "rag_top1_errors.json").write_text(
        json.dumps(top1_errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top3_misses = [r for r in rows if not r["top3_hit"]]
    (RESULT_DIR / "rag_top3_misses.json").write_text(
        json.dumps(top3_misses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# MoonCARE S05 RAG Retrieval Baseline",
        "",
        f"- 数据集：{data['dataset_name']} v{data['version']}",
        f"- 测试样例：{n}",
        "- 检索模式：KnowledgeAgent 关键词 fallback（强制关闭向量检索与外部 API）",
        f"- Top-1 Hit Rate：{top1:.3f}",
        f"- Top-3 Hit Rate：{top3:.3f}",
        f"- MRR@3：{mrr3:.3f}",
        f"- Retrieval Signal Coverage：{coverage:.3f}",
        f"- Top-1 错误：{summary['top1_error_count']}",
        f"- Top-3 未命中：{summary['top3_miss_count']}",
        "",
        "## 实验边界",
        "",
        "本轮只验证“问题 → 本地知识卡片”的检索表现，不调用外部 LLM。",
        "因此本结果不能被描述为最终回答准确率、幻觉率或医学正确率。",
        "",
        "## 后续",
        "",
        "如接入可用 Embedding / LLM API，可在相同数据集上追加向量检索与最终回答 groundedness 对比。",
    ]
    (RESULT_DIR / "rag_report.md").write_text("\n".join(report), encoding="utf-8")

    print("=" * 64)
    print("MoonCARE S05 RAG Retrieval Baseline")
    print("=" * 64)
    print(f"Cases: {n}")
    print(f"Top-1 Hit Rate: {top1:.3f}")
    print(f"Top-3 Hit Rate: {top3:.3f}")
    print(f"MRR@3: {mrr3:.3f}")
    print(f"Retrieval Signal Coverage: {coverage:.3f}")
    print(f"Top-1 Errors: {summary['top1_error_count']}")
    print(f"Top-3 Misses: {summary['top3_miss_count']}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
