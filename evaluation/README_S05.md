# MoonCARE S05 RAG Retrieval Baseline v1

本实验在**不调用任何外部 LLM / Embedding API**的情况下，
复用项目 `KnowledgeAgent` 自身的关键词 fallback 检索逻辑，
评测固定问题能否命中期望知识卡。

## 放置

将压缩包中的 `evaluation` 文件夹合并到 MoonCARE 项目根目录现有的 `evaluation` 中。

应新增：

- `evaluation/datasets/rag_cases_v1.json`
- `evaluation/scripts/eval_rag.py`
- `evaluation/README_S05.md`

## 运行

在 MoonCARE 根目录执行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_rag.py
```

## 输出

会生成：

- `evaluation/results/rag_raw_results.csv`
- `evaluation/results/rag_summary.json`
- `evaluation/results/rag_top1_errors.json`
- `evaluation/results/rag_top3_misses.json`
- `evaluation/results/rag_report.md`

## 指标

- Top-1 Hit Rate：第 1 个检索结果是否为 gold 知识卡。
- Top-3 Hit Rate：gold 知识卡是否出现在前 3 个结果。
- MRR@3：正确知识卡越靠前得分越高。
- Retrieval Signal Coverage：Top-1 检索分数是否大于 0。

## 边界

这是**检索基线**，不是最终回答准确率，也不是幻觉率实验。
数据为合成改写测试问题，不是真实用户数据。
