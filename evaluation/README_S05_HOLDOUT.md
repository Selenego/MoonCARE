# S05 Unseen Paraphrase Challenge Set v1

这是第一轮 RAG 优化完成后新增的 20 条合成改写问题。

用途：检查优化后的关键词检索是否只对原 30 条回归集有效，还是对新的措辞也能保持表现。

注意：
- 这些问题没有加入原 30 条 `rag_cases_v1.json`；
- 但它们是在优化完成之后才生成，所以不能称为“预先冻结的独立 holdout”；
- 更准确的名称是“优化后未见改写挑战集”。

运行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_rag_holdout.py
```

输出：

- `evaluation/results/rag_holdout_raw_results.csv`
- `evaluation/results/rag_holdout_summary.json`
- `evaluation/results/rag_holdout_errors.json`
- `evaluation/results/rag_holdout_report.md`
