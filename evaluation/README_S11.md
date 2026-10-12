# MoonCARE S11 软件链路 E2E 基线

这是赶比赛阶段的**离线软件链路 E2E 基线**，不是完整的硬件端到端实验。

验证链路：
- Perception
- Safety / Router
- Knowledge / Support / Intervention
- AgentService Action Suggestions
- 结构化 trace 与本地耗时

为了避免外部 LLM API 干扰：
- Support / Intervention 使用确定性离线 stub；
- Knowledge 使用项目当前真实 KnowledgeAgent 的本地知识卡检索；
- 不访问外部模型 API。

运行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_e2e_software_chain.py
```

输出：
- `evaluation/results/e2e_software_chain_raw_results.csv`
- `evaluation/results/e2e_software_chain_summary.json`
- `evaluation/results/e2e_software_chain_errors.json`
- `evaluation/results/e2e_software_chain_traces.jsonl`
- `evaluation/results/e2e_software_chain_report.md`

注意：正式 S11 仍需要真实硬件/生产写回/用户反馈回流和 3 次完整演示日志时，才可以宣称“完整 E2E 已完成”。
