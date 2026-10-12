# MoonCARE S12 异常与失败恢复基线

本轮覆盖 7 类软件层故障/缺失场景：

- LLM 超时：普通支持
- LLM 超时：明确行动请求
- LLM 超时：危机场景
- 知识链异常
- RAG 空结果
- 周期信息缺失
- 传感器数据缺失

运行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_failure_recovery.py
```

输出：

- `failure_recovery_raw_results.csv`
- `failure_recovery_summary.json`
- `failure_recovery_errors.json`
- `failure_recovery_report.md`

边界：这是软件层故障注入，不等同于真实 BLE、数据库或生产网络故障恢复。
