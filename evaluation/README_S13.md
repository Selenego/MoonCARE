# MoonCARE S13 本地软件链性能基线

本实验用于快速量化当前离线 Agent 软件链的基础性能。

覆盖：
- Perception
- Router
- Knowledge/Support/Intervention 离线路径
- Action Suggestions
- 串行 P50/P95
- 1/5/10 workers 小规模并发

运行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_performance.py
```

注意：
本结果不包含真实外部 LLM、网络、数据库、BLE 或生产服务，因此只能写为
“本地离线软件链性能基线”，不能当作生产 API 延迟。
