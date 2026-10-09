# MoonCARE S03 Tool Capability Baseline

- 数据集：MoonCARE_S03_Tool_Capability_Baseline_v1 v1.0
- 检查项：14
- 源码证据检查通过：14/14 (100.0%)
- 存在真实可执行能力证据：9 项
- 仅 Action Suggestion：3 项
- Agent/MCP 直接 Tool Calling 证据：2 项
- 已确认架构缺口：2 项

## 关键结论

1. Menstrual、Diary、Music、Report 等均存在真实后端 API 或 Service 实现。
2. KnowledgeAgent 已存在本地 RAG 检索链路，但属于 Agent 内部能力，尚未证明已纳入统一 Tool Registry。
3. Awareness Memory 通过 JSON-RPC `tools/call` 调用 MCP，是当前最明确的 Tool Calling 证据。
4. Breathing / Diary / Music 在聊天 Agent 中主要表现为 Action Suggestion + 前端路由，不应描述成 Agent 已直接执行工具。
5. 当前源码未发现统一 ToolExecutor / ToolRegistry，后续应补充 Tool 选择、参数、执行、失败恢复的运行时测试。

> 边界：本轮是源码静态能力基线，不是最终 Tool Calling 正确率实验，也不是线上用户数据。