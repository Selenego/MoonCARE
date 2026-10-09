# MoonCARE S03 Tool Capability Baseline v1

这是一套 S03 的“源码能力审计”实验包。

目标不是证明所有能力已经形成完整 Tool Calling，而是先区分：

- 真实后端 API / Service
- Agent 内部 RAG
- MCP Tool Calling
- 仅 Action Suggestion
- 当前架构缺口

## 放置方式

把压缩包中的 `evaluation` 文件夹合并到 MoonCARE 项目根目录下已有的 `evaluation`。

最终至少应存在：

- `evaluation/datasets/tool_cases_v1.json`
- `evaluation/scripts/eval_tools.py`

## 运行

在 MoonCARE 项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_tools.py
```

## 输出

脚本会生成：

- `evaluation/results/tool_eval.csv`
- `evaluation/results/tool_summary.json`
- `evaluation/results/tool_report.md`

## 实验边界

本轮属于 S03 的“能力基线 / 源码审计”，不是最终运行时 Tool Calling 正确率实验。

不能把：
- “存在 API”写成“Agent 已成功调用 Tool”
- “Action Suggestion”写成“Tool 已执行”
- “源码检查通过率”写成“真实用户调用成功率”
