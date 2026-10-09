# S03 后续运行时验证清单

当前先完成静态能力基线；如果还有时间，再补以下运行时验证。

- [ ] Menstrual：调用周期记录/预测接口，保存请求、响应和异常。
- [ ] Diary：创建一条测试日记，验证数据库写入及返回 ID。
- [ ] Music：调用推荐接口，检查推荐列表、来源和空库 fallback。
- [ ] Report：构造固定 PSST 输入，验证报告可重复生成。
- [ ] RAG：固定若干知识问题，记录检索卡片与最终答案一致性。
- [ ] Awareness MCP：有 MCP 环境时测试 recall / record 成功与失败分支。
- [ ] Breathing：明确记录当前为前端 Action Suggestion，而非后端 Tool。
- [ ] 统一 ToolExecutor：若后续重构，再新增统一调用日志、参数验证和失败恢复。
