# MoonCARE S05 RAG Retrieval Baseline

- 数据集：MoonCARE_S05_RAG_Retrieval_Baseline_v1 v1.0
- 测试样例：30
- 检索模式：KnowledgeAgent 关键词 fallback（强制关闭向量检索与外部 API）
- Top-1 Hit Rate：0.767
- Top-3 Hit Rate：0.767
- MRR@3：0.767
- Retrieval Signal Coverage：1.000
- Top-1 错误：7
- Top-3 未命中：7

## 实验边界

本轮只验证“问题 → 本地知识卡片”的检索表现，不调用外部 LLM。
因此本结果不能被描述为最终回答准确率、幻觉率或医学正确率。

## 后续

如接入可用 Embedding / LLM API，可在相同数据集上追加向量检索与最终回答 groundedness 对比。