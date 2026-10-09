# MoonCARE S05 Unseen Paraphrase Challenge Set

- 数据集：MoonCARE_S05_RAG_Unseen_Paraphrase_Set_v1 v1.0
- 测试样例：20
- 检索模式：KnowledgeAgent 关键词 fallback（关闭向量检索与外部 API）
- Top-1 Hit Rate：0.750
- Top-3 Hit Rate：0.850
- MRR@3：0.792
- Retrieval Signal Coverage：1.000
- Top-1 错误：5
- Top-3 未命中：3

## 解释边界

该集合是在第一轮优化完成后构造的20条新改写问题，用于做额外回归/挑战测试。它没有参与前述30条训练式调参，但由于是在优化后生成，不能声称为预先冻结、独立盲测或真实用户测试。

因此更准确的描述是“优化后新增20条未参与第一轮调参的合成改写挑战样例”，
不能描述成真实用户集、独立盲测集或预先冻结的 holdout。