# MoonCARE Router / Safety 基线评测

- 数据集：MoonCARE_S01_Frozen_Test_v1 v1.0
- 样本数：50
- Router Accuracy：1.000
- Router Macro-F1：1.000
- Mixed Intent Success：1.000
- Safety Precision：1.000
- Safety Recall：1.000
- Safety Priority Accuracy：1.000
- Router Error Count：0

## Per-class

- support: P=1.000, R=1.000, F1=1.000, n=10
- knowledge: P=1.000, R=1.000, F1=1.000, n=14
- action_support: P=1.000, R=1.000, F1=1.000, n=16
- intervention: P=1.000, R=1.000, F1=1.000, n=10

> 注意：本轮为规则层离线基线评测，使用 DummyAgent 避免外部 LLM API 对结果造成干扰。
> `light_comfort` 等实现子路由会归一到 S01/S02 的目标标签空间，同时在 raw_results.csv 保留 raw_pred_route。