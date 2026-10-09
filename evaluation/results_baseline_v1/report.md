# MoonCARE Router / Safety 基线评测

- 数据集：MoonCARE_S01_Frozen_Test_v1 v1.0
- 样本数：50
- Router Accuracy：0.560
- Router Macro-F1：0.606
- Mixed Intent Success：0.300
- Safety Precision：1.000
- Safety Recall：0.900
- Safety Priority Accuracy：0.900
- Router Error Count：22

## Per-class

- support: P=0.545, R=0.600, F1=0.571, n=10
- knowledge: P=0.500, R=0.714, F1=0.588, n=14
- action_support: P=1.000, R=0.188, F1=0.316, n=16
- intervention: P=1.000, R=0.900, F1=0.947, n=10

> 注意：本轮为规则层离线基线评测，使用 DummyAgent 避免外部 LLM API 对结果造成干扰。