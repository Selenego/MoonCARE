# MoonCARE S07 个体基线迁移与鲁棒性验证

本实验使用**模拟纵向回放数据**，验证：
- 个人权重是否随有效观测量平滑增加；
- 是否覆盖群体参考、混合基线、个人占优三阶段；
- 缺失数据是否错误更新 baseline；
- 极端异常值是否把 baseline 带偏；
- 个人权重是否避免过快达到 1。

运行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_baseline_migration.py
```

本结果只能说明算法行为，不代表真实个体化效果提升。
