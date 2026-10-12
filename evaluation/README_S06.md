# MoonCARE S06 冷启动验证

本包新增一个独立、可测试的冷启动规则原型，并提供合成边界情景。

运行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_cold_start.py
```

指标：
- Cold-start Calculable Rate
- Missing-data Degradation Consistency
- Confidence-Completeness Consistency
- Over-inference Count
- Rule Consistency Accuracy

注意：这是合成规则单元测试，不代表真实用户或临床效果；当前不宣称已经全面接入线上 Agent 主链。
