# MoonCARE S09 JITAI v1

本包补充一个最小、显式、可解释的规则型 JITAI 决策模块，并提供 30 个合成单元测试情景。

## 合并到项目

将压缩包里的 `backend` 和 `evaluation` 两个文件夹分别合并到 MoonCARE 项目根目录。

新增：

- `backend/app/services/jitai_policy_service.py`
- `evaluation/datasets/JITAI_scenarios_v1.json`
- `evaluation/scripts/eval_jitai.py`
- `evaluation/docs/JITAI_rules_v1.md`

## 运行

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_jitai.py
```

## 输出

- `evaluation/results/jitai_raw_results.csv`
- `evaluation/results/jitai_summary.json`
- `evaluation/results/jitai_errors.json`
- `evaluation/results/jitai_rule_coverage.json`
- `evaluation/results/jitai_report.md`

## 注意

这是规则一致性 / 情景覆盖单元测试，不是临床效果实验。
如果当前聊天主链还没有调用 `JITAIPolicyService`，就不能写成“线上主链已全面接入 JITAI”；
更准确的说法是“已形成可测试的 JITAI v1 决策模块，主链集成为下一步”。
