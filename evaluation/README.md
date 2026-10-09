# MoonCARE Evaluation Pack v1

这是用于明日前快速完成 S01 / S02 / S04 基线交付的最小评测包。

## 文件

- `datasets/test_v1.json`：50 条冻结测试样例
- `datasets/label_guideline.md`：标注规范
- `datasets/dispute_log.csv`：第二标注者分歧记录模板
- `scripts/eval_router_safety.py`：Router + Safety 自动评测脚本
- `results/`：脚本运行后自动生成结果

## 放置位置

把整个 `evaluation` 文件夹复制到 MoonCARE 项目根目录：

```text
MoonCARE/
├─ backend/
├─ frontend/
├─ evaluation/
│  ├─ datasets/
│  ├─ scripts/
│  └─ results/
└─ ...
```

## 运行

在 MoonCARE 根目录执行：

```powershell
.\.venv\Scripts\python.exe .\evaluation\scripts\eval_router_safety.py
```

脚本不会调用真实 LLM API。
它会给 Router 注入 DummyAgent，只评测规则层路由与 Perception/Safety。

## 输出

运行后查看：

- `evaluation/results/raw_results.csv`
- `evaluation/results/summary.json`
- `evaluation/results/confusion_matrix.csv`
- `evaluation/results/top_errors.json`
- `evaluation/results/safety_false_negatives.json`
- `evaluation/results/report.md`

## 重要边界

1. 测试样例全部是 synthetic_test，不是真实用户数据。
2. 当前 test_v1 是第一轮标注草案，还没有完成“两名标注者独立复核”。
3. 不要把本脚本的结果描述成 LLM 回复质量，它只评估当前规则 Router / Safety 基线。
