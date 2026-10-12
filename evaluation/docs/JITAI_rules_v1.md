# MoonCARE JITAI v1 规则说明

决策顺序固定为：

**Safety → Need → Availability → Burden → Preference**

1. Safety：`risk_level` 为 `high/crisis` 时阻断普通推荐链，返回 `safety_support`。
2. Need：用户主动求助、困扰度 >= 4、或出现生理偏离，至少满足一项才进入后续决策；数据不足且不是主动求助时不主动干预。
3. Availability：用户当前不方便时返回 `no_intervention`。
4. Burden：针对主动推送设置 120 分钟冷却时间；24 小时最多 3 次主动提醒。用户主动求助不受该频控阻断。
5. Preference：在当前需求类型允许的低负担选项里优先选择用户偏好；无合适偏好时采用可解释默认项。

需求类型与候选：
- `calm` → breathing / music / emotional_support
- `reflect` → diary / emotional_support
- `knowledge` → knowledge / emotional_support
- `support` → emotional_support / breathing / music

本规则是比赛阶段的可解释 JITAI v1 工程实现，不表示临床有效性。
