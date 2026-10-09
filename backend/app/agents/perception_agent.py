from app.utils.safety import contains_crisis_signal


class PerceptionAgent:
    def analyze(self, message: str, cycle_phase: str = None, sensor_data: dict = None) -> dict:
        """Analyze the user turn before any routing or model call."""
        sensor_data = sensor_data or {}
        text = (message or "").lower()

        risk_level = "low"
        high_keywords = [
            "没有意义", "撑不住了", "想消失", "我真的不行了", "崩溃到不行",
            "娌℃湁鎰忎箟", "鎾戜笉浣忎簡", "鎯虫秷澶", "鎴戠湡鐨勪笉琛屼簡", "宕╂簝鍒颁笉琛",
        ]
        medium_keywords = [
            "烦躁", "难受", "难过", "想哭", "崩溃", "焦虑", "紧张", "低落", "敏感", "委屈", "心里堵",
            "鐑﹁簛", "闅惧彈", "鎯冲摥", "宕╂簝", "鐒﹁檻", "绱", "浣庤惤", "鏁忔劅",
        ]

        support_context = self._extract_support_context(text, cycle_phase)

        # 组合式危机信号：单个词“危险”或“控制不住”过宽，
        # 但二者同时出现时应优先进入 Safety，避免高风险漏检。
        danger_markers = ("危险", "不安全", "出事")
        loss_of_control_markers = ("控制不住自己", "控制不住", "无法控制自己", "失控")
        has_danger_loss_of_control = (
            any(marker in text for marker in danger_markers)
            and any(marker in text for marker in loss_of_control_markers)
        )

        if contains_crisis_signal(text) or has_danger_loss_of_control:
            risk_level = "crisis"
        elif any(keyword in text for keyword in high_keywords):
            risk_level = "high"
        elif any(keyword in text for keyword in medium_keywords):
            risk_level = "medium"

        return {
            "risk_level": risk_level,
            "cycle_phase": cycle_phase or "经前期",
            "sensor_data": sensor_data,
            "emotion_summary": message,
            "support_context": support_context,
        }

    def _extract_support_context(self, text: str, cycle_phase: str = None) -> dict:
        """Extract lightweight menstrual, body, and emotion signals for support tone."""
        body_signal_keywords = {
            "pain": [
                "肚子疼", "肚子痛", "腹痛", "小腹痛", "痛经", "绞痛", "疼", "痛", "腰酸", "头痛",
                "鑲氬瓙鐥", "鑵圭棝", "鐥涚粡", "缁炵棝", "鐤", "鐥", "鑵伴吀", "澶寸棝",
            ],
            "bloating": ["腹胀", "胀", "水肿", "乳房胀", "鑵硅儉", "鑳", "姘磋偪", "涔虫埧鑳"],
            "fatigue": ["累", "疲惫", "乏力", "没精神", "困", "绱", "鐤叉儷", "涔忓姏", "娌＄簿绁", "鍥"],
            "sleep_change": ["睡不好", "失眠", "睡不着", "嗜睡", "鐫′笉濂", "澶辩湢", "鐫′笉鐫", "鍡滅潯"],
            "appetite": ["想吃", "没胃口", "暴食", "食欲", "鎯冲悆", "娌¤儍鍙", "鏆撮", "椋熸"],
        }
        emotion_signal_keywords = {
            "sad": ["难过", "低落", "伤心", "沮丧", "闅捐繃", "浣庤惤", "浼ゅ績", "娌抚"],
            "tearful": ["想哭", "容易哭", "委屈", "敏感", "鎯冲摥", "瀹规槗鍝", "濮斿眻", "鏁忔劅"],
            "irritable": ["烦躁", "易怒", "火大", "生气", "鐑﹁簛", "鏄撴", "鐏ぇ", "鐢熸皵"],
            "anxious": ["焦虑", "紧张", "慌", "不安", "担心", "鐒﹁檻", "绱у紶", "鎱", "涓嶅畨", "鎷呭績"],
            "helpless": ["无助", "撑不住", "不知道怎么办", "鏃犲姪", "鎾戜笉浣", "涓嶇煡閬撴"],
        }
        menstrual_keywords = [
            "经前", "月经前", "姨妈前", "经期", "月经", "例假", "姨妈", "来大姨妈", "黄体期",
            "pms", "pmdd", "缁忓墠", "鏈堢粡鍓", "濮ㄥ鍓", "缁忔湡", "鏈堢粡", "渚嬪亣",
            "濮ㄥ", "鏉ュぇ濮ㄥ", "榛勪綋鏈",
        ]
        menstrual_phases = {
            "经前期", "黄体期", "经期", "月经期", "luteal", "menstrual",
            "缁忓墠鏈", "榛勪綋鏈", "缁忔湡", "鏈堢粡鏈",
        }

        body_signals = self._matched_signal_names(text, body_signal_keywords)
        emotion_signals = self._matched_signal_names(text, emotion_signal_keywords)
        menstrual_related = (
            any(keyword in text for keyword in menstrual_keywords)
            or bool(body_signals)
            or (cycle_phase in menstrual_phases)
        )

        return {
            "menstrual_related": menstrual_related,
            "body_signals": body_signals,
            "emotion_signals": emotion_signals,
        }

    def _matched_signal_names(self, text: str, signal_keywords: dict) -> list[str]:
        """Return signal names whose keyword set appears in text."""
        return [
            signal
            for signal, keywords in signal_keywords.items()
            if any(keyword in text for keyword in keywords)
        ]
