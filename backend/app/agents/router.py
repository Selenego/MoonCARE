import logging
from typing import Any, Dict, Optional, Tuple

try:
    from app.agents.support_agent import SupportAgent
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.agents.intervention_agent import InterventionAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    SupportAgent = None
    KnowledgeAgent = None
    InterventionAgent = None

from app.utils.safety import SAFE_INTERVENTION_FALLBACK


logger = logging.getLogger(__name__)

SUPPORT_HANDOFF_FALLBACK = "我在。你可以继续说，我会先跟着你现在最明显的感受。"
ACTION_SUPPORT_HANDOFF_FALLBACK = "我先陪你把眼前这一步理一下。先别急着继续往前冲，我们把现在最小的一步找出来。"
LIGHT_COMFORT_FALLBACK = "好的，那我们不做了。允许自己在这几分钟里摆烂和发呆。我可以就这么陪你安静地待一会。你想吐槽什么，我都在听。"
KNOWLEDGE_FALLBACK_REPLY = (
    "这个问题我先给一个谨慎答复：经前/经期情绪突然变化，可能和激素水平变化、睡眠、疼痛、压力"
    "以及当天事件叠加有关，个体差异会很大。这里仅供参考，不代表诊断；如果波动明显影响生活，"
    "可以记录周期、情绪和身体症状，并咨询专业医生。你也可以告诉我它通常发生在经前几天，"
    "我帮你一起梳理规律。"
)


class Router:
    """Route chat turns through knowledge and emotional support agents."""

    def __init__(self) -> None:
        self._support: Optional[Any] = None
        self._knowledge: Optional[Any] = None
        self._intervention: Optional[Any] = None

    def _safe_intervention_fallback(self) -> str:
        """Return the crisis-safe intervention copy."""
        return SAFE_INTERVENTION_FALLBACK

    def _knowledge_fallback(self, message: str = "") -> str:
        """Return a cautious non-diagnostic answer when knowledge generation is unavailable."""
        compact = "".join((message or "").split())
        if "头晕" in compact:
            return (
                "经期头晕可能和疼痛、睡眠不足、进食少、出血量变化或身体紧张叠在一起有关。"
                "先坐下或躺一会儿，补一点温水；如果头晕明显、快晕倒、心慌或出血异常，建议尽快联系医生。"
                "以上仅供参考。"
            )
        if any(term in compact for term in ("肚子疼", "肚子痛", "腹痛", "痛经", "小腹痛")):
            return (
                "经期腹痛常见原因之一是子宫收缩带来的不适，也可能被睡眠、压力和受凉感放大。"
                "可以先热敷小腹、放慢活动强度；如果疼痛剧烈、和平时明显不同或伴随异常出血，建议咨询医生。以上仅供参考。"
            )
        return KNOWLEDGE_FALLBACK_REPLY

    @property
    def support(self) -> Optional[Any]:
        """Lazily initialize the support agent."""
        if self._support is None and SupportAgent is not None:
            try:
                self._support = SupportAgent()
            except Exception as exc:
                logger.exception("[Router] Failed to initialize SupportAgent: %s", exc)
                self._support = None
        return self._support

    @property
    def knowledge(self) -> Optional[Any]:
        """Lazily initialize the knowledge agent."""
        if self._knowledge is None and KnowledgeAgent is not None:
            try:
                self._knowledge = KnowledgeAgent()
            except Exception as exc:
                logger.exception("[Router] Failed to initialize KnowledgeAgent: %s", exc)
                self._knowledge = None
        return self._knowledge

    @property
    def intervention(self) -> Optional[Any]:
        """Lazily initialize the intervention agent."""
        if self._intervention is None and InterventionAgent is not None:
            try:
                self._intervention = InterventionAgent()
            except Exception as exc:
                logger.exception("[Router] Failed to initialize InterventionAgent: %s", exc)
                self._intervention = None
        return self._intervention

    def _looks_like_knowledge_question(self, message: str) -> bool:
        """Return whether the turn primarily asks for PMS or menstrual-cycle explanation."""
        compact = "".join((message or "").split())
        if not compact:
            return False

        # 明确的即时行动请求优先交给 action_support，而不是被知识关键词抢占。
        if self._looks_like_action_request(message):
            return False

        explanation_markers = (
            "为什么", "什么是", "是什么", "怎么回事", "原因", "会不会", "正常吗", "是不是",
            "有关系吗", "有关吗", "有关系么", "有关么", "和什么有关", "什么有关",
            "区别", "差别", "原理", "机制",
        )

        health_context_markers = (
            "经前", "经期", "月经", "姨妈", "pms", "pmdd", "激素", "周期",
            "情绪", "烦躁", "失眠", "焦虑", "抑郁", "难过", "易怒", "头痛", "腹痛", "疲劳", "乳房", "水肿",
            "运动", "饮食", "睡眠", "休息", "胸痛", "腰酸", "腰痛", "想吐", "恶心", "头晕", "心慌",
            "心悸", "出汗", "潮热", "盗汗", "抽筋", "痉挛", "腹胀", "便秘", "腹泻", "尿频", "尿急",
            "乳房胀痛", "胸胀", "乳痛", "小腹", "肚子", "痛经", "量多", "量少", "推迟", "提前",
            "不来", "停经", "闭经", "淋漓", "血块", "发黑", "褐色", "鲜红", "暗红",
            "脾气", "暴躁", "想哭", "低落", "压抑", "紧张", "不安", "敏感", "多疑",
            "记忆力", "注意力", "专注力", "困", "乏力", "没精神", "疲惫", "累", "嗜睡",
            "想吃", "食欲", "暴食", "胃口", "体重", "发胖", "浮肿",
        )

        # “经期可以运动吗”这类适宜性问题需要知识解释；
        # 但“可以陪我聊一会儿吗”不能因为“可以”被误判为 knowledge。
        menstrual_or_body_markers = (
            "经前", "经期", "月经", "姨妈", "pms", "pmdd", "激素", "周期",
            "运动", "饮食", "睡眠", "腹痛", "痛经", "头痛", "头晕", "出血", "乳房", "水肿", "疲劳",
        )

        lower_compact = compact.lower()
        has_health_context = any(marker in lower_compact for marker in health_context_markers)
        has_explanation_marker = any(marker in compact for marker in explanation_markers)

        if has_explanation_marker and has_health_context:
            return True

        if ("pms" in lower_compact or "pmdd" in lower_compact) and any(
            marker in compact for marker in ("是什么", "区别", "差别", "什么意思")
        ):
            return True

        if any(marker in lower_compact for marker in menstrual_or_body_markers):
            if (
                ("可以" in compact or "能" in compact)
                and (compact.endswith("吗") or "吗？" in compact or "吗?" in compact)
            ):
                return True

        return False

    def _looks_like_negative_feedback(self, message: str) -> bool:
        """Return whether the user is explicitly rejecting an action suggestion."""
        compact = "".join((message or "").split())
        if not compact:
            return False

        # 不使用裸子串“别”或“不行”，避免“别人/特别/区别”等误命中。
        negative_feedback_markers = (
            "不想动", "不想做", "没时间做", "做不到",
            "不要了", "不要这个", "不要建议", "不想听建议", "不想听这些",
            "别推荐", "别给建议", "别说了", "先别说", "不用了", "不必了",
            "够了", "停下", "别继续", "不想试", "不想练",
        )
        return any(marker in compact for marker in negative_feedback_markers)

    def _looks_like_action_request(self, message: str) -> bool:
        """Return whether the user is explicitly asking for immediate next steps."""
        compact = "".join((message or "").split())
        markers = (
            "我该怎么做", "该怎么做", "我该怎么办", "该怎么办", "怎么办", "怎么做", "怎么处理",
            "接下来怎么做", "接下来怎么办", "现在怎么办", "现在怎么做",
            "能做什么", "可以做什么", "马上能做", "马上可以做",
            "帮我想办法", "给我点建议", "给我几个步骤", "给我几个马上能做的步骤",
            "下一步", "最小的下一步", "简单的下一步",
            "照顾计划", "简单的照顾计划",
            "怎么面对", "怎么回复", "怎么沟通", "怎么开口",
            "怎么缓一缓", "先缓解", "怎么先缓解",
            "要不要先休息", "想先稳下来", "先稳下来",
        )
        return any(marker in compact for marker in markers)

    def _run_knowledge(self, message: str, state: Dict[str, Any]) -> Tuple[str, str]:
        """Run the knowledge agent or return a cautious knowledge fallback."""
        if self.knowledge is None:
            return self._knowledge_fallback(message), "knowledge_fallback"
        try:
            return self.knowledge.respond(message, state), "knowledge"
        except Exception as exc:
            logger.exception("[Router] KnowledgeAgent error: %s", exc)
            return self._knowledge_fallback(message), "knowledge_fallback"

    def _run_support(
        self,
        message: str,
        state: Dict[str, Any],
        support_intent: str = "support",
    ) -> Tuple[str, str]:
        """Run the support agent or return a brief human handoff copy."""
        support_state = dict(state or {})
        support_state["support_intent"] = support_intent
        if self.support is None:
            if support_intent == "action_support":
                return ACTION_SUPPORT_HANDOFF_FALLBACK, "action_support_fallback"
            return SUPPORT_HANDOFF_FALLBACK, "support_fallback"
        try:
            return self.support.respond(message, support_state), support_intent
        except Exception as exc:
            logger.exception("[Router] SupportAgent error: %s", exc)
            if support_intent == "action_support":
                return ACTION_SUPPORT_HANDOFF_FALLBACK, "action_support_fallback"
            return SUPPORT_HANDOFF_FALLBACK, "support_fallback"

    def route(self, message: str, state: Dict[str, Any], agent_mode: str = "auto") -> Tuple[str, str]:
        """Route one chat message across the existing three-mode system."""
        message = message or ""
        state = state or {}
        risk_level = state.get("risk_level", "low")
        agent_mode = agent_mode if agent_mode in {"auto", "support", "knowledge"} else "auto"

        if risk_level in {"high", "crisis"}:
            if self.intervention is None:
                return self._safe_intervention_fallback(), "intervention_fallback"
            try:
                return self.intervention.respond(message, state), "intervention"
            except Exception as exc:
                logger.exception("[Router] InterventionAgent error: %s", exc)
                return self._safe_intervention_fallback(), "intervention_fallback"

        # 负反馈检测：如果用户拒绝行动建议，切换到轻安抚陪伴模式
        if self._looks_like_negative_feedback(message):
            return LIGHT_COMFORT_FALLBACK, "light_comfort"

        if agent_mode == "knowledge":
            return self._run_knowledge(message, state)

        if agent_mode == "support":
            support_intent = "action_support" if self._looks_like_action_request(message) else "support"
            return self._run_support(message, state, support_intent=support_intent)

        # 自动模式下，明确的即时行动请求优先于知识解释。
        if self._looks_like_action_request(message):
            return self._run_support(message, state, support_intent="action_support")

        if self._looks_like_knowledge_question(message):
            return self._run_knowledge(message, state)

        return self._run_support(message, state, support_intent="support")
