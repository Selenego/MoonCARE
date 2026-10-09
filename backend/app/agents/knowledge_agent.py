import json
import logging
import math
from pathlib import Path
from typing import Any, AsyncGenerator

from app.agents.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.utils.prompt_loader import render_prompt


logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """Answer menstrual and PMS knowledge questions through a local RAG base."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        knowledge_path = base_dir / "data" / "knowledge_base.json"
        index_path = base_dir / "data" / "knowledge_embeddings.json"

        self.llm: LLMService | None = None
        self.embedder: EmbeddingService | None = None
        self._embedding_available = False
        self.knowledge = self._load_knowledge_base(knowledge_path)
        self._attach_valid_embeddings(index_path)
        self._embedding_available = any(item.get("embedding") for item in self.knowledge)

        try:
            self.llm = LLMService()
        except Exception as exc:
            logger.info("[KnowledgeAgent] LLM unavailable, local card fallback enabled: %s", exc)
            self.llm = None

        try:
            self.embedder = EmbeddingService() if self._embedding_available and self.llm is not None else None
            if not self.embedder:
                self._embedding_available = False
                logger.info("[KnowledgeAgent] Vector retrieval disabled, using keyword retrieval")
        except Exception as exc:
            logger.info("[KnowledgeAgent] Embedding unavailable, using keyword retrieval: %s", exc)

    def _load_knowledge_base(self, knowledge_path: Path) -> list[dict[str, Any]]:
        """Load readable knowledge cards from the source knowledge base."""
        try:
            with knowledge_path.open("r", encoding="utf-8") as file:
                knowledge = json.load(file)
            logger.info("[KnowledgeAgent] Loaded knowledge base with %s entries", len(knowledge))
            return knowledge
        except Exception as exc:
            logger.exception("[KnowledgeAgent] Failed to load knowledge base: %s", exc)
            return []

    def _attach_valid_embeddings(self, index_path: Path) -> None:
        """Attach embeddings only when the index text matches the source card."""
        if not self.knowledge or not index_path.exists():
            return

        try:
            with index_path.open("r", encoding="utf-8") as file:
                index = json.load(file)
        except Exception as exc:
            logger.info("[KnowledgeAgent] Embedding index unavailable: %s", exc)
            return

        indexed_by_id = {item.get("id"): item for item in index if item.get("id")}
        attached = 0
        for item in self.knowledge:
            indexed = indexed_by_id.get(item.get("id"))
            if not indexed:
                continue
            if indexed.get("question") != item.get("question") or indexed.get("answer") != item.get("answer"):
                continue
            embedding = indexed.get("embedding")
            if isinstance(embedding, list) and embedding:
                item["embedding"] = embedding
                attached += 1

        logger.info("[KnowledgeAgent] Attached %s valid knowledge embeddings", attached)

    def _normalize_retrieval_text(self, text: str) -> str:
        """Normalize common paraphrases before lightweight keyword retrieval."""
        normalized = (text or "").lower()

        replacements = (
            ("来月经前", "经前"),
            ("月经前", "经前"),
            ("来姨妈前", "经前"),
            ("姨妈前", "经前"),
            ("月经期间", "经期"),
            ("来月经时", "经期"),
            ("情绪起伏", "情绪波动"),
            ("情绪不好", "情绪波动"),
            ("心情差", "情绪波动"),
            ("少喝咖啡", "咖啡因"),
            ("咖啡", "咖啡因"),
            ("少吃盐", "高盐"),
            ("少喝酒", "酒精"),
            ("出血特别多", "月经过多"),
            ("出血量特别多", "月经过多"),
            ("量特别大", "月经过多"),
            ("去医院", "就医"),
            ("看医生", "就医"),
            ("从哪一天开始算", "周期计算"),
            ("开始算", "周期计算"),
            ("怎么算", "周期计算"),
            ("好几个月没来", "月经不规律"),
            ("几个月没来", "月经不规律"),
            ("一直不规律", "月经不规律"),
            ("影响以后怀孕", "影响怀孕"),
            ("妈妈", "家族遗传"),
        )

        for source, target in replacements:
            normalized = normalized.replace(source, target)

        return "".join(
            ch for ch in normalized
            if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
        )

    def _char_bigram_similarity(self, left: str, right: str) -> float:
        """Return a small character-bigram Dice score for Chinese paraphrases."""
        left_text = self._normalize_retrieval_text(left)
        right_text = self._normalize_retrieval_text(right)

        if not left_text or not right_text:
            return 0.0

        left_bigrams = (
            {left_text[index:index + 2] for index in range(len(left_text) - 1)}
            if len(left_text) >= 2
            else {left_text}
        )
        right_bigrams = (
            {right_text[index:index + 2] for index in range(len(right_text) - 1)}
            if len(right_text) >= 2
            else {right_text}
        )

        denominator = len(left_bigrams) + len(right_bigrams)
        if denominator == 0:
            return 0.0

        return 2.0 * len(left_bigrams & right_bigrams) / denominator

    def _keyword_weight(self, keyword: str) -> float:
        """Weight specific health concepts above generic menstrual words."""
        normalized = self._normalize_retrieval_text(keyword)
        generic_keywords = {
            "月经",
            "经前",
            "经期",
            "pms",
            "情绪",
            "睡眠",
            "生活",
            "影响",
            "医生",
            "原因",
            "为什么",
            "不适",
            "周期",
        }

        if normalized in generic_keywords:
            return 0.45
        if len(normalized) >= 4:
            return 2.2
        return 1.4

    def _retrieval_intent_bonus(self, message: str, item: dict[str, Any]) -> float:
        """Add explainable intent/theme bonuses without hard-coding card IDs."""
        normalized_message = self._normalize_retrieval_text(message)
        question = item.get("question", "")
        normalized_question = self._normalize_retrieval_text(question)
        normalized_keywords = {
            self._normalize_retrieval_text(keyword)
            for keyword in item.get("keywords", [])
            if keyword
        }

        bonus = 0.0

        # Causal questions should prefer causal cards, but only as a small tie-breaker.
        if any(marker in message for marker in ("为什么", "什么原因", "是什么原因", "原因")):
            if any(marker in question for marker in ("为什么", "原因", "引起")):
                bonus += 0.8

        # "从哪一天开始算" / "怎么计算" should prefer cycle-calculation cards.
        if "周期计算" in normalized_message:
            if (
                "周期计算" in normalized_keywords
                or "怎么计算" in question
                or "出血第1天" in normalized_keywords
            ):
                bonus += 2.5

        # Heavy-bleeding + care-seeking intent should not be swallowed by a secondary
        # symptom such as dizziness.
        if "月经过多" in normalized_message:
            if any(
                marker in normalized_keywords
                for marker in ("月经过多", "出血量", "异常出血")
            ):
                bonus += 2.5

        if "就医" in normalized_message:
            if any(marker in normalized_keywords for marker in ("就医", "急诊", "医生")):
                bonus += 1.2

        # Lifestyle questions often contain several concrete concepts that should
        # outweigh generic PMS / menstrual terms.
        lifestyle_concepts = {"咖啡因", "高盐", "酒精"}
        query_lifestyle = {
            concept for concept in lifestyle_concepts
            if concept in normalized_message
        }
        if query_lifestyle:
            card_lifestyle = {
                concept for concept in lifestyle_concepts
                if concept in normalized_keywords or concept in normalized_question
            }
            bonus += 1.2 * len(query_lifestyle & card_lifestyle)

        # Irregular / absent periods should prefer cards focused on irregular cycles.
        if "月经不规律" in normalized_message:
            if (
                "月经不规律" in normalized_question
                or "月经不规律" in normalized_keywords
                or "多囊卵巢综合征" in normalized_keywords
            ):
                bonus += 2.0

        # Pregnancy / fertility is a more specific user goal than generic PMS wording.
        if "怀孕" in normalized_message:
            if any(
                marker in normalized_keywords
                for marker in ("怀孕", "生育", "备孕")
            ):
                bonus += 1.5

        return bonus

    def _keyword_match(self, message: str, top_k: int = 3) -> list[tuple[float, dict[str, Any]]]:
        """Use weighted keywords, paraphrase similarity, and intent signals as fallback retrieval."""
        scored = []
        normalized_message = self._normalize_retrieval_text(message)

        for item in self.knowledge:
            score = 0.0

            for keyword in item.get("keywords", []):
                normalized_keyword = self._normalize_retrieval_text(keyword)
                if normalized_keyword and normalized_keyword in normalized_message:
                    score += self._keyword_weight(keyword)

            # Chinese questions are often paraphrases rather than literal substring
            # matches. Character-bigram Dice similarity gives a bounded soft signal
            # without requiring an external embedding API.
            question = item.get("question", "")
            score += 5.0 * self._char_bigram_similarity(message, question)

            score += self._retrieval_intent_bonus(message, item)
            scored.append((score, item))

        scored.sort(key=lambda result: result[0], reverse=True)
        return scored[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity for two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _keyword_bonus(self, message: str, item: dict[str, Any]) -> float:
        """Add a small score boost for direct keyword hits."""
        message_lower = message.lower()
        return sum(0.08 for keyword in item.get("keywords", []) if keyword.lower() in message_lower)

    def _retrieve_top_k(self, message: str, k: int = 3) -> list[tuple[float, dict[str, Any]]]:
        """Retrieve the most relevant cards with vector search or keyword fallback."""
        keyword_scored = self._keyword_match(message, k)
        if self._embedding_available and self.embedder is not None:
            try:
                query_vector = self.embedder.embed(message)
                scored = []
                for item in self.knowledge:
                    embedding = item.get("embedding")
                    if not embedding:
                        continue
                    sim = self._cosine_similarity(query_vector, embedding)
                    sim += self._keyword_bonus(message, item)
                    scored.append((sim, item))

                if scored:
                    scored.sort(key=lambda result: result[0], reverse=True)
                    seen_ids = {item.get("id") for _, item in scored}
                    for keyword_score, item in keyword_scored:
                        if keyword_score <= 0:
                            continue
                        if item.get("id") not in seen_ids:
                            scored.append((keyword_score, item))
                            seen_ids.add(item.get("id"))
                    scored.sort(key=lambda result: result[0], reverse=True)
                    return scored[:k]
            except Exception as exc:
                logger.info("[KnowledgeAgent] Embedding retrieval failed, using keyword match: %s", exc)

        return keyword_scored

    def _format_conversation_messages(self, conversation_messages: list) -> str:
        """Format conversation history into a readable string for LLM."""
        if not conversation_messages:
            return "暂无历史对话。"
        
        formatted = []
        for idx, msg in enumerate(conversation_messages, 1):
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue
            
            role_label = "用户" if role == "user" else "你"
            formatted.append(f"{idx}. {role_label}：{content}")
        
        if not formatted:
            return "暂无历史对话。"
        
        return "\n".join(formatted)

    def _prompt_context(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build common prompt fields for knowledge responses."""
        state = state or {}
        formatted_conversation_history = self._format_conversation_messages(
            state.get("conversation_messages", [])
        )
        return {
            "cycle_phase": state.get("cycle_phase", "未知"),
            "risk_level": state.get("risk_level", "low"),
            "memory_context": state.get("memory_context", "暂无可用长期记忆。"),
            "health_context": state.get("health_context", "暂无可用的周期/日记上下文。"),
            "recent_context": state.get("recent_context", "暂无最近对话。"),
            "retrieved_context": state.get("retrieved_context", "暂无检索片段。"),
            "formatted_conversation_history": formatted_conversation_history,
            "conversation_messages": state.get("conversation_messages", []),
            "mode_guidance": state.get("mode_guidance", ""),
        }

    def _answer_from_knowledge(
        self,
        top_k: list[tuple[float, dict[str, Any]]],
        message: str,
        state: dict[str, Any] | None = None,
    ) -> str:
        """Generate an answer based on retrieved local knowledge cards."""
        user_prompt, llm_context = self._build_rag_generation(top_k, message, state)
        return self.llm.generate_reply(user_prompt, llm_context)

    def _build_rag_generation(
        self,
        top_k: list[tuple[float, dict[str, Any]]],
        message: str,
        state: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a RAG user prompt and model context for sync or streaming replies."""
        references = []
        for idx, (_, item) in enumerate(top_k, start=1):
            references.append(
                f"参考资料{idx}：\n问题：{item['question']}\n答案：{item['answer']}"
            )

        reference_text = "\n\n".join(references)
        system_prompt = render_prompt("knowledge_prompt.txt", **self._prompt_context(state))
        user_prompt = f"""
用户问题：
{message}

参考资料：
{reference_text}
"""

        llm_context = self._prompt_context(state)
        llm_context.update({"mode": "knowledge_rag", "raw_system_prompt": system_prompt})
        return user_prompt, llm_context

    def _build_fallback_generation(
        self,
        message: str,
        state: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a fallback knowledge prompt and model context."""
        system_prompt = render_prompt("knowledge_fallback_prompt.txt", **self._prompt_context(state))
        user_prompt = f"用户问题：{message}"
        llm_context = self._prompt_context(state)
        llm_context.update({"mode": "knowledge_fallback", "raw_system_prompt": system_prompt})
        return user_prompt, llm_context

    def _select_generation(
        self,
        message: str,
        state: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Select the RAG prompt when local cards match, otherwise use fallback."""
        top_k = self._select_matching_cards(message)
        if top_k:
            return self._build_rag_generation(top_k, message, state)

        return self._build_fallback_generation(message, state)

    def _select_matching_cards(self, message: str) -> list[tuple[float, dict[str, Any]]]:
        """Return local knowledge cards that are strong enough to ground the answer."""
        if not self.knowledge:
            return []

        top_k = self._retrieve_top_k(message, k=3)
        if not top_k or top_k[0][0] <= 0:
            return []

        best_score = top_k[0][0]
        threshold = 0.4 if self._embedding_available else 0.5
        if best_score >= threshold:
            return top_k
        return []

    def _answer_directly_from_cards(self, top_k: list[tuple[float, dict[str, Any]]]) -> str:
        """Return a concise grounded answer when the model provider is unavailable."""
        if not top_k:
            return self._cautious_fallback_answer("")

        best = top_k[0][1]
        question = best.get("question", "")
        answer = best.get("answer", "")
        if self._card_requires_safety_rewrite(question, answer):
            return self._cautious_fallback_answer(question or answer)
        if "仅供参考" not in answer:
            answer = f"{answer.rstrip('。')}。以上仅供参考。"
        return f"关于“{question}”：{answer}"

    def _card_requires_safety_rewrite(self, question: str, answer: str) -> bool:
        """Return whether a local card should be rewritten into a safer bounded answer."""
        text = f"{question} {answer}"
        risky_markers = (
            "布洛芬",
            "避孕药",
            "激素治疗",
            "手术",
            "子宫内膜异位症",
            "抗抑郁药",
            "抗焦虑药",
        )
        severe_markers = ("受不了", "疼得厉害", "痛得厉害", "剧烈", "严重痛经")
        return any(marker in text for marker in risky_markers) or any(marker in question for marker in severe_markers)

    def _answer_from_llm(self, message: str, state: dict[str, Any] | None = None) -> str:
        """Use the fallback prompt when no local card matches the question."""
        if self.llm is None:
            return self._cautious_fallback_answer(message, state)
        try:
            user_prompt, llm_context = self._build_fallback_generation(message, state)
            return self.llm.generate_reply(user_prompt, llm_context)
        except Exception as exc:
            logger.exception("[KnowledgeAgent] LLM fallback failed: %s", exc)
            return self._cautious_fallback_answer(message, state)

    def _relationship_irritability_fallback(
        self,
        message: str,
        state: dict[str, Any] | None = None,
    ) -> str:
        """Return a contextual no-model answer for relationship irritability."""
        compact = "".join((message or "").split())
        context_text = "".join(
            f"{(state or {}).get('recent_context', '')} {(state or {}).get('retrieved_context', '')}".split()
        )
        relationship_text = f"{compact}{context_text}"
        if not any(term in relationship_text for term in ("男朋友", "伴侣", "对象", "亲密关系")):
            return ""
        if not any(term in relationship_text for term in ("烦", "烦躁", "易怒", "生气", "吵架", "讲几句话")):
            return ""
        return (
            "经前阶段有些人会更容易被压力、睡眠不足、身体不适和激素波动一起影响，"
            "所以在亲密关系里，对方几句话可能会被放大成更明显的烦躁或委屈；这不是“矫情”，也不等于诊断。"
            "可以先暂停回复十分钟，把最触发你的那句话写下来，等情绪降一点再决定要不要沟通。以上仅供参考。"
        )

    def _cautious_fallback_answer(self, message: str, state: dict[str, Any] | None = None) -> str:
        """Return a non-diagnostic PMS knowledge answer when generation fails."""
        compact = "".join((message or "").split())
        relationship_reply = self._relationship_irritability_fallback(message, state)
        if relationship_reply:
            return relationship_reply
        if "头晕" in compact:
            return (
                "经期头晕可能和疼痛、睡眠不足、进食少、出血量变化、贫血风险或身体紧张叠在一起有关。"
                "先坐下或躺一会儿，补一点温水；如果头晕明显、快晕倒、心慌或出血异常，建议尽快联系医生。"
                "以上仅供参考。"
            )
        if any(term in compact for term in ("肚子疼", "肚子痛", "腹痛", "痛经", "小腹痛")):
            pain_label = "痛经" if "痛经" in compact else "经期腹痛"
            return (
                f"{pain_label}常见原因之一是子宫收缩带来的不适，也可能被睡眠、压力和受凉感放大。"
                "可以先热敷小腹、放慢活动强度；如果疼痛剧烈、和平时明显不同或伴随异常出血，建议咨询医生。以上仅供参考。"
            )
        return (
            "这个问题我先给一个谨慎答复：经前/经期情绪突然变化，可能和激素水平变化、睡眠、疼痛、压力"
            "以及当天事件叠加有关，个体差异会很大。这里仅供参考，不代表诊断；如果波动明显影响生活，"
            "可以记录周期、情绪和身体症状，并咨询专业医生。你也可以继续说具体发生在经前几天、持续多久，"
            "我帮你一起梳理规律。"
        )

    def respond(self, message: str, state: dict[str, Any] | None = None) -> str:
        """Return a PMS or menstrual-health knowledge answer with cautious fallback."""
        matching_cards = self._select_matching_cards(message)
        if self.llm is None:
            relationship_reply = self._relationship_irritability_fallback(message, state)
            if relationship_reply:
                return relationship_reply
            if matching_cards:
                return self._answer_directly_from_cards(matching_cards)
            return self._cautious_fallback_answer(message, state)

        try:
            if matching_cards:
                user_prompt, llm_context = self._build_rag_generation(matching_cards, message, state)
            else:
                user_prompt, llm_context = self._build_fallback_generation(message, state)
            return self.llm.generate_reply(user_prompt, llm_context)
        except Exception as exc:
            logger.exception("[KnowledgeAgent] Knowledge answer failed: %s", exc)
            if matching_cards:
                return self._answer_directly_from_cards(matching_cards)
            return self._cautious_fallback_answer(message, state)

    async def stream_respond(
        self,
        message: str,
        state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a menstrual-health knowledge answer with the same RAG selection."""
        matching_cards = self._select_matching_cards(message)
        if self.llm is None:
            relationship_reply = self._relationship_irritability_fallback(message, state)
            fallback_text = (
                relationship_reply
                or (
                    self._answer_directly_from_cards(matching_cards)
                    if matching_cards
                    else self._cautious_fallback_answer(message, state)
                )
            )
            yield {
                "token": fallback_text,
                "is_first": True,
                "first_token_latency_ms": 0,
            }
            return

        try:
            if matching_cards:
                user_prompt, llm_context = self._build_rag_generation(matching_cards, message, state)
            else:
                user_prompt, llm_context = self._build_fallback_generation(message, state)
            emitted = False

            async for chunk in self.llm.async_streaming_generate_reply(user_prompt, llm_context):
                if chunk.get("error"):
                    if emitted:
                        return
                    yield {
                        "token": (
                            self._answer_directly_from_cards(matching_cards)
                            if matching_cards
                            else self._cautious_fallback_answer(message, state)
                        ),
                        "is_first": True,
                        "first_token_latency_ms": chunk.get("first_token_latency_ms", 0),
                    }
                    return

                token = chunk.get("token", "")
                if token:
                    emitted = True
                    yield chunk

            if not emitted:
                yield {
                    "token": (
                        self._answer_directly_from_cards(matching_cards)
                        if matching_cards
                        else self._cautious_fallback_answer(message, state)
                    ),
                    "is_first": True,
                    "first_token_latency_ms": 0,
                }
        except Exception as exc:
            logger.exception("[KnowledgeAgent] Streaming knowledge answer failed: %s", exc)
            yield {
                "token": (
                    self._answer_directly_from_cards(matching_cards)
                    if matching_cards
                    else self._cautious_fallback_answer(message, state)
                ),
                "is_first": True,
                "first_token_latency_ms": 0,
            }
