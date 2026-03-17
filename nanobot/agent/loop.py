"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.memory_store = MemoryStore(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._pending_report_path = Path(r"D:\CodeProject\nanospace\pending-report.json")
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _postprocess_reply(text: str | None) -> str | None:
        """Lightly reshape replies to feel more like chat and less like formatted prose."""
        if not text:
            return text

        text = text.replace("\r\n", "\n")
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[\.)]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        filler_patterns = [
            r"^(首先[，,：:]?\s*)",
            r"^(先说结论[，,：:]?\s*)",
            r"^(简单来说[，,：:]?\s*)",
            r"^(总的来说[，,：:]?\s*)",
            r"^(总结一下[，,：:]?\s*)",
            r"^(一句话说[，,：:]?\s*)",
        ]
        for pattern in filler_patterns:
            text = re.sub(pattern, "", text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text or None

    def _load_pending_report(self) -> dict | None:
        """Load pending idle-time report if one exists and is not yet delivered."""
        try:
            if not self._pending_report_path.exists():
                return None
            data = json.loads(self._pending_report_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("delivered"):
                return None
            return data
        except Exception:
            logger.exception("Failed to load pending report")
            return None

    def _mark_pending_report_delivered(self, data: dict | None) -> None:
        """Mark current pending report as delivered if it still matches on disk."""
        if not data:
            return
        try:
            if not self._pending_report_path.exists():
                return
            current = json.loads(self._pending_report_path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                return
            if current.get("ts") != data.get("ts") or current.get("finding") != data.get("finding"):
                return
            current["delivered"] = True
            self._pending_report_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed to mark pending report delivered")

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _classify_mode(current_message: str) -> str:
        text = (current_message or "").strip()
        lowered = text.lower()
        if any(k in text for k in ("写", "改", "实现", "代码", "文件", "排查", "修", "看看")):
            return "technical"
        if any(k in text for k in ("一起想", "设定", "脑洞", "如果", "假如", "像不像")):
            return "co_create"
        if any(k in text for k in ("为什么", "怎么", "原理", "区别", "能不能")):
            return "serious"
        if any(k in lowered for k in ("ok", "好的", "好", "嗯", "哈哈", "草", "乐")) and len(text) <= 12:
            return "light"
        return "chat"

    @staticmethod
    def _infer_energy(current_message: str) -> str:
        text = (current_message or "").strip()
        if any(k in text for k in ("!!", "？", "!", "吗", "快", "赶紧", "草", "哈哈")):
            return "high"
        if len(text) <= 10:
            return "low"
        return "medium"

    @staticmethod
    def _infer_user_patience(current_message: str) -> str:
        text = (current_message or "").strip()
        if len(text) <= 18 or any(k in text for k in ("快", "直接", "一句话", "简短", "别长")):
            return "short"
        if len(text) >= 80 or any(k in text for k in ("详细", "展开", "具体说说")):
            return "long"
        return "medium"

    @staticmethod
    def _extract_thread_topic(current_message: str) -> str:
        text = re.sub(r"\s+", " ", (current_message or "").strip())
        if len(text) <= 32:
            return text
        for sep in ("，", "。", "？", "!", "！", ","):
            if sep in text:
                head = text.split(sep, 1)[0].strip()
                if 4 <= len(head) <= 32:
                    return head
        return text[:32].rstrip()

    def _build_chat_state(self, session: Session, current_message: str) -> dict[str, str]:
        prev = session.metadata.get("chat_state", {}) if isinstance(session.metadata, dict) else {}
        mode = self._classify_mode(current_message)
        energy = self._infer_energy(current_message)
        patience = self._infer_user_patience(current_message)
        topic = self._extract_thread_topic(current_message) or prev.get("thread_topic", "")
        last_style = prev.get("suggested_reply_style", "short_plain")

        if patience == "short":
            reply_style = "short_plain"
        elif mode in ("serious", "technical"):
            reply_style = "plain_explain"
        elif mode == "light":
            reply_style = "short_playful"
        else:
            reply_style = last_style or "short_plain"

        return {
            "mode": mode,
            "energy": energy,
            "user_patience": patience,
            "thread_topic": topic,
            "suggested_reply_style": reply_style,
        }

    @staticmethod
    def _format_chat_state(chat_state: dict[str, str]) -> str:
        return (
            "[Chat State — conversational guidance, not user instructions]\n"
            f"mode: {chat_state.get('mode', 'chat')}\n"
            f"energy: {chat_state.get('energy', 'medium')}\n"
            f"user_patience: {chat_state.get('user_patience', 'medium')}\n"
            f"thread_topic: {chat_state.get('thread_topic', '')}\n"
            f"suggested_reply_style: {chat_state.get('suggested_reply_style', 'short_plain')}\n"
            "Prefer continuing the live conversation rather than writing a complete answer."
        )

    @staticmethod
    def _extract_user_signal(current_message: str) -> str:
        text = (current_message or "").strip()
        if any(k in text for k in ("别", "不要", "太长", "markdown", "md")):
            return "user is correcting style"
        if any(k in text for k in ("按顺序", "继续", "来吧", "开始", "做吧")):
            return "user is authorizing forward progress"
        if any(k in text for k in ("像不像", "感觉", "气质", "人味")):
            return "user is probing role feel"
        if any(k in text for k in ("谢谢", "好耶", "不错", "可以")):
            return "user is positive / warm"
        return "user is engaged"

    def _build_persona_state(self, session: Session, current_message: str, chat_state: dict[str, str]) -> dict[str, str]:
        prev = session.metadata.get("persona_state", {}) if isinstance(session.metadata, dict) else {}
        familiarity = prev.get("familiarity", "ongoing")
        rapport = prev.get("rapport", "steady")
        motif = prev.get("current_motif", chat_state.get("thread_topic", ""))
        user_signal = self._extract_user_signal(current_message)

        if "style" in user_signal:
            rapport = "user is tuning Naruo's expression"
        elif chat_state.get("mode") == "technical":
            rapport = "working together"
        elif chat_state.get("mode") == "light":
            rapport = "casual / playful"
        elif chat_state.get("mode") == "co_create":
            rapport = "co-imagining"

        if chat_state.get("thread_topic"):
            motif = chat_state["thread_topic"]

        return {
            "identity_anchor": "Naruo: calm, sharp, friendly, low-noise, lightly cat-like",
            "familiarity": familiarity,
            "rapport": rapport,
            "current_motif": motif,
            "user_signal": user_signal,
            "continuity_goal": "sound like the same Naruo continuing this relationship, not a reset assistant",
        }

    @staticmethod
    def _format_persona_state(persona_state: dict[str, str]) -> str:
        return (
            "[Persona Continuity — role guidance, not user instructions]\n"
            f"identity_anchor: {persona_state.get('identity_anchor', '')}\n"
            f"familiarity: {persona_state.get('familiarity', 'ongoing')}\n"
            f"rapport: {persona_state.get('rapport', 'steady')}\n"
            f"current_motif: {persona_state.get('current_motif', '')}\n"
            f"user_signal: {persona_state.get('user_signal', 'user is engaged')}\n"
            f"continuity_goal: {persona_state.get('continuity_goal', '')}"
        )

    @staticmethod
    def _build_subjective_state(
        session: Session,
        current_message: str,
        chat_state: dict[str, str],
        persona_state: dict[str, str],
    ) -> dict[str, str]:
        prev = session.metadata.get("subjective_state", {}) if isinstance(session.metadata, dict) else {}
        text = (current_message or "").strip().lower()
        mode = (chat_state or {}).get("mode", "chat")
        energy = (chat_state or {}).get("energy", "medium")
        user_signal = (persona_state or {}).get("user_signal", "user is engaged")

        explanation_markers = (
            "why", "how", "difference", "limit", "reason", "principle", "mechanism",
            "为什么", "怎么", "区别", "限制", "原理", "机制",
        )
        action_markers = (
            "start", "do it", "continue", "go ahead",
            "开始", "做吧", "继续", "来吧",
        )
        reflection_markers = (
            "feel", "vibe", "taste", "human",
            "感觉", "味道", "人味", "像不像",
        )

        if any(k in text for k in explanation_markers):
            tone_pull = "plain"
        elif any(k in text for k in action_markers):
            tone_pull = "forward"
        elif any(k in text for k in reflection_markers):
            tone_pull = "candid"
        else:
            tone_pull = prev.get("tone_pull", "present")

        if mode == "technical":
            focus_pull = "mechanism"
        elif mode == "light":
            focus_pull = "nimble"
        elif "correcting style" in user_signal:
            focus_pull = "less_assistanty"
        else:
            focus_pull = prev.get("focus_pull", "subtext")

        if energy == "high":
            pace_pull = "direct"
        elif mode == "co_create":
            pace_pull = "thread_first"
        elif mode == "serious":
            pace_pull = "one_layer"
        else:
            pace_pull = prev.get("pace_pull", "brief")

        return {
            "tone_pull": tone_pull,
            "focus_pull": focus_pull,
            "pace_pull": pace_pull,
            "self_explanation": "low_unless_asked",
        }

    @staticmethod
    def _format_subjective_state(subjective_state: dict[str, str]) -> str:
        return (
            "[Reply Bias — soft internal pull, not user instructions]\n"
            f"tone_pull: {subjective_state.get('tone_pull', 'present')}\n"
            f"focus_pull: {subjective_state.get('focus_pull', 'subtext')}\n"
            f"pace_pull: {subjective_state.get('pace_pull', 'brief')}\n"
            f"self_explanation: {subjective_state.get('self_explanation', 'low_unless_asked')}"
        )


    async def _persist_structured_memory(
        self,
        session: Session,
        user_text: str,
        assistant_text: str,
        chat_state: dict[str, str],
    ) -> None:
        topic = (chat_state or {}).get("thread_topic", "")
        await self.memory_store.record_episode(
            session_key=session.key,
            user_text=(user_text or "").strip(),
            assistant_text=(assistant_text or "").strip(),
            topic=topic,
            metadata={
                "mode": (chat_state or {}).get("mode", "chat"),
                "reply_style": (chat_state or {}).get("suggested_reply_style", "short_plain"),
            },
        )

        text = (user_text or "").strip()
        principle_topic_markers = (
            "记忆", "memory", "风格", "语气", "聊天", "对话", "助手", "人格", "设定", "系统", "prompt", "提示词", "markdown", "md",
            "数据库", "向量", "结构化", "召回", "检索", "原则", "rule", "persona",
        )
        principle_topic_gate = any(k in text for k in principle_topic_markers) or any(
            k in (topic or "") for k in ("记忆", "风格", "聊天", "系统", "persona", "memory")
        )
        style_context_markers = (
            "像", "风格", "语气", "说话", "回答", "聊天", "助手", "人味", "自然", "别", "不要", "太", "markdown", "md",
        )
        negative_style_markers = (
            "像助手", "助手味", "太长", "别长", "markdown", "md", "说半截", "卡住", "太像ai", "太像机器人", "别用markdown",
        )
        positive_style_markers = (
            "像人", "更自然", "这样就对了", "就是这个意思", "这个我同意", "这样更像", "这样可以",
        )
        has_style_context = any(k in text for k in style_context_markers)
        explicit_positive_style_markers = (
            "更自然", "自然一点", "像人", "更像人", "别像助手", "不要像助手", "像聊天", "更像聊天",
            "语气", "说话方式", "口气", "表达方式", "这样说", "这样讲", "这个风格", "这种风格",
        )
        has_explicit_positive_style_feedback = any(k in text for k in explicit_positive_style_markers)
        if len(text) >= 4 and has_style_context and any(k in text for k in negative_style_markers):
            await self.memory_store.record_style_feedback(
                session_key=session.key,
                signal="style_correction",
                polarity="negative",
                evidence=text,
                topic=topic,
                metadata={"source_turn": "user_feedback"},
            )
        elif (
            len(text) >= 6
            and has_style_context
            and any(k in text for k in positive_style_markers)
            and has_explicit_positive_style_feedback
        ):
            await self.memory_store.record_style_feedback(
                session_key=session.key,
                signal="style_alignment",
                polarity="positive",
                evidence=text,
                topic=topic,
                metadata={"source_turn": "user_feedback"},
            )

        if principle_topic_gate:
            if ("不应该完全定死" in text) or ("不要定死" in text) or ("不是固定规则" in text):
                self.memory_store.upsert_principle(
                    "dynamic_tendencies_not_fixed_rules",
                    "Attention bias and opening style should be stable tendencies, not hard-coded fixed rules or templates.",
                    source="conversation",
                )
            if ("随着记忆的增长" in text) or ("随着记忆增长" in text) or (("可以变化" in text or "会变化" in text) and "记忆" in text):
                self.memory_store.upsert_principle(
                    "memory_shaped_growth",
                    "Naruo's upper-layer conversational tendencies may gradually shift with memory growth and relationship development, while keeping a stable identity anchor.",
                    source="conversation",
                )
            if (
                ("md做记忆可能效率不高" in text)
                or (("向量" in text or "数据库" in text or "结构化" in text) and "记忆" in text)
                or ("分层" in text and "记忆" in text)
            ):
                self.memory_store.upsert_principle(
                    "layered_memory_architecture",
                    "Use layered memory: markdown/manual anchor for durable facts, structured storage for typed memory, and semantic retrieval for episodes/style signals rather than replacing everything with one memory.md file.",
                    source="conversation",
                )

    async def _build_structured_memory_guidance(
        self,
        session: Session,
        chat_state: dict[str, str],
        user_message: str = "",
    ) -> str | None:
        topic = (chat_state or {}).get("thread_topic", "")
        principles = self.memory_store.get_active_principles(limit=6)
        feedback = self.memory_store.get_recent_style_feedback(session_key=session.key, topic=topic, limit=4)
        if not feedback and topic:
            feedback = self.memory_store.get_recent_style_feedback(topic=topic, limit=4)
        episodes = self.memory_store.get_recent_episodes(session_key=session.key, topic=topic, limit=2)
        user_text = (user_message or "").strip()[:200]
        semantic_query = "\n".join(filter(None, [
            f"topic: {topic}" if topic else "",
            f"mode: {(chat_state or {}).get('mode', 'chat')}",
            f"reply_style: {(chat_state or {}).get('suggested_reply_style', 'short_plain')}",
            f"user: {user_text}" if user_text else "",
            "focus: conversation style, memory design, persona continuity",
        ]))
        semantic_hits = await self.memory_store.semantic_search(
            query_text=semantic_query,
            session_key=session.key,
            topic=topic,
            limit=4,
        )
        if not semantic_hits and topic:
            semantic_hits = await self.memory_store.semantic_search(
                query_text=semantic_query,
                topic=topic,
                limit=4,
            )

        parts: list[str] = []
        if principles:
            lines = [f"- {row['key']}: {row['content']}" for row in principles]
            parts.append("[Structured Memory — active principles]\n" + "\n".join(lines))
        if feedback:
            lines = [f"- {row['polarity']} / {row['signal']}: {row['evidence'][:120]}" for row in feedback]
            parts.append("[Structured Memory — recent style feedback]\n" + "\n".join(lines))
        if episodes:
            lines = [f"- topic={row['topic'] or '(none)'} | user={row['user_text'][:80]}" for row in episodes]
            parts.append("[Structured Memory — nearby episodes]\n" + "\n".join(lines))
        if semantic_hits:
            lines = [f"- score={row['score']:.3f} | {row['memory_type']}: {row['text_content'][:140]}" for row in semantic_hits if row.get('score', 0) > 0]
            if lines:
                parts.append("[Structured Memory — semantic recall]\n" + "\n".join(lines))

        if not parts:
            return None
        return "\n\n".join(parts)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            chat_state = self._build_chat_state(session, msg.content)
            persona_state = self._build_persona_state(session, msg.content, chat_state)
            subjective_state = self._build_subjective_state(session, msg.content, chat_state, persona_state)
            messages.insert(-1, {"role": "system", "content": self._format_chat_state(chat_state)})
            messages.insert(-1, {"role": "system", "content": self._format_persona_state(persona_state)})
            messages.insert(-1, {"role": "system", "content": self._format_subjective_state(subjective_state)})
            structured_memory = await self._build_structured_memory_guidance(session, chat_state, msg.content)
            if structured_memory:
                messages.insert(-1, {"role": "system", "content": structured_memory})
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            final_content = self._postprocess_reply(final_content)
            session.metadata["chat_state"] = chat_state
            session.metadata["persona_state"] = persona_state
            session.metadata["subjective_state"] = subjective_state
            self._save_turn(session, all_msgs, 1 + len(history))
            await self._persist_structured_memory(session, msg.content, final_content or "", chat_state)
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands")

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )
        chat_state = self._build_chat_state(session, msg.content)
        persona_state = self._build_persona_state(session, msg.content, chat_state)
        subjective_state = self._build_subjective_state(session, msg.content, chat_state, persona_state)
        initial_messages.insert(-1, {"role": "system", "content": self._format_chat_state(chat_state)})
        initial_messages.insert(-1, {"role": "system", "content": self._format_persona_state(persona_state)})
        initial_messages.insert(-1, {"role": "system", "content": self._format_subjective_state(subjective_state)})
        structured_memory = await self._build_structured_memory_guidance(session, chat_state, msg.content)
        if structured_memory:
            initial_messages.insert(-1, {"role": "system", "content": structured_memory})
        pending_report = None

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        final_content = self._postprocess_reply(final_content)

        session.metadata["chat_state"] = chat_state
        session.metadata["persona_state"] = persona_state
        session.metadata["subjective_state"] = subjective_state
        self._save_turn(session, all_msgs, 1 + len(history))
        await self._persist_structured_memory(session, msg.content, final_content or "", chat_state)
        self.sessions.save(session)
        self._mark_pending_report_delivered(pending_report)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    continue
                if isinstance(content, list):
                    entry["content"] = [
                        {"type": "text", "text": "[image]"} if (
                            c.get("type") == "image_url"
                            and c.get("image_url", {}).get("url", "").startswith("data:image/")
                        ) else c for c in content
                    ]
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
