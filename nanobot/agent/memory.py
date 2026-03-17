"""Memory system for persistent agent memory."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import sqlite_vec

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    """Persistent memory store: markdown memory + history log + lightweight structured sqlite."""

    _EMBED_DIM = 1536  # OpenAI text-embedding-3-small dimension

    def __init__(self, workspace: Path, config: dict[str, Any] | None = None):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.db_file = self.memory_dir / "memory.db"
        self.config = config or {}
        self._embedding_cache: dict[str, list[float]] = {}
        # Try to load embedding config from nanobot config if not provided
        if not self.config.get("embedding"):
            try:
                from nanobot.config.loader import load_config
                cfg = load_config()
                if hasattr(cfg, "model_dump"):
                    cfg_dict = cfg.model_dump()
                else:
                    cfg_dict = dict(cfg) if cfg else {}
                self.config = cfg_dict
            except Exception:
                pass
        self._init_db()

    def _get_embedding_config(self) -> dict[str, Any]:
        """Get embedding configuration from config."""
        emb = self.config.get("embedding", {})
        # Support both camelCase and snake_case
        api_key = emb.get("apiKey") or emb.get("api_key", "")
        api_base = emb.get("apiBase") or emb.get("api_base", "https://api.openai.com/v1")
        model = emb.get("model", "text-embedding-3-small")
        provider = emb.get("provider", "openai")
        
        if api_key:
            return {
                "provider": provider,
                "model": model,
                "apiKey": api_key,
                "apiBase": api_base.rstrip("/"),
            }
        return {}

    async def _embed_text_async(self, text: str) -> list[float]:
        """Get embedding for text using configured API or fallback to hash."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        emb_config = self._get_embedding_config()

        if emb_config and emb_config.get("apiKey"):
            try:
                import aiohttp
                api_key = emb_config["apiKey"]
                api_base = emb_config.get("apiBase", "https://api.openai.com/v1").rstrip("/")
                model = emb_config.get("model", "text-embedding-3-small")

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{api_base}/embeddings",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"input": text, "model": model},
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            embedding = data["data"][0]["embedding"]
                            self._embedding_cache[text] = embedding
                            return embedding
                        else:
                            logger.warning("Embedding API error: status {}", resp.status)
            except Exception as e:
                logger.warning("Embedding API failed: {}", e)

        # Fallback to hash embedding
        return self._embed_text_hash(text)

    def _embed_text_hash(self, text: str) -> list[float]:
        """Fallback hash-based embedding when API is not available."""
        vec = np.zeros(self._EMBED_DIM, dtype=np.float32)
        for token in self._tokenize(text):
            h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self._EMBED_DIM
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            weight = 1.0 + (h[5] / 255.0) * 0.25
            vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.astype(float).tolist()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'long_term',
                    status TEXT NOT NULL DEFAULT 'active',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'manual',
                    updated_at TEXT NOT NULL,
                    UNIQUE(key, scope)
                );

                CREATE TABLE IF NOT EXISTS principles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'manual',
                    updated_at TEXT NOT NULL,
                    UNIQUE(key)
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_key TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT '',
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'conversation',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS style_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    polarity TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'conversation',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    ref_id INTEGER NOT NULL,
                    session_key TEXT NOT NULL DEFAULT '',
                    topic TEXT NOT NULL DEFAULT '',
                    text_content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_session_ts ON episodes(session_key, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_episodes_topic_ts ON episodes(topic, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_style_feedback_topic_ts ON style_feedback(topic, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_semantic_memory_type_ref ON semantic_memory(memory_type, ref_id);
                CREATE INDEX IF NOT EXISTS idx_semantic_memory_session_topic ON semantic_memory(session_key, topic);
                """
            )
            # Create sqlite-vec virtual table for embeddings (cosine distance)
            try:
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS semantic_embeddings USING vec0(embedding float[1536] distance_metric=cosine)")
            except Exception as e:
                logger.warning("Could not create sqlite-vec table (may already exist): {}", e)
            conn.commit()

    def _tokenize(self, text: str) -> list[str]:
        text = (text or "").lower()
        ascii_tokens = re.findall(r"[a-z0-9_]+", text)
        cjk_chunks = re.findall(r"[\u4e00-\u9fff]{1,8}", text)
        cjk_bigrams: list[str] = []
        for chunk in cjk_chunks:
            if len(chunk) == 1:
                cjk_bigrams.append(chunk)
            else:
                cjk_bigrams.extend(chunk[i:i+2] for i in range(len(chunk) - 1))
        return ascii_tokens + cjk_bigrams

    def _serialize_f32(self, vector: list[float]) -> bytes:
        """Serialize a list of floats into compact bytes for sqlite-vec."""
        return struct.pack(f"{len(vector)}f", *vector)

    async def _insert_semantic_memory(
        self,
        memory_type: str,
        ref_id: int,
        text_content: str,
        *,
        session_key: str = "",
        topic: str = "",
        created_at: str | None = None,
    ) -> None:
        created_at = created_at or datetime.now().isoformat()
        embedding = await self._embed_text_async(text_content)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO semantic_memory (memory_type, ref_id, session_key, topic, text_content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_type, ref_id, session_key, topic, text_content, created_at),
            )
            row_id = cursor.lastrowid
            # Insert embedding into sqlite-vec virtual table
            embedding_blob = self._serialize_f32(embedding)
            conn.execute(
                "INSERT INTO semantic_embeddings(rowid, embedding) VALUES (?, ?)",
                (row_id, embedding_blob),
            )
            conn.commit()

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def upsert_fact(
        self,
        key: str,
        value: str,
        *,
        scope: str = "long_term",
        status: str = "active",
        confidence: float = 1.0,
        source: str = "manual",
    ) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO facts (key, value, scope, status, confidence, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key, scope) DO UPDATE SET
                    value=excluded.value,
                    status=excluded.status,
                    confidence=excluded.confidence,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (key, value, scope, status, confidence, source, now),
            )
            conn.commit()

    def upsert_principle(
        self,
        key: str,
        content: str,
        *,
        status: str = "active",
        confidence: float = 1.0,
        source: str = "manual",
    ) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO principles (key, content, status, confidence, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    content=excluded.content,
                    status=excluded.status,
                    confidence=excluded.confidence,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (key, content, status, confidence, source, now),
            )
            conn.commit()

    async def record_episode(
        self,
        session_key: str,
        user_text: str,
        assistant_text: str,
        *,
        topic: str = "",
        source: str = "conversation",
        metadata: dict[str, Any] | None = None,
        ts: str | None = None,
    ) -> None:
        metadata = metadata or {}
        ts = ts or datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO episodes (session_key, ts, topic, user_text, assistant_text, source, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_key, ts, topic, user_text, assistant_text, source, json.dumps(metadata, ensure_ascii=False)),
            )
            ref_id = int(cursor.lastrowid)
            conn.commit()
        semantic_text = f"topic: {topic}\nuser: {user_text}\nassistant: {assistant_text}"
        await self._insert_semantic_memory(
            "episode",
            ref_id,
            semantic_text,
            session_key=session_key,
            topic=topic,
            created_at=ts,
        )

    async def record_style_feedback(
        self,
        session_key: str,
        signal: str,
        polarity: str,
        evidence: str,
        *,
        topic: str = "",
        source: str = "conversation",
        metadata: dict[str, Any] | None = None,
        ts: str | None = None,
    ) -> None:
        metadata = metadata or {}
        ts = ts or datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO style_feedback (ts, session_key, signal, polarity, evidence, topic, source, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, session_key, signal, polarity, evidence, topic, source, json.dumps(metadata, ensure_ascii=False)),
            )
            ref_id = int(cursor.lastrowid)
            conn.commit()
        semantic_text = f"topic: {topic}\nsignal: {signal}\npolarity: {polarity}\nevidence: {evidence}"
        await self._insert_semantic_memory(
            "style_feedback",
            ref_id,
            semantic_text,
            session_key=session_key,
            topic=topic,
            created_at=ts,
        )

    def get_active_principles(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, content, confidence, source, updated_at
                FROM principles
                WHERE status = 'active'
                ORDER BY updated_at DESC, confidence DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_style_feedback(
        self,
        *,
        session_key: str | None = None,
        topic: str = "",
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if session_key:
            clauses.append("session_key = ?")
            params.append(session_key)
        if topic:
            clauses.append("topic = ?")
            params.append(topic)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT ts, session_key, signal, polarity, evidence, topic, metadata_json
            FROM style_feedback
            {where}
            ORDER BY ts DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_recent_episodes(
        self,
        *,
        session_key: str | None = None,
        topic: str = "",
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if session_key:
            clauses.append("session_key = ?")
            params.append(session_key)
        if topic:
            clauses.append("topic = ?")
            params.append(topic)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT ts, session_key, topic, user_text, assistant_text, metadata_json
            FROM episodes
            {where}
            ORDER BY ts DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    async def semantic_search(
        self,
        query_text: str,
        *,
        memory_types: tuple[str, ...] = ("episode", "style_feedback"),
        session_key: str | None = None,
        topic: str = "",
        limit: int = 4,
        min_score: float = 0.18,
    ) -> list[dict[str, Any]]:
        query_embedding = await self._embed_text_async(query_text)
        query_blob = self._serialize_f32(query_embedding)

        # sqlite-vec requires k parameter for MATCH queries
        k = limit * 2

        # Build filter conditions for semantic_memory table
        filter_clauses = []
        filter_params: list[Any] = []
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            filter_clauses.append(f"memory_type IN ({placeholders})")
            filter_params.extend(memory_types)
        if session_key:
            filter_clauses.append("session_key = ?")
            filter_params.append(session_key)
        if topic:
            filter_clauses.append("topic = ?")
            filter_params.append(topic)

        # First, get vector search results from sqlite-vec
        vec_query = """
            SELECT rowid, distance
            FROM semantic_embeddings
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
        """
        vec_params = [query_blob, k]

        with self._connect() as conn:
            vec_results = conn.execute(vec_query, vec_params).fetchall()

        if not vec_results:
            return []

        # Then, get corresponding semantic_memory records
        row_ids = [r[0] for r in vec_results]
        id_placeholders = ",".join("?" for _ in row_ids)
        mem_query = f"""
            SELECT id, memory_type, ref_id, session_key, topic, text_content, created_at
            FROM semantic_memory
            WHERE id IN ({id_placeholders})
        """
        if filter_clauses:
            mem_query += f" AND {' AND '.join(filter_clauses)}"
        mem_params = row_ids + filter_params

        mem_rows = conn.execute(mem_query, mem_params).fetchall()

        # Build lookup by id
        mem_by_id = {row[0]: dict(row) for row in mem_rows}

        # Combine results
        scored: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for rowid, distance in vec_results:
            if rowid not in mem_by_id:
                continue
            item = mem_by_id[rowid]
            key = (item.get("memory_type", ""), int(item.get("ref_id", 0)))
            if key in seen:
                continue
            seen.add(key)
            # Convert distance to similarity score (cosine distance = 1 - cosine similarity)
            score = 1.0 - distance
            if score < min_score:
                continue
            item["score"] = score
            scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def debug_summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            counts = {
                "facts": conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
                "principles": conn.execute("SELECT COUNT(*) FROM principles").fetchone()[0],
                "episodes": conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "style_feedback": conn.execute("SELECT COUNT(*) FROM style_feedback").fetchone()[0],
                "semantic_memory": conn.execute("SELECT COUNT(*) FROM semantic_memory").fetchone()[0],
            }
            recent_principles = [dict(r) for r in conn.execute(
                "SELECT key, content, updated_at FROM principles ORDER BY updated_at DESC LIMIT 5"
            ).fetchall()]
            recent_feedback = [dict(r) for r in conn.execute(
                "SELECT ts, signal, polarity, evidence, topic FROM style_feedback ORDER BY ts DESC LIMIT 5"
            ).fetchall()]
            recent_episodes = [dict(r) for r in conn.execute(
                "SELECT ts, topic, user_text, assistant_text FROM episodes ORDER BY ts DESC LIMIT 5"
            ).fetchall()]
        return {
            "db_file": str(self.db_file),
            "counts": counts,
            "recent_principles": recent_principles,
            "recent_feedback": recent_feedback,
            "recent_episodes": recent_episodes,
        }

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning("Memory consolidation: unexpected arguments type {}", type(args).__name__)
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False
