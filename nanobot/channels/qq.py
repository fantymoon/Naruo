"""QQ channel implementation using botpy SDK."""
import random
import asyncio
import mimetypes
import urllib.request
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import QQConfig

try:
    import botpy
    from botpy.message import C2CMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None

if TYPE_CHECKING:
    from botpy.message import C2CMessage


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            # Disable botpy's file log — nanobot uses loguru; default "botpy.log" fails on read-only fs
            super().__init__(intents=intents, ext_handlers=False)

        async def on_ready(self):
            logger.info("QQ bot ready: {}", self.robot.name)

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_message(message)

        async def on_direct_message_create(self, message):
            await channel._on_message(message)

    return _Bot


class QQChannel(BaseChannel):
    """QQ channel using botpy SDK with WebSocket connection."""

    name = "qq"

    def __init__(self, config: QQConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: QQConfig = config
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._media_dir = Path.home() / ".nanobot" / "workspace" / "tmp" / "qq_media"

    async def start(self) -> None:
        """Start the QQ bot."""
        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.secret:
            logger.error("QQ app_id and secret not configured")
            return

        self._running = True
        BotClass = _make_bot_class(self)
        self._client = BotClass()

        logger.info("QQ bot started (C2C private message)")
        await self._run_bot()

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect."""
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=self.config.secret)
            except Exception as e:
                logger.warning("QQ bot error: {}", e)
            if self._running:
                logger.info("Reconnecting QQ bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the QQ bot."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        logger.info("QQ bot stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through QQ."""
        if not self._client:
            logger.warning("QQ client not initialized")
            return
        try:
            msg_id = msg.metadata.get("message_id")
            msg_seq = random.randint(0, 1000000)
            await self._client.api.post_c2c_message(
                openid=msg.chat_id,
                msg_type=0,
                content=msg.content,
                msg_id=msg_id,
                msg_seq=msg_seq,
            )
        except Exception as e:
            logger.error("Error sending QQ message: {}", e)

    def _extract_image_urls(self, data: "C2CMessage") -> list[str]:
        urls: list[str] = []
        attachments = getattr(data, "attachments", None) or []
        for item in attachments:
            if isinstance(item, dict):
                url = item.get("url") or item.get("proxy_url") or item.get("download_url")
                content_type = item.get("content_type") or item.get("contentType") or ""
                filename = item.get("filename") or ""
            else:
                url = getattr(item, "url", None) or getattr(item, "proxy_url", None) or getattr(item, "download_url", None)
                content_type = getattr(item, "content_type", None) or getattr(item, "contentType", None) or ""
                filename = getattr(item, "filename", None) or ""

            if not url:
                continue

            lowered = f"{content_type} {filename} {url}".lower()
            if "image" in lowered or any(ext in lowered for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
                urls.append(url)
        return urls

    def _download_media_sync(self, url: str, message_id: str, index: int) -> str | None:
        try:
            self._media_dir.mkdir(parents=True, exist_ok=True)
            parsed = urlparse(url)
            suffix = Path(parsed.path).suffix
            req = urllib.request.Request(url, headers={"User-Agent": "nanobot/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content_type = (resp.headers.get_content_type() or "").lower()
                if not suffix:
                    if content_type and content_type != "application/octet-stream":
                        suffix = mimetypes.guess_extension(content_type) or ""
                    if not suffix:
                        guessed, _ = mimetypes.guess_type(url)
                        suffix = mimetypes.guess_extension(guessed or "") or ".img"
                target = self._media_dir / f"{message_id}_{index}{suffix}"
                with open(target, "wb") as f:
                    f.write(resp.read())
            logger.info("QQ media downloaded: message_id={}, index={}, content_type={}, path={}", message_id, index, content_type or "unknown", target)
            return str(target)
        except Exception as e:
            logger.warning("Failed to download QQ media {}: {}", url, e)
            return None

    async def _download_media(self, urls: list[str], message_id: str) -> list[str]:
        files: list[str] = []
        for i, url in enumerate(urls, start=1):
            path = await asyncio.to_thread(self._download_media_sync, url, message_id, i)
            if path:
                files.append(path)
        return files

    async def _on_message(self, data: "C2CMessage") -> None:
        """Handle incoming message from QQ."""
        try:
            # Dedup by message ID
            if data.id in self._processed_ids:
                return
            self._processed_ids.append(data.id)

            author = data.author
            user_id = str(getattr(author, 'id', None) or getattr(author, 'user_openid', 'unknown'))
            content = (data.content or "").strip()
            image_urls = self._extract_image_urls(data)
            media_files = await self._download_media(image_urls, data.id) if image_urls else []
            logger.info("QQ inbound message: id={}, has_text={}, image_urls={}, media_files={}", data.id, bool(content), len(image_urls), len(media_files))
            if not content and media_files:
                content = "[user sent an image]"
            if not content and not media_files:
                return

            metadata: dict[str, Any] = {"message_id": data.id}
            if image_urls:
                metadata["qq_image_urls"] = image_urls

            await self._handle_message(
                sender_id=user_id,
                chat_id=user_id,
                content=content,
                media=media_files,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Error handling QQ message")
