import asyncio
import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable '{name}' is not set.")
    return value


BOT_TOKEN = require_env("BOT_TOKEN")
GEMINI_API_KEY = require_env("GEMINI_API_KEY")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "")).strip().rstrip("/")
WEB_SERVER_HOST = os.getenv("WEB_SERVER_HOST", "0.0.0.0")
WEB_SERVER_PORT = int(os.getenv("PORT", os.getenv("WEB_SERVER_PORT", "7860")))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip() or "/telegram/webhook"
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", hashlib.sha256(BOT_TOKEN.encode()).hexdigest())
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "bot_memory.sqlite3")

# Free-tier daily quotas (approx): gemini-1.5-flash-8b=1500, gemini-1.5-flash=1500,
# gemini-2.0-flash=1500, gemini-2.5-flash=50.
# Order: highest free quota first so quota exhaustion on one model falls over to the next.
MODEL_CANDIDATES = (
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
)

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_MEMORY_MESSAGE_CHARS = int(os.getenv("MAX_MEMORY_MESSAGE_CHARS", "2500"))
IMAGE_CONTEXT_TTL_SECONDS = int(os.getenv("IMAGE_CONTEXT_TTL_SECONDS", "3600"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1400"))
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", "3.0"))
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_SAFE_CHUNK_SIZE = 3800

# Sentinel values returned by _extract_text
_SAFETY_BLOCKED = "__SAFETY_BLOCKED__"
_RECITATION_BLOCKED = "__RECITATION_BLOCKED__"

TEXT_MODE_BUTTON = "💬 Text Mode"
PHOTO_MODE_BUTTON = "🖼 Photo Mode"

SYSTEM_INSTRUCTION = (
    "You are a friendly and helpful AI assistant in Telegram. "
    "Respond naturally and directly, without phrases like 'I am a language model' unless the user asks. "
    "Take recent conversation history into account. "
    "If the user sends a short follow-up like 'ok', 'go on', 'continue', or 'solve them', "
    "connect it to the previous context rather than treating it as a brand-new question. "
    "If the user clearly starts a completely new topic, do not carry over old tasks, photos, or previous context. "
    "If the user already sent an image and then sends a text clarification related to the photo, "
    "treat it as a follow-up to the most recent image. "
    "If the answer is long, give the useful summary first, then the details. "
    "Respond in English by default unless the user asks otherwise."
)

WORD_RE = re.compile(r"[a-zA-Z0-9]+")
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "they", "them", "his", "her",
    "not", "no", "so", "as", "just", "also", "then", "than", "more", "very",
    "what", "how", "who", "when", "where", "which", "there", "here", "now",
}
CONTINUATION_PHRASES = {
    "ok", "go on", "continue", "and then", "what next", "keep going",
    "solve it", "solve them", "solve this", "explain", "elaborate",
    "tell me more", "more details", "more", "next", "proceed",
    "what was that", "what did you say", "repeat", "summarize",
    "make it shorter", "make it brief", "finish it", "complete it",
    "what do you think", "and", "so", "well", "right",
}
PHOTO_CONTINUATION_PHRASES = {
    "solve it", "solve them", "what is in the photo", "what does it say",
    "read it", "analyze it", "explain the task", "answer based on the photo",
    "continue solving", "what does this say", "describe it",
}
COMMON_WORD_ENDINGS = (
    "tion", "sion", "ness", "ment", "ing", "ive", "ous", "ful", "less",
    "ible", "able", "ance", "ence", "ity", "ies", "ied", "ers", "est",
    "ily", "eed", "ed", "er", "es", "ly", "al", "ic", "en", "ry", "ty",
    "fy", "ze", "se", "ce",
)


@dataclass
class PromptRoutingDecision:
    reset_context: bool
    use_image_context: bool


def normalize_prompt(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def extract_keywords(text: str) -> set[str]:
    words = {m.group(0).lower() for m in WORD_RE.finditer(text or "")}
    return {stem_word(w) for w in words if len(w) >= 3 and w not in STOP_WORDS}


def stem_word(word: str) -> str:
    normalized = word.lower()
    for ending in COMMON_WORD_ENDINGS:
        if len(normalized) > len(ending) + 3 and normalized.endswith(ending):
            return normalized[: -len(ending)]
    return normalized


def contains_any_phrase(text: str, phrases: set[str]) -> bool:
    normalized = normalize_prompt(text)
    return any(phrase in normalized for phrase in phrases)


def is_short_followup(text: str) -> bool:
    normalized = normalize_prompt(text)
    return len(normalized) <= 40 or len(extract_keywords(normalized)) <= 2


def recent_topic_overlap(prompt: str, recent_texts: list[str]) -> float:
    prompt_keywords = extract_keywords(prompt)
    recent_keywords = extract_keywords(" ".join(recent_texts))
    if not prompt_keywords or not recent_keywords:
        return 0.0
    shared = prompt_keywords & recent_keywords
    return len(shared) / max(1, len(prompt_keywords))


def decide_prompt_routing(
    prompt: str, recent_texts: list[str], has_image_context: bool
) -> PromptRoutingDecision:
    normalized = normalize_prompt(prompt)
    if not normalized:
        return PromptRoutingDecision(reset_context=False, use_image_context=False)
    if contains_any_phrase(normalized, CONTINUATION_PHRASES):
        return PromptRoutingDecision(
            reset_context=False,
            use_image_context=has_image_context
            and contains_any_phrase(normalized, PHOTO_CONTINUATION_PHRASES),
        )
    overlap = recent_topic_overlap(normalized, recent_texts)
    standalone_prompt = len(normalized) >= 20 or len(extract_keywords(normalized)) >= 3
    clear_topic_shift = standalone_prompt and overlap < 0.2
    if clear_topic_shift:
        return PromptRoutingDecision(reset_context=True, use_image_context=False)
    if has_image_context and contains_any_phrase(normalized, PHOTO_CONTINUATION_PHRASES):
        return PromptRoutingDecision(reset_context=False, use_image_context=True)
    if has_image_context and is_short_followup(normalized) and overlap >= 0.2:
        return PromptRoutingDecision(reset_context=False, use_image_context=True)
    return PromptRoutingDecision(reset_context=False, use_image_context=False)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_gemini_bot")


class UserState(StatesGroup):
    waiting_text = State()
    waiting_photo = State()


@dataclass
class ImageContext:
    image_bytes: bytes
    mime_type: str
    saved_at: float


# ---------------------------------------------------------------------------
# Per-user rate limiter — prevents spam and protects API quota
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, min_interval: float = 3.0) -> None:
        self.min_interval = min_interval
        self._last: dict[int, float] = {}

    def check(self, user_id: int) -> tuple[bool, float]:
        now = time.time()
        last = self._last.get(user_id, 0.0)
        elapsed = now - last
        if elapsed < self.min_interval:
            return False, round(self.min_interval - elapsed, 1)
        self._last[user_id] = now
        return True, 0.0


# ---------------------------------------------------------------------------
# Conversation memory (SQLite)
# ---------------------------------------------------------------------------

class ConversationMemory:
    def __init__(self, db_path: str, max_messages: int, max_message_chars: int) -> None:
        self.db_path = db_path
        self.max_messages = max_messages
        self.max_message_chars = max_message_chars
        self._lock = threading.Lock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_conv_chat_id
                ON conversation_messages (chat_id, id)"""
            )

    def _clip(self, text: str) -> str:
        cleaned = (text or "").strip()
        if len(cleaned) <= self.max_message_chars:
            return cleaned
        return cleaned[: self.max_message_chars].rstrip() + "\n\n[Truncated]"

    def _prune(self, conn: sqlite3.Connection, chat_id: int) -> None:
        rows = conn.execute(
            "SELECT id FROM conversation_messages WHERE chat_id=? ORDER BY id DESC",
            (chat_id,),
        ).fetchall()
        stale = rows[self.max_messages :]
        if stale:
            conn.executemany(
                "DELETE FROM conversation_messages WHERE id=?",
                [(r["id"],) for r in stale],
            )

    def _add_sync(self, chat_id: int, user_text: str, assistant_text: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation_messages (chat_id, role, content) VALUES (?,?,?)",
                (chat_id, "user", self._clip(user_text)),
            )
            conn.execute(
                "INSERT INTO conversation_messages (chat_id, role, content) VALUES (?,?,?)",
                (chat_id, "assistant", self._clip(assistant_text)),
            )
            self._prune(conn, chat_id)

    async def add_exchange(self, chat_id: int, user_text: str, assistant_text: str) -> None:
        await asyncio.to_thread(self._add_sync, chat_id, user_text, assistant_text)

    def _get_contents_sync(self, chat_id: int) -> list[types.Content]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversation_messages WHERE chat_id=? ORDER BY id ASC",
                (chat_id,),
            ).fetchall()
        contents: list[types.Content] = []
        for row in rows:
            part = types.Part(text=row["content"])
            if row["role"] == "assistant":
                contents.append(types.ModelContent(parts=[part]))
            else:
                contents.append(types.UserContent(parts=[part]))
        return contents

    async def get_contents(self, chat_id: int) -> list[types.Content]:
        return await asyncio.to_thread(self._get_contents_sync, chat_id)

    def _get_recent_texts_sync(self, chat_id: int, limit: int) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT content FROM conversation_messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [row["content"] for row in reversed(rows)]

    async def get_recent_texts(self, chat_id: int, limit: int = 6) -> list[str]:
        return await asyncio.to_thread(self._get_recent_texts_sync, chat_id, limit)

    def _clear_sync(self, chat_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM conversation_messages WHERE chat_id=?", (chat_id,))

    async def clear_chat(self, chat_id: int) -> None:
        await asyncio.to_thread(self._clear_sync, chat_id)


class ImageSessionStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: dict[int, ImageContext] = {}

    def remember(self, chat_id: int, image_bytes: bytes, mime_type: str) -> None:
        self._items[chat_id] = ImageContext(
            image_bytes=image_bytes, mime_type=mime_type, saved_at=time.time()
        )

    def get(self, chat_id: int) -> ImageContext | None:
        ctx = self._items.get(chat_id)
        if ctx is None:
            return None
        if time.time() - ctx.saved_at > self.ttl_seconds:
            self._items.pop(chat_id, None)
            return None
        return ctx

    def clear(self, chat_id: int) -> None:
        self._items.pop(chat_id, None)


class RepeatingChatAction:
    def __init__(
        self,
        bot: Bot,
        chat_id: int,
        action: ChatAction = ChatAction.TYPING,
        interval: float = 4.0,
    ) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.action = action
        self.interval = interval
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            await self.bot.send_chat_action(self.chat_id, self.action)
            await asyncio.sleep(self.interval)

    async def __aenter__(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def __aexit__(self, *_: Any) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=TEXT_MODE_BUTTON), KeyboardButton(text=PHOTO_MODE_BUTTON)]],
        resize_keyboard=True,
        input_field_placeholder="Choose a mode",
    )


def split_text(text: str, chunk_size: int = TELEGRAM_SAFE_CHUNK_SIZE) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return ["Could not generate a response."]
    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > TELEGRAM_TEXT_LIMIT:
        split_at = remaining.rfind("\n\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = remaining.rfind(". ", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = remaining.rfind(" ", 0, chunk_size)
        if split_at <= 0:
            split_at = chunk_size
        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:chunk_size]
            split_at = chunk_size
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


async def answer_in_chunks(message: Message, text: str) -> None:
    for chunk in split_text(text):
        await message.answer(chunk)


# ---------------------------------------------------------------------------
# Gemini service — multi-model fallback with smart error detection
# ---------------------------------------------------------------------------

class GeminiService:
    def __init__(
        self,
        api_key: str,
        models: tuple[str, ...],
        memory: ConversationMemory,
        timeout_seconds: int = 90,
        max_output_tokens: int = 1400,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.models = models
        self.memory = memory
        self.timeout_seconds = timeout_seconds
        self.generation_config = types.GenerateContentConfig(
            systemInstruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            maxOutputTokens=max_output_tokens,
        )

    async def ask_text(self, chat_id: int, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return "Empty prompt. Please type something."
        logger.info("Text request chat_id=%s chars=%s", chat_id, len(prompt))
        contents = await self.memory.get_contents(chat_id)
        contents.append(types.UserContent(parts=[types.Part(text=prompt)]))
        reply = await self._generate(contents)
        if not reply.startswith("__"):
            await self.memory.add_exchange(chat_id, prompt, reply)
        return reply

    async def ask_image(
        self,
        chat_id: int,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/jpeg",
    ) -> str:
        if not image_bytes:
            return "Could not read the image. Please try again."
        prompt = (prompt or "").strip() or "What is in this image? Describe it in detail."
        logger.info(
            "Photo request chat_id=%s prompt_chars=%s image_bytes=%s",
            chat_id, len(prompt), len(image_bytes),
        )
        contents = await self.memory.get_contents(chat_id)
        contents.append(
            types.UserContent(
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part(text=prompt),
                ]
            )
        )
        reply = await self._generate(contents)
        if not reply.startswith("__"):
            await self.memory.add_exchange(chat_id, f"[Photo] {prompt}", reply)
        return reply

    async def _generate(self, contents: list[types.Content]) -> str:
        last_error: Exception | None = None

        for attempt, model_name in enumerate(self.models):
            if attempt > 0:
                delay = 2.0 * attempt  # 2s, 4s, 6s between retries
                logger.info("Waiting %.1fs before trying next model (%s)...", delay, model_name)
                await asyncio.sleep(delay)

            try:
                logger.info("Trying model: %s (attempt %d/%d)", model_name, attempt + 1, len(self.models))
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model_name,
                        contents=contents,
                        config=self.generation_config,
                    ),
                    timeout=self.timeout_seconds,
                )

                text = self._extract_text(response)

                if text == _SAFETY_BLOCKED:
                    logger.info("Safety filter blocked content on model %s.", model_name)
                    return (
                        "This content could not be processed due to safety guidelines.\n\n"
                        "Please try with a different image or question."
                    )

                if text == _RECITATION_BLOCKED:
                    logger.info("Recitation block on model %s.", model_name)
                    return (
                        "The response was blocked to avoid reproducing protected content.\n"
                        "Try rephrasing your request."
                    )

                if text:
                    logger.info("Success with model %s.", model_name)
                    return text

                logger.warning("Model %s returned empty response — trying next.", model_name)

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError("Request timed out")
                logger.warning("Timeout on model %s.", model_name)
                continue

            except Exception as exc:
                last_error = exc
                logger.warning("Model %s error: %s", model_name, exc, exc_info=False)
                if self._should_try_next(exc):
                    continue
                break

        # All models failed — show a specific, honest message
        logger.error("All %d models exhausted. Last error: %r", len(self.models), last_error)
        error_str = str(last_error or "").lower()

        if "timeout" in error_str or "deadline exceeded" in error_str:
            return (
                "The request took too long to process.\n\n"
                "Try shortening your message or breaking it into smaller parts."
            )

        if any(k in error_str for k in ("429", "quota", "rate limit", "resource exhausted")):
            return (
                "The daily request limit for the AI service has been reached.\n\n"
                "This limit resets every 24 hours. Please try again later.\n"
                "Use /new to start a fresh conversation when you come back."
            )

        return (
            "The AI service is not responding right now.\n\n"
            "Please try again in a moment. If it keeps failing, use /new to start fresh."
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                feedback = getattr(response, "prompt_feedback", None)
                if feedback:
                    block_reason = str(getattr(feedback, "block_reason", "") or "")
                    if block_reason and block_reason not in ("0", "BLOCK_REASON_UNSPECIFIED", ""):
                        return _SAFETY_BLOCKED

            for candidate in candidates:
                finish_reason = getattr(candidate, "finish_reason", None)
                finish_str = str(finish_reason or "").upper()
                if "SAFETY" in finish_str or finish_str == "2":
                    return _SAFETY_BLOCKED
                if "RECITATION" in finish_str or finish_str == "4":
                    return _RECITATION_BLOCKED
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) or []
                chunks = [
                    p_text.strip()
                    for part in parts
                    if isinstance(p_text := getattr(part, "text", None), str) and p_text.strip()
                ]
                if chunks:
                    return "\n".join(chunks)
        except Exception:
            pass
        return ""

    @staticmethod
    def _should_try_next(exc: Exception) -> bool:
        msg = str(exc).lower()
        retryable = (
            "429", "quota", "rate limit", "resource exhausted",
            "temporarily unavailable", "service unavailable",
            "internal", "unavailable", "404", "not found", "model",
            "permission", "billing", "invalid", "not supported",
        )
        return any(k in msg for k in retryable)


# ---------------------------------------------------------------------------
# App-level instances
# ---------------------------------------------------------------------------

conversation_memory = ConversationMemory(
    db_path=MEMORY_DB_PATH,
    max_messages=MAX_HISTORY_MESSAGES,
    max_message_chars=MAX_MEMORY_MESSAGE_CHARS,
)
image_context_store = ImageSessionStore(ttl_seconds=IMAGE_CONTEXT_TTL_SECONDS)
rate_limiter = RateLimiter(min_interval=RATE_LIMIT_SECONDS)

gemini_service = GeminiService(
    api_key=GEMINI_API_KEY,
    models=MODEL_CANDIDATES,
    memory=conversation_memory,
    timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    max_output_tokens=MAX_OUTPUT_TOKENS,
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


# ---------------------------------------------------------------------------
# Webhook helpers
# ---------------------------------------------------------------------------

def webhook_url() -> str:
    return f"{WEBHOOK_BASE_URL}{WEBHOOK_PATH}"


def running_in_webhook_mode() -> bool:
    return bool(WEBHOOK_BASE_URL)


async def configure_webhook_with_retry(attempts: int = 10, delay_seconds: int = 6) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            await bot.set_webhook(
                url=webhook_url(),
                secret_token=WEBHOOK_SECRET,
                allowed_updates=dp.resolve_used_update_types(),
            )
            logger.info("Webhook registered: %s", webhook_url())
            return True
        except Exception as exc:
            logger.warning("Webhook attempt %s/%s failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                await asyncio.sleep(delay_seconds)
    logger.error("Webhook could not be registered.")
    return False


async def disable_webhook_for_polling() -> None:
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        logger.info("Webhook disabled — polling mode.")
    except Exception as exc:
        logger.warning("Could not disable webhook: %s", exc)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

async def _check_rate_limit(message: Message) -> bool:
    allowed, wait_time = rate_limiter.check(message.from_user.id)
    if not allowed:
        await message.answer(
            f"Please slow down a bit. Wait {wait_time}s before sending another request."
        )
    return allowed


async def process_text_prompt(message: Message, state: FSMContext, prompt: str) -> None:
    await state.set_state(UserState.waiting_text)
    async with RepeatingChatAction(message.bot, message.chat.id):
        reply = await gemini_service.ask_text(chat_id=message.chat.id, prompt=prompt)
    await answer_in_chunks(message, reply)


async def process_photo_prompt(
    message: Message,
    state: FSMContext,
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> None:
    await state.set_state(UserState.waiting_photo)
    async with RepeatingChatAction(message.bot, message.chat.id):
        reply = await gemini_service.ask_image(
            chat_id=message.chat.id,
            image_bytes=image_bytes,
            prompt=prompt,
            mime_type=mime_type,
        )
    await answer_in_chunks(message, reply)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@dp.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Hello! I'm an AI assistant powered by Google Gemini.\n\n"
        "Send me a text message or a photo and I'll do my best to help.\n\n"
        "Use /new to clear conversation memory.\n"
        "Use /help to see all available commands.",
        reply_markup=main_menu(),
    )


@dp.message(Command("help"))
async def help_handler(message: Message) -> None:
    await message.answer(
        "Available commands:\n\n"
        "/start — Start the bot\n"
        "/new — Clear conversation memory and start fresh\n"
        "/cancel — Reset mode (memory is kept)\n"
        "/help — Show this message\n\n"
        "Modes:\n"
        "💬 Text Mode — Ask any text question\n"
        "🖼 Photo Mode — Send a photo for analysis\n\n"
        "Tips:\n"
        "• The bot remembers your last 12 messages.\n"
        "• For very long questions, break them into smaller parts.\n"
        "• Some images may not be supported due to content guidelines.\n"
        "• If the AI isn't responding, the daily quota may be reached — try again in a few hours.",
        reply_markup=main_menu(),
    )


@dp.message(Command("cancel"))
async def cancel_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Mode reset. Conversation memory is preserved.\n"
        "Use /new to start completely fresh.",
        reply_markup=main_menu(),
    )


@dp.message(Command(commands=["new", "reset", "clear"]))
async def new_dialog_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await conversation_memory.clear_chat(message.chat.id)
    image_context_store.clear(message.chat.id)
    await message.answer(
        "Conversation memory cleared. Starting fresh.",
        reply_markup=main_menu(),
    )


@dp.message(F.text == TEXT_MODE_BUTTON)
async def text_mode_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_text)
    await message.answer("Text mode enabled. Go ahead and type your question.")


@dp.message(F.text == PHOTO_MODE_BUTTON)
async def photo_mode_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_photo)
    await message.answer(
        "Photo mode enabled.\n"
        "Send a photo (with an optional caption), or ask a follow-up about the last photo."
    )


@dp.message(F.photo)
async def photo_message_handler(message: Message, state: FSMContext) -> None:
    if not await _check_rate_limit(message):
        return
    try:
        photo = message.photo[-1]
        file_buffer = await message.bot.download(photo)
        if not isinstance(file_buffer, BytesIO):
            await message.answer("Could not download the photo. Please try again.")
            return
        image_bytes = file_buffer.getvalue()
        image_context_store.remember(message.chat.id, image_bytes, "image/jpeg")
        prompt = (message.caption or "").strip() or "What is in this image? Describe it in detail."
        await process_photo_prompt(
            message=message, state=state,
            prompt=prompt, image_bytes=image_bytes, mime_type="image/jpeg",
        )
    except Exception as exc:
        logger.exception("Error processing photo: %s", exc)
        await message.answer("An error occurred while processing the photo. Please try again.")


@dp.message(F.text)
async def text_message_handler(message: Message, state: FSMContext) -> None:
    prompt = (message.text or "").strip()
    if not prompt:
        await message.answer("Please type a message or send a photo.")
        return
    if not await _check_rate_limit(message):
        return

    current_state = await state.get_state()
    recent_texts = await conversation_memory.get_recent_texts(message.chat.id, limit=6)
    image_context = image_context_store.get(message.chat.id)
    routing = decide_prompt_routing(
        prompt=prompt,
        recent_texts=recent_texts,
        has_image_context=image_context is not None,
    )

    try:
        if routing.reset_context:
            logger.info("New topic detected chat_id=%s — clearing context.", message.chat.id)
            await conversation_memory.clear_chat(message.chat.id)
            image_context_store.clear(message.chat.id)
            await state.set_state(UserState.waiting_text)
        elif (
            current_state == UserState.waiting_photo.state
            and image_context is not None
            and routing.use_image_context
        ):
            logger.info("Using last photo as context chat_id=%s.", message.chat.id)
            await process_photo_prompt(
                message=message, state=state, prompt=prompt,
                image_bytes=image_context.image_bytes, mime_type=image_context.mime_type,
            )
            return

        await process_text_prompt(message=message, state=state, prompt=prompt)

    except Exception as exc:
        logger.exception("Error processing text: %s", exc)
        await message.answer(
            "Something went wrong. Please try again.\n"
            "If it keeps failing, use /new to start fresh."
        )


@dp.message()
async def fallback_handler(message: Message) -> None:
    await message.answer(
        "I can work with text messages and photos.\n"
        "Send me a message to get started.",
        reply_markup=main_menu(),
    )


# ---------------------------------------------------------------------------
# Web endpoints
# ---------------------------------------------------------------------------

async def healthcheck(_: web.Request) -> web.Response:
    mode = "webhook" if running_in_webhook_mode() else "polling"
    return web.json_response({"ok": True, "mode": mode})


async def telegram_check(_: web.Request) -> web.Response:
    try:
        me = await asyncio.wait_for(bot.get_me(request_timeout=8), timeout=10)
        return web.json_response(
            {"ok": True, "telegram_api": True, "bot_username": me.username, "bot_id": me.id}
        )
    except Exception as exc:
        return web.json_response(
            {"ok": False, "telegram_api": False, "error": str(exc)}, status=503
        )


def create_web_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", healthcheck)
    app.router.add_get("/healthz", healthcheck)
    app.router.add_get("/telegram-check", telegram_check)
    SimpleRequestHandler(
        dispatcher=dp, bot=bot, secret_token=WEBHOOK_SECRET
    ).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)
    return app


async def run_webhook() -> None:
    app = create_web_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)
    await site.start()
    logger.info("Bot started in webhook mode on %s:%s", WEB_SERVER_HOST, WEB_SERVER_PORT)
    await configure_webhook_with_retry()
    await asyncio.Event().wait()


async def run_polling() -> None:
    logger.info("Bot starting in polling mode...")
    await disable_webhook_for_polling()
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


async def main() -> None:
    if running_in_webhook_mode():
        await run_webhook()
    else:
        await run_polling()


if __name__ == "__main__":
    asyncio.run(main())
