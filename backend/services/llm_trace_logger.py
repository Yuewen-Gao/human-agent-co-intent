"""Session-scoped, fail-open traces for Map Task Guide LLM calls."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # Available in the Linux backend container and on macOS local tests.
    import fcntl
except ImportError:  # pragma: no cover - defensive fallback for non-POSIX hosts.
    fcntl = None


LOGGER = logging.getLogger(__name__)
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_WRITE_LOCK = threading.Lock()
_SECRET_KEY_PARTS = ("api_key", "apikey", "authorization", "token", "secret", "password", "credential")


@dataclass(frozen=True)
class TraceContext:
    """Stable identifiers for one Map Task Guide model invocation."""

    session_id: str
    participant_id: str
    turn_id: str
    stage: str
    trigger_context: str


@dataclass(frozen=True)
class TracedCompletion:
    """Provider output plus the trace identity used by later result records."""

    response: str
    context: TraceContext
    call_id: str
    trace_enabled: bool


def trace_logging_enabled() -> bool:
    return os.getenv("LLM_TRACE_LOGGING", "").strip().lower() in {"1", "true", "yes", "on"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    """Return a detached JSON-compatible value without modifying caller input."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _redact_secrets(value: Any, key: str = "") -> Any:
    """Copy provider options for logging without preserving credentials."""
    if any(part in key.lower().replace("-", "_") for part in _SECRET_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): _redact_secrets(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item, key) for item in value]
    return _json_value(value)


def _redact_error(error: Exception | str) -> str:
    text = f"{error.__class__.__name__}: {error}" if isinstance(error, Exception) else str(error)
    text = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|authorization|token|secret|password|credential)[\"']?\s*[:=]\s*)[\"'][^\"']*[\"']",
        r'\1"[REDACTED]"',
        text,
    )
    text = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|authorization|token|secret|password|credential)[\"']?\s*[:=]\s*)[^\s,;}]+",
        r"\1[REDACTED]",
        text,
    )
    return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)


def _logs_root(logs_root: Path | str | None) -> Path:
    if logs_root is not None:
        return Path(logs_root).resolve()
    from services.action_logger import LOGS_BASE_DIR

    return Path(LOGS_BASE_DIR).resolve()


def _trace_path(context: TraceContext, logs_root: Path | str | None) -> Path:
    if not _SESSION_ID.fullmatch(context.session_id or ""):
        raise ValueError("invalid session identifier for LLM trace")
    root = _logs_root(logs_root)
    path = (root / context.session_id / "llm_requests.jsonl").resolve()
    if path.parent.parent != root:
        raise ValueError("LLM trace path escapes the logs root")
    return path


def _provider_metadata(llm_client: Any, requested_model: str | None) -> dict[str, str | None]:
    class_name = llm_client.__class__.__name__
    provider = class_name[:-6] if class_name.endswith("Client") else class_name
    effective_model = requested_model if isinstance(requested_model, str) and requested_model.strip() else None
    if effective_model is None:
        effective_model = getattr(llm_client, "default_model", None)
    if effective_model is None:
        effective_model = getattr(llm_client, "deployment", None)
    if effective_model is None and class_name == "ClaudeClient":
        effective_model = "claude-3-5-sonnet-20241022"
    if effective_model is None and class_name == "MockLLMClient":
        effective_model = "mock"
    return {
        "provider": provider,
        "requested_model": requested_model,
        "effective_model": str(effective_model) if effective_model is not None else None,
    }


def _append_event(
    context: TraceContext,
    event: Mapping[str, Any],
    *,
    logs_root: Path | str | None,
    durable: bool = False,
) -> bool:
    """Append exactly one JSON record. Failures are intentionally non-fatal."""
    try:
        path = _trace_path(context, logs_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_json_value(event), ensure_ascii=False, separators=(",", ":")) + "\n"
        with _WRITE_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    if durable:
                        os.fsync(handle.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True
    except Exception as error:  # Trace failure must not alter agent behavior.
        LOGGER.warning("LLM trace write failed (%s)", error.__class__.__name__)
        return False


def trace_chat_completion(
    llm_client: Any,
    context: TraceContext,
    *,
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    logs_root: Path | str | None = None,
    enabled: bool | None = None,
    **kwargs: Any,
) -> TracedCompletion:
    """Call an unchanged provider and persist request/response records when enabled."""
    is_enabled = trace_logging_enabled() if enabled is None else enabled
    call_id = str(uuid.uuid4())
    if is_enabled:
        request_event = {
            "event": "request",
            "timestamp": _timestamp(),
            "call_id": call_id,
            "session_id": context.session_id,
            "participant_id": context.participant_id,
            "turn_id": context.turn_id,
            "stage": context.stage,
            "trigger_context": context.trigger_context,
            **_provider_metadata(llm_client, model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Serialize inside _append_event's fail-open boundary. Do not deep-copy
            # caller objects here: an unusual but provider-valid object must not
            # prevent the original completion request from being sent.
            "request_messages": messages,
            "request_options": _redact_secrets(kwargs),
        }
        _append_event(context, request_event, logs_root=logs_root, durable=True)

    started = time.perf_counter()
    try:
        response = llm_client.chat_completions_create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    except Exception as error:
        if is_enabled:
            _append_event(context, {
                "event": "response",
                "timestamp": _timestamp(),
                "call_id": call_id,
                "session_id": context.session_id,
                "participant_id": context.participant_id,
                "turn_id": context.turn_id,
                "stage": context.stage,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "raw_response": None,
                "error": _redact_error(error),
            }, logs_root=logs_root)
        raise

    text = response if isinstance(response, str) else str(response)
    if is_enabled:
        _append_event(context, {
            "event": "response",
            "timestamp": _timestamp(),
            "call_id": call_id,
            "session_id": context.session_id,
            "participant_id": context.participant_id,
            "turn_id": context.turn_id,
            "stage": context.stage,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "raw_response": text,
            "error": None,
        }, logs_root=logs_root)
    return TracedCompletion(text, context, call_id, is_enabled)


def record_trace_result(
    completion: TracedCompletion,
    parsed_result: Any = None,
    *,
    execution_result: Any = None,
    error: str | None = None,
    logs_root: Path | str | None = None,
) -> None:
    """Append a parsed or execution result without changing the original call."""
    if not completion.trace_enabled:
        return
    _append_event(completion.context, {
        "event": "result",
        "timestamp": _timestamp(),
        "call_id": completion.call_id,
        "session_id": completion.context.session_id,
        "participant_id": completion.context.participant_id,
        "turn_id": completion.context.turn_id,
        "stage": completion.context.stage,
        "parsed_result": parsed_result,
        "execution_result": execution_result,
        "error": _redact_error(error) if error else None,
    }, logs_root=logs_root)
