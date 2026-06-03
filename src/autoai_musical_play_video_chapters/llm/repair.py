"""LLM response-repair helper functions."""

from __future__ import annotations

import json
import logging

from autoai_musical_play_video_chapters.config import load_settings

_SETTINGS = load_settings()
EMPTY_REPAIR_SAFETY_MARGIN = _SETTINGS.empty_repair_safety_margin
NUM_CTX_HARD_CAP = _SETTINGS.num_ctx_hard_cap


def _schema_text(schema):
    """Render a schema compactly for repair prompts."""
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_repair_prompt(schema, original_prompt):
    """Create a concise repair prompt for invalid JSON responses."""
    return (
        "your previous response did not match the schema; return only valid JSON matching: "
        f"{_schema_text(schema)}\n\nOriginal task:\n{original_prompt}"
    )


def build_empty_response_repair_prompt(schema, original_prompt):
    """Create a no-think JSON-only prompt for empty-response repair."""
    if schema is not None:
        prefix = "Respond with JSON ONLY. Do NOT think. Do NOT explain. Output must match this schema: "
        prefix += _schema_text(schema)
    else:
        prefix = "Respond with JSON ONLY. Do NOT think. Do NOT explain."
    return f"{prefix}\n\n{original_prompt}"


def clamp_empty_response_budget(num_ctx, num_predict, prompt_tokens_est):
    """Keep repair-call budgets within the configured context ceilings."""
    num_ctx = max(1, int(num_ctx))
    num_predict = max(1, int(num_predict))
    prompt_tokens_est = int(prompt_tokens_est)
    needed_ctx = num_predict + prompt_tokens_est + EMPTY_REPAIR_SAFETY_MARGIN
    if num_ctx < needed_ctx:
        num_ctx = min(max(num_ctx, needed_ctx), NUM_CTX_HARD_CAP)
    if num_ctx < needed_ctx:
        num_predict = max(1, num_ctx - prompt_tokens_est - EMPTY_REPAIR_SAFETY_MARGIN)
    return num_ctx, num_predict


def empty_response_details(raw):
    """Extract visible content and optional reasoning fields from a model response."""
    data = json.loads(raw)
    choices = data.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("thinking")
    finish_reason = first_choice.get("finish_reason")
    has_reasoning_field = "reasoning" in message or "thinking" in message
    reasoning_chars = len(reasoning or "")
    return data, content, finish_reason, has_reasoning_field, reasoning_chars


def log_empty_response_warning(record):
    """Emit a WARNING-level structured log for empty visible content."""
    logging.warning(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def log_empty_response_error(record):
    """Emit an ERROR-level structured log for an exhausted empty-response repair."""
    logging.error(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
