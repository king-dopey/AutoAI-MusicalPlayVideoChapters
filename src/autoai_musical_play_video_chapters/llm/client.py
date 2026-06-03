"""Low-level LLM transport and retry helper functions."""

from __future__ import annotations

from email.utils import parsedate_to_datetime
import json
import logging
import math
import random
import re
import socket
import time
from urllib import error, request

from autoai_musical_play_video_chapters.config import load_settings
from autoai_musical_play_video_chapters.llm.repair import (
    build_empty_response_repair_prompt,
    build_repair_prompt,
    clamp_empty_response_budget,
    empty_response_details,
    log_empty_response_error,
    log_empty_response_warning,
)
from autoai_musical_play_video_chapters.llm.schemas import (
    safe_response_format,
    validate_json_schema,
)

_SETTINGS = load_settings()

BASE_URL = _SETTINGS.base_url
MODEL_DETECT = _SETTINGS.model_detect
MODEL_EXTRACT = _SETTINGS.model_extract
MODEL_SUMMARY = _SETTINGS.model_summary
MODEL_VERIFY = _SETTINGS.model_verify
API_KEY = _SETTINGS.api_key
TEMPERATURE = _SETTINGS.temperature
MAX_RETRIES = _SETTINGS.max_retries
BASE_SLEEP = _SETTINGS.base_sleep
MAX_SLEEP = _SETTINGS.max_sleep
ERR_BODY_CHARS = _SETTINGS.err_body_chars
REQUEST_TIMEOUT = _SETTINGS.request_timeout
TASK_TIMEOUT = _SETTINGS.task_timeout
LOG_RAW_EMPTY = _SETTINGS.log_raw_empty
EMPTY_REPAIR_MAX_STEPS = _SETTINGS.empty_repair_max_steps
VALIDATION_REPAIR_MAX_STEPS = _SETTINGS.validation_repair_max_steps
BOUNDARY_REFINE_FALLBACK_WINDOW = _SETTINGS.boundary_refine_fallback_window
NUM_PREDICT_HARD_CAP = _SETTINGS.num_predict_hard_cap
THINK_DETECT = _SETTINGS.think_detect
THINK_EXTRACT = _SETTINGS.think_extract
THINK_SUMMARY = _SETTINGS.think_summary
THINK_VERIFY = _SETTINGS.think_verify
NUM_CTX_DETECT = _SETTINGS.num_ctx_detect
NUM_CTX_EXTRACT = _SETTINGS.num_ctx_extract
NUM_CTX_SUMMARY = _SETTINGS.num_ctx_summary
NUM_CTX_VERIFY = _SETTINGS.num_ctx_verify
NUM_PREDICT_DETECT = _SETTINGS.num_predict_detect
NUM_PREDICT_EXTRACT = _SETTINGS.num_predict_extract
NUM_PREDICT_SUMMARY = _SETTINGS.num_predict_summary
NUM_PREDICT_VERIFY = _SETTINGS.num_predict_verify
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class LLMTaskError(RuntimeError):
    """Raised when an LLM task cannot be completed after retries."""


class LLMResponseValidationError(LLMTaskError):
    """Raised when a response fails schema or candidate validation."""

    def __init__(
        self,
        message,
        *,
        received_value=None,
        allowed_values=None,
        repair_step=None,
        schema_kind=None,
        last_payload=None,
    ):
        """Initialize a validation error with structured repair metadata."""
        super().__init__(message)
        self.received_value = received_value
        self.allowed_values = allowed_values
        self.repair_step = repair_step
        self.schema_kind = schema_kind
        self.last_payload = last_payload


class LLMTaskTimeoutError(LLMTaskError):
    """Raised when a logical task exceeds its wall-clock budget."""


class LLMTaskRetryExhaustedError(LLMTaskError):
    """Raised when a task exhausts retries after transient failures."""


class LLMEmptyResponseError(LLMTaskError):
    """Raised when a 2xx response returns empty visible content."""

    def __init__(self, message, *, raw_response_body=""):
        super().__init__(message)
        self.raw_response_body = raw_response_body


def print_server_error_detail(status, hdrs, body, label):
    """Print a compact diagnostic block for failed HTTP interactions."""
    print(f"\n--- {label} ---")
    print(f"HTTP {status}")
    header_items = []
    if hdrs:
        try:
            header_items = list(hdrs.items())[:60]
        except Exception:
            header_items = []
    if header_items:
        print("Headers:")
        for key, value in header_items:
            print(f"{key}: {value}")
    print(f"Body (first {ERR_BODY_CHARS} chars):")
    print((body or "")[:ERR_BODY_CHARS])
    print(f"--- END {label} ---\n")


def http_post_json(url, payload, timeout: float = 3600.0, headers=None):
    """Send a JSON POST request and return status, body, and headers."""
    request_headers = {"Content-Type": "application/json"}
    if API_KEY:
        request_headers["Authorization"] = f"Bearer {API_KEY}"
    if headers:
        request_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw, dict(resp.headers)
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return exc.code, body, dict(getattr(exc, "headers", {}) or {})


def task_deadline(timeout_s=TASK_TIMEOUT):
    """Return a monotonic deadline for a logical task."""
    return time.monotonic() + max(1.0, float(timeout_s))


def task_time_remaining(deadline):
    """Return remaining seconds before a deadline, or 0 when expired."""
    return max(0.0, deadline - time.monotonic())


def parse_retry_after(headers):
    """Parse Retry-After headers into a delay in seconds."""
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    raw = str(raw).strip()
    if raw.isdigit():
        return float(raw)
    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        return None
    if dt is None:
        return None
    return max(0.0, dt.timestamp() - time.time())


def is_retryable_status(status):
    """Return True for HTTP status codes that should be retried."""
    return status in RETRYABLE_STATUS_CODES


def is_retryable_exception(exc):
    """Return True when a transport exception should be retried."""
    return isinstance(exc, (error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError))


def log_llm_call(record):
    """Emit a structured single-line call log."""
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def estimate_prompt_tokens(text):
    """Estimate prompt tokens using a cheap character heuristic."""
    return max(1, int(math.ceil(len(text or "") / 4.0)))


def build_chat_payload(messages, temperature, top_p=None, response_format=None, model=None):
    """Build an OpenAI-compatible chat/completions payload."""
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


def extract_json_payload(text):
    """Extract JSON text from a plain or fenced model reply."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    try:
        json.loads(text)
        return text
    except Exception:
        candidate = None

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fence:
        candidate = fence.group(1).strip()
        json.loads(candidate)
        return candidate

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            json.loads(candidate)
            return candidate

    raise ValueError("No parseable JSON found in model output")


def prompt_with_schema(schema, prompt):
    """Wrap a prompt for strict JSON tasks with schema instructions."""
    return (
        "Respond with JSON only matching this schema.\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"{prompt}"
    )


def build_llm_payload(messages, settings, schema=None, *, num_ctx=None, num_predict=None):
    """Build a request payload for a task-tier LLM call."""
    payload = {
        "model": settings["model"],
        "messages": messages,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "options": {
            "num_ctx": num_ctx if num_ctx is not None else settings["num_ctx"],
            "num_predict": num_predict if num_predict is not None else settings["num_predict"],
            "cache_type_k": _SETTINGS.kv_cache_type,
            "cache_type_v": _SETTINGS.kv_cache_type,
            "num_keep": 256,
            "top_k": settings["top_k"],
            "min_p": settings["min_p"],
            "presence_penalty": settings["presence_penalty"],
            "repeat_penalty": settings["repeat_penalty"],
        },
        "keep_alive": settings["keep_alive"],
    }
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        payload["format"] = schema
    return payload


def prompt_chars_for(prompt):
    """Return a prompt length used for logging."""
    return len(prompt or "")


def task_llm_settings(task):
    """Return request settings for a task tier."""
    if task == "detect":
        return {
            "phase": "detect",
            "model": MODEL_DETECT,
            "think": THINK_DETECT,
            "num_ctx": NUM_CTX_DETECT,
            "num_predict": NUM_PREDICT_DETECT,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "keep_alive": -1,
            "strict_json": True,
        }
    if task == "extract":
        return {
            "phase": "extract",
            "model": MODEL_EXTRACT,
            "think": THINK_EXTRACT,
            "num_ctx": NUM_CTX_EXTRACT,
            "num_predict": NUM_PREDICT_EXTRACT,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "keep_alive": -1,
            "strict_json": True,
        }
    if task == "summary":
        return {
            "phase": "summary",
            "model": MODEL_SUMMARY,
            "think": THINK_SUMMARY,
            "num_ctx": NUM_CTX_SUMMARY,
            "num_predict": NUM_PREDICT_SUMMARY,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "keep_alive": -1,
            "strict_json": False,
        }
    if task == "verify":
        return {
            "phase": "verify",
            "model": MODEL_VERIFY,
            "think": THINK_VERIFY,
            "num_ctx": NUM_CTX_VERIFY,
            "num_predict": NUM_PREDICT_VERIFY,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "keep_alive": "10m",
            "strict_json": True,
        }
    raise ValueError(f"Unknown LLM task tier: {task}")


def llm_chat_json(task, call_name, system_prompt, user_prompt, schema, *, task_deadline, candidate_validator=None, prompt_builder=None, call_state=None, num_predict_override=None, expect_json=True):
    """Call the chat endpoint with retries, validation repair, and schema checks.

    Args:
        task: Task tier key used to resolve model and request settings.
        call_name: Structured call label used in logs.
        system_prompt: System instruction for the request.
        user_prompt: Base user prompt used when no prompt builder is supplied.
        schema: JSON schema dict or callable returning a schema from call state.
        task_deadline: Absolute monotonic deadline for this logical task.
        candidate_validator: Optional post-schema validator for business rules.
        prompt_builder: Optional callable to rebuild prompt from mutable state.
        call_state: Optional mutable state for shrinking and repair steps.
        num_predict_override: Optional output-token budget override.
        expect_json: When False, returns raw content without JSON validation.

    Returns:
        Parsed JSON value for JSON calls, or raw content string when
        expect_json is False.

    Raises:
        LLMTaskTimeoutError: If the logical task exceeds TASK_TIMEOUT.
        LLMEmptyResponseError: If empty-response repair is exhausted.
        LLMResponseValidationError: If validation repair is exhausted.
        LLMTaskRetryExhaustedError: If retryable transport/server failures
            exhaust MAX_RETRIES.
        LLMTaskError: For non-retryable HTTP failure statuses.
    """
    settings = task_llm_settings(task)
    state = dict(call_state or {})
    state.setdefault("num_ctx", settings["num_ctx"])
    state.setdefault("window_size", None)
    prompt_builder = prompt_builder or (lambda _state: user_prompt)
    base_num_predict = num_predict_override if num_predict_override is not None else settings["num_predict"]
    retries = 0
    shrinks_applied = 0
    consecutive_5xx = 0
    last_error = None
    last_status = None
    last_latency_ms = None

    validation_context = state.setdefault("validation_context", {})
    validation_context.setdefault("schema_kind", "enum_int")
    validation_context.setdefault("shrink_window", BOUNDARY_REFINE_FALLBACK_WINDOW)

    def current_schema():
        if callable(schema):
            return schema(state)
        return schema

    def extract_allowed_values(schema_obj):
        if not isinstance(schema_obj, dict):
            return []
        out = []

        def _walk(node):
            if isinstance(node, dict):
                node_type = node.get("type")
                node_enum = node.get("enum")
                if node_type == "integer" and isinstance(node_enum, list):
                    for value in node_enum:
                        if isinstance(value, int) and value not in out:
                            out.append(value)
                for key in ("properties", "items", "anyOf", "oneOf"):
                    child = node.get(key)
                    if isinstance(child, dict):
                        _walk(child)
                    elif isinstance(child, list):
                        for item in child:
                            _walk(item)

        _walk(schema_obj)
        return out

    def raise_validation_error(message, *, received_value=None, schema_obj=None, repair_step=0, last_payload=None):
        allowed_values = validation_context.get("allowed_values")
        if not allowed_values:
            allowed_values = extract_allowed_values(schema_obj or current_schema())
        raise LLMResponseValidationError(
            message,
            received_value=received_value,
            allowed_values=allowed_values,
            repair_step=repair_step,
            schema_kind=validation_context.get("schema_kind", "enum_int"),
            last_payload=last_payload,
        )

    def parse_and_validate(raw_text, schema_obj, *, repair_step=0):
        data = json.loads(raw_text)
        choices = data.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") or {}
        content = message.get("content") or ""
        payload_text = extract_json_payload(content)
        value = json.loads(payload_text)
        try:
            validate_json_schema(value, schema_obj, error_cls=LLMResponseValidationError)
            if candidate_validator is not None:
                candidate_validator(value)
        except LLMResponseValidationError as exc:
            raise_validation_error(
                str(exc),
                received_value=value,
                schema_obj=schema_obj,
                repair_step=repair_step,
                last_payload=value,
            )
        return value

    def log_validation_line(outcome, repair_step, received_value, allowed_values, *, level="info", extra=None):
        record = {
            "phase": settings["phase"],
            "task": call_name,
            "model": settings["model"],
            "outcome": outcome,
            "repair_step": repair_step,
            "received_value": received_value,
            "allowed_count": len(allowed_values or []),
            "schema_kind": validation_context.get("schema_kind", "enum_int"),
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if level == "warning":
            logging.warning(line)
        else:
            print(line)

    def build_validation_repair_message(received_value, allowed_values):
        return (
            f"Your previous response was {json.dumps(received_value, ensure_ascii=False)}, which is not in the allowed set. "
            f"The allowed values are exactly {json.dumps(allowed_values, ensure_ascii=False)}. "
            "Return ONLY JSON matching the schema."
        )

    def issue_request(prompt_text, *, num_ctx_used, num_predict_used, think_used, schema_prompt=True, extra_messages=None, schema_obj=None):
        request_prompt = prompt_text
        used_schema = schema_obj if schema_obj is not None else current_schema()
        if settings["strict_json"] and expect_json and schema_prompt:
            request_prompt = prompt_with_schema(used_schema, request_prompt)
        request_chars = prompt_chars_for(request_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": request_prompt})
        payload = build_llm_payload(
            messages,
            settings,
            schema=used_schema if settings["strict_json"] and expect_json else None,
            num_ctx=num_ctx_used,
            num_predict=num_predict_used,
        )
        headers = {"X-Ollama-Think": "true" if think_used else "false"}
        request_timeout = min(REQUEST_TIMEOUT, max(1.0, task_time_remaining(task_deadline)))
        request_started = time.monotonic()
        status, raw, hdrs = http_post_json(BASE_URL + "/chat/completions", payload, timeout=request_timeout, headers=headers)
        latency_ms = int((time.monotonic() - request_started) * 1000)
        return {
            "status": status,
            "raw": raw,
            "hdrs": hdrs,
            "latency_ms": latency_ms,
            "prompt_chars": request_chars,
            "request_prompt": request_prompt,
            "num_ctx_used": num_ctx_used,
            "num_predict_used": num_predict_used,
            "think_used": think_used,
            "schema_obj": used_schema,
        }

    def repair_empty_response(original_prompt, original_num_ctx, original_num_predict, original_think, initial_raw):
        if EMPTY_REPAIR_MAX_STEPS <= 0:
            if LOG_RAW_EMPTY:
                log_empty_response_error({
                    "phase": settings["phase"],
                    "task": call_name,
                    "model": settings["model"],
                    "outcome": "empty_content_raw",
                    "repair_step": 0,
                    "raw_response_chars": len(initial_raw or ""),
                    "raw_response_body": (initial_raw or "")[:1000],
                })
            raise LLMEmptyResponseError(f"{call_name} failed with empty model response", raw_response_body=initial_raw or "")

        repair_num_ctx = original_num_ctx
        repair_num_predict = original_num_predict
        last_empty_raw = initial_raw or ""

        for repair_step in range(1, EMPTY_REPAIR_MAX_STEPS + 1):
            repair_num_predict = min(max(1, repair_num_predict * 2), NUM_PREDICT_HARD_CAP)
            if repair_step == 1:
                repair_think = original_think
                repair_prompt = original_prompt
                schema_prompt = True
            elif repair_step == 2:
                repair_think = False
                repair_prompt = original_prompt
                schema_prompt = True
            else:
                repair_think = False
                repair_prompt = build_empty_response_repair_prompt(schema, original_prompt)
                schema_prompt = False

            prompt_for_budget = repair_prompt
            repair_schema = current_schema()
            if settings["strict_json"] and expect_json and schema_prompt:
                prompt_for_budget = prompt_with_schema(repair_schema, repair_prompt)
            prompt_tokens_est = estimate_prompt_tokens(prompt_for_budget)
            repair_num_ctx, repair_num_predict = clamp_empty_response_budget(repair_num_ctx, repair_num_predict, prompt_tokens_est)

            issue = issue_request(
                repair_prompt,
                num_ctx_used=repair_num_ctx,
                num_predict_used=repair_num_predict,
                think_used=repair_think,
                schema_prompt=schema_prompt,
                schema_obj=repair_schema,
            )
            status = issue["status"]
            raw = issue["raw"]
            hdrs = issue["hdrs"]
            last_latency_ms = issue["latency_ms"]

            if not (200 <= status < 300):
                print_server_error_detail(status, hdrs, raw, f"SERVER ERROR DETAIL ({call_name} empty-response repair step {repair_step})")
                raise LLMTaskError(f"{call_name} empty-response repair step {repair_step} failed with HTTP {status}")

            try:
                _, content, finish_reason, has_reasoning_field, reasoning_chars = empty_response_details(raw)
            except Exception:
                raise

            if not str(content).strip():
                last_empty_raw = raw or ""
                log_empty_response_warning({
                    "phase": settings["phase"],
                    "task": call_name,
                    "model": settings["model"],
                    "think": repair_think,
                    "outcome": "empty_content",
                    "repair_step": repair_step,
                    "num_ctx_used": issue["num_ctx_used"],
                    "num_predict_used": issue["num_predict_used"],
                    "finish_reason": finish_reason,
                    "has_reasoning_field": has_reasoning_field,
                    "reasoning_chars": reasoning_chars,
                    "prompt_chars": issue["prompt_chars"],
                    "latency_ms": last_latency_ms,
                    "http_status": status,
                    "retries": retries,
                    "shrinks_applied": shrinks_applied,
                })
                repair_num_ctx = issue["num_ctx_used"]
                repair_num_predict = issue["num_predict_used"]
                if repair_step >= EMPTY_REPAIR_MAX_STEPS:
                    if LOG_RAW_EMPTY:
                        log_empty_response_error({
                            "phase": settings["phase"],
                            "task": call_name,
                            "model": settings["model"],
                            "outcome": "empty_content_raw",
                            "repair_step": repair_step,
                            "raw_response_chars": len(last_empty_raw),
                            "raw_response_body": last_empty_raw[:1000],
                        })
                    raise LLMEmptyResponseError(f"{call_name} failed after {EMPTY_REPAIR_MAX_STEPS} empty-response repairs", raw_response_body=last_empty_raw)
                continue

            return issue

        if LOG_RAW_EMPTY:
            log_empty_response_error({
                "phase": settings["phase"],
                "task": call_name,
                "model": settings["model"],
                "outcome": "empty_content_raw",
                "repair_step": EMPTY_REPAIR_MAX_STEPS,
                "raw_response_chars": len(last_empty_raw),
                "raw_response_body": last_empty_raw[:1000],
            })
        raise LLMEmptyResponseError(f"{call_name} failed with empty model response", raw_response_body=last_empty_raw)

    def run_validation_repair_ladder(original_prompt, validation_error, *, schema_obj):
        if VALIDATION_REPAIR_MAX_STEPS <= 0:
            raise validation_error

        allowed_values = list(validation_error.allowed_values or extract_allowed_values(schema_obj))
        last_exc = validation_error
        current_num_ctx = state.get("num_ctx", settings["num_ctx"])
        current_num_predict = base_num_predict
        current_prompt = original_prompt

        for repair_step in range(1, VALIDATION_REPAIR_MAX_STEPS + 1):
            step_schema = current_schema()
            allowed_values = extract_allowed_values(step_schema)
            validation_context["allowed_values"] = allowed_values
            validation_hint = build_validation_repair_message(last_exc.received_value, allowed_values)

            step_think = settings["think"]
            step_num_predict = current_num_predict
            step_prompt = current_prompt
            extra_messages = [{"role": "user", "content": validation_hint}]

            if repair_step == 2:
                step_think = True
                step_num_predict = min(max(1, current_num_predict * 2), NUM_PREDICT_HARD_CAP)
            elif repair_step >= 3:
                step_think = True
                shrinker = validation_context.get("step3_shrink")
                if callable(shrinker):
                    shrinker(state)
                step_schema = current_schema()
                allowed_values = extract_allowed_values(step_schema)
                validation_context["allowed_values"] = allowed_values
                step_prompt = prompt_builder(state)
                step_num_predict = min(max(1, current_num_predict * 2), NUM_PREDICT_HARD_CAP)

            issue = issue_request(
                step_prompt,
                num_ctx_used=current_num_ctx,
                num_predict_used=step_num_predict,
                think_used=step_think,
                schema_prompt=True,
                extra_messages=extra_messages,
                schema_obj=step_schema,
            )

            status = issue["status"]
            raw = issue["raw"]
            hdrs = issue["hdrs"]

            if not (200 <= status < 300):
                print_server_error_detail(status, hdrs, raw, f"SERVER ERROR DETAIL ({call_name} validation-repair step {repair_step})")
                raise LLMTaskError(f"{call_name} validation repair step {repair_step} failed with HTTP {status}")

            try:
                value = parse_and_validate(raw, step_schema, repair_step=repair_step)
                log_validation_line(
                    "validation_repair",
                    repair_step,
                    value,
                    allowed_values,
                )
                return value, issue
            except LLMResponseValidationError as exc:
                last_exc = exc
                log_validation_line(
                    "validation_repair",
                    repair_step,
                    exc.received_value,
                    allowed_values,
                )
                current_num_predict = step_num_predict
                current_prompt = step_prompt
                if repair_step >= VALIDATION_REPAIR_MAX_STEPS:
                    raise exc

        raise last_exc

    while True:
        if retries >= MAX_RETRIES:
            break

        if task_time_remaining(task_deadline) <= 0:
            raise LLMTaskTimeoutError(f"{call_name} exceeded TASK_TIMEOUT")

        prompt = prompt_builder(state)
        schema_obj = current_schema()
        validation_context["allowed_values"] = extract_allowed_values(schema_obj)
        request_prompt = prompt
        if settings["strict_json"] and expect_json:
            request_prompt = prompt_with_schema(schema_obj, request_prompt)
        request_chars = prompt_chars_for(request_prompt)
        payload = build_llm_payload(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_prompt},
            ],
            settings,
            schema=schema_obj if settings["strict_json"] and expect_json else None,
            num_ctx=state.get("num_ctx", settings["num_ctx"]),
            num_predict=base_num_predict,
        )

        headers = {"X-Ollama-Think": "true" if settings["think"] else "false"}
        request_timeout = min(REQUEST_TIMEOUT, max(1.0, task_time_remaining(task_deadline)))
        request_started = time.monotonic()

        try:
            status, raw, hdrs = http_post_json(BASE_URL + "/chat/completions", payload, timeout=request_timeout, headers=headers)
            last_latency_ms = int((time.monotonic() - request_started) * 1000)
            last_status = status
            if 500 <= status <= 599:
                consecutive_5xx += 1
            else:
                consecutive_5xx = 0
        except Exception as exc:
            last_latency_ms = int((time.monotonic() - request_started) * 1000)
            last_status = "transport_error"
            last_error = repr(exc)
            if not is_retryable_exception(exc):
                log_llm_call({
                    "phase": settings["phase"],
                    "task": call_name,
                    "model": settings["model"],
                    "think": settings["think"],
                    "num_ctx": state.get("num_ctx", settings["num_ctx"]),
                    "prompt_chars": request_chars,
                    "latency_ms": last_latency_ms,
                    "http_status": last_status,
                    "retries": retries,
                    "shrinks_applied": shrinks_applied,
                })
                raise
            retries += 1
            sleep_cap = min(MAX_SLEEP, BASE_SLEEP * (2 ** retries))
            time.sleep(random.uniform(0.0, sleep_cap))
            continue

        if is_retryable_status(status):
            print_server_error_detail(status, hdrs, raw, f"SERVER ERROR DETAIL ({call_name})")
            last_error = f"HTTP {status}"
            retries += 1
            retry_after = parse_retry_after(hdrs)
            if 500 <= status <= 599:
                consecutive_5xx += 1
            else:
                consecutive_5xx = 0
            if consecutive_5xx >= 2:
                old_ctx = int(state.get("num_ctx", settings["num_ctx"]))
                new_ctx = max(1024, old_ctx // 2)
                state["num_ctx"] = new_ctx
                shrink_bits = [f"num_ctx {old_ctx}->{new_ctx}"]
                if task in {"detect", "extract"} and state.get("window_size"):
                    old_window = int(state["window_size"])
                    new_window = max(1, old_window // 2)
                    state["window_size"] = new_window
                    shrink_bits.append(f"window_size {old_window}->{new_window}")
                shrinks_applied += 1
                print(f"LLM shrink task={call_name} {' '.join(shrink_bits)}")
                consecutive_5xx = 0
            if retries >= MAX_RETRIES:
                break
            sleep_cap = min(MAX_SLEEP, BASE_SLEEP * (2 ** retries))
            sleep_s = retry_after if retry_after is not None else random.uniform(0.0, sleep_cap)
            time.sleep(sleep_s)
            continue

        if status < 200 or status >= 300:
            print_server_error_detail(status, hdrs, raw, f"SERVER ERROR DETAIL ({call_name})")
            log_llm_call({
                "phase": settings["phase"],
                "task": call_name,
                "model": settings["model"],
                "think": settings["think"],
                "num_ctx": state.get("num_ctx", settings["num_ctx"]),
                "prompt_chars": request_chars,
                "latency_ms": last_latency_ms,
                "http_status": status,
                "retries": retries,
                "shrinks_applied": shrinks_applied,
            })
            raise LLMTaskError(f"{call_name} failed with HTTP {status}")

        try:
            data = json.loads(raw)
            choices = data.get("choices") or []
            first_choice = choices[0] if choices else {}
            message = first_choice.get("message") or {}
            content = message.get("content") or ""
            if not str(content).strip():
                log_empty_response_warning({
                    "phase": settings["phase"],
                    "task": call_name,
                    "model": settings["model"],
                    "think": settings["think"],
                    "outcome": "empty_content",
                    "repair_step": 0,
                    "num_ctx_used": state.get("num_ctx", settings["num_ctx"]),
                    "num_predict_used": base_num_predict,
                    "finish_reason": first_choice.get("finish_reason"),
                    "has_reasoning_field": "reasoning" in message or "thinking" in message,
                    "reasoning_chars": len((message.get("reasoning") if message.get("reasoning") is not None else message.get("thinking")) or ""),
                    "prompt_chars": request_chars,
                    "latency_ms": last_latency_ms,
                    "http_status": status,
                    "retries": retries,
                    "shrinks_applied": shrinks_applied,
                })
                repair_response = repair_empty_response(
                    prompt,
                    state.get("num_ctx", settings["num_ctx"]),
                    base_num_predict,
                    settings["think"],
                    raw,
                )
                status = repair_response["status"]
                raw = repair_response["raw"]
                hdrs = repair_response["hdrs"]
                last_latency_ms = repair_response["latency_ms"]
                request_chars = repair_response["prompt_chars"]
                data = json.loads(raw)
                choices = data.get("choices") or []
                first_choice = choices[0] if choices else {}
                message = first_choice.get("message") or {}
                content = message.get("content") or ""
            if not expect_json:
                log_llm_call({
                    "phase": settings["phase"],
                    "task": call_name,
                    "model": settings["model"],
                    "think": settings["think"],
                    "num_ctx": state.get("num_ctx", settings["num_ctx"]),
                    "prompt_chars": request_chars,
                    "latency_ms": last_latency_ms,
                    "http_status": status,
                    "retries": retries,
                    "shrinks_applied": shrinks_applied,
                })
                return content
            try:
                value = parse_and_validate(raw, schema_obj, repair_step=0)
            except LLMResponseValidationError as validation_exc:
                log_validation_line(
                    "validation_repair",
                    0,
                    validation_exc.received_value,
                    validation_exc.allowed_values or validation_context.get("allowed_values") or [],
                )
                repaired_value, repair_issue = run_validation_repair_ladder(prompt, validation_exc, schema_obj=schema_obj)
                value = repaired_value
                status = repair_issue["status"]
                last_latency_ms = repair_issue["latency_ms"]
                request_chars = repair_issue["prompt_chars"]
            log_llm_call({
                "phase": settings["phase"],
                "task": call_name,
                "model": settings["model"],
                "think": settings["think"],
                "num_ctx": state.get("num_ctx", settings["num_ctx"]),
                "prompt_chars": request_chars,
                "latency_ms": last_latency_ms,
                "http_status": status,
                "retries": retries,
                "shrinks_applied": shrinks_applied,
            })
            return value
        except LLMEmptyResponseError:
            raise
        except LLMResponseValidationError:
            raise
        except Exception as exc:
            last_error = repr(exc)
            repair_prompt = build_repair_prompt(schema_obj, prompt)
            repair_payload = build_llm_payload(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_prompt},
                ],
                settings,
                schema=schema_obj if settings["strict_json"] and expect_json else None,
                num_ctx=state.get("num_ctx", settings["num_ctx"]),
                num_predict=base_num_predict,
            )
            repair_started = time.monotonic()
            try:
                repair_status, repair_raw, repair_hdrs = http_post_json(BASE_URL + "/chat/completions", repair_payload, timeout=request_timeout, headers=headers)
                last_latency_ms = int((time.monotonic() - repair_started) * 1000)
                last_status = repair_status
                if 500 <= repair_status <= 599:
                    consecutive_5xx += 1
                else:
                    consecutive_5xx = 0
                if 200 <= repair_status < 300:
                    try:
                        value = parse_and_validate(repair_raw, schema_obj, repair_step=0)
                    except LLMResponseValidationError as repair_validation_exc:
                        log_validation_line(
                            "validation_repair",
                            0,
                            repair_validation_exc.received_value,
                            repair_validation_exc.allowed_values or validation_context.get("allowed_values") or [],
                        )
                        value, repair_issue = run_validation_repair_ladder(prompt, repair_validation_exc, schema_obj=schema_obj)
                        last_latency_ms = repair_issue["latency_ms"]
                        request_chars = repair_issue["prompt_chars"]
                    log_llm_call({
                        "phase": settings["phase"],
                        "task": call_name,
                        "model": settings["model"],
                        "think": settings["think"],
                        "num_ctx": state.get("num_ctx", settings["num_ctx"]),
                        "prompt_chars": request_chars,
                        "latency_ms": last_latency_ms,
                        "http_status": repair_status,
                        "retries": retries,
                        "shrinks_applied": shrinks_applied,
                    })
                    return value
                print_server_error_detail(repair_status, repair_hdrs, repair_raw, f"SERVER ERROR DETAIL ({call_name} repair)")
                last_error = f"HTTP {repair_status}"
            except Exception as repair_exc:
                last_error = repr(repair_exc)

        retries += 1
        if consecutive_5xx >= 2:
            old_ctx = int(state.get("num_ctx", settings["num_ctx"]))
            new_ctx = max(1024, old_ctx // 2)
            state["num_ctx"] = new_ctx
            shrink_bits = [f"num_ctx {old_ctx}->{new_ctx}"]
            if task in {"detect", "extract"} and state.get("window_size"):
                old_window = int(state["window_size"])
                new_window = max(1, old_window // 2)
                state["window_size"] = new_window
                shrink_bits.append(f"window_size {old_window}->{new_window}")
            shrinks_applied += 1
            print(f"LLM shrink task={call_name} {' '.join(shrink_bits)}")
            consecutive_5xx = 0
        if retries >= MAX_RETRIES:
            break
        sleep_cap = min(MAX_SLEEP, BASE_SLEEP * (2 ** retries))
        time.sleep(random.uniform(0.0, sleep_cap))

    log_llm_call({
        "phase": settings["phase"],
        "task": call_name,
        "model": settings["model"],
        "think": settings["think"],
        "num_ctx": state.get("num_ctx", settings["num_ctx"]),
        "prompt_chars": request_chars,
        "latency_ms": last_latency_ms,
        "http_status": last_status,
        "retries": retries,
        "shrinks_applied": shrinks_applied,
    })
    raise LLMTaskRetryExhaustedError(f"{call_name} failed after {MAX_RETRIES} retries. Last error: {last_error}")


def llm_chat_json_legacy(task_name, system_prompt, user_prompt, schema, *, temperature, top_p, task_deadline, candidate_validator=None, response_format=True):
    """Call the chat endpoint with retries, schema validation, and repair."""
    prompt_text = system_prompt + "\n" + user_prompt
    prompt_tokens_est = estimate_prompt_tokens(prompt_text)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        if task_time_remaining(task_deadline) <= 0:
            raise LLMTaskTimeoutError(f"{task_name} exceeded TASK_TIMEOUT")

        request_timeout = min(REQUEST_TIMEOUT, max(1.0, task_time_remaining(task_deadline)))
        payload = build_chat_payload(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            response_format=safe_response_format() if response_format else None,
        )

        request_started = time.monotonic()
        try:
            status, raw, hdrs = http_post_json(BASE_URL + "/chat/completions", payload, timeout=request_timeout)
            latency_ms = int((time.monotonic() - request_started) * 1000)
            print(f"LLM task={task_name} attempt={attempt} prompt_tokens_est={prompt_tokens_est} status={status} latency_ms={latency_ms}")
        except Exception as exc:
            latency_ms = int((time.monotonic() - request_started) * 1000)
            print(f"LLM task={task_name} attempt={attempt} prompt_tokens_est={prompt_tokens_est} status=transport_error latency_ms={latency_ms}")
            if is_retryable_exception(exc):
                last_error = repr(exc)
                sleep_cap = min(MAX_SLEEP, BASE_SLEEP * (2 ** attempt))
                sleep_s = random.uniform(0.0, sleep_cap)
                print(f"LLM task={task_name} retryable transport error: {last_error}")
                print(f"Sleeping {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            raise

        if is_retryable_status(status):
            print_server_error_detail(status, hdrs, raw, f"SERVER ERROR DETAIL ({task_name})")
            last_error = f"HTTP {status}"
            retry_after = parse_retry_after(hdrs)
            sleep_cap = min(MAX_SLEEP, BASE_SLEEP * (2 ** attempt))
            sleep_s = retry_after if retry_after is not None else random.uniform(0.0, sleep_cap)
            print(f"LLM task={task_name} retryable HTTP status={status} latency_ms={latency_ms}")
            print(f"Sleeping {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            continue

        if status < 200 or status >= 300:
            print_server_error_detail(status, hdrs, raw, f"SERVER ERROR DETAIL ({task_name})")
            raise LLMTaskError(f"{task_name} failed with HTTP {status}")

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            payload_text = extract_json_payload(content)
            value = json.loads(payload_text)
            validate_json_schema(value, schema, error_cls=LLMResponseValidationError)
            if candidate_validator is not None:
                candidate_validator(value)
            return value
        except Exception as exc:
            last_error = repr(exc)
            repair_prompt = build_repair_prompt(schema, user_prompt)
            repair_payload = build_chat_payload(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                response_format=safe_response_format() if response_format else None,
            )
            repair_started = time.monotonic()
            try:
                repair_status, repair_raw, repair_hdrs = http_post_json(BASE_URL + "/chat/completions", repair_payload, timeout=request_timeout)
                repair_latency_ms = int((time.monotonic() - repair_started) * 1000)
                print(f"LLM task={task_name} attempt={attempt}.repair prompt_tokens_est={prompt_tokens_est} status={repair_status} latency_ms={repair_latency_ms}")
                if 200 <= repair_status < 300:
                    repair_data = json.loads(repair_raw)
                    repair_content = repair_data["choices"][0]["message"]["content"]
                    payload_text = extract_json_payload(repair_content)
                    value = json.loads(payload_text)
                    validate_json_schema(value, schema, error_cls=LLMResponseValidationError)
                    if candidate_validator is not None:
                        candidate_validator(value)
                    return value
                if is_retryable_status(repair_status):
                    print_server_error_detail(repair_status, repair_hdrs, repair_raw, f"SERVER ERROR DETAIL ({task_name} repair)")
                else:
                    print_server_error_detail(repair_status, repair_hdrs, repair_raw, f"SERVER ERROR DETAIL ({task_name} repair)")
                last_error = f"HTTP {repair_status}"
            except Exception as repair_exc:
                last_error = repr(repair_exc)

        sleep_cap = min(MAX_SLEEP, BASE_SLEEP * (2 ** attempt))
        sleep_s = random.uniform(0.0, sleep_cap)
        print(f"LLM task={task_name} attempt={attempt} failed: {last_error}")
        print(f"Sleeping {sleep_s:.1f}s...")
        time.sleep(sleep_s)

    raise LLMTaskError(f"{task_name} failed after {MAX_RETRIES} retries. Last error: {last_error}")


def chat_json(system_prompt, user_prompt):
    """Compatibility wrapper for callers that still expect a raw JSON HTTP call."""
    payload = build_chat_payload(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
    )
    return http_post_json(BASE_URL + "/chat/completions", payload, timeout=REQUEST_TIMEOUT)
