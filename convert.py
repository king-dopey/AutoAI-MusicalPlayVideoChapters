"""Detect song boundaries and build story summaries from subtitle data.

This script parses flyer metadata, subtitle cues, and extracted audio
features to identify song spans for a musical. It then extracts likely
lyric cues, writes review artifacts, and generates chapter-style story
summaries with an LLM API.

Inputs:
- Environment variables for API endpoints and tuning parameters.
- Files in WORKDIR such as input.srt, flyer.txt, and input.mp4.

Outputs:
- JSON and Markdown artifacts including blocks.json, songs.json,
  songs_review.md, lyrics_by_song.md, and story summary files.

Side Effects:
- Calls an external chat-completions HTTP API.
- Invokes ffmpeg to extract mono PCM audio.
- Reads and writes multiple files in WORKDIR.

External Dependencies:
- ffmpeg available on PATH.
- numpy for numeric feature extraction.
"""

import os, re, json, time, random, wave, subprocess, math, socket, logging
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib import request, error

import numpy as np

# =========================
# Config
# =========================
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
MODEL = os.getenv("MODEL", "")
MODEL_DETECT = os.getenv("MODEL_DETECT", MODEL or "qwen3-coder:30b")
MODEL_EXTRACT = os.getenv("MODEL_EXTRACT", MODEL or "qwen3-coder:30b")
MODEL_SUMMARY = os.getenv("MODEL_SUMMARY", MODEL or "qwen3.6:35b-a3b")
MODEL_VERIFY = os.getenv("MODEL_VERIFY", MODEL or "nemotron-cascade-2:30b")
API_KEY = os.getenv("API_KEY", "")
KV_CACHE_TYPE = os.getenv("KV_CACHE_TYPE", "q8_0")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
BASE_SLEEP = float(os.getenv("BASE_SLEEP", "2.0"))
MAX_SLEEP = float(os.getenv("MAX_SLEEP", "45.0"))
ERR_BODY_CHARS = int(os.getenv("ERR_BODY_CHARS", "6000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))
TASK_TIMEOUT = float(os.getenv("TASK_TIMEOUT", "1800"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "12000"))
MIN_BOUNDARY_SHIFT_MS = int(os.getenv("MIN_BOUNDARY_SHIFT_MS", "1500"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
RESUME = os.getenv("RESUME", "1") == "1"
ENABLE_VERIFIER = os.getenv("ENABLE_VERIFIER", "0") == "1"
LOG_RAW_EMPTY = os.getenv("LOG_RAW_EMPTY", "0") == "1"
EMPTY_REPAIR_MAX_STEPS = max(0, int(os.getenv("EMPTY_REPAIR_MAX_STEPS", "3")))
NUM_PREDICT_HARD_CAP = max(1, int(os.getenv("NUM_PREDICT_HARD_CAP", "16384")))
NUM_CTX_HARD_CAP = max(1, int(os.getenv("NUM_CTX_HARD_CAP", "32768")))
EMPTY_REPAIR_SAFETY_MARGIN = 256
THINK_DETECT = os.getenv("THINK_DETECT", "0") == "1"
THINK_EXTRACT = os.getenv("THINK_EXTRACT", "0") == "1"
THINK_SUMMARY = os.getenv("THINK_SUMMARY", "1") == "1"
THINK_VERIFY = os.getenv("THINK_VERIFY", "1") == "1"

NUM_CTX_DETECT = int(os.getenv("NUM_CTX_DETECT", "16384"))
NUM_CTX_EXTRACT = int(os.getenv("NUM_CTX_EXTRACT", "16384"))
NUM_CTX_SUMMARY = int(os.getenv("NUM_CTX_SUMMARY", "32768"))
NUM_CTX_VERIFY = int(os.getenv("NUM_CTX_VERIFY", "16384"))
NUM_PREDICT_DETECT = int(os.getenv("NUM_PREDICT_DETECT", "512"))
NUM_PREDICT_EXTRACT = int(os.getenv("NUM_PREDICT_EXTRACT", "2048"))
NUM_PREDICT_SUMMARY = int(os.getenv("NUM_PREDICT_SUMMARY", "8192"))
NUM_PREDICT_VERIFY = int(os.getenv("NUM_PREDICT_VERIFY", "4096"))

# Audio / segmentation
AUDIO_SR = int(os.getenv("AUDIO_SR", "16000"))
FRAME_SEC = float(os.getenv("FRAME_SEC", "1.0"))
FRAME_HOP_SEC = float(os.getenv("FRAME_HOP_SEC", "0.25"))

SHORT_GAP_MS = int(os.getenv("SHORT_GAP_MS", "1600"))
LONG_GAP_MS = int(os.getenv("LONG_GAP_MS", "5000"))
BED_MIN_RATIO = float(os.getenv("BED_MIN_RATIO", "0.55"))
BED_MIN_RMS_N = float(os.getenv("BED_MIN_RMS_N", "0.20"))

SEARCH_BLOCKS = int(os.getenv("SEARCH_BLOCKS", "6"))
SEARCH_STRIDE = int(os.getenv("SEARCH_STRIDE", "3"))

BOUNDARY_CONTEXT_BLOCKS = int(os.getenv("BOUNDARY_CONTEXT_BLOCKS", "1"))
LYRICS_WINDOW_CUES = int(os.getenv("LYRICS_WINDOW_CUES", "90"))
LYRICS_WINDOW_OVERLAP = int(os.getenv("LYRICS_WINDOW_OVERLAP", "15"))

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class LLMTaskError(RuntimeError):
    """Raised when an LLM task cannot be completed after retries."""


class LLMResponseValidationError(LLMTaskError):
    """Raised when a response fails schema or candidate validation."""


class LLMTaskTimeoutError(LLMTaskError):
    """Raised when a logical task exceeds its wall-clock budget."""


class LLMTaskRetryExhaustedError(LLMTaskError):
    """Raised when a task exhausts retries after transient failures."""


class LLMEmptyResponseError(LLMTaskError):
    """Raised when a 2xx response returns empty visible content."""

    def __init__(self, message, *, raw_response_body=""):
        super().__init__(message)
        self.raw_response_body = raw_response_body


TASK_BOUNDARY_NEIGHBORS = 2

SYSTEM = """You are a careful musical-theatre transcript analyst.

You are given:
1) an ordered song list from a flyer
2) timestamped subtitle cues from an SRT
3) audio-derived hints such as music-bed likelihood, cue continuity, and gap analysis

Your job is to identify song boundaries and lyric cues.

Rules:
- Follow the flyer song order exactly.
- Prefer explicit evidence from the SRT text.
- Use the audio hints to distinguish sung sections from spoken dialogue.
- A song may be surrounded by spoken dialogue.
- If uncertain, make the best reasonable estimate and lower confidence.
- Do not invent songs not present in the flyer.
- Return JSON only.
"""

SUMMARY_SYSTEM = """You are an expert musical-story analyst.

You will be given:
1) a flyer with plot summary and song order
2) per-song lyrics and metadata

Your task:
Summarize the musical's story as a sequence of chapter summaries, where each song is a chapter.

Rules:
- Follow the song order exactly.
- Treat each song as one chapter.
- Summaries should describe story events, character motivations, and changes caused by the song.
- Preserve facts from the inputs.
- Do not invent plot events unsupported by the lyrics or flyer.
- Maintain continuity across chapters.
- Return JSON only.
"""

# =========================
# HTTP helpers
# =========================
def print_server_error_detail(status, hdrs, body, label):
    """Print a compact diagnostic block for failed HTTP interactions.

    Args:
        status: HTTP status code.
        hdrs: Response headers mapping or similar object.
        body: Response body text.
        label: Human-readable section label.
    """
    print(f"\n--- {label} ---")
    print(f"HTTP {status}")
    try:
        if hdrs:
            print("Headers:")
            for k, v in list(hdrs.items())[:60]:
                print(f"{k}: {v}")
    except Exception:
        pass
    print(f"Body (first {ERR_BODY_CHARS} chars):")
    print((body or "")[:ERR_BODY_CHARS])
    print(f"--- END {label} ---\n")

def http_post_json(url, payload, timeout: float = 3600.0, headers=None):
    """Send a JSON POST request and return status, body, and headers.

    Args:
        url: Endpoint URL.
        payload: JSON-serializable request body.
        timeout: Request timeout in seconds.

    Returns:
        A tuple of (status_code, response_text, response_headers).
    """
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
    except error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, body, dict(getattr(e, "headers", {}) or {})


def estimate_prompt_tokens(text):
    """Estimate prompt tokens using a cheap character heuristic."""
    return max(1, int(math.ceil(len(text or "") / 4.0)))


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
        if dt is None:
            return None
        return max(0.0, dt.timestamp() - time.time())
    except Exception:
        return None


def is_retryable_status(status):
    """Return True for HTTP status codes that should be retried."""
    return status in RETRYABLE_STATUS_CODES


def is_retryable_exception(exc):
    """Return True when a transport exception should be retried."""
    return isinstance(exc, (error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError))


def build_chat_payload(messages, temperature, top_p=None, response_format=None, model=None):
    """Build an OpenAI-compatible chat/completions payload."""
    payload = {
        "model": model or MODEL,
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
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fence:
        candidate = fence.group(1).strip()
        json.loads(candidate)
        return candidate

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            json.loads(candidate)
            return candidate

    raise ValueError("No parseable JSON found in model output")


def validate_json_schema(value, schema, path="$"):
    """Validate a JSON value against a minimal in-code schema subset."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate_json_schema(value, option, path)
                return
            except Exception:
                pass
        raise LLMResponseValidationError(f"{path}: did not match any allowed schema option")

    if "oneOf" in schema:
        matches = 0
        for option in schema["oneOf"]:
            try:
                validate_json_schema(value, option, path)
                matches += 1
            except Exception:
                pass
        if matches != 1:
            raise LLMResponseValidationError(f"{path}: did not match exactly one allowed schema option")
        return

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        for item_type in schema_type:
            try:
                validate_json_schema(value, {**schema, "type": item_type}, path)
                return
            except Exception:
                pass
        raise LLMResponseValidationError(f"{path}: did not match any allowed type")

    if schema_type == "object":
        if not isinstance(value, dict):
            raise LLMResponseValidationError(f"{path}: expected object")
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise LLMResponseValidationError(f"{path}: missing required key {key}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                validate_json_schema(value[key], subschema, f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            allowed = set(properties)
            extras = [key for key in value if key not in allowed]
            if extras:
                raise LLMResponseValidationError(f"{path}: unexpected keys {extras}")
    elif schema_type == "array":
        if not isinstance(value, list):
            raise LLMResponseValidationError(f"{path}: expected array")
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                validate_json_schema(item, item_schema, f"{path}[{idx}]")
    elif schema_type == "string":
        if not isinstance(value, str):
            raise LLMResponseValidationError(f"{path}: expected string")
        enum = schema.get("enum")
        if enum and value not in enum:
            raise LLMResponseValidationError(f"{path}: value {value!r} not in enum")
        min_len = schema.get("minLength")
        if min_len is not None and len(value) < min_len:
            raise LLMResponseValidationError(f"{path}: string too short")
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise LLMResponseValidationError(f"{path}: expected integer")
        enum = schema.get("enum")
        if enum and value not in enum:
            raise LLMResponseValidationError(f"{path}: value {value!r} not in enum")
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise LLMResponseValidationError(f"{path}: value below minimum")
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            raise LLMResponseValidationError(f"{path}: value above maximum")
    elif schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise LLMResponseValidationError(f"{path}: expected number")
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise LLMResponseValidationError(f"{path}: value below minimum")
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            raise LLMResponseValidationError(f"{path}: value above maximum")
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            raise LLMResponseValidationError(f"{path}: expected boolean")
    elif schema_type == "null":
        if value is not None:
            raise LLMResponseValidationError(f"{path}: expected null")


def schema_text(schema):
    """Render a schema compactly for repair prompts."""
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_repair_prompt(schema, original_prompt):
    """Create a concise repair prompt for invalid JSON responses."""
    return (
        "your previous response did not match the schema; return only valid JSON matching: "
        f"{schema_text(schema)}\n\nOriginal task:\n{original_prompt}"
    )


def build_empty_response_repair_prompt(schema, original_prompt):
    """Create a no-think JSON-only prompt for empty-response repair."""
    if schema is not None:
        prefix = "Respond with JSON ONLY. Do NOT think. Do NOT explain. Output must match this schema: "
        prefix += schema_text(schema)
    else:
        prefix = "Respond with JSON ONLY. Do NOT think. Do NOT explain."
    return f"{prefix}\n\n{original_prompt}"


def strict_json_response_format(schema):
    """Return a JSON schema response-format payload for compatible endpoints."""
    return {"type": "json_schema", "json_schema": schema}


def safe_response_format():
    """Compatibility shim for the legacy helper."""
    return {"type": "json_object"}


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


def prompt_with_schema(schema, prompt):
    """Wrap a prompt for strict JSON tasks with schema instructions."""
    return (
        "Respond with JSON only matching this schema.\n"
        f"Schema:\n{schema_text(schema)}\n\n"
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
            "cache_type_k": KV_CACHE_TYPE,
            "cache_type_v": KV_CACHE_TYPE,
            "num_keep": 256,
            "top_k": settings["top_k"],
            "min_p": settings["min_p"],
            "presence_penalty": settings["presence_penalty"],
            "repeat_penalty": settings["repeat_penalty"],
        },
        "keep_alive": settings["keep_alive"],
    }
    if schema is not None:
        payload["response_format"] = strict_json_response_format(schema)
        payload["format"] = schema
    return payload


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


def log_llm_call(record):
    """Emit a structured single-line call log."""
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def prompt_chars_for(prompt):
    """Return a prompt length used for logging."""
    return len(prompt or "")


def llm_chat_json(task, call_name, system_prompt, user_prompt, schema, *, task_deadline, candidate_validator=None, prompt_builder=None, call_state=None, num_predict_override=None, expect_json=True):
    """Call the chat endpoint with retries, schema validation, and repair."""
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

    def issue_request(prompt_text, *, num_ctx_used, num_predict_used, think_used, schema_prompt=True):
        request_prompt = prompt_text
        if settings["strict_json"] and expect_json and schema_prompt:
            request_prompt = prompt_with_schema(schema, request_prompt)
        request_chars = prompt_chars_for(request_prompt)
        payload = build_llm_payload(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_prompt},
            ],
            settings,
            schema=schema if settings["strict_json"] and expect_json else None,
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
            if settings["strict_json"] and expect_json and schema_prompt:
                prompt_for_budget = prompt_with_schema(schema, repair_prompt)
            prompt_tokens_est = estimate_prompt_tokens(prompt_for_budget)
            repair_num_ctx, repair_num_predict = clamp_empty_response_budget(repair_num_ctx, repair_num_predict, prompt_tokens_est)

            issue = issue_request(
                repair_prompt,
                num_ctx_used=repair_num_ctx,
                num_predict_used=repair_num_predict,
                think_used=repair_think,
                schema_prompt=schema_prompt,
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

    while True:
        if retries >= MAX_RETRIES:
            break

        if task_time_remaining(task_deadline) <= 0:
            raise LLMTaskTimeoutError(f"{call_name} exceeded TASK_TIMEOUT")

        prompt = prompt_builder(state)
        if settings["strict_json"] and expect_json:
            prompt = prompt_with_schema(schema, prompt)
        request_chars = prompt_chars_for(prompt)
        payload = build_llm_payload(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            settings,
            schema=schema if settings["strict_json"] and expect_json else None,
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
            payload_text = extract_json_payload(content)
            value = json.loads(payload_text)
            validate_json_schema(value, schema)
            if candidate_validator is not None:
                candidate_validator(value)
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
        except Exception as exc:
            last_error = repr(exc)
            repair_prompt = build_repair_prompt(schema, prompt)
            repair_payload = build_llm_payload(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_prompt},
                ],
                settings,
                schema=schema if settings["strict_json"] and expect_json else None,
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
                    repair_data = json.loads(repair_raw)
                    repair_content = repair_data["choices"][0]["message"]["content"]
                    payload_text = extract_json_payload(repair_content)
                    value = json.loads(payload_text)
                    validate_json_schema(value, schema)
                    if candidate_validator is not None:
                        candidate_validator(value)
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
            validate_json_schema(value, schema)
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
                    validate_json_schema(value, schema)
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
    payload = build_chat_payload([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], temperature=TEMPERATURE)
    return http_post_json(BASE_URL + "/chat/completions", payload, timeout=REQUEST_TIMEOUT)

# =========================
# JSON helpers
# =========================
def parse_json_from_text(text):
    """Parse JSON from raw text or fenced content.

    Args:
        text: Model output text that may wrap JSON in prose.

    Returns:
        Parsed JSON value.

    Raises:
        ValueError: If no parseable JSON payload is found.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fence:
        return json.loads(fence.group(1).strip())

    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        return json.loads(m.group(1))

    raise ValueError("No parseable JSON found in model output")

def load_json(path, default):
    """Load JSON from disk and return a fallback on failure.

    Args:
        path: File path to read.
        default: Value to return if loading fails.

    Returns:
        Parsed JSON content or the provided default.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    """Write an object to disk as UTF-8 formatted JSON.

    Args:
        path: File path to write.
        obj: JSON-serializable object.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =========================
# General helpers
# =========================
def normalize_space(s):
    """Collapse consecutive whitespace and trim leading/trailing space.

    Args:
        s: Input string or None.

    Returns:
        Normalized single-space string.
    """
    return re.sub(r"\s+", " ", s or "").strip()

def ts_to_ms(ts):
    """Convert an SRT timestamp string to milliseconds.

    Args:
        ts: Timestamp in HH:MM:SS,mmm format.

    Returns:
        Timestamp in milliseconds.
    """
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return ((int(h) * 3600 + int(m) * 60 + int(s)) * 1000) + int(ms)

def ms_to_srt(ms):
    """Convert milliseconds to an SRT timestamp string.

    Args:
        ms: Millisecond timestamp.

    Returns:
        Timestamp in HH:MM:SS,mmm format, or None.
    """
    if ms is None:
        return None
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def ms_to_clock(ms):
    """Convert milliseconds to an HH:MM:SS clock string.

    Args:
        ms: Millisecond timestamp.

    Returns:
        Timestamp in HH:MM:SS format, or None.
    """
    if ms is None:
        return None
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    return f"{h:02d}:{m:02d}:{s:02d}"

def duration_clock(start_ms, end_ms):
    """Compute a non-negative duration string between two timestamps.

    Args:
        start_ms: Start timestamp in milliseconds.
        end_ms: End timestamp in milliseconds.

    Returns:
        Duration as HH:MM:SS, or None when bounds are missing.
    """
    if start_ms is None or end_ms is None:
        return None
    d = max(0, end_ms - start_ms)
    return ms_to_clock(d)

def clamp(v, lo, hi):
    """Clamp a numeric value to a closed interval.

    Args:
        v: Input value.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Value constrained to [lo, hi].
    """
    return max(lo, min(hi, v))

def confidence_rank(c):
    """Map a confidence label to a sortable rank.

    Args:
        c: Confidence label.

    Returns:
        Integer rank where high > medium > low.
    """
    return {"low": 1, "medium": 2, "high": 3}.get((c or "low").lower(), 1)

def downgrade_confidence(c):
    """Reduce confidence by one level, bottoming out at low.

    Args:
        c: Confidence label.

    Returns:
        Downgraded confidence label.
    """
    if c == "high":
        return "medium"
    if c == "medium":
        return "low"
    return "low"


def confidence_score_to_label(score):
    """Map a numeric confidence score to the existing low/medium/high labels."""
    try:
        score = float(score)
    except Exception:
        return "low"
    if score >= 0.85:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"

# =========================
# SRT parsing
# =========================
def parse_srt(srt_text):
    """Parse SRT text into normalized cue dictionaries.

    Args:
        srt_text: Full SRT file content.

    Returns:
        List of cue dictionaries with ids, timestamps, and text.
    """
    blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.M)
    cues = []
    cue_id = 1
    ts_re = re.compile(r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})")

    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        ts_idx = None
        ts_match = None
        for i, ln in enumerate(lines):
            m = ts_re.search(ln)
            if m:
                ts_idx = i
                ts_match = m
                break
        if ts_idx is None:
            continue

        text_lines = lines[ts_idx + 1:]
        text = " ".join(re.sub(r"</?i>|</?b>|</?u>|<[^>]+>", "", ln).strip() for ln in text_lines).strip()

        cues.append({
            "cue_id": cue_id,
            "start_ms": ts_to_ms(ts_match.group("start")),
            "end_ms": ts_to_ms(ts_match.group("end")),
            "start": ts_match.group("start"),
            "end": ts_match.group("end"),
            "text": normalize_space(text),
        })
        cue_id += 1

    return cues

# =========================
# Flyer parsing
# =========================
def parse_flyer_songs(flyer_text):
    """Extract ordered song metadata from flyer text.

    Args:
        flyer_text: Full flyer content.

    Returns:
        Song dictionaries with index, act, title, and performers.
    """
    lines = [ln.strip() for ln in flyer_text.splitlines()]
    songs = []
    current_act = None
    in_song_breakdown = False

    for line in lines:
        if "SONG BREAKDOWN" in line.upper():
            in_song_breakdown = True
            continue
        if not in_song_breakdown:
            continue

        upper = line.upper()
        if upper == "ACT 1":
            current_act = 1
            continue
        if upper == "ACT 2":
            current_act = 2
            continue
        if line.startswith("- "):
            body = line[2:].strip()
            if ":" in body:
                title, performers = body.split(":", 1)
            else:
                title, performers = body, ""
            songs.append({
                "index": len(songs) + 1,
                "act": current_act,
                "title": normalize_space(title),
                "performers": normalize_space(performers),
            })

    return songs

def parse_flyer_plot_summary(flyer_text):
    """Extract and normalize the plot-summary section from flyer text.

    Args:
        flyer_text: Full flyer content.

    Returns:
        Flattened plot summary text, or an empty string.
    """
    m = re.search(
        r"INTO THE WOODS PLOT SUMMARY\s*(.*?)(?:## Page 4|INTO THE WOODS DIRECTOR|INTO THE WOODS: SONG BREAKDOWN)",
        flyer_text,
        re.S | re.I
    )
    if m:
        return normalize_space(m.group(1))
    return ""

# =========================
# Audio extraction + analysis
# =========================
def ensure_audio_wav(input_media, wav_path):
    """Extract mono PCM WAV audio with ffmpeg when missing.

    Args:
        input_media: Source media path.
        wav_path: Destination WAV path.

    Returns:
        Path to the extracted WAV file.

    Raises:
        subprocess.CalledProcessError: If ffmpeg extraction fails.
    """
    if os.path.exists(wav_path):
        return wav_path
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_media,
        "-ac", "1",
        "-ar", str(AUDIO_SR),
        "-vn",
        "-acodec", "pcm_s16le",
        wav_path
    ]
    print("Extracting audio with ffmpeg...")
    subprocess.run(cmd, check=True)
    return wav_path

def load_wav_mono(path):
    """Load a WAV file and return mono float samples and sample rate.

    Args:
        path: WAV file path.

    Returns:
        Tuple of (samples, sample_rate).

    Raises:
        SystemExit: If WAV sample width is not 16-bit PCM.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise SystemExit("Expected 16-bit PCM wav after ffmpeg extraction.")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, framerate

def compute_basic_features(x, sr):
    """Compute lightweight time and spectral features for a signal.

    Args:
        x: Audio samples.
        sr: Sample rate.

    Returns:
        Feature dictionary with rms, zcr, centroid, and flatness.
    """
    if x is None or len(x) == 0:
        return {"rms": 0.0, "zcr": 0.0, "centroid": 0.0, "flatness": 1.0}

    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))

    sb = np.signbit(x)
    zcr = float(np.mean(sb[1:] != sb[:-1])) if len(x) > 1 else 0.0

    win = np.hanning(len(x)).astype(np.float32)
    mag = np.abs(np.fft.rfft(x * win)) + 1e-10
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)

    centroid = float((freqs * mag).sum() / mag.sum()) if mag.sum() > 0 else 0.0
    flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag)) if np.mean(mag) > 0 else 1.0

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "flatness": flatness,
    }

def percentile_bounds(arr, lo=10, hi=90):
    """Return robust percentile bounds with a nonzero span.

    Args:
        arr: Numeric array-like input.
        lo: Lower percentile.
        hi: Upper percentile.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return 0.0, 1.0
    a = float(np.percentile(arr, lo))
    b = float(np.percentile(arr, hi))
    if b <= a:
        b = a + 1e-6
    return a, b

def norm01(v, lo, hi):
    """Normalize a value to the [0, 1] interval.

    Args:
        v: Value to normalize.
        lo: Lower calibration bound.
        hi: Upper calibration bound.

    Returns:
        Clipped normalized value.
    """
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo))))

def build_audio_model(y, sr):
    """Build global audio normalization statistics from frame scans.

    Args:
        y: Mono audio samples.
        sr: Sample rate.

    Returns:
        Dictionary containing audio data and normalization bounds.

    Raises:
        SystemExit: If frame parameters are invalid.
    """
    frame = int(FRAME_SEC * sr)
    hop = int(FRAME_HOP_SEC * sr)
    if frame <= 0 or hop <= 0:
        raise SystemExit("Bad audio frame settings")

    rms_vals, zcr_vals, centroid_vals, flatness_vals = [], [], [], []
    i = 0
    while i + frame <= len(y):
        feats = compute_basic_features(y[i:i + frame], sr)
        rms_vals.append(feats["rms"])
        zcr_vals.append(feats["zcr"])
        centroid_vals.append(feats["centroid"])
        flatness_vals.append(feats["flatness"])
        i += hop

    rms_lo, rms_hi = percentile_bounds(rms_vals, 10, 90)
    zcr_lo, zcr_hi = percentile_bounds(zcr_vals, 10, 90)
    cen_lo, cen_hi = percentile_bounds(centroid_vals, 10, 90)
    flat_lo, flat_hi = percentile_bounds(flatness_vals, 10, 90)

    silence_thr = float(np.percentile(np.asarray(rms_vals) if len(rms_vals) else np.array([0.0]), 20))

    return {
        "y": y,
        "sr": sr,
        "rms_lo": rms_lo,
        "rms_hi": rms_hi,
        "zcr_lo": zcr_lo,
        "zcr_hi": zcr_hi,
        "cen_lo": cen_lo,
        "cen_hi": cen_hi,
        "flat_lo": flat_lo,
        "flat_hi": flat_hi,
        "silence_thr": silence_thr,
    }

def ms_to_sample(ms, sr):
    """Convert milliseconds to a sample index at a given rate.

    Args:
        ms: Time in milliseconds.
        sr: Sample rate.

    Returns:
        Integer sample index.
    """
    return int((ms / 1000.0) * sr)

def slice_audio(audio_model, start_ms, end_ms):
    """Slice audio samples between two millisecond timestamps.

    Args:
        audio_model: Audio model containing samples and sample rate.
        start_ms: Start timestamp.
        end_ms: End timestamp.

    Returns:
        Audio slice as a float32 NumPy array.
    """
    y = audio_model["y"]
    sr = audio_model["sr"]
    s = clamp(ms_to_sample(start_ms, sr), 0, len(y))
    e = clamp(ms_to_sample(end_ms, sr), 0, len(y))
    if e <= s:
        return np.zeros(0, dtype=np.float32)
    return y[s:e]

@lru_cache(maxsize=50000)
def span_stats_cached(start_ms, end_ms, audio_key=None):
    """Keep a placeholder cached signature for legacy compatibility.

    Args:
        start_ms: Start timestamp.
        end_ms: End timestamp.
        audio_key: Optional cache key placeholder.

    Returns:
        Empty dictionary.
    """
    # audio_key ignored; just for lru signature compatibility if needed
    return {}

def attach_span_stats(audio_model):
    """Create a cached span-statistics function bound to an audio model.

    Args:
        audio_model: Audio model with samples and normalization bounds.

    Returns:
        Function that computes robust statistics for [start_ms, end_ms].
    """
    @lru_cache(maxsize=50000)
    def _span_stats(start_ms, end_ms):
        """Compute cached statistics for one audio span.

        Args:
            start_ms: Start timestamp in milliseconds.
            end_ms: End timestamp in milliseconds.

        Returns:
            Dictionary of normalized and raw span-level audio features.
        """
        x = slice_audio(audio_model, start_ms, end_ms)
        sr = audio_model["sr"]

        if len(x) == 0:
            return {
                "rms": 0.0,
                "rms_n": 0.0,
                "zcr": 0.0,
                "zcr_n": 0.0,
                "centroid": 0.0,
                "centroid_n": 0.0,
                "flatness": 1.0,
                "flatness_n": 1.0,
                "nonsilent_ratio": 0.0,
                "music_bed_score": 0.0,
            }

        # Break into 250 ms mini-frames for robustness
        sub = int(0.25 * sr)
        if sub <= 0:
            sub = len(x)

        rms_list = []
        feats_acc = {"zcr": [], "centroid": [], "flatness": []}

        i = 0
        while i < len(x):
            part = x[i:i + sub]
            if len(part) < max(64, sub // 4):
                break
            feats = compute_basic_features(part, sr)
            rms_list.append(feats["rms"])
            feats_acc["zcr"].append(feats["zcr"])
            feats_acc["centroid"].append(feats["centroid"])
            feats_acc["flatness"].append(feats["flatness"])
            i += sub

        if not rms_list:
            feats = compute_basic_features(x, sr)
            rms_list = [feats["rms"]]
            feats_acc["zcr"] = [feats["zcr"]]
            feats_acc["centroid"] = [feats["centroid"]]
            feats_acc["flatness"] = [feats["flatness"]]

        rms = float(np.mean(rms_list))
        zcr = float(np.mean(feats_acc["zcr"]))
        centroid = float(np.mean(feats_acc["centroid"]))
        flatness = float(np.mean(feats_acc["flatness"]))

        rms_n = norm01(rms, audio_model["rms_lo"], audio_model["rms_hi"])
        zcr_n = norm01(zcr, audio_model["zcr_lo"], audio_model["zcr_hi"])
        centroid_n = norm01(centroid, audio_model["cen_lo"], audio_model["cen_hi"])
        flatness_n = norm01(flatness, audio_model["flat_lo"], audio_model["flat_hi"])

        nonsilent_ratio = float(np.mean(np.asarray(rms_list) > audio_model["silence_thr"]))
        music_bed_score = 0.65 * nonsilent_ratio + 0.35 * rms_n

        return {
            "rms": rms,
            "rms_n": rms_n,
            "zcr": zcr,
            "zcr_n": zcr_n,
            "centroid": centroid,
            "centroid_n": centroid_n,
            "flatness": flatness,
            "flatness_n": flatness_n,
            "nonsilent_ratio": nonsilent_ratio,
            "music_bed_score": float(max(0.0, min(1.0, music_bed_score))),
        }

    return _span_stats

# =========================
# Cue / block analysis
# =========================
def slug_tokens(s):
    """Tokenize text into lowercase alphanumeric terms.

    Args:
        s: Input string.

    Returns:
        List of token strings.
    """
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]

GENERIC_TITLE_TOKENS = {
    "part", "parts", "act", "opening", "finale", "reprise",
    "the", "a", "an", "of", "and", "i", "ii", "iii", "iv",
    "v", "vi", "vii", "viii", "ix", "x"
}

def title_tokens(title):
    """Return filtered title tokens useful for fuzzy anchor matching.

    Args:
        title: Song title.

    Returns:
        Title tokens excluding very short and generic words.
    """
    return [t for t in slug_tokens(title) if len(t) > 2 and t not in GENERIC_TITLE_TOKENS]

def attach_cue_audio_features(cues, span_stats):
    """Augment cues with per-cue audio and adjacency gap features.

    Args:
        cues: Cue dictionaries to update in place.
        span_stats: Function returning stats for a time span.
    """
    for i, c in enumerate(cues):
        feats = span_stats(c["start_ms"], c["end_ms"])
        dur_s = max(0.05, (c["end_ms"] - c["start_ms"]) / 1000.0)
        c["dur_s"] = round(dur_s, 3)
        c["chars"] = len(c["text"])
        c["chars_per_sec"] = round(c["chars"] / dur_s, 3)
        c["audio"] = feats

        prev_gap = 0 if i == 0 else max(0, c["start_ms"] - cues[i - 1]["end_ms"])
        next_gap = 0 if i == len(cues) - 1 else max(0, cues[i + 1]["start_ms"] - c["end_ms"])

        c["gap_before_ms"] = prev_gap
        c["gap_after_ms"] = next_gap

    for i, c in enumerate(cues):
        if i == 0:
            c["bed_before"] = 0.0
        else:
            gs = span_stats(cues[i - 1]["end_ms"], c["start_ms"]) if c["gap_before_ms"] > 0 else {"music_bed_score": 1.0}
            c["bed_before"] = round(gs["music_bed_score"], 3)

        if i == len(cues) - 1:
            c["bed_after"] = 0.0
        else:
            gs = span_stats(c["end_ms"], cues[i + 1]["start_ms"]) if c["gap_after_ms"] > 0 else {"music_bed_score": 1.0}
            c["bed_after"] = round(gs["music_bed_score"], 3)

def build_blocks(cues, span_stats):
    """Group cues into continuity blocks using gap and audio heuristics.

    Args:
        cues: Ordered subtitle cue list.
        span_stats: Function returning stats for a time span.

    Returns:
        List of block dictionaries.
    """
    blocks = []
    if not cues:
        return blocks

    start_i = 0
    block_id = 1

    for i in range(len(cues) - 1):
        a = cues[i]
        b = cues[i + 1]
        gap_ms = max(0, b["start_ms"] - a["end_ms"])
        gap_feats = span_stats(a["end_ms"], b["start_ms"]) if gap_ms > 0 else {"music_bed_score": 1.0, "rms_n": 1.0}

        # Keep continuity when short gaps or music bed suggests singing.
        same_block = (
            gap_ms <= SHORT_GAP_MS or
            (gap_ms <= LONG_GAP_MS and gap_feats["music_bed_score"] >= BED_MIN_RATIO and gap_feats["rms_n"] >= BED_MIN_RMS_N)
        )

        if not same_block:
            blocks.append(make_block(block_id, cues[start_i:i + 1], span_stats))
            block_id += 1
            start_i = i + 1

    blocks.append(make_block(block_id, cues[start_i:], span_stats))
    return blocks

def make_block(block_id, subset, span_stats):
    """Build one aggregate block record from a cue subset.

    Args:
        block_id: Sequential block identifier.
        subset: Consecutive cues in this block.
        span_stats: Function returning stats for a time span.

    Returns:
        Block summary dictionary.
    """
    start_ms = subset[0]["start_ms"]
    end_ms = subset[-1]["end_ms"]
    dur_s = max(0.1, (end_ms - start_ms) / 1000.0)

    gap_scores = []
    for i in range(len(subset) - 1):
        gs = span_stats(subset[i]["end_ms"], subset[i + 1]["start_ms"]) if subset[i + 1]["start_ms"] > subset[i]["end_ms"] else {"music_bed_score": 1.0}
        gap_scores.append(gs["music_bed_score"])

    music_bed_ratio = float(np.mean(np.asarray(gap_scores) >= BED_MIN_RATIO)) if gap_scores else 1.0
    mean_gap_music = float(np.mean(gap_scores)) if gap_scores else 1.0
    avg_rms_n = float(np.mean([c["audio"]["rms_n"] for c in subset])) if subset else 0.0

    cue_density = len(subset) / dur_s
    text_density = sum(len(c["text"]) for c in subset) / dur_s

    song_like_score = (
        0.40 * music_bed_ratio +
        0.25 * mean_gap_music +
        0.20 * min(1.0, cue_density / 0.50) +
        0.15 * min(1.0, text_density / 18.0)
    )

    excerpt = " | ".join([c["text"] for c in subset[:4] if c["text"]])[:500]

    return {
        "block_id": block_id,
        "start_cue_id": subset[0]["cue_id"],
        "end_cue_id": subset[-1]["cue_id"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration": duration_clock(start_ms, end_ms),
        "cue_count": len(subset),
        "music_bed_ratio": round(music_bed_ratio, 3),
        "mean_gap_music": round(mean_gap_music, 3),
        "avg_rms_n": round(avg_rms_n, 3),
        "song_like_score": round(song_like_score, 3),
        "excerpt": excerpt,
    }

def block_title_anchor_hits(block, title):
    """Find shared tokens between a block excerpt and a song title.

    Args:
        block: Block dictionary with excerpt text.
        title: Song title.

    Returns:
        Sorted list of overlapping anchor tokens.
    """
    block_toks = set(slug_tokens(block["excerpt"]))
    tt = set(title_tokens(title))
    if not tt:
        return []
    return sorted(list(block_toks & tt))

def find_cue_index_by_id(cues, cue_id):
    """Locate the list index for a cue id.

    Args:
        cues: Ordered cue list.
        cue_id: Cue identifier.

    Returns:
        Zero-based index, or None when not found.
    """
    for i, c in enumerate(cues):
        if c["cue_id"] == cue_id:
            return i
    return None

def find_block_index_for_cue(blocks, cue_id):
    """Find the block index that contains a cue id.

    Args:
        blocks: Ordered block list.
        cue_id: Cue identifier.

    Returns:
        Zero-based block index, defaulting to 0.
    """
    for i, b in enumerate(blocks):
        if b["start_cue_id"] <= cue_id <= b["end_cue_id"]:
            return i
    return 0

def cue_map(cues):
    """Build a cue-id lookup table.

    Args:
        cues: Cue list.

    Returns:
        Dictionary keyed by cue_id.
    """
    return {c["cue_id"]: c for c in cues}

# =========================
# Prompt builders
# =========================
def song_list_text(songs):
    """Format song metadata as numbered text lines for prompts.

    Args:
        songs: Ordered song dictionaries.

    Returns:
        Newline-separated song list text.
    """
    return "\n".join([f'{s["index"]}. Act {s["act"]} - {s["title"]} : {s["performers"]}' for s in songs])


def one_sentence_summary(text):
    """Condense a summary down to its first sentence when possible."""
    cleaned = normalize_space(text)
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return parts[0].strip()


BLOCK_SEARCH_SCHEMA = {
    "type": "object",
    "required": ["selected_block_id", "confidence", "reason"],
    "properties": {
        "selected_block_id": {"type": "integer"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}

BOUNDARY_SELECTION_SCHEMA = {
    "type": "object",
    "required": ["found", "start_candidate_index", "end_candidate_index", "confidence", "reason"],
    "properties": {
        "found": {"type": "boolean"},
        "start_candidate_index": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "end_candidate_index": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}

LYRICS_WINDOW_SCHEMA = {
    "type": "object",
    "required": ["lyrics_cue_ids", "confidence", "reason"],
    "properties": {
        "lyrics_cue_ids": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}

CHAPTER_SCHEMA = {
    "type": "object",
    "required": ["index", "act", "song_title", "title", "summary", "themes", "characters"],
    "properties": {
        "index": {"type": "integer"},
        "act": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "song_title": {"type": "string"},
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "themes": {"type": "array", "items": {"type": "string"}},
        "characters": {"type": "array", "items": {"type": "string"}},
        "chapter_title": {"type": "string"},
        "key_characters": {"type": "array", "items": {"type": "string"}},
        "key_events": {"type": "array", "items": {"type": "string"}},
        "continuity_notes": {"type": "array", "items": {"type": "string"}},
        "story_role": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "additionalProperties": False,
}

FINAL_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["overall_summary", "act_summaries"],
    "properties": {
        "overall_summary": {"type": "string"},
        "act_summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["act", "summary"],
                "properties": {
                    "act": {"type": "integer"},
                    "summary": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def build_boundary_candidates(cues, blocks):
    """Build ordered boundary candidates from block starts and long cue gaps."""
    candidates = []
    seen = set()

    def add_candidate(cue_id, ms, kind, source_block_id, label):
        if cue_id in seen:
            return
        seen.add(cue_id)
        candidates.append({
            "cue_id": cue_id,
            "ms": ms,
            "kind": kind,
            "source_block_id": source_block_id,
            "label": label,
        })

    for block in blocks:
        add_candidate(
            block["start_cue_id"],
            block["start_ms"],
            "block_start",
            block["block_id"],
            block.get("excerpt", "")[:120],
        )

    for i in range(len(cues) - 1):
        gap_ms = max(0, cues[i + 1]["start_ms"] - cues[i]["end_ms"])
        if gap_ms >= LONG_GAP_MS:
            add_candidate(
                cues[i + 1]["cue_id"],
                cues[i + 1]["start_ms"],
                "long_gap",
                0,
                f"gap {gap_ms}ms after cue {cues[i]['cue_id']}",
            )

    if cues:
        add_candidate(len(cues) + 1, cues[-1]["end_ms"] + 1, "end_of_file", 0, "end of file")

    candidates.sort(key=lambda x: (x["cue_id"], x["ms"]))
    for idx, candidate in enumerate(candidates, start=1):
        candidate["candidate_index"] = idx
    return candidates


def format_boundary_candidates(candidates):
    """Render candidate boundaries for the LLM prompt."""
    rows = []
    for c in candidates:
        rows.append(
            f'{c["candidate_index"]}. cue_id={c["cue_id"]} ms={ms_to_clock(c["ms"])} kind={c["kind"]} block={c["source_block_id"]} label={c["label"]}'
        )
    return "\n".join(rows)


def choose_candidate_blocks(blocks, start_block_idx, limit=SEARCH_BLOCKS):
    """Select a sliding window of candidate blocks."""
    return blocks[start_block_idx:start_block_idx + limit]


def candidate_index_lookup(candidates):
    """Build a lookup table for candidate indexes."""
    return {c["candidate_index"]: c for c in candidates}


def _sanitize_lyrics_ids(resp, subset, start_cue_id, end_cue_id):
    """Drop lyric cue IDs that are not present in the supplied window."""
    valid_ids = {c["cue_id"] for c in subset if start_cue_id <= c["cue_id"] <= end_cue_id}
    original = list(resp.get("lyrics_cue_ids", []) or [])
    filtered = []
    dropped = []
    seen = set()

    for cue_id in original:
        if not isinstance(cue_id, int):
            continue
        if cue_id not in valid_ids:
            dropped.append(cue_id)
            continue
        if cue_id in seen:
            continue
        seen.add(cue_id)
        filtered.append(cue_id)

    if dropped:
        print(f"LLM task=lyrics_window dropped_unknown_ids={dropped}")
    resp["lyrics_cue_ids"] = filtered


def shrink_search_blocks(current):
    """Reduce a search window for an oversized detection prompt."""
    return max(1, current - max(1, current // 3))


def shrink_boundary_context(current):
    """Reduce boundary context for an oversized detection prompt."""
    return max(0, current - 1)


def shrink_lyrics_window(window_cues, overlap):
    """Reduce lyric window size and expand overlap proportionally."""
    reduced_window = max(24, int(window_cues * 0.75))
    if reduced_window >= window_cues and window_cues > 24:
        reduced_window = window_cues - 1
    delta = max(1, window_cues - reduced_window)
    reduced_overlap = min(reduced_window - 1, max(overlap + delta // 2, overlap))
    return reduced_window, reduced_overlap

def format_block_summaries(blocks, target_song):
    """Render compact block summaries with title-anchor hints.

    Args:
        blocks: Candidate block dictionaries.
        target_song: Song dictionary used for token anchor hits.

    Returns:
        Newline-separated summary rows.
    """
    rows = []
    for b in blocks:
        # Accept legacy block dicts that used "end_m".
        end_ms = b.get("end_ms", b.get("end_m", b.get("start_ms", 0)))
        hits = block_title_anchor_hits(b, target_song["title"])
        rows.append(
            f'BLOCK {b["block_id"]} | cues {b["start_cue_id"]}-{b["end_cue_id"]} | '
            f'{ms_to_clock(b["start_ms"])}-{ms_to_clock(end_ms)} | dur {b["duration"]} | '
            f'music_bed_ratio {b["music_bed_ratio"]} | song_like_score {b["song_like_score"]} | '
            f'title_anchor_hits {hits} | excerpt: {b["excerpt"]}'
        )
    return "\n".join(rows)

def block_search_prompt(target_song, prev_song, next_songs, search_cue_id, candidate_blocks, all_songs):
    """Build the prompt for selecting a candidate song block.

    Args:
        target_song: Song currently being searched.
        prev_song: Previous song dictionary or None.
        next_songs: Nearby upcoming songs for context.
        search_cue_id: Earliest cue id to consider.
        candidate_blocks: Candidate block subset.
        all_songs: Full flyer song ordering.

    Returns:
        Prompt text for block selection.
    """
    return f"""Find where the next flyer song most likely occurs.

Ordered flyer song list:
{song_list_text(all_songs)}

Current target song:
{json.dumps(target_song, ensure_ascii=False)}

Previous song:
{json.dumps(prev_song, ensure_ascii=False) if prev_song else "null"}

Upcoming songs after target:
{json.dumps(next_songs, ensure_ascii=False)}

We are searching at or after cue_id:
{search_cue_id}

Candidate subtitle/audio blocks:
{format_block_summaries(candidate_blocks, target_song)}

Return JSON only:
{{
  "selected_block_id": integer or 0,
  "confidence": "high|medium|low",
  "reason": "brief explanation"
}}

Rules:
- Select the block most likely to contain the target song.
- Use flyer order strictly.
- Spoken dialogue blocks should return 0 if they do not contain the target song.
- Prefer blocks with musical continuity and lyric-like subtitle flow.
- Do not skip ahead to a later song unless the target clearly is not present in these blocks.
"""

def cue_rows_for_prompt(cues_subset):
    """Format cues as TSV-like rows for LLM boundary tasks.

    Args:
        cues_subset: Cue dictionaries in the current window.

    Returns:
        Newline-separated cue rows.
    """
    rows = []
    for c in cues_subset:
        rows.append(
            f'{c["cue_id"]}\t{ms_to_clock(c["start_ms"])}\t{ms_to_clock(c["end_ms"])}\t'
            f'{c["dur_s"]:.2f}\t{c["gap_before_ms"]/1000.0:.2f}\t{c["gap_after_ms"]/1000.0:.2f}\t'
            f'{c["bed_before"]:.2f}\t{c["bed_after"]:.2f}\t{c["audio"]["rms_n"]:.2f}\t'
            f'{c["chars_per_sec"]:.2f}\t{c["text"]}'
        )
    return "\n".join(rows)

def boundary_refine_prompt(target_song, prev_song, next_song, search_cue_id, cues_subset, candidates):
    """Build the prompt for precise start/end cue refinement.

    Args:
        target_song: Song currently being refined.
        prev_song: Previous song dictionary or None.
        next_song: Next song dictionary or None.
        search_cue_id: Earliest cue id allowed.
        cues_subset: Cue context around candidate bounds.

    Returns:
        Prompt text for boundary refinement.
    """
    return f"""Identify the exact cue boundaries for this song inside the provided context.

Target song:
{json.dumps(target_song, ensure_ascii=False)}

Previous song:
{json.dumps(prev_song, ensure_ascii=False) if prev_song else "null"}

Next song:
{json.dumps(next_song, ensure_ascii=False) if next_song else "null"}

Current search starts at or after cue_id:
{search_cue_id}

Subtitle cues are TSV with columns:
cue_id  start  end  dur_s  gap_before_s  gap_after_s  bed_before  bed_after  rms_n  chars_per_sec  text

Context cues:
{cue_rows_for_prompt(cues_subset)}

Boundary candidates (choose by index only):
{format_boundary_candidates(candidates)}

Return JSON only:
{{
    "found": true or false,
    "start_candidate_index": integer or null,
    "end_candidate_index": integer or null,
        "confidence": number from 0.0 to 1.0,
  "reason": "brief explanation"
}}

Rules:
- choose the earliest candidate that clearly marks the start of the song.
- choose the latest candidate that clearly marks the end of the song.
- Exclude spoken dialogue before/after the song.
- If the song is not present here, return found=false and null candidate indexes.
- Use the audio hints to separate sung flow from spoken dialogue.
"""

def lyrics_window_prompt(target_song, song_start_cue_id, song_end_cue_id, cues_subset):
    """Build the prompt for selecting lyric-only cues in a window.

    Args:
        target_song: Song currently being processed.
        song_start_cue_id: Lower bound cue id.
        song_end_cue_id: Upper bound cue id.
        cues_subset: Cue window to classify.

    Returns:
        Prompt text for lyric cue extraction.
    """
    return f"""Within this already-bounded song region, identify which cues are sung lyrics.

Target song:
{json.dumps(target_song, ensure_ascii=False)}

Bounded song cue range:
{song_start_cue_id} to {song_end_cue_id}

Subtitle cues are TSV with columns:
cue_id  start  end  dur_s  gap_before_s  gap_after_s  bed_before  bed_after  rms_n  chars_per_sec  text

Window:
{cue_rows_for_prompt(cues_subset)}

Return JSON only:
{{
  "lyrics_cue_ids": [integers],
  "confidence": "high|medium|low",
  "reason": "brief explanation"
}}

Rules:
- Include only sung lyric cues for the target song.
- Exclude spoken dialogue, banter, scene text, and obvious non-lyric cues.
- If a cue is ambiguous but likely sung, include it.
- Return only cue_ids visible in this window.
"""

def chapter_prompt(flyer_plot_summary, prior_chapters, song_record):
    """Build the prompt for one song-as-chapter story summary.

    Args:
        flyer_plot_summary: Global plot summary from flyer text.
        prior_chapters: Previously generated chapter summaries.
        song_record: Current song detection record.

    Returns:
        Prompt text for chapter generation.
    """
    return f"""Create a chapter-style story summary for this song.

Global plot summary from flyer:
{flyer_plot_summary}

Prior chapter continuity, in order:
{json.dumps(prior_chapters, ensure_ascii=False, indent=2) if prior_chapters else "[]"}

Current song record:
{json.dumps(song_record, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "index": {song_record["index"]},
    "act": {song_record["act"] if song_record["act"] is not None else "null"},
  "song_title": {json.dumps(song_record["song_title"])},
    "title": "short readable chapter title",
    "summary": "1-3 paragraph story summary of what happens in this song",
    "themes": ["theme 1", "theme 2"],
    "characters": ["names"],
    "chapter_title": "short readable chapter title",
    "key_characters": ["names"],
    "key_events": ["event 1", "event 2"],
    "continuity_notes": ["important carry-forward facts"],
    "story_role": "setup|decision|conflict|turning point|aftermath|finale",
    "confidence": "high|medium|low"
}}
"""

def final_assembly_prompt(chapters, flyer_plot_summary):
    """Build the prompt for overall and per-act story synthesis.

    Args:
        chapters: Song chapter summaries.
        flyer_plot_summary: Global plot summary from flyer text.

    Returns:
        Prompt text for final story assembly.
    """
    return f"""Create a polished overall story summary of the musical based on these song-chapters.

Flyer plot summary:
{flyer_plot_summary}

Chapter summaries:
{json.dumps(chapters, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "overall_summary": "multi-paragraph summary of the full musical",
  "act_summaries": [
    {{"act": 1, "summary": "..." }},
    {{"act": 2, "summary": "..." }}
  ]
}}
"""

# =========================
# Detection pipeline
# =========================
def refine_song_in_blocks(cues, blocks, boundary_candidates, song, prev_song, next_song, selected_block_id, search_cue_id, task_deadline):
    """Refine selected block context into exact song cue boundaries."""
    block_idx = selected_block_id - 1
    context_blocks = BOUNDARY_CONTEXT_BLOCKS

    while True:
        lo = max(0, block_idx - context_blocks)
        hi = min(len(blocks), block_idx + context_blocks + 1)
        cue_lo = blocks[lo]["start_cue_id"]
        cue_hi = blocks[hi - 1]["end_cue_id"]

        subset = [c for c in cues if cue_lo <= c["cue_id"] <= cue_hi and c["cue_id"] >= search_cue_id]
        candidate_pool = [c for c in boundary_candidates if cue_lo <= c["cue_id"] <= cue_hi + 1 and c["cue_id"] >= search_cue_id]
        if not subset or not candidate_pool:
            if context_blocks <= 0:
                return None
            context_blocks = shrink_boundary_context(context_blocks)
            continue

        prompt = boundary_refine_prompt(song, prev_song, next_song, search_cue_id, subset, candidate_pool)
        prompt_tokens_est = estimate_prompt_tokens(SYSTEM + "\n" + prompt)
        while prompt_tokens_est > MAX_PROMPT_TOKENS and context_blocks > 0:
            new_context = shrink_boundary_context(context_blocks)
            if new_context == context_blocks:
                break
            print(f"LLM auto-shrink task=boundary_refine boundary_context_blocks {context_blocks}->{new_context} prompt_tokens_est={prompt_tokens_est}")
            context_blocks = new_context
            lo = max(0, block_idx - context_blocks)
            hi = min(len(blocks), block_idx + context_blocks + 1)
            cue_lo = blocks[lo]["start_cue_id"]
            cue_hi = blocks[hi - 1]["end_cue_id"]
            subset = [c for c in cues if cue_lo <= c["cue_id"] <= cue_hi and c["cue_id"] >= search_cue_id]
            candidate_pool = [c for c in boundary_candidates if cue_lo <= c["cue_id"] <= cue_hi + 1 and c["cue_id"] >= search_cue_id]
            prompt = boundary_refine_prompt(song, prev_song, next_song, search_cue_id, subset, candidate_pool)
            prompt_tokens_est = estimate_prompt_tokens(SYSTEM + "\n" + prompt)

        if not subset or not candidate_pool:
            return None

        lookup = candidate_index_lookup(candidate_pool)

        def candidate_validator(resp):
            if not resp.get("found", False):
                return
            start_idx = resp.get("start_candidate_index")
            end_idx = resp.get("end_candidate_index")
            if not isinstance(start_idx, int) or not isinstance(end_idx, int):
                raise LLMResponseValidationError("boundary candidate indexes must be integers")
            if start_idx not in lookup or end_idx not in lookup:
                raise LLMResponseValidationError("boundary candidate indexes must come from the supplied set")

        resp = llm_chat_json(
            "detect",
            "boundary_refine",
            SYSTEM,
            prompt,
            BOUNDARY_SELECTION_SCHEMA,
            task_deadline=task_deadline,
            candidate_validator=candidate_validator,
        )

        if not resp.get("found", False):
            return None

        start_idx = resp.get("start_candidate_index")
        end_idx = resp.get("end_candidate_index")
        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            return None
        if start_idx not in lookup or end_idx not in lookup:
            return None

        start_candidate = lookup[start_idx]
        end_candidate = lookup[end_idx]
        confidence_score = float(resp.get("confidence") or 0.0)
        confidence_label = confidence_score_to_label(confidence_score)

        if ENABLE_VERIFIER:
            candidate_indexes = [c["candidate_index"] for c in candidate_pool]
            edge_choice = start_idx == min(candidate_indexes) or end_idx == max(candidate_indexes)
            if edge_choice or confidence_score < 0.7:
                verify_start = max(1, start_idx - TASK_BOUNDARY_NEIGHBORS)
                verify_end = min(max(candidate_indexes), end_idx + TASK_BOUNDARY_NEIGHBORS)
                verify_candidates = [c for c in candidate_pool if verify_start <= c["candidate_index"] <= verify_end]
                if verify_candidates:
                    verify_lookup = candidate_index_lookup(verify_candidates)
                    verify_prompt = boundary_refine_prompt(song, prev_song, next_song, search_cue_id, subset, verify_candidates)
                    verify_prompt += f"\n\nVerification pass: confirm or revise the chosen boundaries. Original choice was start candidate {start_idx} and end candidate {end_idx}."

                    def verify_candidate_validator(resp):
                        if not resp.get("found", False):
                            return
                        verify_start_idx = resp.get("start_candidate_index")
                        verify_end_idx = resp.get("end_candidate_index")
                        if verify_start_idx not in verify_lookup or verify_end_idx not in verify_lookup:
                            raise LLMResponseValidationError("verification indexes must come from the supplied neighborhood")

                    verify_resp = llm_chat_json(
                        "verify",
                        "boundary_verify",
                        SYSTEM,
                        verify_prompt,
                        BOUNDARY_SELECTION_SCHEMA,
                        task_deadline=task_deadline,
                        candidate_validator=verify_candidate_validator,
                    )
                    if verify_resp.get("found", False):
                        verify_start_idx = verify_resp.get("start_candidate_index")
                        verify_end_idx = verify_resp.get("end_candidate_index")
                        if isinstance(verify_start_idx, int) and isinstance(verify_end_idx, int):
                            if verify_start_idx in lookup and verify_end_idx in lookup:
                                new_start = lookup[verify_start_idx]
                                new_end = lookup[verify_end_idx]
                                start_shift = abs(new_start["ms"] - start_candidate["ms"])
                                end_shift = abs(new_end["ms"] - end_candidate["ms"])
                                if start_shift > MIN_BOUNDARY_SHIFT_MS:
                                    start_candidate = new_start
                                if end_shift > MIN_BOUNDARY_SHIFT_MS:
                                    end_candidate = new_end
                                confidence_score = float(verify_resp.get("confidence") or confidence_score)
                                confidence_label = confidence_score_to_label(confidence_score)

        start_cue_id = start_candidate["cue_id"]
        end_cue_id = max(start_cue_id, end_candidate["cue_id"] - 1)
        confidence = confidence_label
        reason = normalize_space(resp.get("reason") or "")

        if end_cue_id < start_cue_id:
            end_cue_id = start_cue_id
            confidence = downgrade_confidence(confidence)

        return {
            "start_cue_id": start_cue_id,
            "end_cue_id": end_cue_id,
            "confidence": confidence,
            "notes": reason,
            "selected_block_id": selected_block_id,
        }


def extract_lyrics_for_song(cues, song, start_cue_id, end_cue_id, task_deadline):
    """Extract lyric cue ids from a bounded song region.

    Args:
        cues: Full cue list.
        song: Target song dictionary.
        start_cue_id: Song start cue id.
        end_cue_id: Song end cue id.

    Returns:
        Tuple of (lyric_ids, confidence, reason_text).
    """
    start_idx = find_cue_index_by_id(cues, start_cue_id)
    end_idx = find_cue_index_by_id(cues, end_cue_id)
    if start_idx is None or end_idx is None or end_idx < start_idx:
        return [], "low", "Invalid cue bounds"

    lyric_ids = []
    seen_ids = set()
    confs = []
    reasons = []

    wstart = start_idx
    while wstart <= end_idx:
        window_state = {
            "window_size": LYRICS_WINDOW_CUES,
            "overlap": LYRICS_WINDOW_OVERLAP,
            "window_start": wstart,
        }

        def window_subset(state):
            window_size = int(state.get("window_size") or LYRICS_WINDOW_CUES)
            window_start = int(state.get("window_start") or wstart)
            window_end = min(end_idx + 1, window_start + window_size)
            return cues[window_start:window_end]

        def window_prompt_builder(state):
            subset = window_subset(state)
            return lyrics_window_prompt(song, start_cue_id, end_cue_id, subset)

        prompt = window_prompt_builder(window_state)
        prompt_tokens_est = estimate_prompt_tokens(SYSTEM + "\n" + prompt)
        while prompt_tokens_est > MAX_PROMPT_TOKENS and int(window_state["window_size"]) > 24:
            new_window_cues, new_overlap = shrink_lyrics_window(int(window_state["window_size"]), int(window_state["overlap"]))
            if new_window_cues == int(window_state["window_size"]) and new_overlap == int(window_state["overlap"]):
                break
            print(f"LLM auto-shrink task=lyrics_window window_cues {window_state['window_size']}->{new_window_cues} overlap {window_state['overlap']}->{new_overlap} prompt_tokens_est={prompt_tokens_est}")
            window_state["window_size"] = new_window_cues
            window_state["overlap"] = new_overlap
            prompt = window_prompt_builder(window_state)
            prompt_tokens_est = estimate_prompt_tokens(SYSTEM + "\n" + prompt)

        resp = llm_chat_json(
            "extract",
            "lyrics_window",
            SYSTEM,
            prompt,
            LYRICS_WINDOW_SCHEMA,
            task_deadline=task_deadline,
            candidate_validator=lambda r, state=window_state: _sanitize_lyrics_ids(r, window_subset(state), start_cue_id, end_cue_id),
            prompt_builder=window_prompt_builder,
            call_state=window_state,
        )

        ids = resp.get("lyrics_cue_ids") if isinstance(resp, dict) else []
        if isinstance(ids, list):
            for x in ids:
                if isinstance(x, int) and x not in seen_ids:
                    seen_ids.add(x)
                    lyric_ids.append(x)

        confs.append((resp.get("confidence") or "low").lower() if isinstance(resp, dict) else "low")
        rr = normalize_space(resp.get("reason") or "") if isinstance(resp, dict) else ""
        if rr:
            reasons.append(rr)

        current_subset = window_subset(window_state)
        if current_subset and current_subset[-1]["cue_id"] >= end_idx:
            break
        wstart += max(1, int(window_state["window_size"]) - int(window_state["overlap"]))

    lyric_ids = sorted(set(lyric_ids))

    # Fallback: if no lyric cues were identified, use the bounded span
    if not lyric_ids:
        lyric_ids = [c["cue_id"] for c in cues[start_idx:end_idx + 1] if c["text"]]

    overall = "low"
    if "high" in confs:
        overall = "high"
    elif "medium" in confs:
        overall = "medium"

    return lyric_ids, overall, "; ".join(reasons[:3])

def detect_songs(cues, blocks, songs, workdir):
    """Detect all songs in order and persist incremental progress.

    Args:
        cues: Parsed subtitle cues.
        blocks: Audio-backed cue blocks.
        songs: Ordered song list from flyer.
        workdir: Working directory for progress and outputs.

    Returns:
        Postprocessed song result list.
    """
    progress_path = os.path.join(workdir, "enhanced_progress.json")
    results_path = os.path.join(workdir, "songs.json")
    progress_default = {
        "phase": "songs",
        "next_song_index": 1,
        "search_cue_id": 1,
        "results": []
    }
    progress = load_json(progress_path, progress_default) if RESUME else progress_default

    results = progress.get("results", [])
    next_song_index = int(progress.get("next_song_index", 1))
    search_cue_id = int(progress.get("search_cue_id", 1))

    c_map = cue_map(cues)
    boundary_candidates = build_boundary_candidates(cues, blocks)

    for si in range(next_song_index, len(songs) + 1):
        song = songs[si - 1]
        prev_song = songs[si - 2] if si > 1 else None
        next_song = songs[si] if si < len(songs) else None
        song_deadline = task_deadline()

        print(f"Detecting song {si}/{len(songs)}: {song['title']}")

        start_block_idx = find_block_index_for_cue(blocks, search_cue_id)
        found = None
        block_search_conf = "low"
        block_search_reason = ""

        look_idx = start_block_idx
        while look_idx < len(blocks):
            search_state = {"window_size": SEARCH_BLOCKS}

            def search_candidates(state):
                return choose_candidate_blocks(blocks, look_idx, int(state.get("window_size") or SEARCH_BLOCKS))

            candidate_blocks = search_candidates(search_state)
            if not candidate_blocks:
                break

            def search_prompt_builder(state):
                blocks_subset = search_candidates(state)
                return block_search_prompt(
                    song,
                    prev_song,
                    songs[si:si + 3],
                    search_cue_id,
                    blocks_subset,
                    songs,
                )

            prompt = search_prompt_builder(search_state)
            prompt_tokens_est = estimate_prompt_tokens(SYSTEM + "\n" + prompt)
            while prompt_tokens_est > MAX_PROMPT_TOKENS and int(search_state["window_size"]) > 1:
                new_search_blocks = shrink_search_blocks(int(search_state["window_size"]))
                if new_search_blocks == int(search_state["window_size"]):
                    break
                print(f"LLM auto-shrink task=song_search search_blocks {search_state['window_size']}->{new_search_blocks} prompt_tokens_est={prompt_tokens_est}")
                search_state["window_size"] = new_search_blocks
                candidate_blocks = search_candidates(search_state)
                if not candidate_blocks:
                    break
                prompt = search_prompt_builder(search_state)
                prompt_tokens_est = estimate_prompt_tokens(SYSTEM + "\n" + prompt)

            if not candidate_blocks:
                break

            def search_validator(resp):
                selected = resp.get("selected_block_id")
                if selected == 0:
                    return
                current_lookup = {b["block_id"]: b for b in search_candidates(search_state)}
                if selected not in current_lookup:
                    raise LLMResponseValidationError("selected_block_id must come from the supplied candidate blocks")

            search_resp = llm_chat_json(
                "detect",
                "song_search",
                SYSTEM,
                prompt,
                BLOCK_SEARCH_SCHEMA,
                task_deadline=song_deadline,
                candidate_validator=search_validator,
                prompt_builder=search_prompt_builder,
                call_state=search_state,
            )

            selected_block_id = search_resp.get("selected_block_id") if isinstance(search_resp.get("selected_block_id"), int) else 0
            block_search_conf = (search_resp.get("confidence") or "low").lower()
            block_search_reason = normalize_space(search_resp.get("reason") or "")

            current_lookup = {b["block_id"]: b for b in search_candidates(search_state)}
            if selected_block_id and selected_block_id in current_lookup:
                found = refine_song_in_blocks(cues, blocks, boundary_candidates, song, prev_song, next_song, selected_block_id, search_cue_id, song_deadline)
                if found:
                    break

            look_idx += SEARCH_STRIDE

        # Use a deterministic fallback so processing continues on weak cues.
        if not found:
            # Fallback: choose the best-looking nearby block, low confidence
            nearby = choose_candidate_blocks(blocks, start_block_idx, SEARCH_BLOCKS)
            if nearby:
                fallback = sorted(nearby, key=lambda b: (-b["song_like_score"], b["block_id"]))[0]
                found = {
                    "start_cue_id": fallback["start_cue_id"],
                    "end_cue_id": fallback["end_cue_id"],
                    "confidence": "low",
                    "notes": "Fallback from audio-backed block ranking",
                    "selected_block_id": fallback["block_id"],
                }
            else:
                found = {
                    "start_cue_id": search_cue_id,
                    "end_cue_id": search_cue_id,
                    "confidence": "low",
                    "notes": "Fallback: no candidate block found",
                    "selected_block_id": 0,
                }

        lyric_ids, lyric_conf, lyric_reason = extract_lyrics_for_song(
                cues, song, found["start_cue_id"], found["end_cue_id"], song_deadline
        )

        start_cue = c_map.get(found["start_cue_id"])
        end_cue = c_map.get(found["end_cue_id"])

        combined_conf = "low"
        if "high" in [found["confidence"], block_search_conf, lyric_conf]:
            combined_conf = "high"
        elif "medium" in [found["confidence"], block_search_conf, lyric_conf]:
            combined_conf = "medium"

        lyrics_text = "\n".join([c_map[cid]["text"] for cid in lyric_ids if cid in c_map]).strip()

        result = {
            "index": song["index"],
            "act": song["act"],
            "song_title": song["title"],
            "performers": song["performers"],
            "start_cue_id": found["start_cue_id"],
            "end_cue_id": found["end_cue_id"],
            "start_time": ms_to_srt(start_cue["start_ms"]) if start_cue else None,
            "end_time": ms_to_srt(end_cue["end_ms"]) if end_cue else None,
            "duration": duration_clock(start_cue["start_ms"], end_cue["end_ms"]) if start_cue and end_cue else None,
            "confidence": combined_conf,
            "selected_block_id": found["selected_block_id"],
            "lyrics_cue_ids": lyric_ids,
            "lyrics": lyrics_text,
            "notes": "; ".join([x for x in [block_search_reason, found["notes"], lyric_reason] if x]),
        }

        results.append(result)
        search_cue_id = max(search_cue_id, (found["end_cue_id"] or search_cue_id)) + 1

        save_json(results_path, results)
        save_json(progress_path, {
            "phase": "songs",
            "next_song_index": si + 1,
            "search_cue_id": search_cue_id,
            "results": results
        })

    results = postprocess_results(results, cues)

    save_json(results_path, results)
    save_json(progress_path, {
        "phase": "songs_done",
        "next_song_index": len(songs) + 1,
        "search_cue_id": search_cue_id,
        "results": results
    })

    return results

def postprocess_results(results, cues):
    """Repair ordering and bounds, then refresh derived song fields.

    Args:
        results: Raw detection result dictionaries.
        cues: Full cue list.

    Returns:
        Cleaned and order-consistent result dictionaries.
    """
    c_map = cue_map(cues)
    fixed = []
    prev_end = 0

    for r in sorted(results, key=lambda x: x["index"]):
        start_id = r.get("start_cue_id")
        end_id = r.get("end_cue_id")
        conf = r.get("confidence", "low")

        if not isinstance(start_id, int):
            start_id = prev_end + 1 if (prev_end + 1) in c_map else prev_end
            conf = "low"
        if not isinstance(end_id, int):
            end_id = start_id
            conf = "low"

        if start_id <= prev_end:
            start_id= prev_end + 1 if (prev_end + 1) in c_map else start_id
            conf = downgrade_confidence(conf)

        if end_id < start_id:
            end_id = start_id
            conf = downgrade_confidence(conf)

        lyric_ids = []
        seen_lyric_ids = set()
        for cid in r.get("lyrics_cue_ids", []):
            if isinstance(cid, int) and start_id <= cid <= end_id and cid in c_map and cid not in seen_lyric_ids:
                seen_lyric_ids.add(cid)
                lyric_ids.append(cid)
        if not lyric_ids:
            lyric_ids = [cid for cid in range(start_id, end_id + 1) if cid in c_map and c_map[cid]["text"]]

        start_cue = c_map.get(start_id)
        end_cue = c_map.get(end_id)

        fixed.append({
            **r,
            "start_cue_id": start_id,
            "end_cue_id": end_id,
            "start_time": ms_to_srt(start_cue["start_ms"]) if start_cue else None,
            "end_time": ms_to_srt(end_cue["end_ms"]) if end_cue else None,
            "duration": duration_clock(start_cue["start_ms"], end_cue["end_ms"]) if start_cue and end_cue else None,
            "confidence": conf,
            "lyrics_cue_ids": lyric_ids,
            "lyrics": "\n".join([c_map[cid]["text"] for cid in lyric_ids if cid in c_map]).strip(),
        })
        prev_end = end_id

    return fixed

# =========================
# Output writers
# =========================
def escape_pipes(s):
    """Escape pipe characters for Markdown table cells.

    Args:
        s: Input text.

    Returns:
        Text with literal pipes escaped.
    """
    return (s or "").replace("|", "\\|")

def write_blocks_json(blocks, path):
    """Write block summaries to JSON.

    Args:
        blocks: Block dictionaries to serialize.
        path: Output file path.
    """
    save_json(path, blocks)

def write_review_md(results, out_path):
    """Write a Markdown timing review table for detected songs.

    Args:
        results: Song detection result dictionaries.
        out_path: Markdown output path.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Song Timing Review\n\n")
        f.write("| # | Act | Song | Start | End | Duration | Confidence | Notes |\n")
        f.write("|---:|---:|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['index']} | {r['act']} | {escape_pipes(r['song_title'])} | "
                f"{r['start_time'] or ''} | {r['end_time'] or ''} | {r['duration'] or ''} | "
                f"{r['confidence']} | {escape_pipes(r.get('notes', '') or '')} |\n"
            )

def write_lyrics_md(results, out_path):
    """Write grouped lyric text by act and song into Markdown.

    Args:
        results: Song detection result dictionaries.
        out_path: Markdown output path.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Lyrics by Song\n\n")
        current_act = None
        for r in results:
            if r["act"] != current_act:
                current_act = r["act"]
                f.write(f"## Act {current_act}\n\n")
            f.write(f"### {r['index']}. {r['song_title']}\n\n")
            f.write(f"- **Performers:** {r['performers'] or 'Unknown'}\n")
            f.write(f"- **Start:** {r['start_time'] or 'Unknown'}\n")
            f.write(f"- **End:** {r['end_time'] or 'Unknown'}\n")
            f.write(f"- **Confidence:** {r['confidence']}\n\n")
            lyrics_lines = []
            seen = set()
            for line in (r.get("lyrics") or "").splitlines():
                normalized = line.rstrip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    lyrics_lines.append(normalized)
            if lyrics_lines:
                f.write("\n".join(lyrics_lines).strip() + "\n\n")
            else:
                f.write("[No lyrics extracted]\n\n")

# =========================
# Story summary
# =========================
def summarize_story(results, flyer_plot_summary, workdir):
    """Generate chapter summaries and an overall story synthesis.

    Args:
        results: Detected song records.
        flyer_plot_summary: Global plot summary from flyer text.
        workdir: Working directory for progress and outputs.
    """
    progress_path = os.path.join(workdir, "song_summary_progress.json")
    out_json = os.path.join(workdir, "song_story_summary.json")
    out_md = os.path.join(workdir, "song_story_summary.md")

    progress_default = {
        "next_song_index": 1,
        "chapters": [],
        "overall": {}
    }
    progress = load_json(progress_path, progress_default) if RESUME else progress_default

    chapters = progress.get("chapters", [])
    next_song_index = int(progress.get("next_song_index", 1))
    narrative_temperature = max(0.2, TEMPERATURE)

    def chapter_context(existing_chapters):
        context = []
        for ch in existing_chapters:
            context.append({
                "index": ch.get("index"),
                "title": ch.get("title") or ch.get("chapter_title") or ch.get("song_title"),
                "summary": one_sentence_summary(ch.get("summary", "")),
            })
        return context

    llm_chat_json(
        "summary",
        "summary_warmup",
        SUMMARY_SYSTEM,
        "Warm up the summary model with one token.",
        None,
        task_deadline=task_deadline(),
        num_predict_override=1,
        expect_json=False,
    )

    for i in range(next_song_index, len(results) + 1):
        song_record = results[i - 1]
        print(f"Summarizing chapter {i}/{len(results)}: {song_record['song_title']}")
        resp = llm_chat_json(
            "summary",
            "chapter_summary",
            SUMMARY_SYSTEM,
            chapter_prompt(flyer_plot_summary, chapter_context(chapters), song_record),
            CHAPTER_SCHEMA,
            task_deadline=task_deadline(),
        )
        resp.setdefault("chapter_title", resp.get("title", ""))
        resp.setdefault("key_characters", resp.get("characters", []))
        resp.setdefault("themes", resp.get("themes", []))
        chapters.append(resp)
        save_json(progress_path, {
            "next_song_index": i + 1,
            "chapters": chapters,
            "overall": progress.get("overall", {})
        })

    print("Creating overall story summary...")
    overall_context = [
        {
            "index": ch.get("index"),
            "title": ch.get("title") or ch.get("chapter_title") or ch.get("song_title"),
            "summary": ch.get("summary", ""),
        }
        for ch in chapters
    ]
    overall = llm_chat_json(
        "summary",
        "story_assembly",
        SUMMARY_SYSTEM,
        final_assembly_prompt(overall_context, flyer_plot_summary),
        FINAL_SUMMARY_SCHEMA,
        task_deadline=task_deadline(),
    )

    save_json(progress_path, {
        "next_song_index": len(results) + 1,
        "chapters": chapters,
        "overall": overall
    })

    write_story_md(chapters, overall, out_md)
    save_json(out_json, {
        "overall_summary": overall.get("overall_summary", ""),
        "act_summaries": overall.get("act_summaries", []),
        "chapters": chapters
    })

def write_story_md(chapters, overall, out_path):
    """Write chapter and overall story summaries to Markdown.

    Args:
        chapters: Per-song chapter summary dictionaries.
        overall: Overall and per-act summary dictionary.
        out_path: Markdown output path.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Song-as-Chapter Story Summary\n\n")

        if overall.get("overall_summary"):
            f.write("## Overall Story\n\n")
            f.write(overall["overall_summary"].strip() + "\n\n")

        act_summaries = overall.get("act_summaries") or []
        if act_summaries:
            f.write("## Act Summaries\n\n")
            for a in act_summaries:
                f.write(f"### Act {a.get('act')}\n\n")
                f.write((a.get("summary") or "").strip() + "\n\n")

        current_act = None
        for ch in chapters:
            if ch["act"] != current_act:
                current_act = ch["act"]
                f.write(f"## Act {current_act}\n\n")

            chapter_title = ch.get("title") or ch.get("chapter_title") or ""
            characters = ch.get("characters") or ch.get("key_characters") or []
            themes = ch.get("themes") or []

            f.write(f"### {ch['index']}. {ch['song_title']}\n\n")
            f.write(f"- **Chapter title:** {chapter_title}\n")
            f.write(f"- **Story role:** {ch.get('story_role', '')}\n")
            f.write(f"- **Confidence:** {ch.get('confidence', '')}\n")
            f.write(f"- **Characters:** {', '.join(characters)}\n")
            if themes:
                f.write(f"- **Themes:** {', '.join(themes)}\n")
            f.write("\n")
            f.write((ch.get("summary") or "").strip() + "\n\n")

            events = ch.get("key_events") or []
            if events:
                f.write("**Key events**\n\n")
                for e in events:
                    f.write(f"- {e}\n")
                f.write("\n")

# =========================
# Main
# =========================
def main():
    """Run the full subtitle-to-song-to-story processing pipeline.

    Raises:
        SystemExit: If required configuration or parsed inputs are missing.
    """
    if not BASE_URL:
        raise SystemExit("Set BASE_URL, e.g. http://<IP>:4000/v1")

    workdir = os.getenv("WORKDIR", "/work")
    media_path = os.getenv("INPUT_MEDIA", os.path.join(workdir, "input.mp4"))
    srt_path = os.getenv("INPUT_SRT", os.path.join(workdir, "input.srt"))
    flyer_path = os.getenv("INPUT_FLYER", os.path.join(workdir, "flyer.txt"))
    wav_path = os.path.join(workdir, "_audio.wav")

    blocks_json_path = os.path.join(workdir, "blocks.json")
    songs_json_path = os.path.join(workdir, "songs.json")
    review_md_path = os.path.join(workdir, "songs_review.md")
    lyrics_md_path = os.path.join(workdir, "lyrics_by_song.md")

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        srt_text = f.read()
    with open(flyer_path, "r", encoding="utf-8", errors="ignore") as f:
        flyer_text = f.read()

    cues = parse_srt(srt_text)
    songs = parse_flyer_songs(flyer_text)
    flyer_plot_summary = parse_flyer_plot_summary(flyer_text)

    if not cues:
        raise SystemExit("No subtitle cues parsed from input.srt")
    if not songs:
        raise SystemExit("No songs parsed from flyer.txt")

    print(f"Parsed {len(cues)} subtitle cues")
    print(f"Parsed {len(songs)} flyer songs")

    ensure_audio_wav(media_path, wav_path)
    y, sr = load_wav_mono(wav_path)
    print(f"Loaded audio: {len(y)/sr/60:.1f} minutes at {sr} Hz")

    audio_model = build_audio_model(y, sr)
    span_stats = attach_span_stats(audio_model)

    attach_cue_audio_features(cues, span_stats)
    blocks = build_blocks(cues, span_stats)
    write_blocks_json(blocks, blocks_json_path)
    print(f"Built {len(blocks)} audio-backed cue blocks")

    results = detect_songs(cues, blocks, songs, workdir)
    save_json(songs_json_path, results)
    write_review_md(results, review_md_path)
    write_lyrics_md(results, lyrics_md_path)

    summarize_story(results, flyer_plot_summary, workdir)

    print(f"Wrote: {blocks_json_path}")
    print(f"Wrote: {songs_json_path}")
    print(f"Wrote: {review_md_path}")
    print(f"Wrote: {lyrics_md_path}")
    print(f"Wrote: {os.path.join(workdir, 'song_story_summary.json')}")
    print(f"Wrote: {os.path.join(workdir, 'song_story_summary.md')}")

if __name__ == "__main__":
    main()
