"""Environment-backed runtime settings for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os


TRUE_VALUE = "1"


def _env_bool(name: str, default: str) -> bool:
    """Read a boolean env var using the existing 1/0 convention."""
    return os.getenv(name, default) == TRUE_VALUE


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    base_url: str
    model: str
    model_detect: str
    model_extract: str
    model_summary: str
    model_verify: str
    api_key: str
    kv_cache_type: str

    temperature: float
    max_retries: int
    base_sleep: float
    max_sleep: float
    err_body_chars: int
    request_timeout: float
    task_timeout: float
    max_prompt_tokens: int
    min_boundary_shift_ms: int
    max_concurrency: int
    resume: bool
    enable_verifier: bool
    log_raw_empty: bool
    empty_repair_max_steps: int
    validation_repair_max_steps: int
    boundary_refine_fallback_window: int
    strict_boundaries: bool
    num_predict_hard_cap: int
    num_ctx_hard_cap: int
    empty_repair_safety_margin: int
    think_detect: bool
    think_extract: bool
    think_summary: bool
    think_verify: bool

    num_ctx_detect: int
    num_ctx_extract: int
    num_ctx_summary: int
    num_ctx_verify: int
    num_predict_detect: int
    num_predict_extract: int
    num_predict_summary: int
    num_predict_verify: int

    audio_sr: int
    frame_sec: float
    frame_hop_sec: float
    short_gap_ms: int
    long_gap_ms: int
    bed_min_ratio: float
    bed_min_rms_n: float
    search_blocks: int
    search_stride: int
    boundary_context_blocks: int
    lyrics_window_cues: int
    lyrics_window_overlap: int


def load_settings() -> Settings:
    """Load settings from environment with legacy defaults and coercions."""
    model = os.getenv("MODEL", "")
    return Settings(
        base_url=os.getenv("BASE_URL", "").rstrip("/"),
        model=model,
        model_detect=os.getenv("MODEL_DETECT", model or "qwen3-coder:30b"),
        model_extract=os.getenv("MODEL_EXTRACT", model or "qwen3-coder:30b"),
        model_summary=os.getenv("MODEL_SUMMARY", model or "qwen3.6:35b-a3b"),
        model_verify=os.getenv("MODEL_VERIFY", model or "nemotron-cascade-2:30b"),
        api_key=os.getenv("API_KEY", ""),
        kv_cache_type=os.getenv("KV_CACHE_TYPE", "q8_0"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        max_retries=int(os.getenv("MAX_RETRIES", "8")),
        base_sleep=float(os.getenv("BASE_SLEEP", "2.0")),
        max_sleep=float(os.getenv("MAX_SLEEP", "45.0")),
        err_body_chars=int(os.getenv("ERR_BODY_CHARS", "6000")),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT", "180")),
        task_timeout=float(os.getenv("TASK_TIMEOUT", "1800")),
        max_prompt_tokens=int(os.getenv("MAX_PROMPT_TOKENS", "12000")),
        min_boundary_shift_ms=int(os.getenv("MIN_BOUNDARY_SHIFT_MS", "1500")),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", "1")),
        resume=_env_bool("RESUME", "1"),
        enable_verifier=_env_bool("ENABLE_VERIFIER", "0"),
        log_raw_empty=_env_bool("LOG_RAW_EMPTY", "0"),
        empty_repair_max_steps=max(0, int(os.getenv("EMPTY_REPAIR_MAX_STEPS", "3"))),
        validation_repair_max_steps=max(0, int(os.getenv("VALIDATION_REPAIR_MAX_STEPS", "3"))),
        boundary_refine_fallback_window=max(1, int(os.getenv("BOUNDARY_REFINE_FALLBACK_WINDOW", "5"))),
        strict_boundaries=_env_bool("STRICT_BOUNDARIES", "0"),
        num_predict_hard_cap=max(1, int(os.getenv("NUM_PREDICT_HARD_CAP", "16384"))),
        num_ctx_hard_cap=max(1, int(os.getenv("NUM_CTX_HARD_CAP", "32768"))),
        empty_repair_safety_margin=256,
        think_detect=_env_bool("THINK_DETECT", "0"),
        think_extract=_env_bool("THINK_EXTRACT", "0"),
        think_summary=_env_bool("THINK_SUMMARY", "1"),
        think_verify=_env_bool("THINK_VERIFY", "1"),
        num_ctx_detect=int(os.getenv("NUM_CTX_DETECT", "16384")),
        num_ctx_extract=int(os.getenv("NUM_CTX_EXTRACT", "16384")),
        num_ctx_summary=int(os.getenv("NUM_CTX_SUMMARY", "32768")),
        num_ctx_verify=int(os.getenv("NUM_CTX_VERIFY", "16384")),
        num_predict_detect=int(os.getenv("NUM_PREDICT_DETECT", "512")),
        num_predict_extract=int(os.getenv("NUM_PREDICT_EXTRACT", "2048")),
        num_predict_summary=int(os.getenv("NUM_PREDICT_SUMMARY", "8192")),
        num_predict_verify=int(os.getenv("NUM_PREDICT_VERIFY", "4096")),
        audio_sr=int(os.getenv("AUDIO_SR", "16000")),
        frame_sec=float(os.getenv("FRAME_SEC", "1.0")),
        frame_hop_sec=float(os.getenv("FRAME_HOP_SEC", "0.25")),
        short_gap_ms=int(os.getenv("SHORT_GAP_MS", "1600")),
        long_gap_ms=int(os.getenv("LONG_GAP_MS", "5000")),
        bed_min_ratio=float(os.getenv("BED_MIN_RATIO", "0.55")),
        bed_min_rms_n=float(os.getenv("BED_MIN_RMS_N", "0.20")),
        search_blocks=int(os.getenv("SEARCH_BLOCKS", "6")),
        search_stride=int(os.getenv("SEARCH_STRIDE", "3")),
        boundary_context_blocks=int(os.getenv("BOUNDARY_CONTEXT_BLOCKS", "1")),
        lyrics_window_cues=int(os.getenv("LYRICS_WINDOW_CUES", "90")),
        lyrics_window_overlap=int(os.getenv("LYRICS_WINDOW_OVERLAP", "15")),
    )
