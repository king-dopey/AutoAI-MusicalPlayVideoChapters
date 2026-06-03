"""Lyric extraction pipeline helpers."""

from __future__ import annotations

import json
import re

from autoai_musical_play_video_chapters.config import load_settings
from autoai_musical_play_video_chapters.llm.schemas import LYRICS_WINDOW_SCHEMA
from autoai_musical_play_video_chapters.pipeline.detect import cue_rows_for_prompt, find_cue_index_by_id

_SETTINGS = load_settings()

LYRICS_WINDOW_CUES = _SETTINGS.lyrics_window_cues
LYRICS_WINDOW_OVERLAP = _SETTINGS.lyrics_window_overlap
MAX_PROMPT_TOKENS = _SETTINGS.max_prompt_tokens


def _normalize_space(text):
    """Collapse consecutive whitespace and trim leading/trailing space."""
    return re.sub(r"\s+", " ", text or "").strip()


def lyrics_window_prompt(target_song, song_start_cue_id, song_end_cue_id, cues_subset):
    """Build the prompt for selecting lyric-only cues in a window."""
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


def shrink_lyrics_window(window_cues, overlap):
    """Reduce lyric window size and expand overlap proportionally."""
    reduced_window = max(24, int(window_cues * 0.75))
    if reduced_window >= window_cues and window_cues > 24:
        reduced_window = window_cues - 1
    delta = max(1, window_cues - reduced_window)
    reduced_overlap = min(reduced_window - 1, max(overlap + delta // 2, overlap))
    return reduced_window, reduced_overlap


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


def extract_lyrics_for_song(
    cues,
    song,
    start_cue_id,
    end_cue_id,
    task_deadline,
    *,
    estimate_prompt_tokens,
    system_prompt,
    llm_chat_json,
):
    """Extract lyric cue ids from a bounded song region."""
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
        prompt_tokens_est = estimate_prompt_tokens(system_prompt + "\n" + prompt)
        while prompt_tokens_est > MAX_PROMPT_TOKENS and int(window_state["window_size"]) > 24:
            new_window_cues, new_overlap = shrink_lyrics_window(
                int(window_state["window_size"]),
                int(window_state["overlap"]),
            )
            if new_window_cues == int(window_state["window_size"]) and new_overlap == int(window_state["overlap"]):
                break
            print(
                f"LLM auto-shrink task=lyrics_window window_cues {window_state['window_size']}->{new_window_cues} overlap {window_state['overlap']}->{new_overlap} prompt_tokens_est={prompt_tokens_est}"
            )
            window_state["window_size"] = new_window_cues
            window_state["overlap"] = new_overlap
            prompt = window_prompt_builder(window_state)
            prompt_tokens_est = estimate_prompt_tokens(system_prompt + "\n" + prompt)

        resp = llm_chat_json(
            "extract",
            "lyrics_window",
            system_prompt,
            prompt,
            LYRICS_WINDOW_SCHEMA,
            task_deadline=task_deadline,
            candidate_validator=lambda r, state=window_state: _sanitize_lyrics_ids(
                r,
                window_subset(state),
                start_cue_id,
                end_cue_id,
            ),
            prompt_builder=window_prompt_builder,
            call_state=window_state,
        )

        ids = resp.get("lyrics_cue_ids") if isinstance(resp, dict) else []
        if isinstance(ids, list):
            for cue_id in ids:
                if isinstance(cue_id, int) and cue_id not in seen_ids:
                    seen_ids.add(cue_id)
                    lyric_ids.append(cue_id)

        confs.append((resp.get("confidence") or "low").lower() if isinstance(resp, dict) else "low")
        reason = _normalize_space(resp.get("reason") or "") if isinstance(resp, dict) else ""
        if reason:
            reasons.append(reason)

        current_subset = window_subset(window_state)
        if current_subset and current_subset[-1]["cue_id"] >= end_idx:
            break
        wstart += max(1, int(window_state["window_size"]) - int(window_state["overlap"]))

    lyric_ids = sorted(set(lyric_ids))

    if not lyric_ids:
        lyric_ids = [cue["cue_id"] for cue in cues[start_idx:end_idx + 1] if cue["text"]]

    overall = "low"
    if "high" in confs:
        overall = "high"
    elif "medium" in confs:
        overall = "medium"

    return lyric_ids, overall, "; ".join(reasons[:3])