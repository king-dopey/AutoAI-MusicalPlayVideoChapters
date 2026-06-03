"""SRT parsing helpers for subtitle cue extraction."""

from __future__ import annotations

import re


def _normalize_space(text: str | None) -> str:
    """Collapse consecutive whitespace and trim leading/trailing space."""
    return re.sub(r"\s+", " ", text or "").strip()


def ts_to_ms(ts: str) -> int:
    """Convert an SRT timestamp string to milliseconds.

    Args:
        ts: Timestamp in HH:MM:SS,mmm format.

    Returns:
        Timestamp in milliseconds.
    """
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return ((int(h) * 3600 + int(m) * 60 + int(s)) * 1000) + int(ms)


def parse_srt(srt_text: str) -> list[dict]:
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
            match = ts_re.search(ln)
            if match:
                ts_idx = i
                ts_match = match
                break
        if ts_idx is None or ts_match is None:
            continue

        text_lines = lines[ts_idx + 1 :]
        text = " ".join(re.sub(r"</?i>|</?b>|</?u>|<[^>]+>", "", ln).strip() for ln in text_lines).strip()

        cues.append(
            {
                "cue_id": cue_id,
                "start_ms": ts_to_ms(ts_match.group("start")),
                "end_ms": ts_to_ms(ts_match.group("end")),
                "start": ts_match.group("start"),
                "end": ts_match.group("end"),
                "text": _normalize_space(text),
            }
        )
        cue_id += 1

    return cues
