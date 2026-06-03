"""Flyer parsing helpers for song order and plot summary extraction."""

from __future__ import annotations

import re


def _normalize_space(text: str | None) -> str:
    """Collapse consecutive whitespace and trim leading/trailing space."""
    return re.sub(r"\s+", " ", text or "").strip()


def parse_flyer_songs(flyer_text: str) -> list[dict]:
    """Extract ordered song metadata from flyer text.

    Args:
        flyer_text: Full flyer content.

    Returns:
        Song dictionaries with index, act, title, and performers.
    """
    lines = [line.strip() for line in flyer_text.splitlines()]
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
            songs.append(
                {
                    "index": len(songs) + 1,
                    "act": current_act,
                    "title": _normalize_space(title),
                    "performers": _normalize_space(performers),
                }
            )

    return songs


def parse_flyer_plot_summary(flyer_text: str) -> str:
    """Extract and normalize the plot-summary section from flyer text.

    Args:
        flyer_text: Full flyer content.

    Returns:
        Flattened plot summary text, or an empty string.
    """
    match = re.search(
        r"INTO THE WOODS PLOT SUMMARY\s*(.*?)(?:## Page 4|INTO THE WOODS DIRECTOR|INTO THE WOODS: SONG BREAKDOWN)",
        flyer_text,
        re.S | re.I,
    )
    if match:
        return _normalize_space(match.group(1))
    return ""
