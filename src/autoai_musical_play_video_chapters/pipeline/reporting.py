"""Reporting and output writers for pipeline artifacts.

This module owns Markdown/JSON artifact generation and story-summary output
assembly that was previously embedded in convert.py.
"""

from __future__ import annotations

import json
import os
import re

from autoai_musical_play_video_chapters.config import load_settings
from autoai_musical_play_video_chapters.llm.client import llm_chat_json, task_deadline
from autoai_musical_play_video_chapters.llm.schemas import CHAPTER_SCHEMA, FINAL_SUMMARY_SCHEMA

_SETTINGS = load_settings()

RESUME = _SETTINGS.resume
TEMPERATURE = _SETTINGS.temperature

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


def _load_json(path: str, default: object) -> object:
    """Load JSON from disk and return fallback on failure."""
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception:
        return default


def _save_json(path: str, obj: object) -> None:
    """Write an object to disk as UTF-8 formatted JSON."""
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(obj, file_handle, ensure_ascii=False, indent=2)


def _normalize_space(text: str | None) -> str:
    """Collapse consecutive whitespace and trim leading/trailing space."""
    return re.sub(r"\s+", " ", text or "").strip()


def _one_sentence_summary(text: str) -> str:
    """Condense a summary to its first sentence when possible."""
    cleaned = _normalize_space(text)
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return parts[0].strip()


def _chapter_prompt(flyer_plot_summary: str, prior_chapters: list[dict], song_record: dict) -> str:
    """Build the prompt for one song-as-chapter story summary."""
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


def _final_assembly_prompt(chapters: list[dict], flyer_plot_summary: str) -> str:
    """Build the prompt for overall and per-act story synthesis."""
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


def _escape_pipes(text: str | None) -> str:
    """Escape pipe characters for Markdown table cells."""
    return (text or "").replace("|", "\\|")


def write_blocks_json(blocks: list[dict], path: str) -> None:
    """Write block summaries to JSON."""
    _save_json(path, blocks)


def write_review_md(results: list[dict], out_path: str) -> None:
    """Write a Markdown timing review table for detected songs."""
    with open(out_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("# Song Timing Review\n\n")
        file_handle.write("| # | Act | Song | Start | End | Duration | Confidence | Notes |\n")
        file_handle.write("|---:|---:|---|---|---|---|---|---|\n")
        for result in results:
            file_handle.write(
                f"| {result['index']} | {result['act']} | {_escape_pipes(result['song_title'])} | "
                f"{result['start_time'] or ''} | {result['end_time'] or ''} | {result['duration'] or ''} | "
                f"{result['confidence']} | {_escape_pipes(result.get('notes', '') or '')} |\n"
            )


def write_lyrics_md(results: list[dict], out_path: str) -> None:
    """Write grouped lyric text by act and song into Markdown."""
    with open(out_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("# Lyrics by Song\n\n")
        current_act = None
        for result in results:
            if result["act"] != current_act:
                current_act = result["act"]
                file_handle.write(f"## Act {current_act}\n\n")
            file_handle.write(f"### {result['index']}. {result['song_title']}\n\n")
            file_handle.write(f"- **Performers:** {result['performers'] or 'Unknown'}\n")
            file_handle.write(f"- **Start:** {result['start_time'] or 'Unknown'}\n")
            file_handle.write(f"- **End:** {result['end_time'] or 'Unknown'}\n")
            file_handle.write(f"- **Confidence:** {result['confidence']}\n\n")
            lyrics_lines = []
            seen = set()
            for line in (result.get("lyrics") or "").splitlines():
                normalized = line.rstrip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    lyrics_lines.append(normalized)
            if lyrics_lines:
                file_handle.write("\n".join(lyrics_lines).strip() + "\n\n")
            else:
                file_handle.write("[No lyrics extracted]\n\n")


def summarize_story(results: list[dict], flyer_plot_summary: str, workdir: str) -> None:
    """Generate chapter summaries and an overall story synthesis."""
    progress_path = os.path.join(workdir, "song_summary_progress.json")
    out_json = os.path.join(workdir, "song_story_summary.json")
    out_md = os.path.join(workdir, "song_story_summary.md")

    progress_default = {
        "next_song_index": 1,
        "chapters": [],
        "overall": {},
    }
    progress = _load_json(progress_path, progress_default) if RESUME else progress_default

    chapters = progress.get("chapters", [])
    next_song_index = int(progress.get("next_song_index", 1))
    narrative_temperature = max(0.2, TEMPERATURE)
    _ = narrative_temperature

    def chapter_context(existing_chapters: list[dict]) -> list[dict]:
        context = []
        for chapter in existing_chapters:
            context.append(
                {
                    "index": chapter.get("index"),
                    "title": chapter.get("title") or chapter.get("chapter_title") or chapter.get("song_title"),
                    "summary": _one_sentence_summary(chapter.get("summary", "")),
                }
            )
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

    for index in range(next_song_index, len(results) + 1):
        song_record = results[index - 1]
        print(f"Summarizing chapter {index}/{len(results)}: {song_record['song_title']}")
        response = llm_chat_json(
            "summary",
            "chapter_summary",
            SUMMARY_SYSTEM,
            _chapter_prompt(flyer_plot_summary, chapter_context(chapters), song_record),
            CHAPTER_SCHEMA,
            task_deadline=task_deadline(),
        )
        response.setdefault("chapter_title", response.get("title", ""))
        response.setdefault("key_characters", response.get("characters", []))
        response.setdefault("themes", response.get("themes", []))
        chapters.append(response)
        _save_json(
            progress_path,
            {
                "next_song_index": index + 1,
                "chapters": chapters,
                "overall": progress.get("overall", {}),
            },
        )

    print("Creating overall story summary...")
    overall_context = [
        {
            "index": chapter.get("index"),
            "title": chapter.get("title") or chapter.get("chapter_title") or chapter.get("song_title"),
            "summary": chapter.get("summary", ""),
        }
        for chapter in chapters
    ]
    overall = llm_chat_json(
        "summary",
        "story_assembly",
        SUMMARY_SYSTEM,
        _final_assembly_prompt(overall_context, flyer_plot_summary),
        FINAL_SUMMARY_SCHEMA,
        task_deadline=task_deadline(),
    )

    _save_json(
        progress_path,
        {
            "next_song_index": len(results) + 1,
            "chapters": chapters,
            "overall": overall,
        },
    )

    write_story_md(chapters, overall, out_md)
    _save_json(
        out_json,
        {
            "overall_summary": overall.get("overall_summary", ""),
            "act_summaries": overall.get("act_summaries", []),
            "chapters": chapters,
        },
    )


def write_story_md(chapters: list[dict], overall: dict, out_path: str) -> None:
    """Write chapter and overall story summaries to Markdown."""
    with open(out_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("# Song-as-Chapter Story Summary\n\n")

        if overall.get("overall_summary"):
            file_handle.write("## Overall Story\n\n")
            file_handle.write(overall["overall_summary"].strip() + "\n\n")

        act_summaries = overall.get("act_summaries") or []
        if act_summaries:
            file_handle.write("## Act Summaries\n\n")
            for act_summary in act_summaries:
                file_handle.write(f"### Act {act_summary.get('act')}\n\n")
                file_handle.write((act_summary.get("summary") or "").strip() + "\n\n")

        current_act = None
        for chapter in chapters:
            if chapter["act"] != current_act:
                current_act = chapter["act"]
                file_handle.write(f"## Act {current_act}\n\n")

            chapter_title = chapter.get("title") or chapter.get("chapter_title") or ""
            characters = chapter.get("characters") or chapter.get("key_characters") or []
            themes = chapter.get("themes") or []

            file_handle.write(f"### {chapter['index']}. {chapter['song_title']}\n\n")
            file_handle.write(f"- **Chapter title:** {chapter_title}\n")
            file_handle.write(f"- **Story role:** {chapter.get('story_role', '')}\n")
            file_handle.write(f"- **Confidence:** {chapter.get('confidence', '')}\n")
            file_handle.write(f"- **Characters:** {', '.join(characters)}\n")
            if themes:
                file_handle.write(f"- **Themes:** {', '.join(themes)}\n")
            file_handle.write("\n")
            file_handle.write((chapter.get("summary") or "").strip() + "\n\n")

            events = chapter.get("key_events") or []
            if events:
                file_handle.write("**Key events**\n\n")
                for event in events:
                    file_handle.write(f"- {event}\n")
                file_handle.write("\n")