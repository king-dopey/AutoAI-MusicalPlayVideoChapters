"""Package CLI entrypoint for subtitle-to-story orchestration."""

from __future__ import annotations

import json
import os
import re
import sys

from .audio import (
    attach_cue_audio_features,
    attach_span_stats,
    build_audio_model,
    build_blocks,
    ensure_audio_wav,
    load_wav_mono,
)
from .config import load_settings
from .io_flyer import parse_flyer_plot_summary, parse_flyer_songs
from .io_srt import parse_srt
from .llm.client import LLMResponseValidationError, estimate_prompt_tokens, llm_chat_json, task_deadline
from .llm.schemas import build_block_search_schema
from .pipeline.detect import (
    SEARCH_BLOCKS,
    SEARCH_STRIDE,
    block_search_prompt,
    boundary_refine_prompt,
    build_candidates_for_detection,
    choose_blocks_for_detection,
    cue_map,
    find_block_index_for_cue,
    refine_song_in_blocks,
    shrink_search_blocks,
)
from .pipeline.extract import extract_lyrics_for_song
from .pipeline.reporting import summarize_story, write_blocks_json, write_lyrics_md, write_review_md

_SETTINGS = load_settings()

BASE_URL = _SETTINGS.base_url
RESUME = _SETTINGS.resume

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


def load_json(path, default):
    """Load JSON from disk and return a fallback on failure."""
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception:
        return default


def save_json(path, obj):
    """Write an object to disk as UTF-8 formatted JSON."""
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(obj, file_handle, ensure_ascii=False, indent=2)


def normalize_space(text):
    """Collapse consecutive whitespace and trim leading/trailing space."""
    return re.sub(r"\s+", " ", text or "").strip()


def ms_to_srt(ms):
    """Convert milliseconds to an SRT timestamp string."""
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
    """Convert milliseconds to an HH:MM:SS clock string."""
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
    """Compute a non-negative duration string between two timestamps."""
    if start_ms is None or end_ms is None:
        return None
    d = max(0, end_ms - start_ms)
    return ms_to_clock(d)


def downgrade_confidence(confidence):
    """Reduce confidence by one level, bottoming out at low."""
    if confidence == "high":
        return "medium"
    if confidence == "medium":
        return "low"
    return "low"


def confidence_score_to_label(score):
    """Map a numeric confidence score to low/medium/high labels."""
    try:
        score = float(score)
    except Exception:
        return "low"
    if score >= 0.85:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def detect_songs(cues, blocks, songs, workdir):
    """Detect all songs in order and persist incremental progress."""
    progress_path = os.path.join(workdir, "enhanced_progress.json")
    results_path = os.path.join(workdir, "songs.json")
    progress_default = {
        "phase": "songs",
        "next_song_index": 1,
        "search_cue_id": 1,
        "results": [],
    }
    progress = load_json(progress_path, progress_default) if RESUME else progress_default

    results = progress.get("results", [])
    next_song_index = int(progress.get("next_song_index", 1))
    search_cue_id = int(progress.get("search_cue_id", 1))

    c_map = cue_map(cues)
    boundary_candidates = build_candidates_for_detection(cues, blocks)
    degraded_count = 0

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
            def step3_shrink_search(state):
                size = int(state.get("window_size") or SEARCH_BLOCKS)
                if size <= 1:
                    return
                state["window_size"] = max(1, size // 2)

            search_state = {
                "window_size": SEARCH_BLOCKS,
                "validation_context": {
                    "schema_kind": "enum_int",
                    "step3_shrink": step3_shrink_search,
                },
            }

            def search_candidates(state):
                return choose_blocks_for_detection(blocks, look_idx, int(state.get("window_size") or SEARCH_BLOCKS))

            candidate_blocks = search_candidates(search_state)
            if not candidate_blocks:
                break

            def search_prompt_builder(state):
                blocks_subset = search_candidates(state)
                allowed_block_ids = [0] + [b["block_id"] for b in blocks_subset]
                return block_search_prompt(
                    song,
                    prev_song,
                    songs[si:si + 3],
                    search_cue_id,
                    blocks_subset,
                    songs,
                    allowed_block_ids,
                )

            def search_schema_builder(state):
                blocks_subset = search_candidates(state)
                return build_block_search_schema([b["block_id"] for b in blocks_subset])

            prompt = search_prompt_builder(search_state)
            prompt_tokens_est = estimate_prompt_tokens(SYSTEM + "\n" + prompt)
            while prompt_tokens_est > _SETTINGS.max_prompt_tokens and int(search_state["window_size"]) > 1:
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
                search_schema_builder,
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
                found = refine_song_in_blocks(
                    cues,
                    blocks,
                    boundary_candidates,
                    song,
                    prev_song,
                    next_song,
                    selected_block_id,
                    search_cue_id,
                    song_deadline,
                    boundary_refine_prompt=boundary_refine_prompt,
                    estimate_prompt_tokens=estimate_prompt_tokens,
                    system_prompt=SYSTEM,
                    llm_chat_json=llm_chat_json,
                    validation_error_cls=LLMResponseValidationError,
                    normalize_space=normalize_space,
                    confidence_score_to_label=confidence_score_to_label,
                    downgrade_confidence=downgrade_confidence,
                )
                if found:
                    break

            look_idx += SEARCH_STRIDE

        if not found:
            nearby = choose_blocks_for_detection(blocks, start_block_idx, SEARCH_BLOCKS)
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
            cues,
            song,
            found["start_cue_id"],
            found["end_cue_id"],
            song_deadline,
            estimate_prompt_tokens=estimate_prompt_tokens,
            system_prompt=SYSTEM,
            llm_chat_json=llm_chat_json,
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
        if found.get("degraded"):
            result["degraded"] = True
            result["degradation_reason"] = found.get("degradation_reason") or found.get("notes") or "boundary_refine degraded choice"
            degraded_count += 1

        results.append(result)
        search_cue_id = max(search_cue_id, (found["end_cue_id"] or search_cue_id)) + 1

        save_json(results_path, results)
        save_json(
            progress_path,
            {
                "phase": "songs",
                "next_song_index": si + 1,
                "search_cue_id": search_cue_id,
                "results": results,
            },
        )

    results = postprocess_results(results, cues)

    save_json(results_path, results)
    save_json(
        progress_path,
        {
            "phase": "songs_done",
            "next_song_index": len(songs) + 1,
            "search_cue_id": search_cue_id,
            "results": results,
        },
    )

    if degraded_count > 0:
        print(f"Detection summary: degraded songs={degraded_count}")

    return results


def postprocess_results(results, cues):
    """Repair ordering and bounds, then refresh derived song fields."""
    c_map = cue_map(cues)
    fixed = []
    prev_end = 0

    for record in sorted(results, key=lambda x: x["index"]):
        start_id = record.get("start_cue_id")
        end_id = record.get("end_cue_id")
        conf = record.get("confidence", "low")

        if not isinstance(start_id, int):
            start_id = prev_end + 1 if (prev_end + 1) in c_map else prev_end
            conf = "low"
        if not isinstance(end_id, int):
            end_id = start_id
            conf = "low"

        if start_id <= prev_end:
            start_id = prev_end + 1 if (prev_end + 1) in c_map else start_id
            conf = downgrade_confidence(conf)

        if end_id < start_id:
            end_id = start_id
            conf = downgrade_confidence(conf)

        lyric_ids = []
        seen_lyric_ids = set()
        for cue_id in record.get("lyrics_cue_ids", []):
            if isinstance(cue_id, int) and start_id <= cue_id <= end_id and cue_id in c_map and cue_id not in seen_lyric_ids:
                seen_lyric_ids.add(cue_id)
                lyric_ids.append(cue_id)
        if not lyric_ids:
            lyric_ids = [cue_id for cue_id in range(start_id, end_id + 1) if cue_id in c_map and c_map[cue_id]["text"]]

        start_cue = c_map.get(start_id)
        end_cue = c_map.get(end_id)

        fixed.append(
            {
                **record,
                "start_cue_id": start_id,
                "end_cue_id": end_id,
                "start_time": ms_to_srt(start_cue["start_ms"]) if start_cue else None,
                "end_time": ms_to_srt(end_cue["end_ms"]) if end_cue else None,
                "duration": duration_clock(start_cue["start_ms"], end_cue["end_ms"]) if start_cue and end_cue else None,
                "confidence": conf,
                "lyrics_cue_ids": lyric_ids,
                "lyrics": "\n".join([c_map[cue_id]["text"] for cue_id in lyric_ids if cue_id in c_map]).strip(),
            }
        )
        prev_end = end_id

    return fixed


def _usage_text() -> str:
    """Return a short CLI usage summary."""
    return (
        "Usage: python -m autoai_musical_play_video_chapters\n"
        "       MusicalPlayVideoChapters\n\n"
        "Default invocation takes no arguments and runs the full pipeline.\n"
        "Behavior is controlled by environment variables such as BASE_URL,\n"
        "WORKDIR, INPUT_MEDIA, INPUT_SRT, INPUT_FLYER, RESUME, and model settings.\n"
        "See README.md for the complete environment variable reference."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the full subtitle-to-song-to-story processing pipeline."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print(_usage_text())
        return 0
    if args:
        print("Unexpected arguments. Use --help for usage.", file=sys.stderr)
        return 2

    if not BASE_URL:
        print("Set BASE_URL, e.g. http://<IP>:4000/v1", file=sys.stderr)
        return 1

    workdir = os.getenv("WORKDIR", "/work")
    media_path = os.getenv("INPUT_MEDIA", os.path.join(workdir, "input.mp4"))
    srt_path = os.getenv("INPUT_SRT", os.path.join(workdir, "input.srt"))
    flyer_path = os.getenv("INPUT_FLYER", os.path.join(workdir, "flyer.txt"))
    wav_path = os.path.join(workdir, "_audio.wav")

    blocks_json_path = os.path.join(workdir, "blocks.json")
    songs_json_path = os.path.join(workdir, "songs.json")
    review_md_path = os.path.join(workdir, "songs_review.md")
    lyrics_md_path = os.path.join(workdir, "lyrics_by_song.md")

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        srt_text = file_handle.read()
    with open(flyer_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        flyer_text = file_handle.read()

    cues = parse_srt(srt_text)
    songs = parse_flyer_songs(flyer_text)
    flyer_plot_summary = parse_flyer_plot_summary(flyer_text)

    if not cues:
        print("No subtitle cues parsed from input.srt", file=sys.stderr)
        return 1
    if not songs:
        print("No songs parsed from flyer.txt", file=sys.stderr)
        return 1

    print(f"Parsed {len(cues)} subtitle cues")
    print(f"Parsed {len(songs)} flyer songs")

    ensure_audio_wav(media_path, wav_path)
    y, sr = load_wav_mono(wav_path)
    print(f"Loaded audio: {len(y) / sr / 60:.1f} minutes at {sr} Hz")

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
    return 0
