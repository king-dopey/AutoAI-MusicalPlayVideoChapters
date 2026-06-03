"""Detection pipeline helpers and boundary refinement logic."""

from __future__ import annotations

import json
import logging
import re

from autoai_musical_play_video_chapters.config import load_settings
from autoai_musical_play_video_chapters.llm.schemas import build_boundary_selection_schema

_SETTINGS = load_settings()

LONG_GAP_MS = _SETTINGS.long_gap_ms
SEARCH_BLOCKS = _SETTINGS.search_blocks
SEARCH_STRIDE = _SETTINGS.search_stride
BOUNDARY_CONTEXT_BLOCKS = _SETTINGS.boundary_context_blocks
MAX_PROMPT_TOKENS = _SETTINGS.max_prompt_tokens
MIN_BOUNDARY_SHIFT_MS = _SETTINGS.min_boundary_shift_ms
ENABLE_VERIFIER = _SETTINGS.enable_verifier
VALIDATION_REPAIR_MAX_STEPS = _SETTINGS.validation_repair_max_steps
BOUNDARY_REFINE_FALLBACK_WINDOW = _SETTINGS.boundary_refine_fallback_window
STRICT_BOUNDARIES = _SETTINGS.strict_boundaries
TASK_BOUNDARY_NEIGHBORS = 2


def _ms_to_clock(ms):
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


def find_cue_index_by_id(cues, cue_id):
    """Locate the list index for a cue id."""
    for i, cue in enumerate(cues):
        if cue["cue_id"] == cue_id:
            return i
    return None


def find_block_index_for_cue(blocks, cue_id):
    """Find the block index that contains a cue id."""
    for i, block in enumerate(blocks):
        if block["start_cue_id"] <= cue_id <= block["end_cue_id"]:
            return i
    return 0


def cue_map(cues):
    """Build a cue-id lookup table."""
    return {cue["cue_id"]: cue for cue in cues}


def slug_tokens(text):
    """Tokenize text into lowercase alphanumeric terms."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [token for token in text.split() if token]


GENERIC_TITLE_TOKENS = {
    "part",
    "parts",
    "act",
    "opening",
    "finale",
    "reprise",
    "the",
    "a",
    "an",
    "of",
    "and",
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
}


def title_tokens(title):
    """Return filtered title tokens useful for fuzzy anchor matching."""
    return [token for token in slug_tokens(title) if len(token) > 2 and token not in GENERIC_TITLE_TOKENS]


def block_title_anchor_hits(block, title):
    """Find shared tokens between a block excerpt and a song title."""
    block_tokens = set(slug_tokens(block["excerpt"]))
    title_token_set = set(title_tokens(title))
    if not title_token_set:
        return []
    return sorted(list(block_tokens & title_token_set))


def song_list_text(songs):
    """Format ordered song metadata as numbered prompt lines."""
    return "\n".join([f'{song["index"]}. Act {song["act"]} - {song["title"]} : {song["performers"]}' for song in songs])


def shrink_search_blocks(current):
    """Reduce search window size for oversized prompts."""
    return max(1, current - max(1, current // 3))


def shrink_boundary_context(current):
    """Reduce boundary context size for oversized prompts."""
    return max(0, current - 1)


def format_block_summaries(blocks, target_song):
    """Render compact block summaries with title-anchor hints."""
    rows = []
    for block in blocks:
        end_ms = block.get("end_ms", block.get("end_m", block.get("start_ms", 0)))
        hits = block_title_anchor_hits(block, target_song["title"])
        rows.append(
            f'BLOCK {block["block_id"]} | cues {block["start_cue_id"]}-{block["end_cue_id"]} | '
            f'{_ms_to_clock(block["start_ms"])}-{_ms_to_clock(end_ms)} | dur {block["duration"]} | '
            f'music_bed_ratio {block["music_bed_ratio"]} | song_like_score {block["song_like_score"]} | '
            f'title_anchor_hits {hits} | excerpt: {block["excerpt"]}'
        )
    return "\n".join(rows)


def block_search_prompt(
    target_song,
    prev_song,
    next_songs,
    search_cue_id,
    candidate_blocks,
    all_songs,
    allowed_block_ids,
):
    """Build the prompt for selecting a candidate song block."""
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

Allowed block IDs (exact set):
{json.dumps(allowed_block_ids, ensure_ascii=False)}

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
    """Format cues as TSV-like rows for LLM boundary tasks."""
    rows = []
    for cue in cues_subset:
        rows.append(
            f'{cue["cue_id"]}\t{_ms_to_clock(cue["start_ms"])}\t{_ms_to_clock(cue["end_ms"])}\t'
            f'{cue["dur_s"]:.2f}\t{cue["gap_before_ms"]/1000.0:.2f}\t{cue["gap_after_ms"]/1000.0:.2f}\t'
            f'{cue["bed_before"]:.2f}\t{cue["bed_after"]:.2f}\t{cue["audio"]["rms_n"]:.2f}\t'
            f'{cue["chars_per_sec"]:.2f}\t{cue["text"]}'
        )
    return "\n".join(rows)


def boundary_refine_prompt(
    target_song,
    prev_song,
    next_song,
    search_cue_id,
    cues_subset,
    candidates,
    allowed_indexes,
):
    """Build the prompt for precise start/end cue refinement."""
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

Allowed candidate indexes (exact set):
{json.dumps(allowed_indexes, ensure_ascii=False)}

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


def build_boundary_candidates(cues, blocks):
    """Build ordered boundary candidates from block starts and long cue gaps."""
    candidates = []
    seen = set()

    def add_candidate(cue_id, ms, kind, source_block_id, label):
        if cue_id in seen:
            return
        seen.add(cue_id)
        candidates.append(
            {
                "cue_id": cue_id,
                "ms": ms,
                "kind": kind,
                "source_block_id": source_block_id,
                "label": label,
            }
        )

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


def build_candidates_for_detection(cues, blocks):
    """Build ordered boundary candidates for detection call sites."""
    return build_boundary_candidates(cues, blocks)


def format_boundary_candidates(candidates):
    """Render candidate boundaries for the LLM prompt."""
    rows = []
    for candidate in candidates:
        rows.append(
            f'{candidate["candidate_index"]}. cue_id={candidate["cue_id"]} ms={_ms_to_clock(candidate["ms"])} kind={candidate["kind"]} block={candidate["source_block_id"]} label={candidate["label"]}'
        )
    return "\n".join(rows)


def choose_candidate_blocks(blocks, start_block_idx, limit=SEARCH_BLOCKS):
    """Select a sliding window of candidate blocks."""
    return blocks[start_block_idx : start_block_idx + limit]


def choose_blocks_for_detection(blocks, start_block_idx, limit=SEARCH_BLOCKS):
    """Select a candidate-block window for detection call sites."""
    return choose_candidate_blocks(blocks, start_block_idx, limit)


def candidate_index_lookup(candidates):
    """Build a lookup table for candidate indexes."""
    return {candidate["candidate_index"]: candidate for candidate in candidates}


def _estimate_ms_from_candidate_index(candidates, candidate_index):
    """Estimate candidate timestamp for an in-set or nearby out-of-set index."""
    if not candidates or not isinstance(candidate_index, int):
        return None

    ordered = sorted(candidates, key=lambda c: c["candidate_index"])
    first_idx = ordered[0]["candidate_index"]
    last_idx = ordered[-1]["candidate_index"]
    if candidate_index <= first_idx:
        return ordered[0]["ms"]
    if candidate_index >= last_idx:
        return ordered[-1]["ms"]

    for i in range(len(ordered) - 1):
        left = ordered[i]
        right = ordered[i + 1]
        if left["candidate_index"] <= candidate_index <= right["candidate_index"]:
            if candidate_index == left["candidate_index"]:
                return left["ms"]
            if candidate_index == right["candidate_index"]:
                return right["ms"]
            span = right["candidate_index"] - left["candidate_index"]
            if span <= 0:
                return left["ms"]
            ratio = (candidate_index - left["candidate_index"]) / span
            return int(left["ms"] + ratio * (right["ms"] - left["ms"]))

    return ordered[len(ordered) // 2]["ms"]


def _choose_degraded_candidate(original_candidates, selected_block_id, validation_exc):
    """Choose a deterministic fallback candidate after repair exhaustion."""
    if not original_candidates:
        return None, "midpoint", None

    ordered = sorted(original_candidates, key=lambda c: c["candidate_index"])

    received = validation_exc.received_value
    received_index = None
    if isinstance(received, int):
        received_index = received
    elif isinstance(received, dict):
        for key in ("start_candidate_index", "end_candidate_index", "selected_block_id"):
            value = received.get(key)
            if isinstance(value, int):
                received_index = value
                break

    if isinstance(received_index, int):
        target_ms = _estimate_ms_from_candidate_index(ordered, received_index)
        if target_ms is not None:
            closest = min(ordered, key=lambda c: (abs(c["ms"] - target_ms), c["candidate_index"]))
            return closest["candidate_index"], "closest", received_index

    if isinstance(selected_block_id, int) and selected_block_id > 0:
        block_matches = [c for c in ordered if c.get("source_block_id") == selected_block_id]
        if block_matches:
            block_matches.sort(key=lambda c: (abs(c["candidate_index"] - selected_block_id), c["candidate_index"]))
            return block_matches[0]["candidate_index"], "anchor", None

    midpoint = ordered[len(ordered) // 2]
    return midpoint["candidate_index"], "midpoint", None


def refine_song_in_blocks(
    cues,
    blocks,
    boundary_candidates,
    song,
    prev_song,
    next_song,
    selected_block_id,
    search_cue_id,
    task_deadline,
    *,
    estimate_prompt_tokens,
    system_prompt,
    llm_chat_json,
    validation_error_cls,
    normalize_space,
    confidence_score_to_label,
    downgrade_confidence,
):
    """Refine selected block context into exact song cue boundaries."""
    block_idx = selected_block_id - 1
    context_blocks = BOUNDARY_CONTEXT_BLOCKS

    while True:
        lo = max(0, block_idx - context_blocks)
        hi = min(len(blocks), block_idx + context_blocks + 1)
        cue_lo = blocks[lo]["start_cue_id"]
        cue_hi = blocks[hi - 1]["end_cue_id"]

        subset = [c for c in cues if cue_lo <= c["cue_id"] <= cue_hi and c["cue_id"] >= search_cue_id]
        candidate_pool = [
            c
            for c in boundary_candidates
            if cue_lo <= c["cue_id"] <= cue_hi + 1 and c["cue_id"] >= search_cue_id
        ]
        if not subset or not candidate_pool:
            if context_blocks <= 0:
                return None
            context_blocks = shrink_boundary_context(context_blocks)
            continue

        refine_state = {
            "subset": subset,
            "candidate_pool": list(candidate_pool),
            "original_candidate_pool": list(candidate_pool),
            "selected_block_id_anchor": selected_block_id,
            "validation_context": {
                "schema_kind": "enum_int",
            },
        }

        def refine_prompt_builder(state):
            current_candidates = state.get("candidate_pool", candidate_pool)
            current_allowed = [c["candidate_index"] for c in current_candidates]
            return boundary_refine_prompt(song, prev_song, next_song, search_cue_id, subset, current_candidates, current_allowed)

        def refine_schema_builder(state):
            current_candidates = state.get("candidate_pool", candidate_pool)
            current_allowed = [c["candidate_index"] for c in current_candidates]
            return build_boundary_selection_schema(current_allowed)

        def step3_shrink_refine(state):
            candidates = list(state.get("candidate_pool") or [])
            if not candidates:
                return
            ordered = sorted(candidates, key=lambda c: c["candidate_index"])
            selected_anchor = state.get("selected_block_id_anchor")
            anchor_candidate = None
            if isinstance(selected_anchor, int):
                for candidate in ordered:
                    if candidate.get("source_block_id") == selected_anchor:
                        anchor_candidate = candidate
                        break
            if anchor_candidate is None:
                anchor_candidate = ordered[len(ordered) // 2]
            anchor_index = anchor_candidate["candidate_index"]
            lo = anchor_index - BOUNDARY_REFINE_FALLBACK_WINDOW
            hi = anchor_index + BOUNDARY_REFINE_FALLBACK_WINDOW
            shrunk = [c for c in ordered if lo <= c["candidate_index"] <= hi]
            if shrunk:
                state["candidate_pool"] = shrunk

        prompt = refine_prompt_builder(refine_state)
        prompt_tokens_est = estimate_prompt_tokens(system_prompt + "\n" + prompt)
        while prompt_tokens_est > MAX_PROMPT_TOKENS and context_blocks > 0:
            new_context = shrink_boundary_context(context_blocks)
            if new_context == context_blocks:
                break
            print(
                f"LLM auto-shrink task=boundary_refine boundary_context_blocks {context_blocks}->{new_context} prompt_tokens_est={prompt_tokens_est}"
            )
            context_blocks = new_context
            lo = max(0, block_idx - context_blocks)
            hi = min(len(blocks), block_idx + context_blocks + 1)
            cue_lo = blocks[lo]["start_cue_id"]
            cue_hi = blocks[hi - 1]["end_cue_id"]
            subset = [c for c in cues if cue_lo <= c["cue_id"] <= cue_hi and c["cue_id"] >= search_cue_id]
            candidate_pool = [
                c
                for c in boundary_candidates
                if cue_lo <= c["cue_id"] <= cue_hi + 1 and c["cue_id"] >= search_cue_id
            ]
            refine_state["subset"] = subset
            refine_state["candidate_pool"] = list(candidate_pool)
            refine_state["original_candidate_pool"] = list(candidate_pool)
            prompt = refine_prompt_builder(refine_state)
            prompt_tokens_est = estimate_prompt_tokens(system_prompt + "\n" + prompt)

        if not subset or not candidate_pool:
            return None

        lookup = candidate_index_lookup(candidate_pool)

        def candidate_validator(resp):
            if not resp.get("found", False):
                return
            start_idx = resp.get("start_candidate_index")
            end_idx = resp.get("end_candidate_index")
            if not isinstance(start_idx, int) or not isinstance(end_idx, int):
                raise validation_error_cls("boundary candidate indexes must be integers")
            if start_idx not in lookup or end_idx not in lookup:
                raise validation_error_cls("boundary candidate indexes must come from the supplied set")

        refine_state["validation_context"]["step3_shrink"] = step3_shrink_refine

        try:
            resp = llm_chat_json(
                "detect",
                "boundary_refine",
                system_prompt,
                prompt,
                refine_schema_builder,
                task_deadline=task_deadline,
                candidate_validator=candidate_validator,
                prompt_builder=refine_prompt_builder,
                call_state=refine_state,
            )
        except validation_error_cls as exc:
            if STRICT_BOUNDARIES:
                raise

            original_candidates = refine_state.get("original_candidate_pool") or list(candidate_pool)
            chosen_index, chosen_source, raw_received = _choose_degraded_candidate(
                original_candidates, selected_block_id, exc
            )
            if chosen_index is None:
                return None
            chosen_lookup = candidate_index_lookup(original_candidates)
            chosen_candidate = chosen_lookup[chosen_index]
            warning_record = {
                "phase": "detect",
                "task": "boundary_refine",
                "outcome": "degraded_choice",
                "repair_step": int(exc.repair_step or VALIDATION_REPAIR_MAX_STEPS),
                "received_value": exc.received_value,
                "allowed_count": len(exc.allowed_values or []),
                "schema_kind": exc.schema_kind or "enum_int",
                "song": song["title"],
                "step": "refine",
                "chosen_index": chosen_index,
                "source": chosen_source,
            }
            logging.warning(json.dumps(warning_record, ensure_ascii=False, separators=(",", ":")))
            reason_bits = [
                "Degraded boundary_refine after validation-repair exhaustion",
                f"source={chosen_source}",
            ]
            if raw_received is not None:
                reason_bits.append(f"received={raw_received}")
            return {
                "start_cue_id": chosen_candidate["cue_id"],
                "end_cue_id": chosen_candidate["cue_id"],
                "confidence": "low",
                "notes": "; ".join(reason_bits),
                "selected_block_id": selected_block_id,
                "degraded": True,
                "degradation_reason": "; ".join(reason_bits),
            }

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
                verify_candidates = [
                    c for c in candidate_pool if verify_start <= c["candidate_index"] <= verify_end
                ]
                if verify_candidates:
                    verify_lookup = candidate_index_lookup(verify_candidates)
                    verify_allowed = [c["candidate_index"] for c in verify_candidates]
                    verify_prompt = boundary_refine_prompt(
                        song,
                        prev_song,
                        next_song,
                        search_cue_id,
                        subset,
                        verify_candidates,
                        verify_allowed,
                    )
                    verify_prompt += (
                        "\n\nVerification pass: confirm or revise the chosen boundaries. "
                        f"Original choice was start candidate {start_idx} and end candidate {end_idx}."
                    )

                    verify_state = {
                        "subset": subset,
                        "candidate_pool": list(verify_candidates),
                        "original_candidate_pool": list(verify_candidates),
                        "validation_context": {
                            "schema_kind": "enum_int",
                        },
                    }

                    def verify_schema_builder(state):
                        current_candidates = state.get("candidate_pool", verify_candidates)
                        current_allowed = [c["candidate_index"] for c in current_candidates]
                        return build_boundary_selection_schema(current_allowed)

                    def verify_prompt_builder(state):
                        current_candidates = state.get("candidate_pool", verify_candidates)
                        current_allowed = [c["candidate_index"] for c in current_candidates]
                        prompt_text = boundary_refine_prompt(
                            song,
                            prev_song,
                            next_song,
                            search_cue_id,
                            subset,
                            current_candidates,
                            current_allowed,
                        )
                        return (
                            prompt_text
                            + "\n\nVerification pass: confirm or revise the chosen boundaries. "
                            + f"Original choice was start candidate {start_idx} and end candidate {end_idx}."
                        )

                    def step3_shrink_verify(state):
                        candidates = list(state.get("candidate_pool") or [])
                        if not candidates:
                            return
                        ordered = sorted(candidates, key=lambda c: c["candidate_index"])
                        anchor_value = int(round((start_idx + end_idx) / 2.0))
                        anchor_candidate = min(
                            ordered,
                            key=lambda c: (abs(c["candidate_index"] - anchor_value), c["candidate_index"]),
                        )
                        lo = anchor_candidate["candidate_index"] - BOUNDARY_REFINE_FALLBACK_WINDOW
                        hi = anchor_candidate["candidate_index"] + BOUNDARY_REFINE_FALLBACK_WINDOW
                        shrunk = [c for c in ordered if lo <= c["candidate_index"] <= hi]
                        if shrunk:
                            state["candidate_pool"] = shrunk

                    verify_state["validation_context"]["step3_shrink"] = step3_shrink_verify

                    def verify_candidate_validator(resp):
                        if not resp.get("found", False):
                            return
                        verify_start_idx = resp.get("start_candidate_index")
                        verify_end_idx = resp.get("end_candidate_index")
                        if verify_start_idx not in verify_lookup or verify_end_idx not in verify_lookup:
                            raise validation_error_cls(
                                "verification indexes must come from the supplied neighborhood"
                            )

                    verify_resp = llm_chat_json(
                        "verify",
                        "boundary_verify",
                        system_prompt,
                        verify_prompt,
                        verify_schema_builder,
                        task_deadline=task_deadline,
                        candidate_validator=verify_candidate_validator,
                        prompt_builder=verify_prompt_builder,
                        call_state=verify_state,
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
            "degraded": False,
            "degradation_reason": "",
        }