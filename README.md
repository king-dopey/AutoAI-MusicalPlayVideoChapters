# Subtitle-to-Story Pipeline

This project processes a musical production's subtitle file and flyer text to:
- detect likely song boundaries,
- extract lyric cues,
- produce review artifacts,
- generate song-by-song story chapters and an overall summary using an LLM API.

The primary entry points are `python -m autoai_musical_play_video_chapters` and the `MusicalPlayVideoChapters` console script.

## What It Does

Given subtitle cues (`input.srt`), flyer metadata (`flyer.txt`), and source media
(`input.mp4` by default), the script:

1. Parses subtitles and flyer song order.
2. Extracts mono WAV audio using `ffmpeg`.
3. Computes audio features to build cue continuity blocks.
4. Uses an LLM to pick and refine song boundaries.
5. Extracts lyric cue IDs and lyric text per song.
6. Builds chapter-style story summaries and a final full-musical summary.

## Requirements

- Python 3.12+ (3.10+ likely works)
- `ffmpeg` on `PATH`
- Python package:
  - `numpy`
- If you use Docker, the image bundles Python 3.12, `ffmpeg`, and `numpy`, so the host only needs Docker.
- Reachable OpenAI-compatible chat endpoint

## Input Files

By default, the script reads from `/work`:

- `input.mp4` (or another media file)
- `input.srt`
- `flyer.txt`

You can override these with environment variables:
- `INPUT_MEDIA`
- `INPUT_SRT`
- `INPUT_FLYER`

## Core Environment Variables

Required:
- `BASE_URL`: Base URL of your OpenAI-compatible API (example:
  `http://host:4000/v1`)

Common optional:
- `MODEL` (global fallback; if set, it overrides the tier defaults below)
- `MODEL_DETECT` (default: `qwen3-coder:30b`)
- `MODEL_EXTRACT` (default: `qwen3-coder:30b`)
- `MODEL_SUMMARY` (default: `qwen3.6:35b-a3b`)
- `MODEL_VERIFY` (default: `nemotron-cascade-2:30b`)
- `API_KEY` (default: empty)
- `TEMPERATURE` (default: `0.2`)
- `WORKDIR` (default: `/work`)
- `RESUME` (default: `1`) for checkpoint resume behavior
- `ENABLE_VERIFIER` (default: `0`) enables the optional boundary verifier pass
- `REQUEST_TIMEOUT` (default: `180`) per HTTP request timeout in seconds
- `TASK_TIMEOUT` (default: `1800`) hard wall-clock cap per logical LLM task
- `MAX_PROMPT_TOKENS` (default: `12000`) prompt budget heuristic for auto-shrinking
- `MIN_BOUNDARY_SHIFT_MS` (default: `1500`) minimum shift needed to accept a revised boundary
- `MAX_CONCURRENCY` (default: `1`) reserved concurrency cap; the pipeline stays serialized by default

Request routing and cache controls:
- `NUM_CTX_DETECT` (default: `16384`) detect-stage context window
- `NUM_CTX_EXTRACT` (default: `16384`) lyric-extraction context window
- `NUM_CTX_SUMMARY` (default: `32768`) summary-stage context window
- `NUM_CTX_VERIFY` (default: `16384`) verifier context window
- `NUM_CTX_HARD_CAP` (default: `32768`) absolute cap used by empty-response repairs when they need more context
- `NUM_PREDICT_DETECT` (default: `512`) detect-stage output budget
- `NUM_PREDICT_EXTRACT` (default: `2048`) lyric-extraction output budget
- `NUM_PREDICT_SUMMARY` (default: `8192`) summary-stage output budget
- `NUM_PREDICT_VERIFY` (default: `4096`) verifier output budget
- `KV_CACHE_TYPE` (default: `q8_0`) forwarded to both KV cache slots in the request options
- `THINK_DETECT` (default: `0`) disables thinking for detect calls when set to `0`
- `THINK_EXTRACT` (default: `0`) disables thinking for extract calls when set to `0`
- `THINK_SUMMARY` (default: `1`) enables thinking for summary calls by default
- `THINK_VERIFY` (default: `1`) enables thinking for verifier calls by default

Retry and error handling:
- `MAX_RETRIES` (default: `8`)
- `BASE_SLEEP` (default: `2.0`)
- `MAX_SLEEP` (default: `45.0`)
- `ERR_BODY_CHARS` (default: `6000`)
- `EMPTY_REPAIR_MAX_STEPS` (default: `3`) bounded empty-response repair attempts before escalation
- `VALIDATION_REPAIR_MAX_STEPS` (default: `3`) bounded validation-error repair attempts for enum-constrained index calls
- `BOUNDARY_REFINE_FALLBACK_WINDOW` (default: `5`) candidate count on each side of the anchor for step-3 boundary refinement shrinking
- `STRICT_BOUNDARIES` (default: `0`) when set to `1`, disables degraded boundary fallback and aborts on exhausted validation repair
- `NUM_PREDICT_HARD_CAP` (default: `16384`) absolute ceiling used by the empty-response repair ladder
- `LOG_RAW_EMPTY` (default: `0`) set to `1` to log the first 1000 raw response characters on final empty-response failure

### Empty-response handling

Thinking-mode models can sometimes spend their full output budget on hidden reasoning and return HTTP 200 with an empty or whitespace-only `choices[0].message.content`. That is classified as `LLMEmptyResponseError`, not as a transport problem.

The centralized helper now uses a repair ladder instead of retrying the same prompt unchanged:
1. Double `num_predict` up to `NUM_PREDICT_HARD_CAP`.
2. Double it again and force `X-Ollama-Think: false` for that one call.
3. Force `X-Ollama-Think: false` again and prepend `Respond with JSON ONLY. Do NOT think. Do NOT explain. Output must match this schema: <schema>` using the schema already sent in `response_format`.

If the empty response still persists after the configured `EMPTY_REPAIR_MAX_STEPS`, the helper raises a single `LLMEmptyResponseError` to the caller without falling back into the generic retry loop. When the repair ladder needs more room, it can raise `num_ctx` to keep `num_predict <= num_ctx - prompt_tokens_est - 256`, up to `NUM_CTX_HARD_CAP`.

Intermittent verifier failures with `http_status:200` and `Empty model response` usually mean thinking-token budget exhaustion. The repair ladder fixes that automatically.

### Validation-error handling

Boundary and index-selection calls now send strict enum-backed schemas (for example, `selected_block_id` and boundary candidate indexes are constrained with explicit integer `enum` lists). This pushes out-of-set rejection into the upstream decoder instead of relying on prompt prose or client-side min/max checks.

When a 2xx response is parseable JSON but fails candidate/schema validation (`LLMResponseValidationError`), the centralized helper runs a dedicated repair ladder and does not use the generic retry loop for that error class:
1. Re-issue with a repair message that includes the model's prior value and the exact allowed enum set.
2. Retry once with `X-Ollama-Think: true` and a larger `num_predict` budget.
3. For boundary refinement/verification, shrink candidates to a tight anchor window (`BOUNDARY_REFINE_FALLBACK_WINDOW`) and rebuild the enum schema before retry.

If the validation ladder is exhausted for a song boundary refine call and `STRICT_BOUNDARIES=0`, the run degrades gracefully instead of aborting:
- the song gets deterministic fallback boundaries,
- `songs.json` adds optional fields `degraded: true` and `degradation_reason`,
- processing continues with the next song.

Set `STRICT_BOUNDARIES=1` for CI or strict batch runs where any exhausted validation repair must fail loud.

Intermittent `LLMResponseValidationError` failures with `http_status:200` and long candidate lists usually mean the model lost track of the allowed range. Enum schema hardening fixes this at the source; the validation ladder is the safety net.

Audio and segmentation tuning:
- `AUDIO_SR` (default: `16000`)
- `FRAME_SEC` (default: `1.0`)
- `FRAME_HOP_SEC` (default: `0.25`)
- `SHORT_GAP_MS` (default: `1600`)
- `LONG_GAP_MS` (default: `5000`)
- `BED_MIN_RATIO` (default: `0.55`)
- `BED_MIN_RMS_N` (default: `0.20`)
- `SEARCH_BLOCKS` (default: `6`)
- `SEARCH_STRIDE` (default: `3`)
- `BOUNDARY_CONTEXT_BLOCKS` (default: `1`)
- `LYRICS_WINDOW_CUES` (default: `90`)
- `LYRICS_WINDOW_OVERLAP` (default: `15`)

## Model Routing

The pipeline now routes each phase to its own model by default:
- Detect/boundary selection uses `MODEL_DETECT`.
- Lyric extraction uses `MODEL_EXTRACT`.
- Chapter summaries and the final story summary use `MODEL_SUMMARY`.
- Optional boundary verification uses `MODEL_VERIFY` when `ENABLE_VERIFIER=1`.

The run order is fixed:
1. Detect and extract are completed first for all songs.
2. A one-token warmup call is sent to the summary model.
3. Chapter summaries and the overall summary run after warmup.

If `MODEL` is set, it acts as a global fallback for all four tier-specific model variables.

## Local Run

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy
```

Run from the `work` folder:

```bash
export BASE_URL="http://<your-server>:4000/v1"
export MODEL_DETECT="qwen3-coder:30b"
export MODEL_EXTRACT="qwen3-coder:30b"
export MODEL_SUMMARY="qwen3.6:35b-a3b"
# Optional verifier:
# export MODEL_VERIFY="nemotron-cascade-2:30b"
# export ENABLE_VERIFIER=1
# export API_KEY="..."  # if required by your endpoint
python -m autoai_musical_play_video_chapters
```

For a short usage summary:

```bash
python -m autoai_musical_play_video_chapters --help
```

If the console script is installed, this works too:

```bash
MusicalPlayVideoChapters
```
```

## Docker Run

Build:

```bash
docker build -t subtitle-to-story .
```

Run (mount current folder as `/work`):

```bash
docker run --rm \
  -e BASE_URL="http://host.docker.internal:4000/v1" \
  -e MODEL_DETECT="qwen3-coder:30b" \
  -e MODEL_EXTRACT="qwen3-coder:30b" \
  -e MODEL_SUMMARY="qwen3.6:35b-a3b" \
  -v "$PWD":/work \
  autoai_musical_play_video_chapters
```

On Linux, if `host.docker.internal` is unavailable, replace it with a reachable
host IP for your LLM endpoint.

Docker run with optional verifier and resume:

```bash
docker run --rm \
  -e BASE_URL="http://<orin>:4000/v1" \
  -e MODEL_DETECT="qwen3-coder:30b" \
  -e MODEL_EXTRACT="qwen3-coder:30b" \
  -e MODEL_SUMMARY="qwen3.6:35b-a3b" \
  -e MODEL_VERIFY="nemotron-cascade-2:30b" \
  -e ENABLE_VERIFIER=1 \
  -e KV_CACHE_TYPE=q8_0 \
  -e RESUME=1 \
  -v "$PWD":/work \
  autoai_musical_play_video_chapters
```

## Outputs

Main outputs in `WORKDIR`:

- `blocks.json`: audio-backed subtitle continuity blocks
- `songs.json`: detected song timings, lyric cue IDs, lyric text
- `songs_review.md`: table for manual timing review
- `lyrics_by_song.md`: extracted lyrics grouped by act/song
- `song_story_summary.json`: chapter summaries and overall story summary
- `song_story_summary.md`: Markdown rendering of story summary

Optional per-song detection fields:
- `degraded` (boolean): present and `true` when boundary refinement exhausted validation repair and a deterministic fallback choice was used.
- `degradation_reason` (string): short reason/source for the degraded fallback decision.

Progress/checkpoint files:

- `enhanced_progress.json`: song detection progress
- `song_summary_progress.json`: chapter-summary progress

## Manual Test

Suggested development smoke test from the local Python environment:

```bash
RESUME=1 \
BASE_URL=http://<orin>:4000/v1 \
MODEL_DETECT=qwen3-coder:30b \
MODEL_EXTRACT=qwen3-coder:30b \
MODEL_SUMMARY=qwen3.6:35b-a3b \
MODEL_VERIFY=nemotron-cascade-2:30b \
ENABLE_VERIFIER=1 \
KV_CACHE_TYPE=q8_0 \
python -m autoai_musical_play_video_chapters
```

Docker form:

```bash
docker run --rm \
  -e BASE_URL="http://<orin>:4000/v1" \
  -e MODEL_DETECT="qwen3-coder:30b" \
  -e MODEL_EXTRACT="qwen3-coder:30b" \
  -e MODEL_SUMMARY="qwen3.6:35b-a3b" \
  -e MODEL_VERIFY="nemotron-cascade-2:30b" \
  -e ENABLE_VERIFIER=1 \
  -e KV_CACHE_TYPE=q8_0 \
  -e RESUME=1 \
  -v "$PWD":/work \
  subtitle-to-story
```

## Notes

- `RESUME=1` reuses progress files and continues where it left off.
- Delete progress files to force a full rerun.
- The script expects an OpenAI-style `/chat/completions` API route.
- If `BASE_URL` is not set, the script exits immediately.