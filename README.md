# Subtitle-to-Story Pipeline

This project processes a musical production's subtitle file and flyer text to:
- detect likely song boundaries,
- extract lyric cues,
- produce review artifacts,
- generate song-by-song story chapters and an overall summary using an LLM API.

The main entry point is `convert.py`.

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
- `MODEL` (default: `qwen3.6:35b-a3b`)
- `API_KEY` (default: empty)
- `TEMPERATURE` (default: `0.2`)
- `WORKDIR` (default: `/work`)
- `RESUME` (default: `1`) for checkpoint resume behavior

Retry and error handling:
- `MAX_RETRIES` (default: `8`)
- `BASE_SLEEP` (default: `2.0`)
- `MAX_SLEEP` (default: `45.0`)
- `ERR_BODY_CHARS` (default: `6000`)

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

## Local Run

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy
```

Run from the `work` folder:

```bash
export BASE_URL="http://<your-server>:4000/v1"
export MODEL="qwen3.6:35b-a3b"
# export API_KEY="..."  # if required by your endpoint
python convert.py
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
  -e MODEL="qwen3.6:35b-a3b" \
  -v "$PWD":/work \
  subtitle-to-story
```

On Linux, if `host.docker.internal` is unavailable, replace it with a reachable
host IP for your LLM endpoint.

## Outputs

Main outputs in `WORKDIR`:

- `blocks.json`: audio-backed subtitle continuity blocks
- `songs.json`: detected song timings, lyric cue IDs, lyric text
- `songs_review.md`: table for manual timing review
- `lyrics_by_song.md`: extracted lyrics grouped by act/song
- `song_story_summary.json`: chapter summaries and overall story summary
- `song_story_summary.md`: Markdown rendering of story summary

Progress/checkpoint files:

- `enhanced_progress.json`: song detection progress
- `song_summary_progress.json`: chapter-summary progress

## Notes

- `RESUME=1` reuses progress files and continues where it left off.
- Delete progress files to force a full rerun.
- The script expects an OpenAI-style `/chat/completions` API route.
- If `BASE_URL` is not set, the script exits immediately.
