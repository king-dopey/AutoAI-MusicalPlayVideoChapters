# Agent Instructions

## Project Shape
- Main pipeline: [convert.py](convert.py) is the production entrypoint.
- Supporting variants: [convert_lyrics.py](convert_lyrics.py), [convert_story.py](convert_story.py), and [convert_story.v2.py](convert_story.v2.py).
- Keep [README.md](README.md) as the source of truth for runtime flags, outputs, and operator-facing behavior.

## Working Rules
- Use Python 3.12+ and the existing dependency set; avoid adding new packages unless strictly necessary.
- Preserve the checkpoint/resume flow and the JSON progress files in the workspace root.
- Keep LLM request handling centralized in convert.py; do not duplicate retry, schema, or repair logic in callers.
- Treat ffmpeg, the subtitle/flyer inputs, and the OpenAI-compatible chat endpoint as required runtime dependencies.
- When creating a commit, use [.github/commit-message.md](.github/commit-message.md) as the body template and keep the message detailed, with what/why/validation.

## Editing Priorities
- Make small, focused changes that preserve the phase order: detect/extract first, then summary, then optional verification.
- Keep response schemas and validators aligned with the LLM prompt structure.
- If you change env vars, defaults, or outputs, update [README.md](README.md) in the same change.
- Prefer linking to existing docs instead of restating them here.

## Validation
- For behavior changes, validate with `python convert.py` from the work directory using `RESUME=1` and a reachable `BASE_URL`.
- If Docker behavior matters, validate the image build as well.
- Watch for intermittent empty 200 responses from thinking-mode models; the helper already owns the repair path for those cases.