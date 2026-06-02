# Copilot Instructions

- Follow [AGENTS.md](../AGENTS.md) if present; it is the workspace-level authority for this repo.
- Main entrypoint: [convert.py](../convert.py).
- Keep LLM request handling centralized in convert.py; do not duplicate retry, schema, or repair logic in callers.
- Preserve the checkpoint/resume flow and the JSON progress files in the workspace root.
- If you change env vars, defaults, or outputs, update [README.md](../README.md) in the same change.
- When asked to create a commit, write a detailed commit message that states what changed, why it changed, and any notable validation.