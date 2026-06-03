# AGENTS.md

This file defines **how** an AI coding agent must work in this repository.
It contains no project facts — those live in [README.md](README.md), which the
agent must read at the start of every task and re-read after any change that
could affect its contents.

## Mandatory Reading Order

Before proposing or making any change, in this order:

1. Read [README.md](README.md) in full. It is the single source of truth for:
   project structure, entry points, environment variables, runtime behavior,
   outputs, models, refactor state, and any temporary scaffolding.
2. Read [.github/commit-message.md](.github/commit-message.md). All commits use it.
3. Read this file.
4. Re-read README.md whenever a task modifies code structure, env vars,
   outputs, commands, or refactor state. **Update README.md in the same
   commit** that changes anything it documents. A commit that drifts the code
   from README.md is incomplete.

## Operating Principles

1. **Truth over progress.** Never claim a task is complete unless every
   verification step in §Verification produces the expected output. If any
   step fails or is skipped, the task is incomplete; say so explicitly.
2. **No placeholders, no stubs, no "without moving logic yet" notes.** A file
   that exists must do its stated job. If logic cannot be moved in this turn,
   do not create the destination file in this turn.
3. **One seam per commit.** A seam is a single responsibility being relocated
   in full. Partial moves are forbidden. Behavior changes never share a commit
   with structural moves.
4. **Behavior preservation.** External behavior (CLI invocation surface, env
   var contract, output filenames, top-level JSON keys, exit codes) must not
   change as a side effect of restructuring. Functional changes are separate
   commits with their own justification in the message.
5. **Read before write.** Inspect every file that imports from or is imported
   by the file you are about to touch. List them in the commit message.
6. **Ask, don't guess.** If a requirement is ambiguous, stop and ask. Do not
   invent defaults, names, or semantics.

## Python Coding Standards (applies to all `src/` code)

1. **PEP 8** naming. **PEP 257** docstrings on every public module, class,
   and function. **PEP 484** type hints on every public function signature.
2. `from __future__ import annotations` at the top of every module.
3. One responsibility per module. Target **< 400 LOC** per file; split when
   exceeded along the responsibility boundary, not by line count.
4. **No `print()` in library code.** Use `logging.getLogger(__name__)`. The
   CLI layer may write user-facing summaries to stdout via the logging
   configuration, not via bare `print`.
5. **Imports** ordered stdlib → third-party → first-party, one group per
   blank-line block. No star imports. Relative imports only within the
   immediate package (one dot).
6. **Exceptions** raised by library code must be specific (custom subclass or
   precise builtin). Never raise bare `Exception`. Never silently `except:`
   without re-raising or logging at `ERROR` with context.
7. **No new runtime dependencies** beyond what README.md lists as current.
   Adding one requires explicit approval before the commit.
8. **Stdlib first.** Prefer `pathlib`, `dataclasses`, `enum`, `logging`,
   `json`, `subprocess`, `urllib`, `concurrent.futures` over third-party
   equivalents.
9. **Data shapes that already exist as dicts stay as dicts** unless a commit
   is explicitly dedicated to introducing a typed model for that record.
10. **Public surface is what gets imported from `__init__.py`.** Everything
    else is private; prefix with `_` if it must live at module top level.
11. **No dead code.** Remove the original definition in the same commit that
    moves it. Leaving both copies "for safety" is a defect.
12. **No commented-out code.** Use version control.

## Refactor Method (when moving code between files)

For each seam:

1. **Locate.** `grep` for every symbol you intend to move. List call sites.
2. **Move atomically.** Cut from source, paste to destination, update every
   import found in step 1. No duplication window.
3. **Delete the original.** Confirm with `grep` that no copy remains in the
   source file.
4. **Wire the public surface.** Re-export from `__init__.py` only if external
   callers need it.
5. **Run §Verification** before claiming completion.

Verification Protocol:

- Agents may only use static, local verification primitives to validate changes. Allowed primitives are:
   - `PYTHONPATH=src -m py_compile <file>`
   - `PYTHONPATH=src -m compileall -q <dir>`
   - `PYTHONPATH=src -c "import ast; ast.parse(open('<file>').read())"`
   - `grep`, `wc -l`, `find`
- Agents MUST NOT use or attempt any of the following as part of verification:
   - `pip` or any `pip install` (editable or otherwise)
   - virtualenv creation or activation (venv, virtualenv, pipx, etc.)
   - mutating `PYTHONPATH` or any runtime import path hacks
   - importing the project package for runtime verification (e.g. `import autoai_musical_play_video_chapters`)
   - executing the project as a module or script (e.g. `python -m autoai_musical_play_video_chapters` or `python convert.py`)
   - building, running, or otherwise operating Docker images or containers
   - any workaround intended to bypass PEP-668 or other host package-management protections

A seam is not done while the original file still contains any portion of the
moved responsibility.

## Verification (run before claiming any task complete)

Run these and paste the actual output into the response. Do not paraphrase.

1. `PYTHONPATH=src python3 -c "import autoai_musical_play_video_chapters"` — must exit 0.
2. `PYTHONPATH=src -m autoai_musical_play_video_chapters --help` (or equivalent
   no-network invocation documented in README.md) — must exit 0.
3. `wc -l convert.py` and `wc -l src/autoai_musical_play_video_chapters/**/*.py`
   — report the numbers. The direction of change for `convert.py` must
   match the task's stated intent.
4. `grep -nE "TODO|FIXME|without moving|placeholder|stub|pass\\s*$" src/`
   on touched files — must return nothing for files claimed as complete.
5. `grep -rn "<symbol>" .` for each symbol moved in this commit — must
   appear in the destination file and nowhere else in `src/` or `convert.py`.
6. For changes that affect runtime behavior, the operator (human) runs the
   full pipeline. The agent does not run external services or build images.
7. docker build -t autoai_musical_play_video_chapters . must build successfully

If any step cannot be run in the current environment, say so explicitly and
list which verification the operator must perform manually before merge.

## What the Agent Must Not Do

- Run the full LLM pipeline without explicit instruction.
- Edit reference/legacy files that README.md marks as out-of-scope.
- Create or retain transitional shims, re-export hacks, or "compat" layers
  that exist only to delay completing a seam.
- Modify `backup.v3/` or any reference output directory listed in README.md.
- Commit secrets, sample media, `.env`, virtualenvs, caches, or generated
  outputs.
- Force-push, rebase shared branches, or amend pushed commits.
- Add dev tooling (`pytest`, `mypy`, `ruff`, `black`, etc.) without approval.
- Claim completion when any §Verification step failed, was skipped, or was
  not actually executed.
- Additionally, agents MUST NOT:
   - run `pip` or install packages as part of verification or execution
   - create or use virtualenvs/venvs
   - import the project package or execute its entrypoints for verification
   - build or run Docker images/containers; these are operator tasks
   - attempt workarounds for system package management protections (e.g. PEP-668)

Chat Output Rules:

- Anything the agent writes to files in the workspace MUST NOT be re-pasted into chat. Files are the authoritative record.
- Agents must report only a one-line status per completed task when communicating results in chat (for example: `GOVERNANCE FIX: DONE` or `BLOCKED — <reason>`).

## Commits

- One seam per commit. One concern per commit.
- Body follows [.github/commit-message.md](.github/commit-message.md) and
  includes **what / why / verification output**.
- If the commit changes anything documented in README.md, the README.md
  update is part of the same commit. A separate "docs" follow-up is a defect.

## When Stuck

Stop. State what is ambiguous. List the specific decisions needed from the
operator. Do not proceed on assumption.
