import os, re, json, time, random
from urllib import request, error

# ====== Config (matches the original “good output” behavior) ======
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")          # e.g. http://<IP>:4000/v1
MODEL = os.getenv("MODEL", "qwen3.6:35b-a3b")
API_KEY = os.getenv("API_KEY", "")                       # optional
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "12000"))     # original default
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))     # original default
STYLE = os.getenv(
    "STYLE",
    "third-person past tense, warm reflective tone, simple language, minimal adjectives"
)

# Retries + error printing
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
BASE_SLEEP = float(os.getenv("BASE_SLEEP", "2.0"))
MAX_SLEEP = float(os.getenv("MAX_SLEEP", "45.0"))
ERR_BODY_CHARS = int(os.getenv("ERR_BODY_CHARS", "6000"))

# Resume support
RESUME = os.getenv("RESUME", "1") == "1"

SYSTEM = """You are an expert narrative editor.

Goal: Convert subtitle transcript (from an .srt) into a readable story (prose).
Input will arrive in chunks. Each chunk may contain timestamps, cue numbers, and broken line wraps.

Rules:
1) Ignore/remove cue numbers and timestamps.
2) Merge broken subtitle lines into complete sentences.
3) Rewrite into a coherent narrative with paragraphs, light scene-setting, and clear chronology.
4) Preserve factual content. Do NOT invent events, dialogue, names, or outcomes.
5) You MAY lightly fix grammar, add punctuation, and replace filler words, but keep the original meaning.
6) Keep proper nouns consistent. If a name is unclear, use a stable placeholder like [Unclear Name 1] and keep it consistent.
7) Maintain a "Continuity Notes" block (private working notes) that tracks: characters, relationships, locations, timeline anchors, recurring objects, and open questions.
8) For each chunk, output ONLY:
   - STORY_TEXT
   - CONTINUITY_NOTES
Do not output any other commentary.
"""

def print_server_error_detail(status, hdrs, body, label):
    print(f"\n--- {label} ---")
    print(f"HTTP {status}")
    try:
        if hdrs:
            print("Headers:")
            for k, v in list(hdrs.items())[:60]:
                print(f"{k}: {v}")
    except Exception:
        pass
    print(f"Body (first {ERR_BODY_CHARS} chars):")
    print((body or "")[:ERR_BODY_CHARS])
    print(f"--- END {label} ---\n")

def http_post_json(url, payload, timeout=3600):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw, dict(resp.headers)
    except error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, body, dict(getattr(e, "headers", {}) or {})

def chat_completions(system, user):
    url = BASE_URL + "/chat/completions"
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    return http_post_json(url, payload)

def srt_to_transcript(srt_text: str) -> str:
    lines = srt_text.splitlines()
    out = []
    ts_re = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}")
    num_re = re.compile(r"^\s*\d+\s*$")
    for ln in lines:
        ln = ln.strip("\ufeff").rstrip()
        if not ln.strip():
            out.append("")
            continue
        if ts_re.match(ln.strip()):
            continue
        if num_re.match(ln.strip()):
            continue
        ln = re.sub(r"</?i>|</?b>|</?u>|<[^>]+>", "", ln).strip()
        out.append(ln)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def chunk_text(text: str, max_chars: int):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= max_chars:
            cur += "\n\n" + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def extract_blocks(model_text: str):
    story = model_text.strip()
    notes = ""
    m1 = re.search(r"STORY_TEXT\s*:?\s*(.*?)(?:\n\s*CONTINUITY_NOTES\s*:|\Z)", model_text, re.S | re.I)
    m2 = re.search(r"CONTINUITY_NOTES\s*:?\s*(.*)\Z", model_text, re.S | re.I)
    if m1: story = m1.group(1).strip()
    if m2: notes = m2.group(1).strip()
    return story, notes

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    in_path = os.getenv("INPUT_SRT", "/work/input.srt")
    out_path = os.getenv("OUTPUT_MD", "/work/story.md")
    notes_path = os.getenv("NOTES_TXT", "/work/continuity_notes.txt")
    progress_path = os.getenv("PROGRESS_JSON", "/work/progress.json")

    if not BASE_URL:
        raise SystemExit("Set BASE_URL, e.g. http://<IP>:4000/v1")

    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        srt = f.read()

    transcript = srt_to_transcript(srt)
    chunks = chunk_text(transcript, CHUNK_CHARS)

    if not chunks:
        raise SystemExit("No chunks created. Check that input.srt is not empty and is valid SRT.")

    # Resume state
    progress_default = {"next_index": 1, "continuity": ""}
    progress = load_json(progress_path, progress_default) if RESUME else progress_default

    start_i = int(progress.get("next_index", 1))
    continuity = (progress.get("continuity", "") or "").strip()

    # If resuming but story.md missing, start over to avoid “no output”
    if RESUME and start_i > 1 and not os.path.exists(out_path):
        print("progress.json indicates a resume, but story.md is missing; restarting from chunk 1.")
        start_i = 1

    # If already finished, say so (common “it ran but produced nothing” case)
    if start_i > len(chunks):
        print(f"Nothing to do: progress next_index={start_i} and total_chunks={len(chunks)}.")
        print("Delete /work/progress.json (and optionally story.md) to restart.")
        return

    # This matches the original “good output” behavior:
    # build in-memory parts from this run and rewrite story.md each chunk.
    story_parts = []

    # If resuming, try to seed story_parts from existing story.md (best-effort)
    if RESUME and start_i > 1 and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            existing = f.read().strip()
        if existing:
            story_parts = [existing]

    print(f"Total chunks: {len(chunks)} | Starting at: {start_i} | CHUNK_CHARS={CHUNK_CHARS}")

    for i in range(start_i, len(chunks) + 1):
        ch = chunks[i - 1]
        user = f"""CHUNK {i}/{len(chunks)}
STYLE: {STYLE}
OUTPUT_FORMAT: markdown

CONTINUITY_NOTES_SO_FAR:
{continuity if continuity else "(none yet)"}

TRANSCRIPT:
{ch}
"""

        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            status, raw, hdrs = chat_completions(SYSTEM, user)

            if status < 200 or status >= 300:
                print_server_error_detail(status, hdrs, raw, "SERVER ERROR DETAIL (non-2xx)")
                last_err = f"HTTP {status}"
            else:
                try:
                    data = json.loads(raw)
                    text = data["choices"][0]["message"]["content"]
                except Exception:
                    print_server_error_detail(status, hdrs, raw, "BAD JSON FROM SERVER (2xx but not parseable)")
                    last_err = "Bad JSON"
                else:
                    story, continuity = extract_blocks(text)
                    story_parts.append(story.strip())

                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write("\n\n".join([p for p in story_parts if p]).strip() + "\n")

                    with open(notes_path, "w", encoding="utf-8") as f:
                        f.write(continuity.strip() + "\n")

                    save_json(progress_path, {"next_index": i + 1, "continuity": continuity})
                    print(f"Processed chunk {i}/{len(chunks)} (attempt {attempt})")
                    last_err = None
                    break

            sleep_s = min(MAX_SLEEP, BASE_SLEEP * (2 ** (attempt - 1)) + random.random())
            print(f"Chunk {i}: attempt {attempt} failed: {last_err}")
            print(f"Sleeping {sleep_s:.1f}s...")
            time.sleep(sleep_s)

        if last_err is not None:
            raise SystemExit(f"Failed chunk {i} after {MAX_RETRIES} retries. Last error: {last_err}")

if __name__ == "__main__":
    main()
