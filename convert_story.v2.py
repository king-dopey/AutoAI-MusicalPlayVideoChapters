import os, re, json, time, random
from urllib import request, error

# =========================
# Config
# =========================
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
MODEL = os.getenv("MODEL", "qwen3.6:35b-a3b")
API_KEY = os.getenv("API_KEY", "")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
BASE_SLEEP = float(os.getenv("BASE_SLEEP", "2.0"))
MAX_SLEEP = float(os.getenv("MAX_SLEEP", "45.0"))
ERR_BODY_CHARS = int(os.getenv("ERR_BODY_CHARS", "6000"))
RESUME = os.getenv("RESUME", "1") == "1"

SYSTEM = """You are an expert musical-story analyst.

You will be given:
1) a flyer with cast, plot summary, and song order
2) lyrics grouped by song
3) optionally structured song metadata

Your task:
Summarize the musical's story as a sequence of chapter summaries, where each song is a chapter.

Rules:
- Follow the song order exactly.
- Treat each song as one chapter.
- Summaries should describe story events, character motivations, and changes caused by the song.
- Preserve facts from the inputs.
- Do not invent plot events that are unsupported by the lyrics or flyer.
- If a lyric is ambiguous, infer conservatively using the flyer plot summary and song placement.
- Maintain continuity across chapters.
- Return JSON only.
"""

# =========================
# HTTP helpers
# =========================
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

def chat_completions(user_prompt):
    url = BASE_URL + "/chat/completions"
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    }
    return http_post_json(url, payload)

def llm_json(user_prompt):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        status, raw, hdrs = chat_completions(user_prompt)

        if status < 200 or status >= 300:
            print_server_error_detail(status, hdrs, raw, "SERVER ERROR DETAIL (non-2xx)")
            last_err = f"HTTP {status}"
        else:
            try:
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
                return parse_json_from_text(content)
            except Exception as e:
                print_server_error_detail(status, hdrs, raw, "BAD JSON FROM SERVER (2xx but not parseable)")
                last_err = repr(e)

        sleep_s = min(MAX_SLEEP, BASE_SLEEP * (2 ** (attempt - 1)) + random.random())
        print(f"LLM attempt {attempt} failed: {last_err}")
        print(f"Sleeping {sleep_s:.1f}s...")
        time.sleep(sleep_s)

    raise SystemExit(f"LLM failed after {MAX_RETRIES} retries. Last error: {last_err}")

# =========================
# Utility
# =========================
def parse_json_from_text(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```json\s*(\\{.*?\\}|$$.*?$$)\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        return json.loads(m.group(1))

    raise ValueError("No parseable JSON found in model output")

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_space(s):
    return re.sub(r"\s+", " ", s or "").strip()

# =========================
# Input parsing
# =========================
def parse_flyer_plot_summary(flyer_text):
    m = re.search(
        r"INTO THE WOODS PLOT SUMMARY\s*(.*?)(?:## Page 4|INTO THE WOODS DIRECTOR|INTO THE WOODS: SONG BREAKDOWN)",
        flyer_text,
        re.S | re.I
    )
    if m:
        return normalize_space(m.group(1))
    return ""

def parse_lyrics_by_song(md_text):
    results = []
    current_act = None
    current_song = None

    lines = md_text.splitlines()
    for line in lines:
        line = line.rstrip()

        act_match = re.match(r"##\s+Act\s+(\d+)", line, re.I)
        if act_match:
            current_act = int(act_match.group(1))
            continue

        song_match = re.match(r"###\s+(\d+)\.\s+(.*)", line)
        if song_match:
            if current_song:
                current_song["lyrics"] = current_song["lyrics"].strip()
                results.append(current_song)
            current_song = {
                "index": int(song_match.group(1)),
                "act": current_act,
                "song_title": normalize_space(song_match.group(2)),
                "performers": "",
                "start_time": "",
                "end_time": "",
                "confidence": "",
                "lyrics": ""
            }
            continue

        if current_song:
            perf = re.match(r"- \*\*Performers:\*\*\s*(.*)", line)
            start = re.match(r"- \*\*Start:\*\*\s*(.*)", line)
            end = re.match(r"- \*\*End:\*\*\s*(.*)", line)
            conf = re.match(r"- \*\*Confidence:\*\*\s*(.*)", line)

            if perf:
                current_song["performers"] = normalize_space(perf.group(1))
            elif start:
                current_song["start_time"] = normalize_space(start.group(1))
            elif end:
                current_song["end_time"] = normalize_space(end.group(1))
            elif conf:
                current_song["confidence"] = normalize_space(conf.group(1))
            elif line.startswith("## "):
                pass
            elif line.startswith("### "):
                pass
            else:
                current_song["lyrics"] += line + "\n"

    if current_song:
        current_song["lyrics"] = current_song["lyrics"].strip()
        results.append(current_song)

    return results

# =========================
# Prompting
# =========================
def chapter_prompt(flyer_plot_summary, prior_chapters, song_record):
    return f"""Create a chapter-style story summary for this song.

Global plot summary from flyer:
{flyer_plot_summary}

Prior chapter continuity:
{json.dumps(prior_chapters[-3:], ensure_ascii=False, indent=2) if prior_chapters else "[]"}

Current song record:
{json.dumps(song_record, ensure_ascii=False, indent=2)}

Return JSON only in this exact shape:
{{
  "index": {song_record["index"]},
  "act": {song_record["act"] if song_record["act"] is not None else "null"},
  "song_title": {json.dumps(song_record["song_title"])},
  "chapter_title": "short readable chapter title",
  "summary": "1-3 paragraph story summary of what happens in this song",
  "story_role": "setup|decision|conflict|turning point|aftermath|finale",
  "key_characters": ["names"],
  "key_events": ["event 1", "event 2"],
  "continuity_notes": ["important carry-forward facts"],
  "confidence": "high|medium|low"
}}

Requirements:
- Treat the song as a story chapter.
- Focus on plot, emotional movement, and consequences.
- Use the flyer plot summary only to clarify, not to overwrite the lyrics.
- Keep the summary grounded in the provided inputs.
- continuity_notes should be short factual bullets useful for the next chapter.
"""

def final_assembly_prompt(chapters, flyer_plot_summary):
    return f"""Create a polished overall story summary of the musical based on these song-chapters.

Flyer plot summary:
{flyer_plot_summary}

Song chapters:
{json.dumps(chapters, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "overall_summary": "multi-paragraph summary of the full musical",
  "act_summaries": [
    {{"act": 1, "summary": "..." }},
    {{"act": 2, "summary": "..." }}
  ]
}}

Requirements:
- Preserve song order.
- Reflect the progression from wish and pursuit to consequence and resolution.
- Do not invent unsupported details.
"""

# =========================
# Writers
# =========================
def write_md(chapters, overall, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Song-as-Chapter Story Summary\n\n")

        if overall.get("overall_summary"):
            f.write("## Overall Story\n\n")
            f.write(overall["overall_summary"].strip() + "\n\n")

        act_summaries = overall.get("act_summaries") or []
        if act_summaries:
            f.write("## Act Summaries\n\n")
            for a in act_summaries:
                f.write(f"### Act {a.get('act')}\n\n")
                f.write((a.get("summary") or "").strip() + "\n\n")

        current_act = None
        for ch in chapters:
            if ch["act"] != current_act:
                current_act = ch["act"]
                f.write(f"## Act {current_act}\n\n")

            f.write(f"### {ch['index']}. {ch['song_title']}\n\n")
            f.write(f"- **Chapter title:** {ch.get('chapter_title', '')}\n")
            f.write(f"- **Story role:** {ch.get('story_role', '')}\n")
            f.write(f"- **Confidence:** {ch.get('confidence', '')}\n")
            f.write(f"- **Characters:** {', '.join(ch.get('key_characters', []))}\n\n")
            f.write((ch.get("summary") or "").strip() + "\n\n")

            events = ch.get("key_events") or []
            if events:
                f.write("**Key events**\n\n")
                for e in events:
                    f.write(f"- {e}\n")
                f.write("\n")

def write_json(chapters, overall, out_path):
    obj = {
        "overall_summary": overall.get("overall_summary", ""),
        "act_summaries": overall.get("act_summaries", []),
        "chapters": chapters
    }
    save_json(out_path, obj)

# =========================
# Main
# =========================
def main():
    if not BASE_URL:
        raise SystemExit("Set BASE_URL, e.g. http://<IP>:4000/v1")

    workdir = os.getenv("WORKDIR", "/work")
    lyrics_path = os.getenv("INPUT_LYRICS", os.path.join(workdir, "lyrics_by_song.md"))
    flyer_path = os.getenv("INPUT_FLYER", os.path.join(workdir, "flyer.txt"))
    songs_json_path = os.getenv("INPUT_SONGS_JSON", os.path.join(workdir, "songs.json"))

    progress_path = os.path.join(workdir, "song_summary_progress.json")
    out_md = os.path.join(workdir, "song_story_summary.md")
    out_json = os.path.join(workdir, "song_story_summary.json")

    with open(lyrics_path, "r", encoding="utf-8", errors="ignore") as f:
        lyrics_md = f.read()

    with open(flyer_path, "r", encoding="utf-8", errors="ignore") as f:
        flyer_text = f.read()

    songs_meta = load_json(songs_json_path, None)
    songs_from_md = parse_lyrics_by_song(lyrics_md)
    flyer_plot_summary = parse_flyer_plot_summary(flyer_text)

    if not songs_from_md:
        raise SystemExit("No songs parsed from lyrics_by_song.md")

    # Merge songs.json metadata if present
    if isinstance(songs_meta, list):
        meta_by_index = {x.get("index"): x for x in songs_meta if isinstance(x, dict)}
        merged = []
        for s in songs_from_md:
            m = meta_by_index.get(s["index"], {})
            merged.append({
                "index": s["index"],
                "act": s.get("act") or m.get("act"),
                "song_title": s.get("song_title") or m.get("song_title"),
                "performers": m.get("performers") or s.get("performers"),
                "start_time": m.get("start_time") or s.get("start_time"),
                "end_time": m.get("end_time") or s.get("end_time"),
                "confidence": m.get("confidence") or s.get("confidence"),
                "lyrics": s.get("lyrics", "")
            })
        songs = merged
    else:
        songs = songs_from_md

    progress_default = {
        "next_song_index": 1,
        "chapters": [],
        "overall": {}
    }
    progress = load_json(progress_path, progress_default) if RESUME else progress_default

    chapters = progress.get("chapters", [])
    next_song_index = int(progress.get("next_song_index", 1))

    for i in range(next_song_index, len(songs) + 1):
        song = songs[i - 1]
        print(f"Summarizing song {i}/{len(songs)}: {song['song_title']}")
        resp = llm_json(chapter_prompt(flyer_plot_summary, chapters, song))
        chapters.append(resp)

        save_json(progress_path, {
            "next_song_index": i + 1,
            "chapters": chapters,
            "overall": progress.get("overall", {})
        })

    print("Creating overall story summary...")
    overall = llm_json(final_assembly_prompt(chapters, flyer_plot_summary))

    save_json(progress_path, {
        "next_song_index": len(songs) + 1,
        "chapters": chapters,
        "overall": overall
    })

    write_md(chapters, overall, out_md)
    write_json(chapters, overall, out_json)

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")

if __name__ == "__main__":
    main()
