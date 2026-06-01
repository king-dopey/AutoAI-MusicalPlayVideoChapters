import os, re, json, time, random
from urllib import request, error

# =========================
# Config
# =========================
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")   # e.g. http://<IP>:4000/v1
MODEL = os.getenv("MODEL", "qwen3.6:35b-a3b")
API_KEY = os.getenv("API_KEY", "")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
BASE_SLEEP = float(os.getenv("BASE_SLEEP", "2.0"))
MAX_SLEEP = float(os.getenv("MAX_SLEEP", "45.0"))
ERR_BODY_CHARS = int(os.getenv("ERR_BODY_CHARS", "6000"))

# Rough pass windows
ROUGH_WINDOW_CUES = int(os.getenv("ROUGH_WINDOW_CUES", "180"))
ROUGH_WINDOW_OVERLAP = int(os.getenv("ROUGH_WINDOW_OVERLAP", "40"))

# Boundary refine context
START_CONTEXT_BEFORE = int(os.getenv("START_CONTEXT_BEFORE", "25"))
START_CONTEXT_AFTER = int(os.getenv("START_CONTEXT_AFTER", "70"))
END_CONTEXT_BEFORE = int(os.getenv("END_CONTEXT_BEFORE", "70"))
END_CONTEXT_AFTER = int(os.getenv("END_CONTEXT_AFTER", "25"))

# Lyrics extraction windows
LYRICS_WINDOW_CUES = int(os.getenv("LYRICS_WINDOW_CUES", "120"))
LYRICS_WINDOW_OVERLAP = int(os.getenv("LYRICS_WINDOW_OVERLAP", "20"))

RESUME = os.getenv("RESUME", "1") == "1"

SYSTEM = """You are a careful musical-theatre transcript analyst.

You are given:
1) an ordered song list from a flyer
2) timestamped subtitle cues from an SRT

Your job is to identify song boundaries and lyric cues.

Rules:
- Follow the flyer song order exactly.
- Prefer the SRT evidence over assumptions.
- Distinguish sung lyrics from spoken dialogue as best as possible.
- If uncertain, make the best reasonable estimate and mark lower confidence.
- Do not invent songs not in the flyer.
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
# JSON helpers
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

# =========================
# Time / text helpers
# =========================
def ts_to_ms(ts):
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return ((int(h) * 3600 + int(m) * 60 + int(s)) * 1000) + int(ms)

def ms_to_srt(ms):
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

def duration_str(start_ms, end_ms):
    if start_ms is None or end_ms is None:
        return None
    return ms_to_clock(max(0, end_ms - start_ms))

def normalize_space(s):
    return re.sub(r"\s+", " ", s or "").strip()

def slug_tokens(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks

GENERIC_TITLE_TOKENS = {
    "part", "parts", "act", "opening", "finale", "reprise",
    "the", "a", "an", "of", "and", "i", "ii", "iii", "iv",
    "v", "vi", "vii", "viii", "ix", "x"
}

def title_tokens(title):
    toks = slug_tokens(title)
    return [t for t in toks if len(t) > 2 and t not in GENERIC_TITLE_TOKENS]

# =========================
# SRT parsing
# =========================
def parse_srt(srt_text):
    blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.M)
    cues = []
    cue_id = 1
    ts_re = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
    )

    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        ts_idx = None
        ts_match = None
        for i, ln in enumerate(lines):
            m = ts_re.search(ln)
            if m:
                ts_idx = i
                ts_match = m
                break

        if ts_idx is None:
            continue

        text_lines = lines[ts_idx + 1:]
        text = " ".join(
            re.sub(r"</?i>|</?b>|</?u>|<[^>]+>", "", ln).strip() for ln in text_lines
        ).strip()

        cues.append({
            "cue_id": cue_id,
            "start_ms": ts_to_ms(ts_match.group("start")),
            "end_ms": ts_to_ms(ts_match.group("end")),
            "start": ts_match.group("start"),
            "end": ts_match.group("end"),
            "text": normalize_space(text),
            "norm_tokens": slug_tokens(text),
        })
        cue_id += 1

    return cues

# =========================
# Flyer parsing
# =========================
def parse_flyer_songs(flyer_text):
    lines = [ln.strip() for ln in flyer_text.splitlines()]
    songs = []
    current_act = None
    in_song_breakdown = False

    for line in lines:
        if "SONG BREAKDOWN" in line.upper():
            in_song_breakdown = True
            continue
        if not in_song_breakdown:
            continue

        upper = line.upper()
        if upper == "ACT 1":
            current_act = 1
            continue
        if upper == "ACT 2":
            current_act = 2
            continue
        if line.startswith("- "):
            body = line[2:].strip()
            if ":" in body:
                title, performers = body.split(":", 1)
            else:
                title, performers = body, ""
            songs.append({
                "index": len(songs) + 1,
                "act": current_act,
                "title": normalize_space(title),
                "performers": normalize_space(performers),
            })

    return songs

# =========================
# Cue helpers
# =========================
def compact_cues(cues_subset):
    rows = []
    for c in cues_subset:
        rows.append(
            f'{c["cue_id"]}\t{ms_to_clock(c["start_ms"])}\t{ms_to_clock(c["end_ms"])}\t{c["text"]}'
        )
    return "\n".join(rows)

def find_cue_index_by_id(cues, cue_id):
    for i, c in enumerate(cues):
        if c["cue_id"] == cue_id:
            return i
    return None

def cue_by_id_map(cues):
    return {c["cue_id"]: c for c in cues}

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def weak_title_anchors(window_cues, songs, topn=10):
    hits = []
    for c in window_cues:
        cue_toks = set(c["norm_tokens"])
        if not cue_toks:
            continue
        for s in songs:
            tt = set(title_tokens(s["title"]))
            if not tt:
                continue
            inter = cue_toks & tt
            if inter:
                hits.append({
                    "cue_id": c["cue_id"],
                    "song_index": s["index"],
                    "song_title": s["title"],
                    "matched_tokens": sorted(list(inter)),
                    "score": len(inter),
                })
    hits.sort(key=lambda x: (-x["score"], x["cue_id"], x["song_index"]))
    dedup = []
    seen = set()
    for h in hits:
        k = (h["cue_id"], h["song_index"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(h)
        if len(dedup) >= topn:
            break
    return dedup

# =========================
# Prompt builders
# =========================
def song_list_text(songs):
    return "\n".join([f'{s["index"]}. Act {s["act"]} - {s["title"]} : {s["performers"]}' for s in songs])

def rough_window_prompt(songs, window_cues, anchors, window_num, total_windows):
    return f"""Analyze this subtitle window from a staged musical.

Ordered flyer song list:
{song_list_text(songs)}

Weak title-anchor hints from this window (may be wrong or absent; use only as hints):
{json.dumps(anchors, ensure_ascii=False)}

Window number:
{window_num}/{total_windows}

Subtitle cues in this window are TSV:
cue_id    start    end    text

Window:
{window_cues}

Return JSON only in this exact shape:
{{
  "songs": [
    {{
      "song_index": integer,
      "song_title": "exact flyer title",
      "rough_start_cue_id": integer or null,
      "rough_end_cue_id": integer or null,
      "confidence": "high|medium|low",
      "reason": "brief explanation"
    }}
  ]
}}

Rules:
- Include only songs with actual evidence in this window.
- A window may contain zero, one, or multiple songs.
- Use the flyer song order exactly.
- rough_start_cue_id is the first cue in this window clearly belonging to that song.
- rough_end_cue_id is the last cue in this window clearly belonging to that song.
- If only part of a song is visible, still include it.
- Do not invent songs or guess from plot alone.
"""

def refine_start_prompt(song, prev_song, next_song, cues_text):
    return f"""Find the exact first subtitle cue where this song begins.

Target song:
{json.dumps(song, ensure_ascii=False)}

Previous song:
{json.dumps(prev_song, ensure_ascii=False) if prev_song else "null"}

Next song:
{json.dumps(next_song, ensure_ascii=False) if next_song else "null"}

Subtitle cues are TSV:
cue_id    start    end    text

Context:
{cues_text}

Return JSON only:
{{
  "start_cue_id": integer or null,
  "confidence": "high|medium|low",
  "reason": "brief explanation"
}}

Guidance:
- Choose the earliest cue that is clearly part of the song.
- Include a sung pickup/opening line.
- Exclude purely spoken dialogue immediately before the song.
- If uncertain, choose the best cue and lower confidence.
"""

def refine_end_prompt(song, prev_song, next_song, cues_text):
    return f"""Find the exact last subtitle cue where this song ends.

Target song:
{json.dumps(song, ensure_ascii=False)}

Previous song:
{json.dumps(prev_song, ensure_ascii=False) if prev_song else "null"}

Next song:
{json.dumps(next_song, ensure_ascii=False) if next_song else "null"}

Subtitle cues are TSV:
cue_id    start    end    text

Context:
{cues_text}

Return JSON only:
{{
  "end_cue_id": integer or null,
  "confidence": "high|medium|low",
  "reason": "brief explanation"
}}

Guidance:
- Choose the latest cue still clearly part of the song.
- Include a sung final line.
- Exclude spoken dialogue immediately after the song.
- If uncertain, choose the best cue and lower confidence.
"""

def lyrics_prompt(song, start_cue_id, end_cue_id, cues_text):
    return f"""Within this already-bounded song span, identify which subtitle cues are sung lyrics.

Target song:
{json.dumps(song, ensure_ascii=False)}

Bounded span:
start_cue_id={start_cue_id}
end_cue_id={end_cue_id}

Subtitle cues are TSV:
cue_id    start    end    text

Window:
{cues_text}

Return JSON only:
{{
  "lyrics_cue_ids": [integers],
  "confidence": "high|medium|low",
  "reason": "brief explanation"
}}

Rules:
- Include only cues that are sung lyrics for the target song.
- Exclude pure spoken dialogue, applause, noise labels, or scene-transition text.
- If a cue is ambiguous but likely sung, include it.
- Return cue_ids only from the visible window.
"""

# =========================
# Rough pass
# =========================
def normalize_rough_resp(resp):
    songs = resp.get("songs") if isinstance(resp, dict) else None
    if not isinstance(songs, list):
        return []
    out = []
    for x in songs:
        if not isinstance(x, dict):
            continue
        idx = x.get("song_index")
        if not isinstance(idx, int):
            continue
        out.append({
            "song_index": idx,
            "song_title": normalize_space(x.get("song_title") or ""),
            "rough_start_cue_id": x.get("rough_start_cue_id") if isinstance(x.get("rough_start_cue_id"), int) else None,
            "rough_end_cue_id": x.get("rough_end_cue_id") if isinstance(x.get("rough_end_cue_id"), int) else None,
            "confidence": (x.get("confidence") or "low").lower(),
            "reason": normalize_space(x.get("reason") or "")
        })
    out.sort(key=lambda z: z["song_index"])
    return out

def confidence_rank(conf):
    return {"low": 1, "medium": 2, "high": 3}.get((conf or "low").lower(), 1)

def run_rough_pass(cues, songs, workdir, progress):
    total_windows = 0
    starts = []
    step = max(1, ROUGH_WINDOW_CUES - ROUGH_WINDOW_OVERLAP)
    i = 0
    while i < len(cues):
        starts.append(i)
        total_windows += 1
        if i + ROUGH_WINDOW_CUES >= len(cues):
            break
        i += step

    rough_hits = progress.get("rough_hits", {})
    next_window = int(progress.get("rough_next_window", 1))

    for wnum, start_idx in enumerate(starts, start=1):
        if wnum < next_window:
            continue

        end_idx = min(len(cues), start_idx + ROUGH_WINDOW_CUES)
        subset = cues[start_idx:end_idx]
        prompt = rough_window_prompt(
            songs,
            compact_cues(subset),
            weak_title_anchors(subset, songs, topn=10),
            wnum,
            total_windows
        )

        print(f"Rough pass window {wnum}/{total_windows}")
        resp = llm_json(prompt)
        items = normalize_rough_resp(resp)

        for item in items:
            idx = str(item["song_index"])
            rough_hits.setdefault(idx, {
                "mentions": 0,
                "rough_start_cue_id": None,
                "rough_end_cue_id": None,
                "best_confidence": "low",
                "reasons": [],
                "window_hits": []
            })
            rh = rough_hits[idx]
            rh["mentions"] += 1
            if item["rough_start_cue_id"] is not None:
                if rh["rough_start_cue_id"] is None or item["rough_start_cue_id"] < rh["rough_start_cue_id"]:
                    rh["rough_start_cue_id"] = item["rough_start_cue_id"]
            if item["rough_end_cue_id"] is not None:
                if rh["rough_end_cue_id"] is None or item["rough_end_cue_id"] > rh["rough_end_cue_id"]:
                    rh["rough_end_cue_id"] = item["rough_end_cue_id"]
            if confidence_rank(item["confidence"]) > confidence_rank(rh["best_confidence"]):
                rh["best_confidence"] = item["confidence"]
            if item["reason"]:
                rh["reasons"].append(item["reason"])
            rh["window_hits"].append({
                "window": wnum,
                "start": item["rough_start_cue_id"],
                "end": item["rough_end_cue_id"],
                "confidence": item["confidence"]
            })

        progress["rough_hits"] = rough_hits
        progress["rough_next_window"] = wnum + 1
        save_json(os.path.join(workdir, "songs_progress_v2.json"), progress)

    return rough_hits

def build_rough_results(cues, songs, rough_hits):
    last_cue_id = cues[-1]["cue_id"]
    rough = []

    for s in songs:
        rh = rough_hits.get(str(s["index"]), {})
        item = {
            "index": s["index"],
            "act": s["act"],
            "song_title": s["title"],
            "performers": s["performers"],
            "rough_start_cue_id": rh.get("rough_start_cue_id"),
            "rough_end_cue_id": rh.get("rough_end_cue_id"),
            "rough_confidence": rh.get("best_confidence", "low"),
            "rough_mentions": rh.get("mentions", 0),
            "rough_notes": "; ".join(rh.get("reasons", [])[:3]),
        }
        rough.append(item)

    # Monotonic cleanup
    prev_end = None
    for i, r in enumerate(rough):
        if r["rough_start_cue_id"] is not None and prev_end is not None:
            if r["rough_start_cue_id"] <= prev_end:
                r["rough_start_cue_id"] = prev_end + 1
        if r["rough_end_cue_id"] is not None and r["rough_start_cue_id"] is not None:
            if r["rough_end_cue_id"] < r["rough_start_cue_id"]:
                r["rough_end_cue_id"] = r["rough_start_cue_id"]
        if r["rough_end_cue_id"] is not None:
            prev_end = r["rough_end_cue_id"]

    # Backfill from neighbors
    for i, r in enumerate(rough):
        prev_r = rough[i - 1] if i > 0 else None
        next_r = rough[i + 1] if i + 1 < len(rough) else None

        if r["rough_start_cue_id"] is None:
            if prev_r and prev_r["rough_end_cue_id"] is not None:
                r["rough_start_cue_id"] = prev_r["rough_end_cue_id"] + 1

        if r["rough_end_cue_id"] is None:
            if next_r and next_r["rough_start_cue_id"] is not None:
                r["rough_end_cue_id"] = next_r["rough_start_cue_id"] - 1

        if r["rough_start_cue_id"] is None and i == 0:
            r["rough_start_cue_id"] = 1
        if r["rough_end_cue_id"] is None and i == len(rough) - 1:
            r["rough_end_cue_id"] = last_cue_id

        if r["rough_start_cue_id"] is not None and r["rough_end_cue_id"] is not None:
            if r["rough_end_cue_id"] < r["rough_start_cue_id"]:
                r["rough_end_cue_id"] = r["rough_start_cue_id"]
                if r["rough_confidence"] == "high":
                    r["rough_confidence"] = "medium"

    return rough

# =========================
# Refine pass
# =========================
def get_song_context_bounds(cues, rough_results, idx, exact_results_so_far):
    current = rough_results[idx]
    prev_exact = exact_results_so_far[-1] if exact_results_so_far else None
    next_rough = rough_results[idx + 1] if idx + 1 < len(rough_results) else None

    start_id = current.get("rough_start_cue_id")
    end_id = current.get("rough_end_cue_id")

    if start_id is None and prev_exact and prev_exact.get("end_cue_id") is not None:
        start_id = prev_exact["end_cue_id"] + 1
    if end_id is None and next_rough and next_rough.get("rough_start_cue_id") is not None:
        end_id = next_rough["rough_start_cue_id"] - 1
    if start_id is None:
        start_id = 1
    if end_id is None:
        end_id = cues[-1]["cue_id"]
    if end_id < start_id:
        end_id = start_id

    return start_id, end_id

def safe_refine_boundary_id(resp, key, cues_subset):
    val = resp.get(key) if isinstance(resp, dict) else None
    if not isinstance(val, int):
        return None
    ids = {c["cue_id"] for c in cues_subset}
    return val if val in ids else None

def extract_lyrics_ids(resp, cues_subset):
    ids = []
    valid = {c["cue_id"] for c in cues_subset}
    raw = resp.get("lyrics_cue_ids") if isinstance(resp, dict) else None
    if not isinstance(raw, list):
        return []
    for x in raw:
        if isinstance(x, int) and x in valid:
            ids.append(x)
    return sorted(set(ids))

def fill_small_text_gaps(cue_map, ids, max_gap=1):
    ids = sorted(set(ids))
    if not ids:
        return []
    out = set(ids)
    for a, b in zip(ids, ids[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            ok = True
            mids = []
            for cid in range(a + 1, b):
                c = cue_map.get(cid)
                if not c or not c["text"]:
                    ok = False
                    break
                mids.append(cid)
            if ok:
                for cid in mids:
                    out.add(cid)
    return sorted(out)

def refine_song(cues, songs, rough_results, idx, exact_results_so_far):
    song = songs[idx]
    prev_song = songs[idx - 1] if idx > 0 else None
    next_song = songs[idx + 1] if idx + 1 < len(songs) else None
    cue_map = cue_by_id_map(cues)

    base_start_id, base_end_id = get_song_context_bounds(cues, rough_results, idx, exact_results_so_far)
    base_start_idx = find_cue_index_by_id(cues, base_start_id) or 0
    base_end_idx = find_cue_index_by_id(cues, base_end_id) or (len(cues) - 1)

    # Refine start
    s_lo = clamp(base_start_idx - START_CONTEXT_BEFORE, 0, len(cues) - 1)
    s_hi = clamp(base_start_idx + START_CONTEXT_AFTER, 0, len(cues) - 1)
    start_subset = cues[s_lo:s_hi + 1]
    start_resp = llm_json(refine_start_prompt(song, prev_song, next_song, compact_cues(start_subset)))
    exact_start = safe_refine_boundary_id(start_resp, "start_cue_id", start_subset)
    start_conf = (start_resp.get("confidence") or "low").lower() if isinstance(start_resp, dict) else "low"
    start_reason = normalize_space(start_resp.get("reason") or "") if isinstance(start_resp, dict) else ""

    if exact_start is None:
        exact_start = base_start_id

    # Refine end
    e_lo = clamp(base_end_idx - END_CONTEXT_BEFORE, 0, len(cues) - 1)
    e_hi = clamp(base_end_idx + END_CONTEXT_AFTER, 0, len(cues) - 1)
    end_subset = cues[e_lo:e_hi + 1]
    end_resp = llm_json(refine_end_prompt(song, prev_song, next_song, compact_cues(end_subset)))
    exact_end = safe_refine_boundary_id(end_resp, "end_cue_id", end_subset)
    end_conf = (end_resp.get("confidence") or "low").lower() if isinstance(end_resp, dict) else "low"
    end_reason = normalize_space(end_resp.get("reason") or "") if isinstance(end_resp, dict) else ""

    if exact_end is None:
        exact_end = base_end_id

    if exact_end < exact_start:
        exact_end = exact_start
        if end_conf == "high":
            end_conf = "medium"

    # Lyrics extraction over bounded span
    span_start_idx = find_cue_index_by_id(cues, exact_start)
    span_end_idx = find_cue_index_by_id(cues, exact_end)
    if span_start_idx is None:
        span_start_idx = 0
    if span_end_idx is None:
        span_end_idx = span_start_idx
    if span_end_idx < span_start_idx:
        span_end_idx = span_start_idx

    lyric_ids = []
    lyric_conf_votes = []
    lyric_reasons = []

    step = max(1, LYRICS_WINDOW_CUES - LYRICS_WINDOW_OVERLAP)
    wstart = span_start_idx
    while wstart <= span_end_idx:
        wend = min(span_end_idx + 1, wstart + LYRICS_WINDOW_CUES)
        subset = cues[wstart:wend]
        resp = llm_json(lyrics_prompt(song, exact_start, exact_end, compact_cues(subset)))
        lyric_ids.extend(extract_lyrics_ids(resp, subset))
        if isinstance(resp, dict):
            lyric_conf_votes.append((resp.get("confidence") or "low").lower())
            rr = normalize_space(resp.get("reason") or "")
            if rr:
                lyric_reasons.append(rr)
        if wend >= span_end_idx + 1:
            break
        wstart += step

    lyric_ids = sorted(set([cid for cid in lyric_ids if exact_start <= cid <= exact_end]))
    lyric_ids = fill_small_text_gaps(cue_map, lyric_ids, max_gap=1)

    # Fallback if no lyric IDs were extracted
    if not lyric_ids:
        lyric_ids = [c["cue_id"] for c in cues[span_start_idx:span_end_idx + 1] if c["text"]]

    lyrics_text = "\n".join([cue_map[cid]["text"] for cid in lyric_ids if cid in cue_map]).strip()

    overall_conf = "low"
    confs = [start_conf, end_conf] + lyric_conf_votes
    if "high" in confs:
        overall_conf = "high"
    elif "medium" in confs:
        overall_conf = "medium"

    notes = "; ".join([x for x in [start_reason, end_reason] + lyric_reasons[:2] if x])

    return {
        "index": song["index"],
        "act": song["act"],
        "song_title": song["title"],
        "performers": song["performers"],
        "rough_start_cue_id": rough_results[idx].get("rough_start_cue_id"),
        "rough_end_cue_id": rough_results[idx].get("rough_end_cue_id"),
        "rough_confidence": rough_results[idx].get("rough_confidence", "low"),
        "start_cue_id": exact_start,
        "end_cue_id": exact_end,
        "start_time": ms_to_srt(cue_map[exact_start]["start_ms"]) if exact_start in cue_map else None,
        "end_time": ms_to_srt(cue_map[exact_end]["end_ms"]) if exact_end in cue_map else None,
        "duration": duration_str(
            cue_map[exact_start]["start_ms"] if exact_start in cue_map else None,
            cue_map[exact_end]["end_ms"] if exact_end in cue_map else None,
        ),
        "confidence": overall_conf,
        "lyrics_cue_ids": lyric_ids,
        "lyrics": lyrics_text,
        "notes": notes
    }

# =========================
# Output writers
# =========================
def escape_pipes(s):
    return (s or "").replace("|", "\\|")

def write_review_md(results, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Song Timing Review\n\n")
        f.write("| # | Act | Song | Start | End | Duration | Confidence | Notes |\n")
        f.write("|---:|---:|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['index']} | {r['act']} | {escape_pipes(r['song_title'])} | "
                f"{r['start_time'] or ''} | {r['end_time'] or ''} | {r['duration'] or ''} | "
                f"{r['confidence']} | {escape_pipes(r['notes'] or '')} |\n"
            )

def write_lyrics_md(results, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Lyrics by Song\n\n")
        current_act = None
        for r in results:
            if r["act"] != current_act:
                current_act = r["act"]
                f.write(f"## Act {current_act}\n\n")
            f.write(f"### {r['index']}. {r['song_title']}\n\n")
            f.write(f"- **Performers:** {r['performers'] or 'Unknown'}\n")
            f.write(f"- **Start:** {r['start_time'] or 'Unknown'}\n")
            f.write(f"- **End:** {r['end_time'] or 'Unknown'}\n")
            f.write(f"- **Confidence:** {r['confidence']}\n\n")
            if r["lyrics"]:
                f.write(r["lyrics"].strip() + "\n\n")
            else:
                f.write("[No lyrics extracted]\n\n")

# =========================
# Main
# =========================
def main():
    if not BASE_URL:
        raise SystemExit("Set BASE_URL, e.g. http://<IP>:4000/v1")

    workdir = os.getenv("WORKDIR", "/work")
    srt_path = os.getenv("INPUT_SRT", os.path.join(workdir, "input.srt"))
    flyer_path = os.getenv("INPUT_FLYER", os.path.join(workdir, "flyer.txt"))

    progress_path = os.path.join(workdir, "songs_progress_v2.json")
    rough_results_path = os.path.join(workdir, "songs_rough.json")
    songs_json_path = os.path.join(workdir, "songs.json")
    review_md_path = os.path.join(workdir, "songs_review.md")
    lyrics_md_path = os.path.join(workdir, "lyrics_by_song.md")

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        srt_text = f.read()

    with open(flyer_path, "r", encoding="utf-8", errors="ignore") as f:
        flyer_text = f.read()

    cues = parse_srt(srt_text)
    songs = parse_flyer_songs(flyer_text)

    if not cues:
        raise SystemExit("No subtitle cues parsed from input.srt")
    if not songs:
        raise SystemExit("No songs parsed from flyer.txt")

    print(f"Parsed {len(cues)} subtitle cues")
    print(f"Parsed {len(songs)} songs from flyer")

    progress_default = {
        "phase": "rough",
        "rough_next_window": 1,
        "rough_hits": {},
        "refine_next_song_index": 1,
        "results": []
    }
    progress = load_json(progress_path, progress_default) if RESUME else progress_default

    # Rough phase
    if progress.get("phase") == "rough":
        rough_hits = run_rough_pass(cues, songs, workdir, progress)
        rough_results = build_rough_results(cues, songs, rough_hits)
        save_json(rough_results_path, rough_results)

        progress["phase"] = "refine"
        progress["refine_next_song_index"] = 1
        save_json(progress_path, progress)
    else:
        rough_results = load_json(rough_results_path, [])

    if not rough_results:
        raise SystemExit("Rough results missing; cannot continue.")

    # Refine phase
    results = progress.get("results", [])
    next_song_index = int(progress.get("refine_next_song_index", 1))

    if next_song_index > len(songs):
        print("Nothing to do: all songs already refined.")
    else:
        for si in range(next_song_index, len(songs) + 1):
            print(f"Refining song {si}/{len(songs)}: {songs[si - 1]['title']}")
            result = refine_song(cues, songs, rough_results, si - 1, results)
            results.append(result)

            save_json(songs_json_path, results)
            write_review_md(results, review_md_path)
            write_lyrics_md(results, lyrics_md_path)

            progress["results"] = results
            progress["refine_next_song_index"] = si + 1
            save_json(progress_path, progress)

    progress["phase"] = "done"
    save_json(progress_path, progress)

    save_json(songs_json_path, results)
    write_review_md(results, review_md_path)
    write_lyrics_md(results, lyrics_md_path)

    print(f"Wrote: {rough_results_path}")
    print(f"Wrote: {songs_json_path}")
    print(f"Wrote: {review_md_path}")
    print(f"Wrote: {lyrics_md_path}")

if __name__ == "__main__":
    main()
