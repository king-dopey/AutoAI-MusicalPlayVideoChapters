"""Audio extraction and continuity-block helpers."""

from __future__ import annotations

import os
from functools import lru_cache
import subprocess
import wave

import numpy as np

from autoai_musical_play_video_chapters.config import load_settings

_SETTINGS = load_settings()

AUDIO_SR = _SETTINGS.audio_sr
FRAME_SEC = _SETTINGS.frame_sec
FRAME_HOP_SEC = _SETTINGS.frame_hop_sec
SHORT_GAP_MS = _SETTINGS.short_gap_ms
LONG_GAP_MS = _SETTINGS.long_gap_ms
BED_MIN_RATIO = _SETTINGS.bed_min_ratio
BED_MIN_RMS_N = _SETTINGS.bed_min_rms_n


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer to a closed interval."""
    return max(lo, min(hi, value))


def _ms_to_clock(ms: int | None) -> str | None:
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


def _duration_clock(start_ms: int | None, end_ms: int | None) -> str | None:
    """Compute a non-negative duration string between two timestamps."""
    if start_ms is None or end_ms is None:
        return None
    duration_ms = max(0, end_ms - start_ms)
    return _ms_to_clock(duration_ms)


def ensure_audio_wav(input_media, wav_path):
    """Extract mono PCM WAV audio with ffmpeg when missing."""
    if os.path.exists(wav_path):
        return wav_path
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        input_media,
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SR),
        "-vn",
        "-acodec",
        "pcm_s16le",
        wav_path,
    ]
    print("Extracting audio with ffmpeg...")
    subprocess.run(cmd, check=True)
    return wav_path


def load_wav_mono(path):
    """Load a WAV file and return mono float samples and sample rate."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise SystemExit("Expected 16-bit PCM wav after ffmpeg extraction.")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, framerate


def compute_basic_features(x, sr):
    """Compute lightweight time and spectral features for a signal."""
    if x is None or len(x) == 0:
        return {"rms": 0.0, "zcr": 0.0, "centroid": 0.0, "flatness": 1.0}

    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))

    sb = np.signbit(x)
    zcr = float(np.mean(sb[1:] != sb[:-1])) if len(x) > 1 else 0.0

    win = np.hanning(len(x)).astype(np.float32)
    mag = np.abs(np.fft.rfft(x * win)) + 1e-10
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)

    centroid = float((freqs * mag).sum() / mag.sum()) if mag.sum() > 0 else 0.0
    flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag)) if np.mean(mag) > 0 else 1.0

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "flatness": flatness,
    }


def percentile_bounds(arr, lo=10, hi=90):
    """Return robust percentile bounds with a nonzero span."""
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return 0.0, 1.0
    lower = float(np.percentile(arr, lo))
    upper = float(np.percentile(arr, hi))
    if upper <= lower:
        upper = lower + 1e-6
    return lower, upper


def norm01(v, lo, hi):
    """Normalize a value to the [0, 1] interval."""
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo))))


def build_audio_model(y, sr):
    """Build global audio normalization statistics from frame scans."""
    frame = int(FRAME_SEC * sr)
    hop = int(FRAME_HOP_SEC * sr)
    if frame <= 0 or hop <= 0:
        raise SystemExit("Bad audio frame settings")

    rms_vals, zcr_vals, centroid_vals, flatness_vals = [], [], [], []
    i = 0
    while i + frame <= len(y):
        feats = compute_basic_features(y[i : i + frame], sr)
        rms_vals.append(feats["rms"])
        zcr_vals.append(feats["zcr"])
        centroid_vals.append(feats["centroid"])
        flatness_vals.append(feats["flatness"])
        i += hop

    rms_lo, rms_hi = percentile_bounds(rms_vals, 10, 90)
    zcr_lo, zcr_hi = percentile_bounds(zcr_vals, 10, 90)
    cen_lo, cen_hi = percentile_bounds(centroid_vals, 10, 90)
    flat_lo, flat_hi = percentile_bounds(flatness_vals, 10, 90)

    silence_thr = float(np.percentile(np.asarray(rms_vals) if len(rms_vals) else np.array([0.0]), 20))

    return {
        "y": y,
        "sr": sr,
        "rms_lo": rms_lo,
        "rms_hi": rms_hi,
        "zcr_lo": zcr_lo,
        "zcr_hi": zcr_hi,
        "cen_lo": cen_lo,
        "cen_hi": cen_hi,
        "flat_lo": flat_lo,
        "flat_hi": flat_hi,
        "silence_thr": silence_thr,
    }


def ms_to_sample(ms, sr):
    """Convert milliseconds to a sample index at a given rate."""
    return int((ms / 1000.0) * sr)


def slice_audio(audio_model, start_ms, end_ms):
    """Slice audio samples between two millisecond timestamps."""
    y = audio_model["y"]
    sr = audio_model["sr"]
    start_sample = _clamp(ms_to_sample(start_ms, sr), 0, len(y))
    end_sample = _clamp(ms_to_sample(end_ms, sr), 0, len(y))
    if end_sample <= start_sample:
        return np.zeros(0, dtype=np.float32)
    return y[start_sample:end_sample]


def attach_span_stats(audio_model):
    """Create a cached span-statistics function bound to an audio model."""

    @lru_cache(maxsize=50000)
    def _span_stats(start_ms, end_ms):
        x = slice_audio(audio_model, start_ms, end_ms)
        sr = audio_model["sr"]

        if len(x) == 0:
            return {
                "rms": 0.0,
                "rms_n": 0.0,
                "zcr": 0.0,
                "zcr_n": 0.0,
                "centroid": 0.0,
                "centroid_n": 0.0,
                "flatness": 1.0,
                "flatness_n": 1.0,
                "nonsilent_ratio": 0.0,
                "music_bed_score": 0.0,
            }

        sub = int(0.25 * sr)
        if sub <= 0:
            sub = len(x)

        rms_list = []
        feats_acc = {"zcr": [], "centroid": [], "flatness": []}

        i = 0
        while i < len(x):
            part = x[i : i + sub]
            if len(part) < max(64, sub // 4):
                break
            feats = compute_basic_features(part, sr)
            rms_list.append(feats["rms"])
            feats_acc["zcr"].append(feats["zcr"])
            feats_acc["centroid"].append(feats["centroid"])
            feats_acc["flatness"].append(feats["flatness"])
            i += sub

        if not rms_list:
            feats = compute_basic_features(x, sr)
            rms_list = [feats["rms"]]
            feats_acc["zcr"] = [feats["zcr"]]
            feats_acc["centroid"] = [feats["centroid"]]
            feats_acc["flatness"] = [feats["flatness"]]

        rms = float(np.mean(rms_list))
        zcr = float(np.mean(feats_acc["zcr"]))
        centroid = float(np.mean(feats_acc["centroid"]))
        flatness = float(np.mean(feats_acc["flatness"]))

        rms_n = norm01(rms, audio_model["rms_lo"], audio_model["rms_hi"])
        zcr_n = norm01(zcr, audio_model["zcr_lo"], audio_model["zcr_hi"])
        centroid_n = norm01(centroid, audio_model["cen_lo"], audio_model["cen_hi"])
        flatness_n = norm01(flatness, audio_model["flat_lo"], audio_model["flat_hi"])

        nonsilent_ratio = float(np.mean(np.asarray(rms_list) > audio_model["silence_thr"]))
        music_bed_score = 0.65 * nonsilent_ratio + 0.35 * rms_n

        return {
            "rms": rms,
            "rms_n": rms_n,
            "zcr": zcr,
            "zcr_n": zcr_n,
            "centroid": centroid,
            "centroid_n": centroid_n,
            "flatness": flatness,
            "flatness_n": flatness_n,
            "nonsilent_ratio": nonsilent_ratio,
            "music_bed_score": float(max(0.0, min(1.0, music_bed_score))),
        }

    return _span_stats


def attach_cue_audio_features(cues, span_stats):
    """Augment cues with per-cue audio and adjacency gap features."""
    for i, cue in enumerate(cues):
        feats = span_stats(cue["start_ms"], cue["end_ms"])
        dur_s = max(0.05, (cue["end_ms"] - cue["start_ms"]) / 1000.0)
        cue["dur_s"] = round(dur_s, 3)
        cue["chars"] = len(cue["text"])
        cue["chars_per_sec"] = round(cue["chars"] / dur_s, 3)
        cue["audio"] = feats

        prev_gap = 0 if i == 0 else max(0, cue["start_ms"] - cues[i - 1]["end_ms"])
        next_gap = 0 if i == len(cues) - 1 else max(0, cues[i + 1]["start_ms"] - cue["end_ms"])

        cue["gap_before_ms"] = prev_gap
        cue["gap_after_ms"] = next_gap

    for i, cue in enumerate(cues):
        if i == 0:
            cue["bed_before"] = 0.0
        else:
            gap_stats = span_stats(cues[i - 1]["end_ms"], cue["start_ms"]) if cue["gap_before_ms"] > 0 else {"music_bed_score": 1.0}
            cue["bed_before"] = round(gap_stats["music_bed_score"], 3)

        if i == len(cues) - 1:
            cue["bed_after"] = 0.0
        else:
            gap_stats = span_stats(cue["end_ms"], cues[i + 1]["start_ms"]) if cue["gap_after_ms"] > 0 else {"music_bed_score": 1.0}
            cue["bed_after"] = round(gap_stats["music_bed_score"], 3)


def build_blocks(cues, span_stats):
    """Group cues into continuity blocks using gap and audio heuristics."""
    blocks = []
    if not cues:
        return blocks

    start_i = 0
    block_id = 1

    for i in range(len(cues) - 1):
        a = cues[i]
        b = cues[i + 1]
        gap_ms = max(0, b["start_ms"] - a["end_ms"])
        gap_feats = span_stats(a["end_ms"], b["start_ms"]) if gap_ms > 0 else {"music_bed_score": 1.0, "rms_n": 1.0}

        same_block = (
            gap_ms <= SHORT_GAP_MS
            or (gap_ms <= LONG_GAP_MS and gap_feats["music_bed_score"] >= BED_MIN_RATIO and gap_feats["rms_n"] >= BED_MIN_RMS_N)
        )

        if not same_block:
            blocks.append(make_block(block_id, cues[start_i : i + 1], span_stats))
            block_id += 1
            start_i = i + 1

    blocks.append(make_block(block_id, cues[start_i:], span_stats))
    return blocks


def make_block(block_id, subset, span_stats):
    """Build one aggregate block record from a cue subset."""
    start_ms = subset[0]["start_ms"]
    end_ms = subset[-1]["end_ms"]
    dur_s = max(0.1, (end_ms - start_ms) / 1000.0)

    gap_scores = []
    for i in range(len(subset) - 1):
        gap_stats = span_stats(subset[i]["end_ms"], subset[i + 1]["start_ms"]) if subset[i + 1]["start_ms"] > subset[i]["end_ms"] else {"music_bed_score": 1.0}
        gap_scores.append(gap_stats["music_bed_score"])

    music_bed_ratio = float(np.mean(np.asarray(gap_scores) >= BED_MIN_RATIO)) if gap_scores else 1.0
    mean_gap_music = float(np.mean(gap_scores)) if gap_scores else 1.0
    avg_rms_n = float(np.mean([cue["audio"]["rms_n"] for cue in subset])) if subset else 0.0

    cue_density = len(subset) / dur_s
    text_density = sum(len(cue["text"]) for cue in subset) / dur_s

    song_like_score = 0.40 * music_bed_ratio + 0.25 * mean_gap_music + 0.20 * min(1.0, cue_density / 0.50) + 0.15 * min(1.0, text_density / 18.0)

    excerpt = " | ".join([cue["text"] for cue in subset[:4] if cue["text"]])[:500]

    return {
        "block_id": block_id,
        "start_cue_id": subset[0]["cue_id"],
        "end_cue_id": subset[-1]["cue_id"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration": _duration_clock(start_ms, end_ms),
        "cue_count": len(subset),
        "music_bed_ratio": round(music_bed_ratio, 3),
        "mean_gap_music": round(mean_gap_music, 3),
        "avg_rms_n": round(avg_rms_n, 3),
        "song_like_score": round(song_like_score, 3),
        "excerpt": excerpt,
    }
