#!/usr/bin/env python3
"""
Librosa QCM Inference Script

Extracts audio features (tempo, RMS energy, ZCR, spectral centroid, MFCCs)
and maps them to QCM choices to determine the best answer.

Usage:
    python librosa_qcm_inference.py --wav audio.wav --question "..." --choices '{"A": "...", "B": "..."}'

Output: JSON to stdout with {answer, confidence, detail}
"""

import sys
import json
import argparse
import numpy as np

try:
    import librosa
except ImportError:
    print(json.dumps({
        "answer": "ERROR",
        "confidence": 0.0,
        "detail": "librosa is not installed. Run: pip install librosa numpy"
    }))
    sys.exit(0)


# --- Feature extraction ---

def extract_features(wav_path, sr=22050):
    """Extract core audio features from a WAV file."""
    y, _sr = librosa.load(wav_path, sr=sr, mono=True)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    # RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_var = float(np.var(rms))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_var = float(np.var(zcr))

    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_cent_mean = float(np.mean(spec_cent))
    spec_cent_var = float(np.var(spec_cent))

    # MFCCs (first 13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1).tolist()  # list of 13 floats
    mfcc_var = mfccs.var(axis=1).tolist()

    return {
        "tempo": tempo,
        "rms_mean": rms_mean,
        "rms_var": rms_var,
        "zcr_mean": zcr_mean,
        "zcr_var": zcr_var,
        "spectral_centroid_mean": spec_cent_mean,
        "spectral_centroid_var": spec_cent_var,
        "mfcc_mean": mfcc_mean,
        "mfcc_var": mfcc_var,
        "duration": len(y) / sr,
        "sample_rate": sr,
    }


# --- Feature interpretation helpers ---

def _is_silence(features):
    """Check if audio is essentially silent."""
    return features["rms_mean"] < 0.01


def _is_speech_likelihood(features):
    """Heuristic: how likely is this speech? Returns 0–1 score."""
    score = 0.0
    # Speech typically has moderate ZCR (consonants + vowels)
    zcr = features["zcr_mean"]
    if 0.03 < zcr < 0.15:
        score += 0.3
    elif 0.01 < zcr < 0.2:
        score += 0.15

    # Speech spectral centroid is typically 1–3 kHz
    sc = features["spectral_centroid_mean"]
    if 800 < sc < 4000:
        score += 0.3
    elif 500 < sc < 5000:
        score += 0.15

    # Speech has moderate RMS
    rms = features["rms_mean"]
    if 0.01 < rms < 0.3:
        score += 0.2

    # Speech tempo is often 100–200 BPM (syllable rate mapped)
    tempo = features["tempo"]
    if 80 < tempo < 250:
        score += 0.2

    return min(score, 1.0)


def _is_music_likelihood(features):
    """Heuristic: how likely is this music? Returns 0–1 score."""
    score = 0.0
    # Music often has clearer tempo structure
    tempo = features["tempo"]
    if 60 < tempo < 200:
        score += 0.3
    elif tempo > 0:
        score += 0.1

    # Music can have broader spectral content
    sc = features["spectral_centroid_mean"]
    if 1000 < sc < 8000:
        score += 0.2

    # Music RMS varies (dynamics)
    rms_var = features["rms_var"]
    if rms_var > 0.001:
        score += 0.2

    # MFCC variance: music tends to have more timbral variation
    mfcc_var = features["mfcc_var"]
    if isinstance(mfcc_var, list) and len(mfcc_var) > 0:
        avg_mfcc_var = np.mean(mfcc_var)
        if avg_mfcc_var > 1.0:
            score += 0.3
        elif avg_mfcc_var > 0.5:
            score += 0.15

    return min(score, 1.0)


def _is_noisy_likelihood(features):
    """Heuristic: how likely is this noise/broadband?"""
    score = 0.0
    # High ZCR suggests noise
    if features["zcr_mean"] > 0.15:
        score += 0.4
    elif features["zcr_mean"] > 0.1:
        score += 0.2

    # High spectral centroid
    if features["spectral_centroid_mean"] > 3000:
        score += 0.3

    # High ZCR variance (irregular)
    if features["zcr_var"] > 0.1:
        score += 0.3

    return min(score, 1.0)


# --- Choice scoring ---

# Built-in keyword → likelihood function mapping
_KEYWORD_FUNCTIONS = {
    "speech": _is_speech_likelihood,
    "parole": _is_speech_likelihood,
    "voix": _is_speech_likelihood,
    "voice": _is_speech_likelihood,
    "music": _is_music_likelihood,
    "musique": _is_music_likelihood,
    "song": _is_music_likelihood,
    "chanson": _is_music_likelihood,
    "noise": _is_noisy_likelihood,
    "bruit": _is_noisy_likelihood,
    "noisy": _is_noisy_likelihood,
    "silence": lambda f: 1.0 if _is_silence(f) else 0.0,
    "quiet": lambda f: max(0, 1.0 - f["rms_mean"] * 10),
    "calme": lambda f: max(0, 1.0 - f["rms_mean"] * 10),
    "loud": lambda f: min(1.0, f["rms_mean"] * 5),
    "fort": lambda f: min(1.0, f["rms_mean"] * 5),
    "fast": lambda f: min(1.0, max(0, (f["tempo"] - 60) / 140)),
    "rapide": lambda f: min(1.0, max(0, (f["tempo"] - 60) / 140)),
    "slow": lambda f: max(0, 1.0 - max(0, (f["tempo"] - 60) / 140)),
    "lent": lambda f: max(0, 1.0 - max(0, (f["tempo"] - 60) / 140)),
    "bright": lambda f: min(1.0, max(0, (f["spectral_centroid_mean"] - 500) / 4000)),
    "brillant": lambda f: min(1.0, max(0, (f["spectral_centroid_mean"] - 500) / 4000)),
    "dark": lambda f: max(0, 1.0 - max(0, (f["spectral_centroid_mean"] - 500) / 4000)),
    "sombre": lambda f: max(0, 1.0 - max(0, (f["spectral_centroid_mean"] - 500) / 4000)),
}


def _find_keyword_in_choice(choice_text):
    """Find a matching keyword function for a choice string."""
    lower = choice_text.lower()
    for keyword, func in _KEYWORD_FUNCTIONS.items():
        if keyword in lower:
            return func
    return None


def _score_choice(features, choice_text):
    """Score how well a choice matches the audio features. Returns 0–1."""
    func = _find_keyword_in_choice(choice_text)
    if func is not None:
        return func(features)

    # Fallback: check for numeric range expectations
    # e.g., "120 BPM", "> 1000 Hz", etc.
    import re
    lower = choice_text.lower()
    # BPM match
    bpm_match = re.search(r'(\d+)\s*bpm', lower)
    if bpm_match:
        target_bpm = int(bpm_match.group(1))
        diff = abs(features["tempo"] - target_bpm)
        return max(0, 1.0 - diff / 100)

    # Hz match
    hz_match = re.search(r'(\d+)\s*hz', lower)
    if hz_match:
        target_hz = int(hz_match.group(1))
        diff = abs(features["spectral_centroid_mean"] - target_hz)
        return max(0, 1.0 - diff / 2000)

    # Generic numeric match (treat as RMS or BPM guess)
    num_match = re.search(r'(\d+\.?\d*)', choice_text)
    if num_match:
        val = float(num_match.group(1))
        # Try matching against tempo first
        if val < 300:  # likely BPM
            diff = abs(features["tempo"] - val)
            return max(0, 1.0 - diff / 80)

    # If nothing matches, return neutral score
    return 0.5


def answer_qcm(features, question, choices):
    """
    Determine the best answer for a QCM based on audio features.

    Args:
        features: dict from extract_features()
        question: the QCM question string
        choices: dict like {"A": "Speech", "B": "Music"}

    Returns:
        dict with {answer, confidence, detail}
    """
    if _is_silence(features):
        # Check if any choice mentions silence
        for key, text in choices.items():
            if any(kw in text.lower() for kw in ["silence", "rien", "nothing", "aucun", "none"]):
                return {
                    "answer": key,
                    "confidence": 0.95,
                    "detail": f"Audio is essentially silent (RMS={features['rms_mean']:.4f})."
                }

    # Score each choice
    scores = {}
    for key, text in choices.items():
        scores[key] = _score_choice(features, text)

    # Pick the best
    if not scores:
        return {
            "answer": "ERROR",
            "confidence": 0.0,
            "detail": "No choices provided."
        }

    best_key = max(scores, key=scores.get)
    best_score = scores[best_key]

    # Confidence: how much better is the best vs the second-best
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        margin = sorted_scores[0] - sorted_scores[1]
        confidence = 0.5 + margin * 0.5  # maps 0–1 margin to 0.5–1.0 confidence
    else:
        confidence = best_score

    confidence = max(0.0, min(1.0, confidence))

    # Build detail string
    detail_parts = [
        f"Tempo: {features['tempo']:.1f} BPM",
        f"RMS: {features['rms_mean']:.4f} (var={features['rms_var']:.4f})",
        f"ZCR: {features['zcr_mean']:.3f} (var={features['zcr_var']:.4f})",
        f"Spectral centroid: {features['spectral_centroid_mean']:.0f} Hz",
        f"MFCCs: 13 coefficients extracted",
        f"Duration: {features['duration']:.1f}s",
        f"Scores: {', '.join(f'{k}={v:.3f}' for k, v in scores.items())}",
    ]
    detail = ". ".join(detail_parts)

    return {
        "answer": best_key,
        "confidence": round(confidence, 4),
        "detail": detail,
    }


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Librosa QCM Inference")
    parser.add_argument("--wav", required=True, help="Path to WAV file")
    parser.add_argument("--question", default="", help="QCM question")
    parser.add_argument("--choices", required=True, help='JSON dict like {"A": "Speech", "B": "Music"}')
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate for analysis")
    args = parser.parse_args()

    try:
        choices = json.loads(args.choices)
    except json.JSONDecodeError:
        print(json.dumps({
            "answer": "ERROR",
            "confidence": 0.0,
            "detail": f"Invalid choices JSON: {args.choices}"
        }))
        sys.exit(0)

    try:
        features = extract_features(args.wav, sr=args.sr)
    except Exception as e:
        print(json.dumps({
            "answer": "ERROR",
            "confidence": 0.0,
            "detail": f"Feature extraction failed: {str(e)}"
        }))
        sys.exit(0)

    result = answer_qcm(features, args.question, choices)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
