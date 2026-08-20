"""
Whisper QCM inference script.

Transcribes audio with OpenAI Whisper, extracts linguistic features
(language, text content, temporal segments, speaker estimation),
then scores each QCM choice against those features to pick the best answer.

Usage:
    python whisper_qcm_inference.py \
        --audio path/to/audio.wav \
        --payload '{"question": "Quelle langue est parlée ?", "choices": {"A": "Français", "B": "Anglais", "C": "Espagnol"}}'

Output: JSON dict with keys: answer (str), confidence (float), detail (str)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

CONFIDENCE_CAP = 0.75  # Probabilistic tier

# ---------------------------------------------------------------------------
# Language name mapping
# ---------------------------------------------------------------------------

_LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "el": "Greek",
    "cs": "Czech", "ro": "Romanian", "hu": "Hungarian", "uk": "Ukrainian",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
    "ca": "Catalan",
}


def _lang_name(code):
    return _LANG_NAMES.get(code, code)


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path, model_size="base"):
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(whisper_result):
    lang_code = whisper_result.get("language", "unknown")
    transcript = whisper_result.get("text", "").strip()
    segments_raw = whisper_result.get("segments", [])

    segments = []
    for seg in segments_raw:
        seg_text = seg.get("text", "").strip()
        if seg_text:
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg_text,
                "duration": seg.get("end", 0) - seg.get("start", 0),
            })

    total_duration = 0
    if segments:
        total_duration = segments[-1]["end"] - segments[0]["start"]

    avg_seg_dur = 0
    if segments:
        avg_seg_dur = sum(s["duration"] for s in segments) / len(segments)

    estimated_speakers = _estimate_speakers(segments, transcript)

    return {
        "language_code": lang_code,
        "language_name": _lang_name(lang_code),
        "transcript": transcript,
        "transcript_lower": transcript.lower(),
        "word_count": len(transcript.split()) if transcript else 0,
        "total_duration_sec": round(total_duration, 2),
        "segment_count": len(segments),
        "segments": segments,
        "avg_segment_duration": round(avg_seg_dur, 2),
        "has_speech": len(transcript) > 0,
        "estimated_speakers": estimated_speakers,
    }


def _estimate_speakers(segments, transcript):
    if not segments:
        return 0

    pauses = 0
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        if gap > 1.5:
            pauses += 1

    conv_markers = [
        r"\byes\b", r"\bno\b", r"\bI think\b", r"\bI agree\b",
        r"\bd'accord\b", r"\boui\b", r"\bnon\b", r"\bje pense\b",
        r"\? ", r"\bexactly\b", r"\bright\b",
    ]
    conv_count = 0
    for marker in conv_markers:
        conv_count += len(re.findall(marker, transcript, re.IGNORECASE))

    speaker_score = pauses + (conv_count // 3)

    if speaker_score >= 4:
        return 3
    elif speaker_score >= 2:
        return 2
    elif speaker_score >= 1 and len(segments) >= 3:
        return 2
    else:
        return 1


# ---------------------------------------------------------------------------
# QCM scoring
# ---------------------------------------------------------------------------

def score_choices(question, choices, features):
    question_lower = question.lower()
    scores = {}

    if any(kw in question_lower for kw in ["langue", "language", "spoken", "parlée", "parlé"]):
        scores = _score_language(choices, features)
    elif any(kw in question_lower for kw in ["parole", "speech", "parle", "talking", "y a-t-il"]):
        scores = _score_speech_presence(choices, features)
    elif any(kw in question_lower for kw in ["long", "court", "short", "duration", "durée", "<", ">"]):
        scores = _score_duration(choices, features)
    elif any(kw in question_lower for kw in ["speaker", "locuteur", "personne", "people", "speaking"]):
        scores = _score_speaker_count(choices, features)
    else:
        scores = _score_topic(choices, features, question)

    return scores


def _score_language(choices, features):
    scores = {}
    detected = features["language_code"]
    detected_name = features["language_name"].lower()

    for key, choice_text in choices.items():
        choice_lower = choice_text.lower()
        detail = f"Detected language: {features['language_name']} ({detected})"

        if detected_name in choice_lower or detected in choice_lower:
            scores[key] = (0.95, f"{detail}. Choice '{key}' matches detected language.")
        elif any(kw in choice_lower for kw in ["autre", "other", "unknown", "none"]):
            scores[key] = (0.05, f"{detail}. Choice '{key}' is 'other' fallback.")
        else:
            lang_aliases = {
                "français": ["french", "francais", "français"],
                "anglais": ["english", "anglais"],
                "espagnol": ["spanish", "español", "espagnol"],
                "allemand": ["german", "allemand"],
                "italien": ["italian", "italien"],
                "portugais": ["portuguese", "português", "portugais"],
            }

            match = False
            for lang, aliases in lang_aliases.items():
                if detected_name in aliases or detected == lang[:2]:
                    if any(a in choice_lower for a in aliases):
                        match = True
                        break
            if match:
                scores[key] = (0.90, f"{detail}. Choice '{key}' matches via alias.")
            else:
                scores[key] = (0.10, f"{detail}. No match for choice '{key}'.")

    return scores


def _score_speech_presence(choices, features):
    scores = {}
    has_speech = features["has_speech"]
    word_count = features["word_count"]

    for key, choice_text in choices.items():
        choice_lower = choice_text.lower()

        is_yes = any(kw in choice_lower for kw in ["oui", "yes", "present", "speech"])
        is_no = any(kw in choice_lower for kw in ["non", "no", "absent", "aucun", "aucune", "silence"])

        if is_yes and has_speech:
            scores[key] = (0.95, f"Speech detected ({word_count} words). Choice '{key}' = Yes.")
        elif is_yes and not has_speech:
            scores[key] = (0.10, f"No speech detected. Choice '{key}' = Yes (unlikely).")
        elif is_no and has_speech:
            scores[key] = (0.10, f"Speech detected ({word_count} words). Choice '{key}' = No (unlikely).")
        elif is_no and not has_speech:
            scores[key] = (0.95, f"No speech detected. Choice '{key}' = No.")
        else:
            scores[key] = (0.50, f"Speech presence unclear for choice '{key}'.")

    return scores


def _score_duration(choices, features):
    scores = {}
    total_dur = features["total_duration_sec"]
    word_count = features["word_count"]

    for key, choice_text in choices.items():
        choice_lower = choice_text.lower()
        detail = f"Total duration: {total_dur:.1f}s, word count: {word_count}"

        threshold = _parse_duration_threshold(choice_lower)

        if threshold is not None:
            if total_dur <= threshold:
                scores[key] = (0.90, f"{detail}. Duration ({total_dur:.1f}s) <= threshold ({threshold}s). Choice '{key}' matches.")
            else:
                scores[key] = (0.15, f"{detail}. Duration ({total_dur:.1f}s) > threshold ({threshold}s). Choice '{key}' unlikely.")
        else:
            is_short = any(kw in choice_lower for kw in ["court", "short", "<", "less", "moins", "petit"])
            is_long = any(kw in choice_lower for kw in ["long", ">", "more", "plus", "grand", "étendu", "longue"])

            if is_short and total_dur < 30:
                scores[key] = (0.85, f"{detail}. Short audio. Choice '{key}' matches.")
            elif is_short and total_dur >= 30:
                scores[key] = (0.15, f"{detail}. Not short. Choice '{key}' unlikely.")
            elif is_long and total_dur >= 30:
                scores[key] = (0.85, f"{detail}. Long audio. Choice '{key}' matches.")
            elif is_long and total_dur < 30:
                scores[key] = (0.15, f"{detail}. Not long. Choice '{key}' unlikely.")
            else:
                scores[key] = (0.50, f"{detail}. Could not classify duration for choice '{key}'.")

    return scores


def _parse_duration_threshold(choice_text):
    patterns = [
        r'[<≤]\s*(\d+)\s*s',
        r'[<≤]\s*(\d+)',
        r'(\d+)\s*sec',
        r'(\d+)\s*second',
        r'moins\s+de\s+(\d+)',
        r'under\s+(\d+)',
        r'less\s+than\s+(\d+)',
        r'(\d+)\s*min',
    ]
    for pattern in patterns:
        m = re.search(pattern, choice_text)
        if m:
            val = int(m.group(1))
            if "min" in pattern:
                val *= 60
            return val
    return None


def _score_speaker_count(choices, features):
    scores = {}
    estimated = features["estimated_speakers"]

    for key, choice_text in choices.items():
        choice_lower = choice_text.lower()
        detail = f"Estimated speakers: {estimated}"

        number_match = re.search(r'(\d+)', choice_text)
        if number_match:
            choice_num = int(number_match.group(1))
            if choice_num == estimated:
                scores[key] = (0.80, f"{detail}. Choice '{key}' matches estimated count.")
            elif abs(choice_num - estimated) <= 1:
                scores[key] = (0.40, f"{detail}. Choice '{key}' is close (±1).")
            else:
                scores[key] = (0.10, f"{detail}. Choice '{key}' does not match.")
        elif any(kw in choice_lower for kw in ["un", "one", "single", "seul", "solo"]):
            scores[key] = (0.80 if estimated == 1 else 0.10, f"{detail}. Choice '{key}' = one speaker.")
        elif any(kw in choice_lower for kw in ["deux", "two", "couple", "pair"]):
            scores[key] = (0.80 if estimated == 2 else 0.10, f"{detail}. Choice '{key}' = two speakers.")
        elif any(kw in choice_lower for kw in ["trois", "three", "plusieurs", "several", "many", "multiple", "more"]):
            scores[key] = (0.80 if estimated >= 3 else 0.10, f"{detail}. Choice '{key}' = three+ speakers.")
        else:
            scores[key] = (0.50, f"{detail}. Could not parse speaker count for choice '{key}'.")

    return scores


def _score_topic(choices, features, question):
    scores = {}
    transcript_lower = features["transcript_lower"]
    word_count = features["word_count"]

    if not transcript_lower:
        for key in choices:
            scores[key] = (0.25, "No speech detected. Cannot determine topic.")
        return scores

    for key, choice_text in choices.items():
        choice_lower = choice_text.lower()

        keywords = _extract_keywords(choice_lower)

        if not keywords:
            scores[key] = (0.30, f"Choice '{key}' has no meaningful keywords to match.")
            continue

        match_count = 0
        matched_words = []
        for kw in keywords:
            if kw in transcript_lower:
                match_count += 1
                matched_words.append(kw)

        match_ratio = match_count / len(keywords) if keywords else 0
        length_factor = min(word_count / 10, 1.0) if word_count > 0 else 0
        raw_score = match_ratio * (0.5 + 0.5 * length_factor)

        if match_ratio > 0.5:
            confidence = min(raw_score + 0.3, 0.95)
            detail = f"Matched {match_count}/{len(keywords)} keywords: {', '.join(matched_words)}. Choice '{key}' likely."
        elif match_ratio > 0:
            confidence = raw_score + 0.1
            detail = f"Partial match ({match_count}/{len(keywords)} keywords): {', '.join(matched_words)}. Choice '{key}' possible."
        else:
            confidence = 0.05
            detail = f"No keyword matches for choice '{key}'."

        scores[key] = (round(confidence, 4), detail)

    return scores


def _extract_keywords(text):
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "of", "to", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "but", "or", "because", "if", "while", "about",
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "sont",
        "était", "étaient", "être", "avoir", "a", "ont", "dans", "pour", "sur",
        "avec", "ce", "cette", "ces", "son", "sa", "ses", "notre", "votre",
        "leur", "ils", "elles", "il", "elle", "nous", "vous", "je", "tu",
        "que", "qui", "dont", "où", "mais", "ou", "donc", "car", "ni", "pas",
        "ne", "plus", "moins", "tres", "bien", "tout", "tous", "toute",
    }

    words = re.findall(r'[a-zàâäéèêëïîôùûüÿçœæ]{3,}', text.lower())
    return [w for w in words if w not in stop_words]

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_whisper_qcm(audio_path, question, choices, model_size="base"):
    whisper_result = transcribe_audio(audio_path, model_size=model_size)
    features = extract_features(whisper_result)
    scores = score_choices(question, choices, features)

    best_key = max(scores, key=lambda k: scores[k][0])
    best_score, best_detail = scores[best_key]

    all_scores = [s for s, _ in scores.values()]
    if len(all_scores) > 1:
        score_range = max(all_scores) - min(all_scores)
        if score_range < 0.1:
            best_score = max(best_score, 0.35)

    best_score = min(best_score, CONFIDENCE_CAP)  # Probabilistic tier cap

    return {
        "answer": best_key,
        "confidence": round(float(best_score), 4),
        "detail": best_detail,
    }


def main():
    parser = argparse.ArgumentParser(description="Whisper QCM – Answer multiple-choice questions about audio using Whisper transcription")
    parser.add_argument("--audio", required=True, help="Path to audio file (WAV, MP3, FLAC, M4A, OGG)")
    parser.add_argument("--payload", required=True, help='JSON: {"question": "...", "choices": {"A": "...", "B": "...", ...}}')
    parser.add_argument("--model", default="base", help="Whisper model size (default: base)")
    args = parser.parse_args()

    payload = json.loads(args.payload)
    result = run_whisper_qcm(args.audio, payload["question"], payload["choices"], model_size=args.model)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
