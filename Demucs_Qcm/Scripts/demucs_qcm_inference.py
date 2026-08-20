#!/usr/bin/env python3
"""Demucs QCM – Source separation based audio QCM answering.
Separates audio into 4 stems (vocals, drums, bass, other) using Demucs htdemucs,
computes RMS energy per stem, and maps the results to QCM choices.
Usage:
    python demucs_qcm_inference.py \
        --audio path/to/audio.wav \
        --payload '{"question": "Which source dominates?", "choices": {"A": "Vocals", "B": "Drums"}}'
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional
import numpy as np
warnings.filterwarnings("ignore")

CONFIDENCE_CAP = 0.75  # Probabilistic tier

# ---------------------------------------------------------------------------
# Demucs separation
# ---------------------------------------------------------------------------
def _load_demucs_model(device: str = "cpu"):
    try:
        from demucs.api import Pretrained
        model = Pretrained("htdemucs", device=device)
        return model
    except Exception:
        pass

    try:
        from demucs.pretrained import get_model
        model = get_model("htdemucs")
        if device == "cuda":
            import torch
            model = model.to(device)
        return model
    except Exception:
        pass

    raise RuntimeError(
        "Could not load Demucs model. Ensure demucs is installed: pip install demucs"
    )

def _separate_audio(model, audio_path: str, device: str = "cpu") -> Dict[str, np.ndarray]:
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 44100:
        resampler = torchaudio.transforms.Resample(sr, 44100)
        waveform = resampler(waveform)
        sr = 44100

    waveform = waveform.to(device)

    try:
        if hasattr(model, 'separate'):
            stems_dict = model.separate(waveform)
            result = {}
            for name, stem_tensor in stems_dict.items():
                result[name.lower()] = stem_tensor.cpu().numpy().flatten()
            return result

        # Older API: apply_model
        from demucs.apply import apply_model
        with torch.no_grad():
            separated = apply_model(model, waveform.unsqueeze(0))

        stem_names = model.sources
        result = {}
        for name, stem_tensor in zip(stem_names, separated[0]):
            result[name.lower()] = stem_tensor.cpu().numpy().flatten()
        return result

    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {e}")

def separate_and_analyze(audio_path: str, device: str = "cpu") -> Dict[str, float]:
    model = _load_demucs_model(device)
    stems = _separate_audio(model, audio_path, device)

    rms_energies = {}
    for stem_name in ["vocals", "drums", "bass", "other"]:
        if stem_name in stems:
            stem_audio = stems[stem_name]
            rms = float(np.sqrt(np.mean(stem_audio ** 2)))
            rms_energies[stem_name] = rms
        else:
            rms_energies[stem_name] = 0.0

    return rms_energies

# ---------------------------------------------------------------------------
# QCM answering logic
# ---------------------------------------------------------------------------
_STEM_KEYWORD_MAP = {
    "vocals": ["vocals"], "vocal": ["vocals"], "voice": ["vocals"],
    "speech": ["vocals"], "parole": ["vocals"], "chant": ["vocals"], "singing": ["vocals"],
    "drums": ["drums"], "drum": ["drums"], "percussion": ["drums"],
    "beat": ["drums"], "rythme": ["drums"], "batterie": ["drums"],
    "bass": ["bass"], "basse": ["bass"], "low": ["bass"], "grave": ["bass"],
    "other": ["other"], "autre": ["other"], "instrument": ["other"],
    "music": ["other"], "musique": ["other"], "background": ["other"],
    "ambient": ["other"], "accompaniment": ["other"], "melody": ["other"],
}

def _keyword_to_stems(keyword: str) -> list:
    keyword_lower = keyword.lower().strip()
    if keyword_lower in _STEM_KEYWORD_MAP:
        return _STEM_KEYWORD_MAP[keyword_lower]
    for kw, stems in _STEM_KEYWORD_MAP.items():
        if kw in keyword_lower or keyword_lower in kw:
            return stems
    return []

def _compute_choice_score(choice_text: str, rms_energies: Dict[str, float]) -> float:
    stems = _keyword_to_stems(choice_text)
    if not stems:
        return sum(rms_energies.values())
    score = sum(rms_energies.get(s, 0.0) for s in stems)
    return score

def _detect_pattern(question: str, choices: Dict[str, str], rms_energies: Dict[str, float]) -> Optional[Dict]:
    q_lower = question.lower()
    choice_values = {k: v.lower() for k, v in choices.items()}

    if any(w in q_lower for w in ["dominant", "prominent", "loudest", "principal", "plus fort"]):
        best_choice = None
        best_score = -1
        for key, text in choice_values.items():
            stems = _keyword_to_stems(text)
            score = sum(rms_energies.get(s, 0.0) for s in stems) if stems else sum(rms_energies.values())
            if score > best_score:
                best_score = score
                best_choice = key
        if best_choice:
            return {
                "pattern": "dominant_source",
                "answer": best_choice,
                "scores": {k: _compute_choice_score(v, rms_energies) for k, v in choices.items()},
            }

    if any(w in q_lower for w in ["speech", "music", "parole", "musique", "vocal", "instrumental"]):
        vocal_energy = rms_energies.get("vocals", 0.0)
        music_energy = (rms_energies.get("drums", 0.0) + rms_energies.get("bass", 0.0) + rms_energies.get("other", 0.0))
        for key, text in choice_values.items():
            if any(w in text for w in ["speech", "parole", "voice", "vocal"]):
                if vocal_energy > music_energy:
                    return {"pattern": "speech_vs_music", "answer": key, "scores": {"speech": vocal_energy, "music": music_energy}}
            if any(w in text for w in ["music", "musique", "instrumental"]):
                if music_energy >= vocal_energy:
                    return {"pattern": "speech_vs_music", "answer": key, "scores": {"speech": vocal_energy, "music": music_energy}}

    if any(w in q_lower for w in ["present", "detect", "contain", "contains", "y a", "est-ce"]):
        for key, text in choice_values.items():
            stems = _keyword_to_stems(text)
            if stems:
                energy = sum(rms_energies.get(s, 0.0) for s in stems)
                if energy > 0.01:
                    return {"pattern": "presence", "answer": key, "scores": {text: energy}}

    return None

def answer_qcm(stems: Dict[str, float], question: str, choices: Dict[str, str]) -> Dict:
    pattern_result = _detect_pattern(question, choices, stems)
    if pattern_result:
        answer = pattern_result["answer"]
        scores = pattern_result["scores"]
        total = sum(scores.values()) if scores else 1.0
        confidence = scores.get(answer, 0.0) / total if total > 0 else 0.0
        confidence = min(confidence, CONFIDENCE_CAP)  # Probabilistic tier cap
        detail = f"Pattern '{pattern_result['pattern']}' matched. "
        detail += f"Stem energies: {', '.join(f'{k}={v:.4f}' for k, v in stems.items())}. "
        detail += f"Scores: {scores}. "
        return {"answer": answer, "confidence": round(confidence, 4), "detail": detail}

    choice_scores = {}
    for key, text in choices.items():
        choice_scores[key] = _compute_choice_score(text, stems)

    if len(set(choice_scores.values())) <= 1:
        sorted_stems = sorted(stems.items(), key=lambda x: x[1], reverse=True)
        if sorted_stems:
            top_stem = sorted_stems[0][0]
            for key, text in choices.items():
                stems_for_choice = _keyword_to_stems(text)
                if top_stem in stems_for_choice:
                    choice_scores[key] = stems[top_stem] + 0.001

    best_choice = max(choice_scores, key=choice_scores.get)
    best_score = choice_scores[best_choice]
    total_score = sum(choice_scores.values()) if sum(choice_scores.values()) > 0 else 1.0
    confidence = best_score / total_score
    confidence = min(confidence, CONFIDENCE_CAP)  # Probabilistic tier cap

    detail = f"Stem energies: {', '.join(f'{k}={v:.4f}' for k, v in stems.items())}. "
    detail += f"Choice scores: {choice_scores}. "
    detail += f"Best match: {best_choice} (choice: {choices[best_choice]})."

    return {"answer": best_choice, "confidence": round(confidence, 4), "detail": detail}

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Demucs QCM – Source separation audio classification")
    parser.add_argument("--audio", required=True, help="Path to audio file (WAV/MP3)")
    parser.add_argument("--payload", required=True, help='JSON string: {"question": ..., "choices": {...}}')
    parser.add_argument("--device", default="cpu", help="Device for Demucs (cpu or cuda)")
    args = parser.parse_args()

    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid payload JSON: {e}"}))
        sys.exit(1)

    try:
        rms_energies = separate_and_analyze(args.audio, device=args.device)
    except Exception as e:
        print(json.dumps({"error": f"Separation failed: {e}"}))
        sys.exit(1)

    result = answer_qcm(rms_energies, payload["question"], payload["choices"])
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
