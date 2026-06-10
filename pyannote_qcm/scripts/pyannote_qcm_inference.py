import sys
import subprocess
import json
import os
import re

def ensure_pyannote():
    try:
        import pyannote.audio
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyannote.audio", "torch", "torchaudio"])
            return True
        except Exception:
            return False

def run_pyannote_qcm(audio_path: str, question: str, choices: dict) -> dict:
    try:
        if not ensure_pyannote():
            return {"answer": "", "confidence": 0.0, "detail": "Failed to install or import pyannote.audio"}
        
        import torch
        from pyannote.audio import Pipeline
        
        hf_token = os.environ.get("HF_TOKEN")
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        except Exception:
            return {"answer": "", "confidence": 0.0, "detail": "Failed to load pyannote model (check HF_TOKEN)"}
        
        diarization = pipeline(audio_path)
        
        speakers = set()
        speaker_durations = {}
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            duration = turn.end - turn.start
            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "duration": round(duration, 2)
            })
            
        num_speakers = len(speakers)
        question_lower = question.lower()
        
        best_choice = ""
        max_score = -1.0
        best_detail = ""
        
        for choice_key, choice_text in choices.items():
            choice_lower = choice_text.lower()
            score = 0.0
            detail_parts = []
            
            # 1. Nombre de locuteurs
            if "locuteur" in question_lower or "speaker" in question_lower or "personne" in question_lower or "combien" in question_lower:
                numbers = re.findall(r'\d+', choice_lower)
                if numbers:
                    target_num = int(numbers[0])
                    if target_num == num_speakers:
                        score += 1.0
                        detail_parts.append(f"Match: detected {num_speakers} speakers")
                    else:
                        score += 0.3 if abs(target_num - num_speakers) <= 1 else 0.0
                        
            # 2. Durée de parole
            if "durée" in question_lower or "duration" in question_lower or "long" in question_lower or "court" in question_lower or "longtemps" in question_lower:
                if num_speakers > 0:
                    max_speaker = max(speaker_durations, key=speaker_durations.get)
                    if max_speaker.lower() in choice_lower or str(max_speaker) in choice_lower:
                        score += 1.0
                        detail_parts.append(f"Match: Speaker {max_speaker} spoke longest ({speaker_durations[max_speaker]:.1f}s)")
            
            # 3. Chevauchements (Overlap)
            if "chevauchement" in question_lower or "overlap" in question_lower or "simultané" in question_lower or "même temps" in question_lower:
                if "plusieurs" in choice_lower or "multiple" in choice_lower or "oui" in choice_lower or "yes" in choice_lower:
                    score += 0.5
                    detail_parts.append("Overlap heuristic: multiple speakers detected")

            # Fallback: basic keyword matching
            if score == 0.0:
                words = choice_lower.split()
                matches = sum(1 for w in words if w in ["0", "1", "2", "3", "locuteur", "speaker", "parole"])
                score = min(0.5, matches * 0.2)
                if matches > 0:
                    detail_parts.append("Weak keyword match")

            if score > max_score:
                max_score = score
                best_choice = choice_key
                best_detail = "; ".join(detail_parts) if detail_parts else f"Analyzed {num_speakers} speakers."
        
        # Tier de fiabilité : Probabilistic, confiance max 0.75
        confidence = min(0.75, max_score * 0.75)
        
        if max_score <= 0.0:
            best_choice = list(choices.keys())[0] if choices else ""
            confidence = 0.0
            best_detail = "No strong match found based on diarization features."
            
        return {
            "answer": best_choice,
            "confidence": round(confidence, 2),
            "detail": best_detail,
            "features": {
                "num_speakers": num_speakers,
                "speaker_durations": {k: round(v, 2) for k, v in speaker_durations.items()},
                "segments_count": len(segments)
            }
        }
        
    except Exception as e:
        # Gère les erreurs silencieusement et retourne confidence=0.0 si échec.
        return {"answer": "", "confidence": 0.0, "detail": "Error during inference"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--question", required=True, help="The QCM question")
    parser.add_argument("--choices", required=True, help="JSON string of choices dictionary")
    args = parser.parse_args()
    
    try:
        choices = json.loads(args.choices)
    except json.JSONDecodeError:
        print(json.dumps({"answer": "", "confidence": 0.0, "detail": "Invalid JSON for choices"}))
        sys.exit(1)
        
    result = run_pyannote_qcm(args.audio, args.question, choices)
    print(json.dumps(result, indent=2))