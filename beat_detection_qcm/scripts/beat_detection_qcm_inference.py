#!/usr/bin/env python3
"""
Beat Detection QCM Inference Script
Uses Librosa to detect tempo and beats, answering QCM questions based on:
- BPM (tempo in beats per minute)
- number of beats detected
- tempo regularity (standard deviation of inter-beat intervals)
- timestamps of the first detected beats
"""

import argparse
import json
import sys
import os
import re
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

def process_qcm(audio_path: str, payload: dict) -> dict:
    """
    Process a QCM request for beat detection.
    
    Args:
        audio_path: Path to the audio file.
        payload: Dictionary containing 'question' and 'choices'.
        
    Returns:
        Dictionary with 'answer', 'confidence', 'detail'.
    """
    question = payload.get("question", "")
    choices = payload.get("choices", {})
    
    default_result = {
        "answer": "",
        "confidence": 0.0,
        "detail": "Échec de l'analyse audio ou fichier introuvable."
    }
    
    if not os.path.exists(audio_path):
        return {**default_result, "detail": f"Fichier audio introuvable: {audio_path}"}
        
    if librosa is None:
        return {**default_result, "detail": "Librosa ou soundfile non installé."}
        
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Fix for librosa >= 0.10: beat_track() returns an np.ndarray for tempo
        bpm = float(np.atleast_1d(tempo)[0])
        
        # Convert beat frames to timestamps
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        num_beats = len(beat_times)
        
        # Calculate inter-beat intervals and their standard deviation
        if num_beats > 1:
            inter_beat_intervals = np.diff(beat_times)
            tempo_regularity = float(np.std(inter_beat_intervals))
        else:
            tempo_regularity = 0.0
            
        first_beats = beat_times[:5].tolist()
        
        features = {
            "bpm": bpm,
            "num_beats": num_beats,
            "tempo_regularity": tempo_regularity,
            "first_beats": first_beats
        }
        
        best_choice = ""
        max_score = -1.0
        best_detail = ""
        
        for choice_key, choice_text in choices.items():
            score = 0.0
            choice_lower = choice_text.lower()
            
            # BPM matching
            if "bpm" in choice_lower or "tempo" in choice_lower:
                numbers = re.findall(r'\d+', choice_text)
                for num in numbers:
                    if abs(float(num) - bpm) < 15:
                        score += 0.8
                        
            # Regularity matching
            if tempo_regularity < 0.10 and any(word in choice_lower for word in ["régulier", "regular", "stable", "constant"]):
                score += 0.8
            elif tempo_regularity >= 0.10 and any(word in choice_lower for word in ["irrégulier", "irregular", "variable", "changeant"]):
                score += 0.8
                
            # Beat count matching
            numbers = re.findall(r'\d+', choice_text)
            for num in numbers:
                if int(num) == num_beats:
                    score += 0.8
                    
            # Fallback: if no specific match, give a small base score to allow selection
            if score == 0.0:
                score = 0.1
                
            if score > max_score:
                max_score = score
                best_choice = choice_key
                best_detail = (f"BPM: {bpm:.1f}, Beats: {num_beats}, "
                               f"Régularité (std): {tempo_regularity:.3f}s, "
                               f"Premiers beats: {first_beats[:3]}")
                
        if not best_choice and choices:
            best_choice = list(choices.keys())[0]
            max_score = 0.5
            best_detail = (f"Analyse effectuée. BPM: {bpm:.1f}, Beats: {num_beats}, "
                           f"Régularité: {tempo_regularity:.3f}s. Aucune correspondance forte.")
            
        # Cap confidence at 0.90 for Analytic tier
        confidence = min(0.90, max_score)
        
        return {
            "answer": best_choice,
            "confidence": confidence,
            "detail": best_detail
        }
        
    except Exception as e:
        return {
            "answer": "",
            "confidence": 0.0,
            "detail": f"Erreur silencieuse lors de l'analyse: {str(e)}"
        }

def main():
    parser = argparse.ArgumentParser(description="Beat Detection QCM Inference")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--payload", required=True, help="JSON string containing 'question' and 'choices'")
    
    args = parser.parse_args()
    
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError:
        print(json.dumps({"answer": "", "confidence": 0.0, "detail": "Payload JSON invalide"}))
        sys.exit(1)
        
    result = process_qcm(args.audio, payload)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()