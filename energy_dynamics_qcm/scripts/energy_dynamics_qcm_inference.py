#!/usr/bin/env python3
"""
Energy Dynamics QCM Inference Script
Uses Librosa to analyze RMS energy and dynamics, answering QCM questions based on:
- Mean RMS (global energy level)
- Min/max RMS
- Dynamic range in dB (difference between max and min RMS)
- Temporal evolution of energy (increasing, decreasing, stable, variable)
"""

import argparse
import json
import sys
import os
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

def process_qcm(audio_path: str, payload: dict) -> dict:
    """
    Process a QCM request for energy dynamics.
    
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
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Mean, min, max RMS
        rms_mean = float(np.mean(rms))
        rms_min = float(np.min(rms))
        rms_max = float(np.max(rms))
        
        # Dynamic range in dB
        # Avoid log(0) by adding a small epsilon
        eps = 1e-10
        rms_min_db = 20 * np.log10(rms_min + eps)
        rms_max_db = 20 * np.log10(rms_max + eps)
        dynamic_range_db = float(rms_max_db - rms_min_db)
        
        # Temporal evolution: split into 4 segments and compare means
        num_segments = 4
        segment_size = len(rms) // num_segments
        segment_means = []
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else len(rms)
            segment_means.append(float(np.mean(rms[start:end])))
            
        # Determine evolution
        first_half_mean = np.mean(segment_means[:2])
        second_half_mean = np.mean(segment_means[2:])
        overall_std = float(np.std(segment_means))
        
        if overall_std < 0.01 * rms_mean:
            evolution = "stable"
        elif second_half_mean > first_half_mean * 1.2:
            evolution = "croissante"
        elif second_half_mean < first_half_mean * 0.8:
            evolution = "décroissante"
        else:
            evolution = "variable"
            
        features = {
            "rms_mean": rms_mean,
            "rms_min": rms_min,
            "rms_max": rms_max,
            "dynamic_range_db": dynamic_range_db,
            "evolution": evolution,
            "segment_means": segment_means
        }
        
        best_choice = ""
        max_score = -1.0
        best_detail = ""
        
        for choice_key, choice_text in choices.items():
            score = 0.0
            choice_lower = choice_text.lower()
            
            # Energy level matching
            if rms_mean < 0.01 and any(word in choice_lower for word in ["silence", "faible", "quiet", "low", "calme"]):
                score += 0.8
            elif rms_mean > 0.1 and any(word in choice_lower for word in ["fort", "loud", "élevé", "high", "intense"]):
                score += 0.8
                
            # Dynamic range matching
            if dynamic_range_db > 20 and any(word in choice_lower for word in ["dynamique", "varié", "variable", "wide", "large"]):
                score += 0.8
            elif dynamic_range_db < 10 and any(word in choice_lower for word in ["compressé", "stable", "constant", "compressed", "flat"]):
                score += 0.8
                
            # Evolution matching
            if evolution == "croissante" and any(word in choice_lower for word in ["croissante", "increasing", "monte", "crescendo"]):
                score += 0.8
            elif evolution == "décroissante" and any(word in choice_lower for word in ["décroissante", "decreasing", "descend", "diminuendo"]):
                score += 0.8
            elif evolution == "stable" and any(word in choice_lower for word in ["stable", "constant", "uniforme"]):
                score += 0.8
            elif evolution == "variable" and any(word in choice_lower for word in ["variable", "fluctuant", "changeant"]):
                score += 0.8
                
            # Fallback: if no specific match, give a small base score to allow selection
            if score == 0.0:
                score = 0.1
                
            if score > max_score:
                max_score = score
                best_choice = choice_key
                best_detail = (f"RMS moyen: {rms_mean:.4f}, Min: {rms_min:.4f}, Max: {rms_max:.4f}, "
                               f"Plage dynamique: {dynamic_range_db:.1f} dB, Évolution: {evolution}")
                
        if not best_choice and choices:
            best_choice = list(choices.keys())[0]
            max_score = 0.5
            best_detail = (f"Analyse effectuée. RMS moyen: {rms_mean:.4f}, Plage dynamique: {dynamic_range_db:.1f} dB, "
                           f"Évolution: {evolution}. Aucune correspondance forte.")
            
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
    parser = argparse.ArgumentParser(description="Energy Dynamics QCM Inference")
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