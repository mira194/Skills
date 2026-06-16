#!/usr/bin/env python3
"""
Spectral Features QCM Inference Script
Uses Librosa to analyze spectral features, answering QCM questions based on:
- mean spectral centroid (brightness)
- mean MFCCs (timbre)
- mean spectral bandwidth (bandwidth)
- mean spectral rolloff (rolloff)
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
    Process a QCM request for spectral features analysis.
    
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
        
        # 1. Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = float(np.mean(spectral_centroid))
        
        # 2. MFCCs (timbre) - mean of first 3 MFCCs (excluding 0th which is energy)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)[1:] # 1, 2, 3
        mean_mfccs = [float(np.mean(m)) for m in mfccs]
        
        # 3. Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        mean_bandwidth = float(np.mean(spectral_bandwidth))
        
        # 4. Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        mean_rolloff = float(np.mean(spectral_rolloff))
        
        best_choice = ""
        max_score = -1.0
        best_detail = ""
        
        for choice_key, choice_text in choices.items():
            score = 0.0
            choice_lower = choice_text.lower()
            
            # Brightness matching
            if any(word in choice_lower for word in ["brillant", "bright", "aigu", "high frequency", "clair"]):
                if mean_centroid > 2000:
                    score += 0.8
            elif any(word in choice_lower for word in ["sombre", "dark", "grave", "low frequency", "mat"]):
                if mean_centroid < 1000:
                    score += 0.8
                    
            # Bandwidth matching
            if any(word in choice_lower for word in ["large", "wide", "étendu", "riche"]):
                if mean_bandwidth > 2000:
                    score += 0.8
            elif any(word in choice_lower for word in ["étroit", "narrow", "concentré", "pur"]):
                if mean_bandwidth < 1000:
                    score += 0.8
                    
            # MFCC/Timbre matching (basic)
            if any(word in choice_lower for word in ["timbre", "voix", "vocal", "parole"]):
                if mean_mfccs[0] < -100: # rough heuristic for vocal presence
                    score += 0.5
                    
            # Fallback: if no specific match, give a small base score to allow selection
            if score == 0.0:
                score = 0.1
                
            if score > max_score:
                max_score = score
                best_choice = choice_key
                best_detail = (f"Centroïde: {mean_centroid:.1f}Hz, "
                               f"MFCCs moy: {[round(m, 2) for m in mean_mfccs]}, "
                               f"Bande passante: {mean_bandwidth:.1f}Hz, "
                               f"Rolloff: {mean_rolloff:.1f}Hz")
                
        if not best_choice and choices:
            best_choice = list(choices.keys())[0]
            max_score = 0.5
            best_detail = (f"Analyse effectuée. Centroïde: {mean_centroid:.1f}Hz, "
                           f"Bande passante: {mean_bandwidth:.1f}Hz, "
                           f"Rolloff: {mean_rolloff:.1f}Hz. Aucune correspondance forte.")
            
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
    parser = argparse.ArgumentParser(description="Spectral Features QCM Inference")
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