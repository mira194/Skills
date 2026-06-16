import argparse
import json
import librosa
import numpy as np
import sys
import os

def estimate_rt60(y, sr):
    """Estimate RT60 (reverberation time) using energy decay heuristic."""
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Normalize RMS
    max_rms = np.max(rms)
    if max_rms < 1e-8:
        return 0.0
    rms = rms / max_rms
    
    # Find the first frame where energy drops below -60dB (0.001)
    idx = np.where(rms < 0.001)[0]
    if len(idx) > 0:
        decay_frames = idx[0]
    else:
        # Fallback: estimate from the overall decay slope or cap at max length
        decay_frames = len(rms)
        
    rt60 = (decay_frames * hop_length) / sr
    return min(rt60, 5.0)

def estimate_noise_floor(y, sr):
    """Estimate background noise floor as the 10th percentile of RMS energy."""
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    noise_floor = float(np.percentile(rms, 10))
    return noise_floor

def classify_environment(rt60, noise_floor):
    """Heuristic classification of the acoustic environment."""
    if rt60 < 0.3:
        if noise_floor < 0.02:
            return "intérieur sec"
        else:
            return "intérieur bruyant"
    elif rt60 < 1.5:
        return "salle réverbérante"
    else:
        return "extérieur ou très réverbérant"

def process_qcm(audio_path, payload):
    """
    Process QCM for environment detection.
    
    Args:
        audio_path (str): Path to the audio file.
        payload (dict): Dictionary containing 'question' and 'choices'.
        
    Returns:
        dict: Result with 'answer', 'confidence', and 'detail'.
    """
    try:
        question = payload.get("question", "").lower()
        choices = payload.get("choices", {})
        
        if not choices:
            return {"answer": "", "confidence": 0.0, "detail": "Aucun choix fourni"}
            
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Extract features
        rt60 = estimate_rt60(y, sr)
        noise_floor = estimate_noise_floor(y, sr)
        env_class = classify_environment(rt60, noise_floor)
        
        detail = f"RT60 estimé: {rt60:.2f}s, Niveau de bruit: {noise_floor:.4f}, Environnement: {env_class}"
        
        # Score each choice
        best_choice = None
        best_score = -1.0
        
        for key, choice_text in choices.items():
            choice_lower = choice_text.lower()
            score = 0.0
            
            # Match environment class
            if any(word in choice_lower for word in ["sec", "dry", "intérieur sec"]):
                if rt60 < 0.4:
                    score += 0.4
            if any(word in choice_lower for word in ["réverbérant", "reverberant", "salle", "hall"]):
                if 0.3 <= rt60 < 1.5:
                    score += 0.4
            if any(word in choice_lower for word in ["extérieur", "outdoor", "dehors"]):
                if rt60 >= 1.0 or noise_floor > 0.03:
                    score += 0.4
                    
            # Match noise level
            if any(word in choice_lower for word in ["bruyant", "noisy", "bruit", "noise"]):
                if noise_floor > 0.02:
                    score += 0.3
            if any(word in choice_lower for word in ["calme", "quiet", "silence", "sec"]):
                if noise_floor < 0.015:
                    score += 0.3
                    
            # Direct string match fallback
            if env_class in choice_lower:
                score += 0.5
                
            if score > best_score:
                best_score = score
                best_choice = key
                
        if best_choice is None:
            best_choice = list(choices.keys())[0]
            best_score = 0.1
            
        # Cap confidence at 0.60 for Heuristic tier
        confidence = min(best_score, 0.60)
        
        return {
            "answer": best_choice,
            "confidence": round(confidence, 2),
            "detail": detail
        }
    except Exception:
        # Silent error handling as requested
        return {
            "answer": "",
            "confidence": 0.0,
            "detail": "Échec de l'analyse"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Environment Detection QCM Inference")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--payload", required=True, help="JSON string containing 'question' and 'choices'")
    args = parser.parse_args()
    
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError:
        payload = {"question": "", "choices": {}}
        
    result = process_qcm(args.audio, payload)
    print(json.dumps(result))