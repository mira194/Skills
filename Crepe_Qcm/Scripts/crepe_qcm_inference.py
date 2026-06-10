import numpy as np
import json
import argparse
import os
import sys

def ensure_crepe():
    try:
        import crepe
        return crepe
    except ImportError:
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "crepe", "librosa", "scipy"])
            import crepe
            return crepe
        except Exception:
            return None

def analyze_audio(wav_path):
    try:
        crepe = ensure_crepe()
        if crepe is None:
            raise ImportError("Failed to install or import crepe")
        
        import librosa
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Load audio at 16kHz (standard for crepe)
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
            
            # Run CREPE with tiny model for CPU compatibility
            # Returns: frequencies, confidence, activation, time
            result = crepe.predict(y, sr, viterbi=True, model_capacity='tiny', step_size=10)
            
            if isinstance(result, tuple) and len(result) >= 2:
                frequencies = result[0]
                confidence = result[1]
            else:
                frequencies = result
                confidence = np.ones_like(frequencies)
                
            # Filter out unvoiced frames (confidence < 0.5 or frequency == 0)
            valid_mask = (frequencies > 0) & (confidence > 0.5)
            
            valid_freqs = frequencies[valid_mask]
            valid_conf = confidence[valid_mask]
            
            if len(valid_freqs) == 0:
                return {
                    "mean_pitch_hz": 0.0,
                    "median_pitch_hz": 0.0,
                    "min_pitch_hz": 0.0,
                    "max_pitch_hz": 0.0,
                    "mean_confidence": 0.0,
                    "pitch_evolution": "variable",
                    "is_sung": False,
                    "error": "No valid pitch detected"
                }
                
            mean_pitch = float(np.mean(valid_freqs))
            median_pitch = float(np.median(valid_freqs))
            min_pitch = float(np.min(valid_freqs))
            max_pitch = float(np.max(valid_freqs))
            mean_conf = float(np.mean(valid_conf))
            
            # Pitch evolution: stable, rising, falling, variable
            try:
                from scipy.stats import linregress
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 2:
                    slope, _, r_value, _, _ = linregress(valid_indices, valid_freqs)
                    normalized_slope = slope / (mean_pitch + 1e-6)
                    
                    if r_value**2 > 0.7:
                        if normalized_slope > 0.05:
                            evolution = "rising"
                        elif normalized_slope < -0.05:
                            evolution = "falling"
                        else:
                            evolution = "stable"
                    else:
                        evolution = "variable"
                else:
                    evolution = "stable"
            except Exception:
                evolution = "variable"
                
            # Sung vs spoken detection based on pitch regularity
            pitch_std = float(np.std(valid_freqs))
            cv = pitch_std / (mean_pitch + 1e-6)
            is_sung = (mean_conf > 0.65) and (evolution in ["stable", "rising", "falling"]) and (cv < 0.4)
            
            return {
                "mean_pitch_hz": round(mean_pitch, 2),
                "median_pitch_hz": round(median_pitch, 2),
                "min_pitch_hz": round(min_pitch, 2),
                "max_pitch_hz": round(max_pitch, 2),
                "mean_confidence": round(mean_conf, 3),
                "pitch_evolution": evolution,
                "is_sung": is_sung,
                "error": None
            }
            
    except Exception as e:
        return {
            "mean_pitch_hz": 0.0,
            "median_pitch_hz": 0.0,
            "min_pitch_hz": 0.0,
            "max_pitch_hz": 0.0,
            "mean_confidence": 0.0,
            "pitch_evolution": "variable",
            "is_sung": False,
            "error": str(e)
        }

def answer_qcm(features, question, choices):
    if features.get("error") or features["mean_confidence"] == 0.0:
        return {
            "answer": None,
            "confidence": 0.0,
            "detail": "Pitch extraction failed or no valid pitch detected."
        }
        
    question_lower = question.lower()
    choices_lower = {k: str(v).lower() for k, v in choices.items()}
    
    scores = {}
    
    for key, choice in choices_lower.items():
        score = 0.5
        
        if "pitch" in question_lower or "hauteur" in question_lower or "hz" in question_lower:
            if "aigu" in choice or "haut" in choice or "high" in choice:
                if features["mean_pitch_hz"] > 300:
                    score += 0.2
            elif "grave" in choice or "bas" in choice or "low" in choice:
                if features["mean_pitch_hz"] < 150:
                    score += 0.2
                    
        if "chant" in choice or "sung" in choice or "chantée" in choice or "chanté" in choice:
            if features["is_sung"]:
                score += 0.25
            else:
                score -= 0.15
                
        if "parl" in choice or "spoken" in choice:
            if not features["is_sung"]:
                score += 0.25
            else:
                score -= 0.15
                
        if "montant" in choice or "rising" in choice or "ascendant" in choice:
            if features["pitch_evolution"] == "rising":
                score += 0.25
                
        if "descendant" in choice or "falling" in choice:
            if features["pitch_evolution"] == "falling":
                score += 0.25
                
        if "stable" in choice:
            if features["pitch_evolution"] == "stable":
                score += 0.25
                
        if "variable" in choice:
            if features["pitch_evolution"] == "variable":
                score += 0.25
                
        scores[key] = score
        
    import math
    exp_scores = {k: math.exp(v) for k, v in scores.items()}
    total = sum(exp_scores.values())
    probs = {k: v / total for k, v in exp_scores.items()}
    
    best_key = max(probs, key=probs.get)
    best_prob = probs[best_key]
    
    # Cap confidence at 0.75 as per Probabilistic tier requirements
    final_confidence = min(best_prob, 0.75)
    
    detail = (f"Pitch analysis: mean={features['mean_pitch_hz']}Hz, median={features['median_pitch_hz']}Hz, "
              f"range=[{features['min_pitch_hz']}, {features['max_pitch_hz']}]Hz, "
              f"confidence={features['mean_confidence']}, evolution={features['pitch_evolution']}, "
              f"is_sung={features['is_sung']}.")
              
    return {
        "answer": best_key,
        "confidence": round(final_confidence, 3),
        "detail": detail
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CREPE QCM Inference")
    parser.add_argument("--wav", required=True, help="Path to WAV file")
    parser.add_argument("--question", required=True, help="QCM Question")
    parser.add_argument("--choices", required=True, help="JSON string of choices")
    
    args = parser.parse_args()
    
    try:
        choices_dict = json.loads(args.choices)
    except json.JSONDecodeError:
        print(json.dumps({"answer": None, "confidence": 0.0, "detail": "Invalid choices JSON"}))
        sys.exit(1)
        
    features = analyze_audio(args.wav)
    result = answer_qcm(features, args.question, choices_dict)
    
    print(json.dumps(result, indent=2))
