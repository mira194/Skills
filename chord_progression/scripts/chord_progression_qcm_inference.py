#!/usr/bin/env python3
"""
Chord Progression QCM Inference Script
Uses Librosa to extract chroma CQT and match against 24 major/minor chord templates,
answering QCM questions based on:
- dominant chord detected
- sequence of chords with timestamps
- number of chord changes
- average confidence of template matching
"""

import argparse
import json
import sys
import os
import re
import warnings
from collections import Counter
import numpy as np

# Suppress librosa warnings for silent error handling
warnings.filterwarnings("ignore")

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

# 24 chord templates (12 major, 12 minor)
# Pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
CHORD_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def generate_chord_templates():
    templates = {}
    for i, root in enumerate(CHORD_NAMES):
        # Major: root, root+4, root+7
        major_template = np.zeros(12)
        major_template[i] = 1.0
        major_template[(i + 4) % 12] = 1.0
        major_template[(i + 7) % 12] = 1.0
        templates[f"{root} majeur"] = major_template / np.linalg.norm(major_template)
        
        # Minor: root, root+3, root+7
        minor_template = np.zeros(12)
        minor_template[i] = 1.0
        minor_template[(i + 3) % 12] = 1.0
        minor_template[(i + 7) % 12] = 1.0
        templates[f"{root} mineur"] = minor_template / np.linalg.norm(minor_template)
        
    return templates

CHORD_TEMPLATES = generate_chord_templates()

def process_qcm(audio_path: str, payload: dict) -> dict:
    """
    Process a QCM request for chord progression detection.
    
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
        
        # Extract chroma CQT
        # hop_length=512 gives ~23ms resolution at 22050Hz, good for chord changes
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        
        # Number of frames
        n_frames = chroma_cqt.shape[1]
        
        # Segment into windows (e.g., 1 second windows) to detect progression
        # 1 second = sr / hop_length frames
        window_frames = max(1, int(sr / 512))
        
        chord_sequence = []
        timestamps = []
        confidences = []
        
        for i in range(0, n_frames, window_frames):
            window_chroma = chroma_cqt[:, i:i+window_frames]
            # Average chroma over the window
            avg_chroma = np.mean(window_chroma, axis=1)
            # Normalize
            norm = np.linalg.norm(avg_chroma)
            if norm > 0:
                avg_chroma = avg_chroma / norm
            else:
                continue
                
            # Match against all 24 templates
            best_chord = ""
            best_sim = -1.0
            
            for chord_name, template in CHORD_TEMPLATES.items():
                # Cosine similarity
                sim = np.dot(avg_chroma, template)
                if sim > best_sim:
                    best_sim = sim
                    best_chord = chord_name
                    
            if best_chord:
                chord_sequence.append(best_chord)
                timestamps.append(i * 512 / sr)
                confidences.append(best_sim)
                
        # Analyze results
        if not chord_sequence:
            return {
                "answer": "",
                "confidence": 0.0,
                "detail": "Aucun accord détecté dans l'audio."
            }
            
        # Dominant chord (most frequent)
        chord_counts = Counter(chord_sequence)
        dominant_chord = chord_counts.most_common(1)[0][0]
        
        # Number of chord changes
        changes = sum(1 for i in range(1, len(chord_sequence)) if chord_sequence[i] != chord_sequence[i-1])
        
        # Average confidence
        avg_conf = float(np.mean(confidences))
        
        # Format sequence for detail (first 5 chords)
        seq_str = ", ".join([f"{c}({t:.1f}s)" for c, t in zip(chord_sequence[:5], timestamps[:5])])
        if len(chord_sequence) > 5:
            seq_str += "..."
            
        best_choice = ""
        max_score = -1.0
        best_detail = (f"Accord dominant: {dominant_chord}, "
                       f"Changements: {changes}, "
                       f"Confiance moy: {avg_conf:.2f}, "
                       f"Séquence: {seq_str}")
        
        for choice_key, choice_text in choices.items():
            score = 0.0
            choice_lower = choice_text.lower()
            
            # Check for dominant chord match
            if dominant_chord.lower() in choice_lower:
                score += 0.8
                
            # Check for chord change count match
            numbers = re.findall(r'\d+', choice_text)
            for num in numbers:
                if int(num) == changes or int(num) == changes + 1 or int(num) == max(1, changes - 1):
                    score += 0.6
                    
            # Check for specific chord names in choice
            for chord_name in CHORD_NAMES:
                if (chord_name.lower() in choice_lower or 
                    f"{chord_name.lower()}m" in choice_lower or 
                    f"{chord_name.lower()} majeur" in choice_lower or 
                    f"{chord_name.lower()} mineur" in choice_lower):
                    if chord_name.lower() in dominant_chord.lower():
                        score += 0.5
                        
            # Fallback
            if score == 0.0:
                score = 0.1
                
            if score > max_score:
                max_score = score
                best_choice = choice_key
                
        if not best_choice and choices:
            best_choice = list(choices.keys())[0]
            max_score = 0.3
            
        # Cap confidence at 0.60 for Heuristic tier
        confidence = min(0.60, max_score)
        
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
    parser = argparse.ArgumentParser(description="Chord Progression QCM Inference")
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