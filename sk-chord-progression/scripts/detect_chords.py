#!/usr/bin/env python3
"""
Détecte la progression d'accords d'un fichier audio en utilisant Librosa (chroma CQT + modèles d'accords).
"""
import argparse
import json
import numpy as np
import librosa

# Définition des modèles d'accords (12 classes de hauteurs)
# Accord majeur : fondamentale, tierce majeure (4), quinte juste (7)
# Accord mineur : fondamentale, tierce mineure (3), quinte juste (7)
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORDS = []
TEMPLATES = []

for i, root in enumerate(ROOTS):
    # Accord majeur
    CHORDS.append(f"{root}maj")
    maj_template = np.zeros(12)
    maj_template[i] = 1.0
    maj_template[(i + 4) % 12] = 1.0
    maj_template[(i + 7) % 12] = 1.0
    TEMPLATES.append(maj_template)
    
    # Accord mineur
    CHORDS.append(f"{root}min")
    min_template = np.zeros(12)
    min_template[i] = 1.0
    min_template[(i + 3) % 12] = 1.0
    min_template[(i + 7) % 12] = 1.0
    TEMPLATES.append(min_template)

TEMPLATES = np.array(TEMPLATES) # Forme : (24, 12)

def detect_chords(audio_path, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extraction du chroma CQT
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # Normalisation du chroma pour la similarité cosinus
    chroma_norm = librosa.util.normalize(chroma, norm=2, axis=0)
    
    # Calcul de la similarité entre chaque frame et chaque modèle d'accord
    # TEMPLATES est (24, 12), chroma_norm est (12, n_frames)
    # Nous voulons les scores de similarité (24, n_frames)
    similarities = np.dot(TEMPLATES, chroma_norm)
    
    # Obtention de l'index de la similarité maximale pour chaque frame
    best_chord_indices = np.argmax(similarities, axis=0)
    
    # Obtention du temps pour chaque frame
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    
    # Conversion des index en noms d'accords
    detected_chords = [CHORDS[idx] for idx in best_chord_indices]
    
    # Simplification de la progression en regroupant les accords identiques consécutifs
    progression = []
    if len(detected_chords) > 0:
        current_chord = detected_chords[0]
        start_time = times[0]
        
        for i in range(1, len(detected_chords)):
            if detected_chords[i] != current_chord:
                progression.append({
                    "chord": current_chord,
                    "start_time": round(float(start_time), 2),
                    "end_time": round(float(times[i]), 2)
                })
                current_chord = detected_chords[i]
                start_time = times[i]
        
        # Ajout du dernier accord
        progression.append({
            "chord": current_chord,
            "start_time": round(float(start_time), 2),
            "end_time": round(float(times[-1]), 2)
        })
    
    return {
        "file": audio_path,
        "duration_seconds": round(float(len(y) / sr), 2),
        "sample_rate": int(sr),
        "progression": progression,
        "total_chords_detected": len(progression)
    }

def main():
    parser = argparse.ArgumentParser(description="Détecte la progression d'accords d'un fichier audio.")
    parser.add_argument("--audio", required=True, help="Chemin vers le fichier audio")
    parser.add_argument("--output", default="chords.json", help="Chemin du fichier JSON de sortie")
    parser.add_argument("--hop_length", type=int, default=512, help="Longueur de saut pour l'extraction chroma")
    
    args = parser.parse_args()
    
    try:
        result = detect_chords(args.audio, hop_length=args.hop_length)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Progression d'accords détectée avec succès et enregistrée dans {args.output}")
    except Exception as e:
        print(f"Erreur lors de la détection des accords : {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()