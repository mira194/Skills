#!/usr/bin/env python3
"""
Estimation du RT60 et caractéristiques acoustiques de l'environnement.
Utilise l'intégration de Schroeder pour estimer la courbe de décroissance d'énergie (EDC).
"""

import argparse
import json
import sys
import numpy as np
import librosa

def estimate_rt60(y, sr):
    """
    Estime le RT60 en utilisant l'intégration de Schroeder.
    """
    # Calcul de l'enveloppe d'énergie
    energy = y ** 2
    
    # Intégration de Schroeder (décroissance cumulative inversée)
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10 * np.log10(edc / np.max(edc) + 1e-10)
    
    # Trouver les points de décroissance de -5 dB à -35 dB (pour extrapoler 60 dB)
    # On utilise une régression linéaire sur cette portion
    valid_indices = np.where((edc_db <= -5) & (edc_db >= -35))[0]
    
    if len(valid_indices) < 10:
        return None, None, None
    
    # Régression linéaire simple
    x = valid_indices / sr  # temps en secondes
    y_vals = edc_db[valid_indices]
    
    # Pente de la décroissance (dB/s)
    coeffs = np.polyfit(x, y_vals, 1)
    slope = coeffs[0]
    
    if slope >= 0:
        return None, None, None
        
    # RT60 = -60 / slope
    rt60 = -60.0 / slope
    
    return rt60, slope, edc_db

def estimate_noise_floor(y):
    """Estime le plancher de bruit en dB."""
    noise_floor = np.percentile(np.abs(y), 10)
    noise_floor_db = 20 * np.log10(noise_floor + 1e-10)
    return noise_floor_db

def classify_room_size(rt60):
    """Heuristique simple pour classifier la taille de la pièce."""
    if rt60 is None:
        return "Indéterminée"
    elif rt60 < 0.3:
        return "Petite pièce (traitée ou très absorbante)"
    elif rt60 < 0.6:
        return "Pièce moyenne (bureau, salon standard)"
    elif rt60 < 1.2:
        return "Grande pièce (hall, salle de classe)"
    else:
        return "Très grande pièce ou espace réverbérant (église, gymnase)"

def main():
    parser = argparse.ArgumentParser(description="Estimation du RT60 et caractéristiques acoustiques")
    parser.add_argument("--audio_path", required=True, help="Chemin vers le fichier audio")
    args = parser.parse_args()

    try:
        y, sr = librosa.load(args.audio_path, sr=22050)
    except Exception as e:
        print(json.dumps({"error": f"Erreur de chargement audio: {str(e)}"}))
        sys.exit(1)

    rt60, slope, edc_db = estimate_rt60(y, sr)
    noise_floor_db = estimate_noise_floor(y)
    room_size = classify_room_size(rt60)

    result = {
        "rt60_seconds": round(rt60, 3) if rt60 is not None else None,
        "noise_floor_db": round(noise_floor_db, 2),
        "estimated_room_size": room_size,
        "audio_duration_seconds": round(len(y) / sr, 2)
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()