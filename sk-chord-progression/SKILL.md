---
name: chord_progression
description: >
  Analyse la progression d'accords d'un fichier audio en utilisant Librosa (chroma CQT + modèles d'accords).
  Utilisez cette compétence lorsque l'utilisateur souhaite détecter des accords, analyser l'harmonie d'un morceau,
  ou extraire une suite d'accords à partir d'un fichier audio (WAV, MP3, etc.).
  Fiabilité : Analytique.
---

# Détection de Progression d'Accords

Détecte et extrait la progression d'accords d'un fichier audio en utilisant l'analyse chromatique CQT et des modèles d'accords (majeurs et mineurs) via Librosa.

## Fonctionnement

1. Charge le fichier audio avec Librosa.
2. Extrait les caractéristiques Chroma CQT (`librosa.feature.chroma_cqt`).
3. Compare chaque frame chromatique à 24 modèles d'accords prédéfinis (12 racines × 2 qualités : majeur et mineur) via un produit scalaire (similarité cosinus).
4. Identifie l'accord le plus probable pour chaque frame.
5. Regroupe les frames consécutives identiques pour produire une progression d'accords segmentée dans le temps.
6. Retourne une sortie JSON structurée adaptée à l'analyse ou aux tâches en aval.

## Dépendances

```bash
pip install librosa numpy
```

## Fichiers

- `scripts/detect_chords.py` – Script principal de détection

## Utilisation

### Utilisation de base (CLI)

```bash
python skills/sk-chord-progression/scripts/detect_chords.py \
    --audio chemin/vers/audio.wav \
    --output progression.json
```

Sortie (`progression.json`) :
```json
{
  "file": "chemin/vers/audio.wav",
  "duration_seconds": 15.5,
  "sample_rate": 22050,
  "progression": [
    {
      "chord": "Cmaj",
      "start_time": 0.0,
      "end_time": 2.32
    },
    {
      "chord": "Amin",
      "start_time": 2.32,
      "end_time": 4.64
    }
  ],
  "total_chords_detected": 2
}
```

### En tant que fonction Python

```python
import sys
sys.path.append("skills/sk-chord-progression/scripts")
from detect_chords import detect_chords

result = detect_chords("audio.wav")
for segment in result["progression"]:
    print(f"[{segment['start_time']}s - {segment['end_time']}s] : {segment['chord']}")
```

## Interprétation des résultats

- **Cmaj, Dmaj, etc.** : Accords majeurs (sonorité joyeuse, stable).
- **Cmin, Dmin, etc.** : Accords mineurs (sonorité triste, mélancolique).
- Les temps (`start_time`, `end_time`) sont arrondis à 2 décimales pour éviter les problèmes d'égalité flottante.

## Limites

- Nécessite `librosa` et `soundfile` (ou `audioread`) installés.
- La détection basée sur des modèles simples (majeur/mineur) peut être imprécise sur des accords complexes (7ème, 9ème, sus4) ou des mixages très denses.
- Pour une meilleure précision, un `hop_length` plus petit peut être utilisé, au détriment du temps de calcul.