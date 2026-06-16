---
name: energy_dynamics_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using Librosa for RMS energy and dynamics analysis.
  Use this skill whenever the user provides an audio file along with a question and choices and wants to
  automatically determine the correct answer based on energy analysis (mean RMS, min/max RMS, dynamic range in dB,
  temporal evolution of energy). Also use when the user mentions energy dynamics, loudness, dynamic range,
  RMS analysis, or audio QCM with librosa energy features.
---

# Energy Dynamics QCM – RMS and Dynamic Range Analysis

Answer multiple-choice questions about audio files using Librosa's RMS energy and dynamic range features.

## How it works

1. Load the audio file and extract energy features using Librosa:
   - **RMS moyen** : Niveau d'énergie global (moyenne des valeurs RMS).
   - **RMS min/max** : Valeurs minimale et maximale de l'enveloppe d'énergie.
   - **Plage dynamique (dB)** : Différence en décibels entre le RMS max et le RMS min.
   - **Évolution temporelle** : Classification de la courbe d'énergie en 4 segments (croissante, décroissante, stable, variable).

2. Analyser les caractéristiques extraites et les mapper aux choix de la QCM via un scoring heuristique.

3. Retourner le meilleur choix avec un score de confiance (plafonné à 0.90 pour le tier Analytic).

## Dependencies

```
librosa
numpy
soundfile
```

**Recommended setup (using venv):**

```bash
cd skills/energy_dynamics_qcm
python3 -m venv .venv
. .venv/bin/activate
pip install librosa numpy soundfile
```

## Files

- `scripts/energy_dynamics_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python scripts/energy_dynamics_qcm_inference.py \
    --audio path/to/audio.wav \
    --payload '{"question": "Quelle est la dynamique du morceau ?", "choices": {"A": "Très compressée et stable", "B": "Large plage dynamique"}}'
```

Output:
```json
{
  "answer": "B",
  "confidence": 0.80,
  "detail": "RMS moyen: 0.1250, Min: 0.0010, Max: 0.4500, Plage dynamique: 53.1 dB, Évolution: variable"
}
```

### As a Python function

```python
from scripts.energy_dynamics_qcm_inference import process_qcm

result = process_qcm(
    audio_path="audio.wav",
    payload={
        "question": "Le volume augmente-t-il ?",
        "choices": {"A": "Oui, il est croissant", "B": "Non, il est stable"}
    }
)
print(result["answer"], result["confidence"], result["detail"])
```

## Feature Interpretation Guide

### RMS Mean (Global Energy)
- **< 0.01** : Silence ou audio très faible.
- **0.01 – 0.1** : Niveau modéré (parole typique, musique calme).
- **> 0.1** : Niveau élevé (musique forte, mix dense).

### Dynamic Range (dB)
- **< 10 dB** : Très compressé, dynamique faible (radio, pop moderne très masterisée).
- **10 – 20 dB** : Dynamique modérée.
- **> 20 dB** : Large plage dynamique (musique classique, jazz, enregistrements non compressés).

### Temporal Evolution
- **Stable** : Écart-type des segments < 1% du RMS moyen.
- **Croissante** : La moyenne de la seconde moitié est > 120% de la première moitié.
- **Décroissante** : La moyenne de la seconde moitié est < 80% de la première moitié.
- **Variable** : Fluctuations sans tendance nette de montée ou descente.

## Reliability Tier

- **Tier** : Analytic
- **Max Confidence** : 0.90
- **Notes** : L'analyse RMS est robuste et déterministe. Les erreurs (fichier manquant, dépendances manquantes) sont gérées silencieusement en retournant `confidence: 0.0`.

## Limitations

- Le calcul de la plage dynamique en dB peut être sensible aux pics très brefs (un seul échantillon proche de 1.0 peut fausser le max).
- Nécessite que `librosa` et `soundfile` soient installés dans l'environnement d'exécution.
- L'audio doit être chargeable par librosa/soundfile.