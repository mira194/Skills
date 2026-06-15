---
name: environment_detection
description: >
  Estimer les caractéristiques acoustiques d'un environnement (RT60, plancher de bruit, taille de pièce) à partir d'un fichier audio en utilisant l'analyse de l'enveloppe RMS et l'estimation de la courbe de décroissance d'énergie (EDC) via `librosa`.
  Utilisez ce skill lorsque l'utilisateur souhaite estimer le temps de réverbération (RT60), classifier la taille approximative d'une pièce, ou analyser le plancher de bruit.
---

# environment_detection

Estimer les caractéristiques acoustiques d'un environnement (RT60, plancher de bruit, taille de pièce) à partir d'un fichier audio en utilisant l'analyse de l'enveloppe RMS et l'estimation de la courbe de décroissance d'énergie (EDC) via `librosa`.

## Description
Ce skill analyse un fichier audio pour en déduire des propriétés acoustiques de l'environnement d'enregistrement. Il calcule notamment le temps de réverbération approximatif (RT60) via l'intégration de Schroeder, estime le plancher de bruit et fournit des heuristiques sur la taille probable de la pièce.

## Usage
Utilisez ce skill lorsque l'utilisateur souhaite :
- Estimer le temps de réverbération (RT60) d'un enregistrement.
- Classifier la taille approximative d'une pièce (petite, moyenne, grande) à partir de ses caractéristiques acoustiques.
- Analyser le plancher de bruit ou la dynamique d'un environnement sonore.

## Dépendances
- `librosa`
- `numpy`
- `scipy`

## Script
Exécutez `python scripts/estimate_rt60.py --audio_path <chemin_vers_audio>`