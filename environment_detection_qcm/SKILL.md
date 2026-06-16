---
name: environment_detection_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using Librosa for RT60 (reverberation time) 
  estimation and heuristic acoustic environment classification. Use this skill whenever the user provides 
  an audio file along with a question and choices and wants to automatically determine the correct answer 
  based on acoustic environment analysis (e.g., "intérieur sec", "salle réverbérante", "extérieur", 
  noise floor levels). Also use when the user mentions reverberation, RT60, acoustic environment, 
  room characteristics, or background noise estimation.
---

# Environment Detection QCM – Acoustic Environment Classification

Answer multiple-choice questions about audio files by estimating reverberation time (RT60), classifying the acoustic environment, and estimating the background noise floor using Librosa signal processing heuristics.

## How it works

1. Load the audio file and compute frame-based RMS energy
2. Estimate RT60 (reverberation time) by analyzing the energy decay curve
3. Estimate the background noise floor using the 10th percentile of RMS energy
4. Classify the environment heuristically (e.g., "intérieur sec", "salle réverbérante", "extérieur")
5. Map the extracted acoustic features to QCM choices using keyword matching and heuristic scoring
6. Return the best-matching choice with a confidence capped at 0.60 (Heuristic tier)

## Dependencies

```
librosa
numpy
```

Install with: `pip install librosa numpy`

## Files

- `scripts/environment_detection_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python scripts/environment_detection_qcm_inference.py \
    --audio path/to/audio.wav \
    --payload '{"question": "Quel est l'environnement acoustique ?", "choices": {"A": "Intérieur sec", "B": "Salle réverbérante", "C": "Extérieur"}}'
```

Output:
```json
{
  "answer": "B",
  "confidence": 0.55,
  "detail": "RT60 estimé: 1.20s, Niveau de bruit: 0.0150, Environnement: salle réverbérante"
}
```

### As a Python function

```python
from scripts.environment_detection_qcm_inference import process_qcm

result = process_qcm(
    audio_path="audio.wav",
    payload={
        "question": "Quel est l'environnement acoustique ?",
        "choices": {
            "A": "Intérieur sec",
            "B": "Salle réverbérante",
            "C": "Extérieur"
        }
    }
)
print(result["answer"], result["confidence"], result["detail"])
```

## Feature Interpretation Guide

### RT60 (Reverberation Time)
- **< 0.3s**: Dry environment (small room, treated studio, anechoic)
- **0.3s – 1.0s**: Moderate reverberation (typical living room, office, classroom)
- **1.0s – 2.5s**: Reverberant space (large hall, church, gymnasium)
- **> 2.5s**: Highly reverberant or outdoor with echoes (cathedral, canyon)

### Noise Floor
- **< 0.01**: Very quiet, low background noise
- **0.01 – 0.03**: Moderate background noise (typical indoor ambient)
- **> 0.03**: Noisy environment (street, wind, heavy HVAC, crowd)

### Environment Classification Heuristics
- **Intérieur sec**: RT60 < 0.3s AND noise floor < 0.02
- **Intérieur bruyant**: RT60 < 0.5s AND noise floor >= 0.02
- **Salle réverbérante**: RT60 between 0.3s and 1.5s
- **Extérieur ou très réverbérant**: RT60 > 1.5s OR high noise floor with specific spectral characteristics

## Scoring Strategy

1. Each choice is evaluated against the extracted features (RT60, noise floor, classified environment)
2. Keywords in the choice text are matched against the classified environment and feature ranges
3. Points are awarded for matching conditions (e.g., +0.4 for matching RT60 range, +0.3 for matching noise level)
4. The choice with the highest score is selected
5. Confidence is calculated as `min(best_score, 0.60)` to reflect the Heuristic reliability tier

## Error Handling

- The script handles all exceptions silently
- If audio loading or processing fails, it returns:
  ```json
  {
    "answer": "",
    "confidence": 0.0,
    "detail": "Échec de l'analyse"
  }
  ```

## Limitations

- RT60 estimation is heuristic and based on energy decay; it may be inaccurate for non-impulsive audio or continuous noise
- Noise floor estimation assumes the quietest 10% of the audio represents the background noise, which may not hold for dynamically mixed content
- Confidence is strictly capped at 0.60 due to the heuristic nature of the analysis
- Requires `librosa` and `numpy` to be installed in the execution environment
- Audio must be loadable by `librosa` (WAV, MP3, FLAC, etc.)