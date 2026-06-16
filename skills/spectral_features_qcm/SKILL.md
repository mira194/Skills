---
name: spectral_features_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using Librosa for spectral feature analysis.
  Use this skill whenever the user provides an audio file along with a question and choices and wants to
  automatically determine the correct answer based on spectral analysis (mean spectral centroid, mean MFCCs, 
  mean spectral bandwidth, mean spectral rolloff). Also use when the user mentions spectral features, 
  brightness, timbre, bandwidth, rolloff, or audio QCM with librosa spectral analysis.
---

# Spectral Features QCM – Audio Spectrum Analysis

Answer multiple-choice questions about audio files using Librosa's spectral feature extraction capabilities.

## How it works

1. Load the audio file and extract spectral features using Librosa:
   - **Mean Spectral Centroid**: Indicates the "brightness" or center of mass of the spectrum.
   - **Mean MFCCs (1-3)**: Represents the timbre or tone color of the sound.
   - **Mean Spectral Bandwidth**: Indicates the width of the band of frequencies present.
   - **Mean Spectral Rolloff**: The frequency below which a specified percentage (default 85%) of the spectral energy lies.

2. Analyze the extracted features and map them to the provided QCM choices using heuristic scoring.

3. Return the best-matching choice with a confidence score (capped at 0.90 for the Analytic tier).

## Dependencies

```
librosa
numpy
soundfile
```

**Recommended setup (using venv):**

```bash
cd skills/spectral_features_qcm
python3 -m venv .venv
. .venv/bin/activate
pip install librosa numpy soundfile
```

## Files

- `scripts/spectral_features_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python scripts/spectral_features_qcm_inference.py \
    --audio path/to/audio.wav \
    --payload '{"question": "Le son est-il brillant ou sombre ?", "choices": {"A": "Brillant (aigu)", "B": "Sombre (grave)"}}'
```

Output:
```json
{
  "answer": "A",
  "confidence": 0.80,
  "detail": "Centroïde: 2500.5Hz, MFCCs moy: [-120.5, 45.2, -15.1], Bande passante: 3200.0Hz, Rolloff: 4100.0Hz"
}
```

### As a Python function

```python
from scripts.spectral_features_qcm_inference import process_qcm

result = process_qcm(
    audio_path="audio.wav",
    payload={
        "question": "Quelle est la largeur spectrale ?",
        "choices": {"A": "Étroite", "B": "Large"}
    }
)
print(result["answer"], result["confidence"], result["detail"])
```

## Feature Interpretation Guide

### Mean Spectral Centroid
- **< 1000 Hz**: Dark, muffled, or bass-heavy sounds (e.g., kick drum, male speech).
- **1000 – 3000 Hz**: Neutral, mid-range sounds (e.g., typical speech, guitar).
- **> 3000 Hz**: Bright, sharp, or treble-heavy sounds (e.g., cymbals, whistles, female speech).

### Mean MFCCs (1-3)
- Captures the spectral envelope shape. Large variations indicate complex timbres (e.g., vowels, instruments), while flat values indicate noise or pure tones.

### Mean Spectral Bandwidth
- **< 1000 Hz**: Narrowband, tonal, or pure sounds (e.g., sine wave, flute).
- **> 2000 Hz**: Wideband, noisy, or complex sounds (e.g., white noise, cymbals, full mix).

### Mean Spectral Rolloff
- Correlates with the highest significant frequency. Low rolloff indicates absence of high frequencies (low-pass filtered or bass-only), while high rolloff indicates rich high-frequency content.

## Reliability Tier

- **Tier**: Analytic
- **Max Confidence**: 0.90
- **Notes**: Spectral features are highly dependent on the audio content and sampling rate. Errors are handled silently, returning `confidence: 0.0` on failure.

## Limitations

- Requires the audio to be loadable by librosa/soundfile.
- Short audio clips may yield unstable spectral estimates.
- Heuristic matching is basic; for complex semantic questions, consider combining with an LLM prompt.