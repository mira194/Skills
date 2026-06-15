---
name: spectral_features
description: >
  Analyze spectral characteristics of an audio file using Librosa (spectral centroid, MFCC).
  Use this skill whenever the user wants to extract spectral features, analyze timbre, brightness,
  or perform spectral analysis on audio files. Also use when the user mentions spectral centroid,
  MFCC extraction, timbral analysis, or audio feature extraction for machine learning or analysis.
  Reliability tier: Analytic.
---

# Spectral Features Extraction

Extract and analyze spectral characteristics of audio files using Librosa.

## How it works

1. Load the audio file using Librosa
2. Extract core spectral features:
   - **Spectral Centroid**: Indicates the "brightness" or center of mass of the spectrum
   - **MFCCs (Mel-Frequency Cepstral Coefficients)**: Captures the timbral fingerprint of the audio
   - **Spectral Rolloff**: Frequency below which a specified percentage of the total spectral energy lies
   - **Spectral Bandwidth**: Width of the band of frequencies centered around the spectral centroid
3. Compute statistical summaries (mean, variance, min, max) over time
4. Return structured JSON output suitable for analysis or downstream tasks

## Dependencies

```bash
pip install librosa numpy
```

## Files

- `scripts/extract_spectral.py` – Main extraction script

## Usage

### Basic usage (CLI)

```bash
python skills/sk-spectral-features/scripts/extract_spectral.py \
    --audio path/to/audio.wav \
    --output features.json
```

Output (`features.json`):
```json
{
  "file": "path/to/audio.wav",
  "duration_seconds": 10.5,
  "sample_rate": 22050,
  "features": {
    "spectral_centroid": {"mean": 1250.5, "var": 45000.2, "min": 200.1, "max": 3500.0},
    "mfcc": {"mfcc_0": {"mean": 0.0, "var": 1.2}, "mfcc_1": {"mean": -5.2, "var": 0.8}},
    "spectral_rolloff": {"mean": 2100.0, "var": 15000.0},
    "spectral_bandwidth": {"mean": 1800.0, "var": 25000.0}
  }
}
```

### As a Python function

```python
import sys
sys.path.append("skills/sk-spectral-features/scripts")
from extract_spectral import extract_spectral_features

features = extract_spectral_features("audio.wav")
print(features["features"]["spectral_centroid"]["mean"])
```

## Feature Interpretation

### Spectral Centroid
- **Low (< 1000 Hz)**: Warm, bass-heavy sounds (male voice, bass guitar, kick drum)
- **Medium (1000–3000 Hz)**: Mid-range content (female voice, most acoustic instruments)
- **High (> 3000 Hz)**: Bright sounds (cymbals, hi-hats, sibilants, noise, synthesizers)

### MFCCs
- First 13 coefficients are typically used
- Coefficient 0 represents overall energy (often excluded in timbral analysis)
- Coefficients 1–12 capture the shape of the spectral envelope (timbre)
- Useful for speech recognition, instrument classification, and music genre analysis

### Spectral Rolloff
- Typically set to 85% or 95% of energy
- Helps distinguish between noise-like sounds (high rolloff) and tonal sounds (low rolloff)

### Spectral Bandwidth
- Measures the spread of frequencies around the centroid
- High bandwidth: complex, noisy, or rich harmonic content
- Low bandwidth: pure tones, simple waveforms

## Limitations

- Requires `librosa` and `soundfile` (or `audioread`) to be installed
- Large files may require chunking to avoid memory issues
- Feature extraction is deterministic but sensitive to sample rate and preprocessing (e.g., normalization)