---
name: energy_dynamics
description: Analyze the energy and dynamic range of an audio file using Librosa. Use this skill whenever the user asks to measure audio energy, RMS (Root Mean Square) amplitude, dynamic range, loudness variations, crest factor, or wants to perform quantitative dynamic analysis on an audio file (WAV, MP3, FLAC, etc.).
---

# Energy Dynamics Analysis

This skill provides a standardized approach to analyzing the energy and dynamic range of audio files using `librosa`.

## Core Metrics

1. **RMS Energy**: Root Mean Square energy, representing the perceived loudness or power of the signal over time.
2. **Crest Factor**: The ratio of peak amplitude to RMS amplitude (in dB), indicating how "peaky" or dynamic the signal is. Higher values mean more dynamic range (e.g., classical music), lower values mean more compressed (e.g., modern pop).
3. **Dynamic Range**: The difference between the highest and lowest significant amplitude levels (e.g., 99th percentile vs 1st percentile of RMS frames).

## Implementation

Use the provided `analyze_dynamics.py` script in the `scripts/` directory, or follow this pattern:

```python
import librosa
import numpy as np
import json
import sys

def analyze_audio_dynamics(audio_path, frame_length=2048, hop_length=512):
    # Load audio at native sample rate
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB (using max RMS as reference to keep values relative)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Compute peak amplitude
    peak_amp = np.max(np.abs(y))
    peak_db = librosa.amplitude_to_db(peak_amp, ref=np.max)
    
    # Crest factor (Peak to mean RMS ratio in dB)
    mean_rms_db = np.mean(rms_db)
    crest_factor_db = peak_db - mean_rms_db
    
    # Dynamic range (99th percentile of RMS vs 1st percentile of RMS)
    rms_percentile_high = np.percentile(rms_db, 99)
    rms_percentile_low = np.percentile(rms_db, 1)
    dynamic_range_db = rms_percentile_high - rms_percentile_low
    
    return {
        "file": audio_path,
        "duration_seconds": float(len(y) / sr),
        "mean_rms_db": float(np.mean(rms_db)),
        "max_rms_db": float(np.max(rms_db)),
        "min_rms_db": float(np.min(rms_db)),
        "peak_db": float(peak_db),
        "crest_factor_db": float(crest_factor_db),
        "dynamic_range_db": float(dynamic_range_db)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Please provide an audio file path"}))
        sys.exit(1)
    
    audio_path = sys.argv[1]
    try:
        result = analyze_audio_dynamics(audio_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
```

## Output Format

Always return the analysis results in a structured JSON format:
```json
{
  "file": "audio.wav",
  "duration_seconds": 10.5,
  "mean_rms_db": -15.2,
  "max_rms_db": -2.1,
  "min_rms_db": -45.0,
  "peak_db": 0.0,
  "crest_factor_db": 15.2,
  "dynamic_range_db": 25.4
}
```

## Dependencies
- `librosa`
- `numpy`
- `soundfile` (or `audioread`)

## Usage Notes
- If the user provides a specific audio file, run the script and present the JSON output clearly.
- Explain what the metrics mean in plain language (e.g., "A crest factor of 15 dB indicates a highly dynamic signal with significant peaks relative to the average loudness").
- If the file is not found or cannot be loaded, return a clear error message suggesting the user verify the file path and format.