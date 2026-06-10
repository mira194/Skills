---
name: crepe_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using CREPE for pitch estimation.
  Use this skill whenever the user provides an audio file along with a question and choices and wants to
  automatically determine the correct answer based on pitch analysis (mean/median pitch in Hz, pitch range,
  confidence, temporal evolution, and sung vs spoken voice detection). Also use when the user mentions
  pitch estimation, fundamental frequency (F0), CREPE, melodic height, or vocal classification.
---

# CREPE QCM – Pitch-Based Audio Classification

Answer multiple-choice questions about audio files using CREPE's state-of-the-art pitch estimation.

## How it works

1. Load the audio file and resample to 16kHz (CREPE standard)
2. Run CREPE pitch estimation using the "tiny" model for CPU compatibility
3. Extract core pitch features:
   - **Mean & Median Pitch (Hz)** – central tendency of the fundamental frequency
   - **Pitch Range (Min/Max Hz)** – melodic span
   - **Mean Confidence** – reliability of the pitch tracking
   - **Temporal Evolution** – stable, rising, falling, or variable pitch contour
   - **Sung vs Spoken Detection** – based on pitch regularity and confidence metrics
4. Map feature patterns to QCM choices using built-in heuristics
5. Return the best-matching choice with probabilistic confidence (capped at 0.75)

## Dependencies

```
crepe
librosa
scipy
numpy
```

Install with: `pip install crepe librosa scipy numpy`
*(Note: CREPE will be automatically installed by the script if missing)*

## Files

- `Scripts/crepe_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python Scripts/crepe_qcm_inference.py \
    --wav path/to/audio.wav \
    --question "Is this a sung or spoken voice?" \
    --choices '{"A": "Sung voice", "B": "Spoken voice"}'
```

Output:
```json
{
  "answer": "A",
  "confidence": 0.72,
  "detail": "Pitch analysis: mean=220.5Hz, median=218.0Hz, range=[180.0, 260.0]Hz, confidence=0.85, evolution=stable, is_sung=True."
}
```

### As a Python function

```python
from Scripts.crepe_qcm_inference import analyze_audio, answer_qcm

# Extract pitch features
features = analyze_audio("audio.wav")
# Returns: {mean_pitch_hz, median_pitch_hz, min_pitch_hz, max_pitch_hz, mean_confidence, pitch_evolution, is_sung, error}

# Answer a QCM
result = answer_qcm(
    features=features,
    question="Is the pitch rising or falling?",
    choices={"A": "Rising", "B": "Falling", "C": "Stable"}
)
print(result["answer"], result["confidence"], result["detail"])
```

## Feature Interpretation Guide

### Pitch (Hz)
- **< 150 Hz**: Typically male speech, bass instruments, or low-pitched content
- **150–300 Hz**: Typical female speech, mid-range instruments
- **> 300 Hz**: High-pitched content, children's voices, soprano singing, or high-frequency instruments

### Mean Confidence
- **> 0.7**: Strong, clear periodic signal (likely sung voice or clear instrumental)
- **0.5–0.7**: Moderate periodicity (typical speech, some musical content)
- **< 0.5**: Weak or noisy signal, unvoiced speech, or percussive content

### Temporal Evolution
- **Stable**: Consistent pitch over time (sustained notes, drones, steady speech)
- **Rising**: Upward pitch contour (questions, rising intonation, glissando up)
- **Falling**: Downward pitch contour (statements, falling intonation, glissando down)
- **Variable**: Highly fluctuating pitch (emotional speech, vibrato, complex melodies)

### Sung vs Spoken Detection
- **Sung**: Higher mean confidence (>0.65), lower coefficient of variation (stable pitch within notes), and clear evolution patterns
- **Spoken**: Lower mean confidence, higher variability, less structured pitch contours

## Reliability Tier

- **Tier**: Probabilistic
- **Max Confidence**: 0.75 (capped to reflect the heuristic nature of rule-based QCM mapping)

## Error Handling

- If CREPE fails to install or run, the script handles errors silently and returns `confidence: 0.0` with an error detail.
- If no valid pitch is detected (e.g., silence or pure noise), returns `confidence: 0.0`.

## Limitations

- Requires audio with clear periodic content for reliable pitch tracking
- "Tiny" model trades some accuracy for CPU compatibility and speed
- Polyphonic audio (multiple simultaneous pitches) will yield mixed/unreliable results
- Confidence is capped at 0.75 due to the probabilistic heuristic mapping
