---
name: chord_progression_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using Librosa for chord progression detection.
  Use this skill whenever the user provides an audio file along with a question and choices and wants to
  automatically determine the correct answer based on chord analysis (dominant chord, chord sequence with
  timestamps, number of chord changes, average template matching confidence). Also use when the user mentions
  chord progression, harmonic analysis, chord detection, or audio QCM with librosa chroma features.
---

# Chord Progression QCM – Harmonic Analysis

Answer multiple-choice questions about audio files using Librosa's chroma CQT feature extraction and 24 major/minor chord templates.

## How it works

1. Load the audio file and extract chroma CQT features using Librosa.
2. Segment the audio into 1-second windows and compute the average chroma profile for each window.
3. Match each window's chroma profile against a set of 24 predefined chord templates (12 major, 12 minor) using cosine similarity.
4. Analyze the extracted features and map them to the provided QCM choices using heuristic scoring based on:
   - Dominant chord detected (most frequent)
   - Sequence of chords with timestamps
   - Number of chord changes
   - Average confidence of template matching
5. Return the best-matching choice with a confidence score (capped at 0.60 for the Heuristic tier).

## Dependencies

```
librosa
numpy
soundfile
```

**Recommended setup (using venv):**

```bash
cd skills/chord_progression_qcm
python3 -m venv .venv
. .venv/bin/activate
pip install librosa numpy soundfile
```

## Files

- `scripts/chord_progression_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python scripts/chord_progression_qcm_inference.py \
    --audio path/to/audio.wav \
    --payload '{"question": "Quel est l'accord dominant ?", "choices": {"A": "C majeur", "B": "A mineur", "C": "G majeur"}}'
```

Output:
```json
{
  "answer": "A",
  "confidence": 0.60,
  "detail": "Accord dominant: C majeur, Changements: 3, Confiance moy: 0.75, Séquence: C majeur(0.0s), C majeur(1.0s), A mineur(2.0s)..."
}
```

### As a Python function

```python
from scripts.chord_progression_qcm_inference import process_qcm

result = process_qcm(
    audio_path="audio.wav",
    payload={
        "question": "Combien de changements d'accords y a-t-il ?",
        "choices": {"A": "1 à 2", "B": "3 à 5", "C": "Plus de 5"}
    }
)
print(result["answer"], result["confidence"], result["detail"])
```

## Feature Interpretation Guide

### Dominant Chord
The most frequently detected chord template across all 1-second windows in the audio file.

### Chord Changes
The number of times the detected chord changes from one 1-second window to the next. A higher number indicates a more harmonically complex or dynamic progression.

### Average Confidence
The mean cosine similarity between the detected chroma profiles and their best-matching chord templates. Values closer to 1.0 indicate clear, strong harmonic content, while lower values suggest ambiguity, noise, or non-chordal content (e.g., percussion, speech).

## Reliability Tier

- **Tier**: Heuristic
- **Max Confidence**: 0.60
- **Notes**: Chord detection is highly dependent on the presence of clear harmonic content (e.g., guitars, pianos, synthesized chords). A cappella speech, solo percussion, or heavily distorted audio may yield unreliable chord estimates. Errors are handled silently, returning `confidence: 0.0` on failure.

## Limitations

- Requires clear harmonic or polyphonic content for accurate chord detection.
- Librosa's `chroma_cqt` may struggle with atonal music, heavy distortion, or dominant percussive elements.
- The 1-second windowing may miss very rapid chord changes (e.g., in fast jazz or complex classical pieces).
- Audio must be loadable by librosa/soundfile.