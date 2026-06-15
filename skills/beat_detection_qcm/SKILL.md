---
name: beat_detection_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using Librosa for beat and tempo detection.
  Use this skill whenever the user provides an audio file along with a question and choices and wants to
  automatically determine the correct answer based on beat analysis (BPM, number of beats, tempo regularity,
  timestamps of first beats). Also use when the user mentions beat detection, tempo estimation, rhythm analysis,
  or audio QCM with librosa beat tracking.
---

# Beat Detection QCM – Tempo and Rhythm Analysis

Answer multiple-choice questions about audio files using Librosa's beat tracking and tempo estimation features.

## How it works

1. Load the audio file and extract beat/tempo features using Librosa:
   - **BPM (Tempo)**: Estimated beats per minute
   - **Number of beats**: Total detected beat frames
   - **Tempo regularity**: Standard deviation of inter-beat intervals (in seconds)
   - **First beats**: Timestamps of the first 5 detected beats

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
cd skills/beat_detection_qcm
python3 -m venv .venv
. .venv/bin/activate
pip install librosa numpy soundfile
```

## Files

- `scripts/beat_detection_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python scripts/beat_detection_qcm_inference.py \
    --audio path/to/audio.wav \
    --payload '{"question": "Quel est le tempo approximatif ?", "choices": {"A": "60 BPM", "B": "120 BPM", "C": "180 BPM"}}'
```

Output:
```json
{
  "answer": "B",
  "confidence": 0.80,
  "detail": "BPM: 118.5, Beats: 240, Régularité (std): 0.045s, Premiers beats: [0.12, 0.62, 1.11]"
}
```

### As a Python function

```python
from scripts.beat_detection_qcm_inference import process_qcm

result = process_qcm(
    audio_path="audio.wav",
    payload={
        "question": "Le rythme est-il régulier ?",
        "choices": {"A": "Oui, très régulier", "B": "Non, très variable"}
    }
)
print(result["answer"], result["confidence"], result["detail"])
```

## Feature Interpretation Guide

### BPM (Tempo)
- **< 60**: Very slow (ambient, drone, adagio)
- **60–90**: Slow to moderate (hip-hop, ballad, walking pace)
- **90–120**: Moderate (pop, house, typical speech rhythm)
- **120–140**: Upbeat (techno, upbeat pop, allegro)
- **> 140**: Fast (drum and bass, hardcore, presto)

### Tempo Regularity (Standard Deviation of Inter-Beat Intervals)
- **< 0.05s**: Highly regular (metronomic, electronic music, drum machine)
- **0.05s – 0.15s**: Moderately regular (human-played music with slight groove)
- **> 0.15s**: Irregular (rubato, free jazz, complex polyrhythms, or poor detection)

### Number of Beats
- Useful for estimating the density of rhythmic events in the audio clip.

## Reliability Tier

- **Tier**: Analytic
- **Max Confidence**: 0.90
- **Notes**: Beat detection is highly dependent on the presence of percussive or rhythmic content. A cappella speech or ambient drones may yield unreliable BPM estimates. Errors are handled silently, returning `confidence: 0.0` on failure.

## Limitations

- Requires rhythmic or percussive content for accurate BPM estimation.
- Librosa's `beat_track` may struggle with highly syncopated or polyrhythmic music.
- The `bpm = float(np.atleast_1d(tempo)[0])` fix is applied to ensure compatibility with librosa >= 0.10.
- Audio must be loadable by librosa/soundfile.