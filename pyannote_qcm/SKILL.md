---
name: pyannote_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using pyannote.audio for speaker diarization.
  Use this skill whenever the user provides an audio file along with a question and choices and wants to
  automatically determine the correct answer based on speaker diarization analysis (number of speakers,
  temporal segments, duration per speaker, speech overlaps). Also use when the user mentions speaker
  diarization, pyannote, speaker counting, or wants to analyze who speaks when in an audio recording.
---

# Pyannote QCM – Speaker Diarization-Based Audio Classification

Answer multiple-choice questions about audio files by performing speaker diarization using `pyannote.audio` (model: `pyannote/speaker-diarization-3.1`) and mapping the extracted speaker features to QCM choices.

## How it works

1. Ensure `pyannote.audio` is installed (installs via pip if missing in the active venv).
2. Load the audio file and run the `pyannote/speaker-diarization-3.1` pipeline.
3. Extract diarization features:
   - **Number of detected speakers**
   - **Temporal segments** of each speaker (start, end, duration)
   - **Duration of speech** per speaker
   - **Speech overlaps** (heuristic estimation)
4. Analyze the question and score each choice based on how well it matches the extracted features.
5. Return the best-matching choice with a probabilistic confidence (capped at 0.75).

## Dependencies

```
pyannote.audio
torch
torchaudio
```

Install with: `pip install pyannote.audio torch torchaudio`

**Prerequisites:**
- **Python 3.8–3.11**
- **ffmpeg** required for audio decoding.
- **Hugging Face Token**: The `pyannote/speaker-diarization-3.1` model requires acceptance of its terms on Hugging Face and a valid `HF_TOKEN` environment variable.

**Quick verification:**
```bash
ffmpeg -version > /dev/null 2>&1 && echo "✓ ffmpeg" || echo "✗ ffmpeg missing"
python3 -c "import pyannote.audio; print('✓ pyannote.audio')" 2>/dev/null || echo "✗ pyannote.audio missing"
```

## Files

- `scripts/pyannote_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python scripts/pyannote_qcm_inference.py \
    --audio path/to/audio.wav \
    --question "How many speakers are in this audio?" \
    --choices '{"A": "One", "B": "Two", "C": "Three or more"}'
```

Output:
```json
{
  "answer": "B",
  "confidence": 0.75,
  "detail": "Match: detected 2 speakers",
  "features": {
    "num_speakers": 2,
    "speaker_durations": {"SPEAKER_00": 15.4, "SPEAKER_01": 12.1},
    "segments_count": 8
  }
}
```

### As a Python function

```python
from scripts.pyannote_qcm_inference import run_pyannote_qcm

result = run_pyannote_qcm(
    audio_path="interview.wav",
    question="Qui a parlé le plus longtemps ?",
    choices={
        "A": "Locuteur 0",
        "B": "Locuteur 1",
        "C": "Ils ont parlé autant"
    }
)
print(result["answer"], result["confidence"])
```

## Question Types & Feature Matching

The script auto-detects question intent from keywords and applies the appropriate scoring strategy.

### 1. Speaker Count
Keywords: `locuteur`, `speaker`, `personne`, `people`, `combien`

Matches the detected number of unique speakers against numeric values in the choices.

### 2. Speech Duration
Keywords: `durée`, `duration`, `long`, `court`, `plus longtemps`

Identifies the speaker with the maximum cumulative duration and matches against choices referencing that speaker.

### 3. Speech Overlaps
Keywords: `chevauchement`, `overlap`, `simultané`, `en même temps`

Provides heuristic scoring. Note: standard diarization outputs partitions, so explicit overlap detection is limited without a dedicated overlap model, but the script accounts for multi-speaker complexity.

### 4. Fallback Keyword Matching
If no specific pattern matches, the script performs basic keyword overlap between the choice text and known diarization terms (e.g., "0", "1", "2", "locuteur", "speaker").

## Reliability & Error Handling

- **Reliability Tier**: Probabilistic. Maximum confidence is strictly capped at **0.75** due to the heuristic nature of mapping diarization features to arbitrary QCM choices.
- **Silent Failure**: If the model fails to load (e.g., missing `HF_TOKEN`), the audio is invalid, or any exception occurs, the script catches the error silently and returns `{"answer": "", "confidence": 0.0, "detail": "Error during inference"}`.
- **Virtual Environment**: The script attempts to auto-install `pyannote.audio` via pip into the active Python environment (e.g., `venv`) if it is not already available.

## Limitations

- Requires a valid Hugging Face token (`HF_TOKEN`) with access to `pyannote/speaker-diarization-3.1`.
- Diarization is compute-intensive; expect 10-60s processing time depending on audio length and hardware.
- Standard diarization models output non-overlapping partitions. True overlap detection requires a separate overlap detection model.
- Speaker labels (e.g., "SPEAKER_00") are arbitrary and may not correspond to "Speaker 1" in a human sense across different runs.
- Audio quality, background noise, and music can significantly degrade diarization accuracy.