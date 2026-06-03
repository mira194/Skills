---
name: demucs_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using Demucs source separation.
  Use this skill whenever the user provides a WAV/MP3 file along with a question and choices and wants to
  automatically determine the correct answer based on separated audio source analysis (vocals, drums, bass, other).
  Also use when the user mentions source separation, Demucs, htdemucs, stem separation, vocal isolation,
  instrument separation, or wants to analyze audio by separating and measuring the energy of individual sources.
  This skill handles any audio classification task where the choices can be mapped to relative energy levels
  of separated stems (e.g., "is vocals dominant?", "which instrument is loudest?", "compare speech vs music energy").
---

# Demucs QCM – Source Separation-Based Audio Classification

Answer multiple-choice questions about audio files by separating the audio into 4 stems using Demucs (htdemucs), computing RMS energy per stem, and mapping the results to QCM choices.

## How it works

1. Load the WAV/MP3 file and run Demucs htdemucs to separate into 4 stems: vocals, drums, bass, other
2. Compute RMS energy for each separated stem
3. Analyze the relative energy distribution across stems
4. Map each QCM choice to expected stem energy patterns using built-in heuristics
5. Return the best-matching choice with confidence based on energy separation

## Dependencies

```
demucs
torch
librosa
numpy
```

Install with: `pip install demucs torch librosa numpy`

## Files

- `scripts/demucs_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python scripts/demucs_qcm_inference.py \
    --audio path/to/audio.wav \
    --question "Which source is most prominent?" \
    --choices '{"A": "Vocals", "B": "Drums", "C": "Bass", "D": "Other"}'
```

Output:
```json
{
  "answer": "A",
  "confidence": 0.78,
  "detail": "Vocals stem has highest RMS energy (0.245) compared to drums (0.112), bass (0.089), other (0.156). Vocal dominance indicates choice A is correct."
}
```

### As a Python function

```python
from scripts.demucs_qcm_inference import separate_and_analyze, answer_qcm

# Separate and get stem energies
stems = separate_and_analyze("audio.wav")
# Returns: {"vocals": rms_float, "drums": rms_float, "bass": rms_float, "other": rms_float}

# Answer a QCM
result = answer_qcm(
    stems=stems,
    question="What is the dominant element?",
    choices={"A": "Vocals", "B": "Drums", "C": "Bass", "D": "Other"}
)
print(result["answer"], result["confidence"], result["detail"])
```

## Stem Energy Interpretation Guide

### RMS Energy per Stem
- **Vocals**: Speech, singing, human voice content
- **Drums**: Percussion, rhythm section, beats
- **Bass**: Bass guitar, kick drum low-end, sub-bass
- **Other**: Instruments, ambient sounds, effects, background

### Common QCM Patterns

#### Dominant Source
- Compare RMS energies directly
- Highest energy stem → dominant source

#### Speech vs Music
- **Speech dominant**: vocals energy >> other stems
- **Music dominant**: (drums + bass + other) >> vocals

#### Vocal presence detection
- vocals > 0.05 RMS → vocals present
- vocals < 0.01 RMS → vocals absent or very quiet

#### Percussive content
- drums > bass and drums > other → percussion-heavy
- drums < 0.01 → little to no percussion

#### Bass-heavy content
- bass > 0.1 → strong bass presence
- bass > vocals → bass-dominant mix

#### Instrumental vs Vocal
- vocals / (vocals + other) > 0.6 → vocal-forward
- vocals / (vocals + other) < 0.3 → instrumental

## Custom Mapping

For questions that don't match built-in patterns, the script uses a scoring approach:
1. Each choice is associated with expected stem energy patterns
2. The actual stem energies are compared to each choice's expected patterns
3. The choice with the best match (lowest distance) is selected
4. Confidence is derived from the margin between the top two choices

## Limitations

- Audio must be loadable by demucs (WAV, MP3, FLAC, etc.)
- Demucs separation is compute-intensive; expect 10-60s processing time depending on audio length
- Separation quality degrades with heavily compressed or low-bitrate audio
- RMS energy alone cannot distinguish between similar-frequency sources within a stem
- Works best for clear distinctions (vocals vs instrumental, speech vs music)
- The htdemucs model requires downloading on first use (~2GB)
