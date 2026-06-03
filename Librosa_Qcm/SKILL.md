---
name: librosa_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using Librosa feature extraction.
  Use this skill whenever the user provides a WAV file along with a question and choices and wants to
  automatically determine the correct answer based on audio feature analysis (tempo, RMS energy,
  zero crossing rate, spectral centroid, MFCCs). Also use when the user mentions audio features,
  spectral analysis, tempo detection, audio QCM with librosa, or wants to classify audio based on
  signal processing characteristics rather than deep learning models. This skill is ideal for
  questions about rhythm, pitch, timbre, loudness, speech vs music distinction, and other
  low-level audio properties.
---

# Librosa QCM – Audio Feature-Based Classification

Answer multiple-choice questions about audio files using Librosa's signal processing features.

## How it works

1. Load the WAV file and extract five core feature sets:
   - **Tempo (BPM)** – beats per minute estimation
   - **RMS Energy** – root mean square amplitude (loudness/dynamic range)
   - **Zero Crossing Rate (ZCR)** – rate of sign changes (voiced vs unvoiced, percussive vs tonal)
   - **Spectral Centroid** – center of mass of the spectrum (brightness)
   - **MFCCs** – mel-frequency cepstral coefficients (timbral fingerprint)

2. Analyze feature statistics (mean, variance, distribution) to characterize the audio

3. Map feature patterns to QCM choices using built-in heuristics and configurable rules

4. Return the best-matching choice with confidence based on feature separation

## Dependencies

```
librosa
numpy
```

Install with: `pip install librosa numpy`

## Files

- `scripts/librosa_qcm_inference.py` – Main inference script

## Usage

### Basic usage (CLI)

```bash
python scripts/librosa_qcm_inference.py \
    --wav path/to/audio.wav \
    --question "Is this speech or music?" \
    --choices '{"A": "Speech", "B": "Music"}'
```

Output:
```json
{
  "answer": "A",
  "confidence": 0.85,
  "detail": "High ZCR variance and MFCC pattern indicate speech. RMS energy is moderate and consistent with vocal content."
}
```

### As a Python function

```python
from scripts.librosa_qcm_inference import analyze_audio, answer_qcm

# Extract features
features = analyze_audio("audio.wav")
# Returns: {tempo, rms_mean, rms_var, zcr_mean, zcr_var, spectral_centroid_mean, mfcc_mean, mfcc_var}

# Answer a QCM
result = answer_qcm(
    features=features,
    question="What type of audio is this?",
    choices={"A": "Speech", "B": "Music", "C": "Silence"}
)
print(result["answer"], result["confidence"], result["detail"])
```

## Feature Interpretation Guide

### Tempo (BPM)
- **60–120 BPM**: Typical speech rhythm or slow music
- **120–180 BPM**: Upbeat music, fast speech
- **< 60 BPM**: Ambient, drone, or very slow content
- **> 180 BPM**: Fast electronic music, percussion-heavy content

### RMS Energy
- **< 0.01**: Near silence or very quiet audio
- **0.01–0.1**: Moderate level (typical speech, quiet music)
- **0.1–0.3**: Loud content (music, shouting, dense mix)
- **> 0.3**: Very loud, clipped, or heavily compressed audio

### Zero Crossing Rate
- **Low ZCR (< 0.05)**: Tonal sounds (sustained notes, bass, vowels)
- **Medium ZCR (0.05–0.15)**: Mixed content (speech with consonants, rhythmic music)
- **High ZCR (> 0.15)**: Percussive sounds, noise, fricatives, cymbals

### Spectral Centroid
- **< 1000 Hz**: Bass-heavy, warm sounds (male voice, bass guitar, kick drum)
- **1000–3000 Hz**: Mid-range content (female voice, most instruments)
- **> 3000 Hz**: Bright sounds (cymbals, hi-hats, sibilants, noise)

### MFCCs
- First 3–5 coefficients capture most timbral information
- Speech typically shows distinct MFCC patterns compared to music
- MFCC variance indicates how much the timbre changes over time

## Common QCM Patterns

### Speech vs Music
- **Speech**: Higher ZCR variance, moderate spectral centroid (1–3 kHz), MFCC pattern with formant structure
- **Music**: More stable tempo, broader spectral content, different MFCC distribution

### Loud vs Quiet
- Use RMS energy mean directly
- High RMS → loud, low RMS → quiet

### Fast vs Slow (tempo-based)
- Use estimated BPM
- Higher BPM → fast, lower BPM → slow

### Bright vs Dark (timbre)
- Use spectral centroid
- Higher centroid → bright, lower → dark

### Noisy vs Clean
- High ZCR + high spectral centroid variance → noisy
- Low ZCR + stable features → clean

### Silence detection
- RMS energy < 0.01 across most of the audio → silence or near-silence

## Custom Mapping

For questions that don't match built-in patterns, the script uses a scoring approach:
1. Each choice is associated with expected feature ranges
2. The actual features are compared to each choice's expected ranges
3. The choice with the best match (lowest distance) is selected
4. Confidence is derived from the margin between the top two choices

## Limitations

- Audio must be loadable by librosa (WAV, MP3, FLAC, etc.)
- Feature-based classification is less accurate than deep learning models for complex tasks
- Works best for clear distinctions (speech vs silence, loud vs quiet)
- May struggle with nuanced musical genre classification
- Tempo estimation requires rhythmic content to be reliable
