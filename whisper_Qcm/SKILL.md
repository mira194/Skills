---
name: whisper_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using OpenAI Whisper transcription.
  Use this skill whenever the user provides an audio file (WAV, MP3, FLAC, M4A, OGG) along with
  a question and choices and wants to automatically determine the correct answer based on speech
  content analysis. Also use when the user mentions speech-to-text QCM, audio transcription QA,
  spoken content classification, language detection from audio, or wants to answer questions about
  what is being said in an audio recording. This skill handles any audio classification or QCM task
  where the choices can be answered by analyzing transcribed text, detected language, temporal
  patterns, or speaker characteristics.
---

# Whisper QCM – Speech-to-Text Multiple Choice

Answer multiple-choice questions about audio files using OpenAI Whisper transcription to extract
linguistic features and map them to QCM choices.

## How it works

1. Load the audio file and transcribe it with Whisper (`base` model by default)
2. Extract features from the transcription result:
   - **Language**: detected language code (ISO 639-1) mapped to human-readable names (FR/EN)
   - **Text content**: full transcript for keyword matching against choices
   - **Temporal segments**: start/end times, durations, pause patterns
   - **Speaker estimation**: heuristic speaker count from segment boundaries and conversational markers
3. Detect question type and score each choice using the appropriate strategy
4. Return the choice with the highest score and its confidence

## Dependencies

```
openai-whisper
torch
numpy
```

Install with: `pip install openai-whisper torch numpy`

**Prerequisites:**
- **Python 3.8–3.11** (3.12+ may have dependency issues with tiktoken/torch)
- **ffmpeg** required for audio decoding

**Quick verification:**
```bash
ffmpeg -version > /dev/null 2>&1 && echo "✓ ffmpeg" || echo "✗ ffmpeg missing"
python3 -c "import whisper; print('✓ whisper')" 2>/dev/null || echo "✗ whisper missing"
python3 -c "import torch; print('✓ torch')" 2>/dev/null || echo "✗ torch missing"
```

## Model Selection

| Model | Parameters | VRAM | Best for |
|-------|-----------|------|----------|
| `tiny` | 39M | ~1 GB | Quick drafts, low-resource |
| `base` | 74M | ~1 GB | **Default for QCM** — fast, good enough |
| `small` | 244M | ~2 GB | Higher accuracy, longer audio |
| `medium` | 769M | ~5 GB | Maximum accuracy |

Use `base` as the default for QCM tasks — it's the sweet spot between speed and accuracy.

## Files

- `scripts/whisper_qcm_inference.py` – Main inference script
- `scripts/generate_test_audio.py` – Utility to generate test WAV files
- `evals/evals.json` – Test cases for evaluation

## Usage

### Basic usage (CLI)

```bash
python scripts/whisper_qcm_inference.py \
    --audio path/to/audio.wav \
    --question "What language is being spoken?" \
    --choices '{"A": "French", "B": "Spanish", "C": "Italian"}'
```

Output:
```json
{
  "answer": "A",
  "confidence": 0.87,
  "detail": "Detected language: French (fr). Choice 'A' matches detected language."
}
```

### As a Python function

```python
from scripts.whisper_qcm_inference import run_whisper_qcm

result = run_whisper_qcm(
    audio_path="audio.wav",
    question="What is the main topic of discussion?",
    choices={
        "A": "Technology and AI",
        "B": "Cooking recipes",
        "C": "Sports news"
    }
)
print(result["answer"], result["confidence"])
```

## Question Types

The script auto-detects question type from keywords and applies the appropriate scoring strategy.

### 1. Language Detection

Keywords: `langue`, `language`, `spoken`, `parlée`, `parlé`

Maps Whisper's detected language code to choice text using direct name matching and FR/EN aliases.

```python
run_whisper_qcm("speech.wav", "Quelle langue est parlée?",
    {"A": "Français", "B": "Anglais", "C": "Espagnol"})
# → Uses language code "fr" to match "Français"
```

### 2. Speech Presence

Keywords: `parole`, `speech`, `parle`, `talking`, `y a-t-il`

Binary yes/no scoring based on whether the transcript is empty or not.

```python
run_whisper_qcm("audio.wav", "Y a-t-il de la parole ?",
    {"A": "Oui", "B": "Non"})
# → Checks if transcript is non-empty
```

### 3. Duration Classification

Keywords: `long`, `court`, `short`, `duration`, `durée`, `<`, `>`

Parses numeric thresholds from choice text (e.g., `< 30s` → 30 seconds) or uses short/long heuristics with a 30s default boundary.

```python
run_whisper_qcm("audio.wav", "Le discours est-il long ou court ?",
    {"A": "< 30s", "B": "> 30s"})
# → Compares total_duration_sec against threshold
```

### 4. Speaker Count Estimation

Keywords: `speaker`, `locuteur`, `personne`, `people`, `speaking`

Heuristic estimation based on segment pause patterns (>1.5s gaps) and conversational turn-taking markers (yes/no, d'accord, question marks).

```python
run_whisper_qcm("interview.wav", "Combien de locuteurs ?",
    {"A": "Un", "B": "Deux", "C": "Trois ou plus"})
# → Analyzes segment patterns for speaker estimation
```

### 5. Topic / Content Matching (fallback)

Any question not matching the above types falls back to keyword matching:
- Extracts meaningful keywords from each choice (removes FR/EN stop words)
- Counts keyword occurrences in the transcript
- Scores by match ratio, scaled by transcript length to avoid false positives on short audio

```python
run_whisper_qcm("lecture.wav", "What is the lecture about?",
    {"A": "Machine learning", "B": "Quantum physics", "C": "Art history"})
# → Keywords matched against transcript content
```

## Feature Extraction Details

### Language Detection

Whisper returns a `language` field with the ISO 639-1 code. The script maps 28+ language codes to human-readable names for choice matching.

### Text Content

- Full transcript available for keyword matching
- Stop-word filtering for both English and French (100+ stop words)
- Unicode-aware keyword extraction (handles accented characters: àâäéèêëïîôùûüÿçœæ)

### Temporal Segments

Each segment includes `start`, `end`, `text`, and computed `duration`. Total duration is calculated from first to last segment.

### Speaker Estimation

Heuristic based on:
- **Pause count**: gaps >1.5s between consecutive segments
- **Conversational markers**: regex patterns for turn-taking phrases in FR/EN
- **Score mapping**: 0→0 speakers, 1→1 speaker, 2→2 speakers, 4+→3+ speakers

**Note**: This is approximate. Whisper doesn't perform true diarization. For accurate speaker separation, use a dedicated model like pyannote.

## Limitations

- Audio quality affects transcription accuracy — noisy audio may give poor results
- Whisper doesn't do true speaker diarization — speaker count is estimated heuristically
- Language detection works best with >3 seconds of clear speech
- Very short audio (<1 second) may not transcribe reliably
- Background music or noise can interfere with speech detection
- The `base` model may struggle with heavy accents or technical jargon
- Topic matching uses keyword overlap, not semantic similarity — choices with synonyms may not match well

## Example Scenarios

### Scenario 1: Language Detection
```python
result = run_whisper_qcm(
    audio_path="speech.wav",
    question="Quelle langue est parlée?",
    choices={"A": "Français", "B": "Espagnol", "C": "Anglais"}
)
# → Uses detected language code to match choice
```

### Scenario 2: Topic Classification
```python
result = run_whisper_qcm(
    audio_path="lecture.wav",
    question="What is the lecture about?",
    choices={
        "A": "Machine learning",
        "B": "Quantum physics",
        "C": "Art history"
    }
)
# → Keywords from choices matched against transcript
```

### Scenario 3: Speaker Count
```python
result = run_whisper_qcm(
    audio_path="interview.wav",
    question="How many people are speaking?",
    choices={"A": "One", "B": "Two", "C": "Three or more"}
)
# → Segment pattern analysis for speaker estimation
```

### Scenario 4: Duration Check
```python
result = run_whisper_qcm(
    audio_path="short_clip.wav",
    question="Le discours est-il long ou court ?",
    choices={"A": "< 30s", "B": "> 30s"}
)
# → Compares total duration against 30s threshold
```
