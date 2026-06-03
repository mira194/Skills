---
name: transcribe
description: >
  Transcribe audio files (WAV/MP3) to text using OpenAI Whisper. Use this skill whenever the user
  wants to convert speech/audio to text, get a transcription, detect the language of an audio file,
  or obtain timestamped segments. Also use when the user mentions Whisper, speech-to-text, STT,
  audio transcription, or wants to extract spoken words from audio. This skill handles any audio
  file containing speech and returns a structured result with full text, detected language, and
  per-segment timestamps.
---

# Transcribe – Whisper Audio Transcription

Transcribe audio files using OpenAI Whisper and return structured results.

## How it works

1. Convert input audio to WAV if needed (MP3 → WAV via FFmpeg)
2. Load the Whisper model (default: `base`)
3. Run transcription with word-level timestamps
4. Return `{text, language, segments}` dict

## Dependencies

```
openai-whisper
torch
ffmpeg
```

Install with:
```bash
pip install openai-whisper torch
# FFmpeg must be installed system-wide: apt install ffmpeg (Linux) / brew install ffmpeg (macOS)
```

## Files

- `scripts/transcribe_audio.py` – Main transcription script

## Usage

### CLI

```bash
python scripts/transcribe_audio.py --audio path/to/audio.wav
python scripts/transcribe_audio.py --audio path/to/audio.mp3 --model medium
```

Output (JSON):
```json
{
  "text": "Full transcription here.",
  "language": "en",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "First sentence."},
    {"start": 3.5, "end": 7.1, "text": "Second sentence."}
  ]
}
```

### As a Python function

```python
from scripts.transcribe_audio import transcribe

result = transcribe("audio.wav")
# result = transcribe("audio.mp3", model_name="small")
print(result["text"])
print(result["language"])
for seg in result["segments"]:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
```

## Model selection

Whisper provides these model sizes (trade accuracy vs speed):

| Model   | Parameters | Speed  | Quality |
|---------|------------|--------|---------|
| tiny    | 39M        | Fastest | Basic   |
| base    | 74M        | Fast   | Good    |
| small   | 244M       | Medium | Better  |
| medium  | 769M       | Slow   | Best    |
| large   | 1550M      | Slowest| Best    |

Default is `base`. Use `small` or `medium` for noisy audio or non-English languages. Use `large` for maximum accuracy on difficult audio.

## Audio format handling

- **WAV**: loaded directly
- **MP3**: converted to WAV via FFmpeg before processing
- **Other formats**: converted via FFmpeg (FLAC, OGG, M4A, etc.)

The script handles conversion automatically — just pass any supported audio path.

## Limitations

- Whisper is optimized for speech; music or non-speech audio will produce garbled text
- Model downloads on first use (~140MB for base, ~1.5GB for large)
- No GPU support without CUDA-enabled PyTorch; CPU inference is slower
- Long files (>30 min) may consume significant memory
