---
name: whisper-transcription
description: Transcribe audio files to text using OpenAI's Whisper model (installable via pip as openai-whisper). Use when the user asks to transcribe, convert speech to text, generate subtitles, detect language from audio, or translate spoken audio to English text. Also use when the user mentions audio files (mp3, wav, flac, m4a, ogg, etc.) and wants text output, SRT/VTT subtitles, or a transcript. Supports all Whisper model sizes (tiny, base, small, medium, large, turbo) with guidance on selecting the right model for the task. Make sure to use this skill whenever the user mentions transcription, speech-to-text, audio to text, subtitles, captions, SRT, VTT, or wants to convert any audio/video file into text.
---

# Whisper Transcription

Transcribe audio files to text using OpenAI's Whisper ASR model running locally.

## Prerequisites

Before transcribing, ensure the environment is ready:

1. **Python 3.8–3.11** — Whisper is tested on these versions. Python 3.12+ may work but can have dependency issues (especially with `tiktoken` or `torch`).
2. **ffmpeg** — Required for audio decoding. Install via:
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: `choco install ffmpeg` or `scoop install ffmpeg`
3. **openai-whisper package**: `pip install -U openai-whisper`
   - If `tiktoken` fails to install, install Rust first or run `pip install setuptools-rust` then retry.
   - If `torch` is not available, install it: `pip install torch torchaudio` (CPU-only is fine: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`).

**Quick verification** — run this before transcribing to catch environment issues early:

```bash
ffmpeg -version > /dev/null 2>&1 && echo "✓ ffmpeg" || echo "✗ ffmpeg missing"
python3 -c "import whisper; print('✓ whisper')" 2>/dev/null || echo "✗ whisper missing"
python3 -c "import torch; print('✓ torch')" 2>/dev/null || echo "✗ torch missing"
```

## Model Selection

Choose the right model based on the trade-off between speed, accuracy, and available compute:

| Model | Parameters | English-only | Multilingual | VRAM | Best for |
|-------|-----------|-------------|-------------|------|----------|
| `tiny` | 39M | `tiny.en` | `tiny` | ~1 GB | Quick drafts, low-resource machines |
| `base` | 74M | `base.en` | `base` | ~1 GB | Basic transcription, fast turnaround |
| `small` | 244M | `small.en` | `small` | ~2 GB | Good balance of speed and accuracy |
| `medium` | 769M | `medium.en` | `medium` | ~5 GB | High accuracy, longer audio |
| `large` | 1550M | N/A | `large` | ~10 GB | Maximum accuracy, multilingual |
| `turbo` | 809M | N/A | `turbo` | ~6 GB | Fast large-model quality (no translation) |

**Rules of thumb:**
- Use `.en` variants for English-only audio — they're faster and more accurate for English.
- Use `turbo` when you want near-large quality but faster inference. Note: turbo does **not** support `--task translate`.
- Use `small` as the default for general-purpose transcription — it's the sweet spot for most use cases.
- If the user has no GPU and limited RAM (<4 GB), stick to `tiny` or `base`.

## Basic Transcription

### Python API

```python
import whisper

# Load model (choose size based on model selection guidance above)
model = whisper.load_model("small")

# Transcribe audio file
result = model.transcribe("audio.mp3")

# Result is a dict with keys:
#   text: full transcription string
#   segments: list of dicts with start, end, text, words (optional)
#   language: detected language code (e.g., "en", "fr")
print(result["text"])
```

### CLI

```bash
whisper audio.mp3 --model small --output_format txt
```

## Output Formats

### Plain Text (default)

```python
result = model.transcribe("audio.mp3")
print(result["text"])
```

Or via CLI: `whisper audio.mp3 --model small --output_format txt`

### SRT Subtitles

```python
from whisper.utils import WriteSRT

result = model.transcribe("audio.mp3")
srt_writer = WriteSRT("output_dir")
srt_writer(result, "audio.mp3", {})
```

Or via CLI: `whisper audio.mp3 --model small --output_format srt`

### VTT Subtitles

```python
from whisper.utils import WriteVTT

result = model.transcribe("audio.mp3")
vtt_writer = WriteVTT("output_dir")
vtt_writer(result, "audio.mp3", {})
```

Or via CLI: `whisper audio.mp3 --model small --output_format vtt`

### JSON (segments with timestamps)

```python
import json

result = model.transcribe("audio.mp3")
with open("transcript.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
```

Or via CLI: `whisper audio.mp3 --model small --output_format json`

## Advanced Options

### Specify Language

If you know the spoken language, pass it to skip auto-detection and improve accuracy:

```python
result = model.transcribe("audio.mp3", language="fr")
```

CLI: `whisper audio.mp3 --model small --language French`

### Translate to English

Transcribe non-English audio and output English text:

```python
result = model.transcribe("audio.mp3", task="translate")
```

CLI: `whisper audio.mp3 --model medium --task translate`

Note: `turbo` does not support translation. Use `medium` or `large`.

### Initial Prompt (context steering)

Provide an initial prompt to guide the model — useful for domain-specific vocabulary, names, or jargon:

```python
result = model.transcribe(
    "audio.mp3",
    initial_prompt="This is a medical lecture about cardiology and electrophysiology."
)
```

### Temperature and Sampling

Control randomness for more consistent output:

```python
result = model.transcribe("audio.mp3", temperature=0.0)  # Deterministic
```

Or with multiple temperatures for fallback (Whisper's default behavior):

```python
result = model.transcribe("audio.mp3", temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
```

### Word-Level Timestamps

Get timestamps for individual words (requires a non-tiny model):

```python
result = model.transcribe("audio.mp3", word_timestamps=True)
# result["segments"] will contain "words" key with per-word timing
```

## Handling Long Audio

Whisper processes audio in 30-second chunks internally. For very long files (hours), consider:

1. **Memory management**: Load the model once, transcribe sequentially.
2. **Progress tracking**: Use the `verbose=True` option to see progress.
3. **Chunking for reliability**: If a file fails, split it with ffmpeg and transcribe segments:

```python
import subprocess

def split_audio(input_path, chunk_duration_sec=1800, output_prefix="chunk"):
    """Split audio into chunks for safer transcription."""
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-f", "segment",
        "-segment_time", str(chunk_duration_sec),
        "-c", "copy",
        f"{output_prefix}_%03d.mp3"
    ], check=True)
```

## Common Issues and Fixes

| Issue | Fix |
|-------|-----|
| `ffmpeg not found` | Install ffmpeg for your platform |
| `CUDA out of memory` | Use a smaller model or run on CPU |
| `tiktoken` install fails | Install Rust or `pip install setuptools-rust` |
| Transcription is garbled | Specify the correct `language` parameter |
| Slow transcription on CPU | Use `tiny` or `base` model, or switch to `turbo` |
| `ModuleNotFoundError: whisper` | Run `pip install -U openai-whisper` |
| Python 3.12 compatibility issues | Use a virtualenv with Python 3.11, or install `torch` nightly |
| No output / empty result | Check audio file is valid: `ffprobe audio.mp3` |

## Workflow for Reliable Transcription

When asked to transcribe a file, follow this sequence:

1. **Check environment** — verify ffmpeg, whisper, and torch are available.
2. **Install missing dependencies** — ffmpeg, whisper, torch as needed.
3. **Select model** — ask the user or choose based on file length, language, and available compute. Default to `small`.
4. **Transcribe** — use the Python API for programmatic control, CLI for quick one-offs.
5. **Save output** — in the requested format, or default to `.txt` alongside the original file.
6. **Report** — tell the user the detected language, model used, and where the transcript was saved.

## Example: Full Transcription Script

A robust transcription script is available at `scripts/transcribe.py`. Use it for command-line transcription with full option support:

```bash
python scripts/transcribe.py audio.mp3 --model small --output-format txt
python scripts/transcribe.py audio.mp3 --model turbo --language fr --output-format srt
python scripts/transcribe.py audio.mp3 --model medium --translate --output-format json
```
