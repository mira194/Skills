#!/usr/bin/env python3
"""
Transcribe audio files using OpenAI Whisper.

Usage:
    python transcribe_audio.py --audio path/to/audio.wav
    python transcribe_audio.py --audio path/to/audio.mp3 --model medium

Output: JSON dict with text, language, and segments.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _convert_to_wav(audio_path: str) -> str:
    """Convert any supported audio to 16kHz mono WAV via FFmpeg."""
    audio_path = str(audio_path)
    if audio_path.lower().endswith(".wav"):
        return audio_path

    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    return wav_path


def transcribe(audio_path: str, model_name: str = "base", device: str = None) -> dict:
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to audio file (WAV, MP3, or other FFmpeg-supported format).
        model_name: Whisper model size (tiny, base, small, medium, large).
        device: 'cpu', 'cuda', or None (auto-detect).

    Returns:
        dict with keys: text (str), language (str), segments (list of {start, end, text})
    """
    import whisper

    # Convert to WAV if needed
    wav_path = _convert_to_wav(audio_path)
    cleanup_temp = wav_path != audio_path

    try:
        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(wav_path)

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            })

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": segments,
        }
    finally:
        if cleanup_temp and os.path.exists(wav_path):
            os.remove(wav_path)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("--audio", required=True, help="Path to audio file (WAV/MP3)")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--device", default=None, help="Device: cpu, cuda, or auto")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    result = transcribe(args.audio, model_name=args.model, device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
