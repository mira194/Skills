#!/usr/bin/env python3
"""
Transcribe an audio file using Whisper with robust error handling.
Usage:
    python transcribe.py audio.mp3 --model small --output-format txt
    python transcribe.py audio.mp3 --model turbo --language fr --output-format srt
    python transcribe.py audio.mp3 --model medium --translate --output-format json
"""

import argparse
import json
import sys
import os
import subprocess


def check_environment():
    """Verify prerequisites and return list of issues."""
    issues = []
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        issues.append("ffmpeg not found — install it for audio decoding")
    try:
        import whisper  # noqa: F401
    except ImportError:
        issues.append("openai-whisper not installed — run: pip install openai-whisper")
    try:
        import torch  # noqa: F401
    except ImportError:
        issues.append("torch not installed — run: pip install torch")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--language", default=None, help="Source language (auto-detect if omitted)")
    parser.add_argument("--output-format", default="txt", choices=["txt", "srt", "vtt", "json"],
                        help="Output format")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--translate", action="store_true", help="Translate to English")
    parser.add_argument("--word-timestamps", action="store_true", help="Include word-level timestamps")
    parser.add_argument("--initial-prompt", default=None, help="Initial prompt for context steering")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = deterministic)")
    args = parser.parse_args()

    # Check environment
    issues = check_environment()
    if issues:
        print("Environment issues found:", file=sys.stderr)
        for issue in issues:
            print(f"  ✗ {issue}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.audio):
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    import whisper

    print(f"Loading model: {args.model}...", file=sys.stderr)
    model = whisper.load_model(args.model)

    print(f"Transcribing: {args.audio}", file=sys.stderr)
    result = model.transcribe(
        args.audio,
        language=args.language,
        task="translate" if args.translate else "transcribe",
        word_timestamps=args.word_timestamps,
        initial_prompt=args.initial_prompt,
        temperature=args.temperature,
        verbose=True,
    )

    # Determine output path
    base = os.path.splitext(os.path.basename(args.audio))[0]
    out_dir = args.output_dir or os.path.dirname(args.audio) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.output_format == "json":
        out_path = os.path.join(out_dir, f"{base}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    elif args.output_format == "srt":
        out_path = os.path.join(out_dir, f"{base}.srt")
        from whisper.utils import WriteSRT
        writer = WriteSRT(out_dir)
        writer(result, args.audio, {})
    elif args.output_format == "vtt":
        out_path = os.path.join(out_dir, f"{base}.vtt")
        from whisper.utils import WriteVTT
        writer = WriteVTT(out_dir)
        writer(result, args.audio, {})
    else:
        out_path = os.path.join(out_dir, f"{base}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

    print(f"\nTranscript saved to: {out_path}", file=sys.stderr)
    print(f"Detected language: {result.get('language', 'unknown')}", file=sys.stderr)


if __name__ == "__main__":
    main()
