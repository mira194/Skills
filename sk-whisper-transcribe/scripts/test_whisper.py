#!/usr/bin/env python3
"""
Test script for whisper-transcription skill.
Run this in an environment with Python 3.8-3.11, ffmpeg, and openai-whisper installed.

Usage:
    python test_whisper.py [--model MODEL] [--audio AUDIO_FILE]

If no audio file is provided, generates a synthetic test using TTS (if available)
or uses a known test URL.
"""

import subprocess
import sys
import os
import json
import time


def check_env():
    """Verify all prerequisites are met."""
    issues = []
    
    # Python version
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 8 or minor > 11:
        issues.append(f"Python {major}.{minor} — Whisper officially supports 3.8-3.11")
    
    # ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ ffmpeg: {result.stdout.splitlines()[0]}")
        else:
            issues.append("ffmpeg found but returned error")
    except FileNotFoundError:
        issues.append("ffmpeg not found in PATH")
    
    # whisper
    try:
        import whisper
        print(f"✓ whisper: {whisper.__version__ if hasattr(whisper, '__version__') else 'installed'}")
    except ImportError:
        issues.append("openai-whisper not installed (pip install openai-whisper)")
    
    # torch
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        device = "CUDA" if has_cuda else "CPU"
        print(f"✓ torch: {torch.__version__} ({device})")
    except ImportError:
        issues.append("torch not installed")
    
    return len(issues) == 0, issues


def test_basic_transcription(audio_path, model="tiny"):
    """Test 1: Basic transcription of an audio file."""
    print(f"\n{'='*60}")
    print(f"TEST 1: Basic transcription (model={model})")
    print(f"{'='*60}")
    
    import whisper
    
    start = time.time()
    model_obj = whisper.load_model(model)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    
    start = time.time()
    result = model_obj.transcribe(audio_path, verbose=False)
    transcribe_time = time.time() - start
    print(f"Transcription completed in {transcribe_time:.2f}s")
    
    text = result.get("text", "").strip()
    language = result.get("language", "unknown")
    
    print(f"Detected language: {language}")
    print(f"Transcript length: {len(text)} chars")
    print(f"First 200 chars: {text[:200]}...")
    
    return {
        "test": "basic_transcription",
        "passed": len(text) > 0,
        "load_time": round(load_time, 2),
        "transcribe_time": round(transcribe_time, 2),
        "language": language,
        "text_length": len(text),
    }


def test_output_formats(audio_path, model="tiny"):
    """Test 2: Multiple output formats."""
    print(f"\n{'='*60}")
    print(f"TEST 2: Output formats")
    print(f"{'='*60}")
    
    import whisper
    from whisper.utils import WriteSRT, WriteVTT
    import tempfile
    
    model_obj = whisper.load_model(model)
    result = model_obj.transcribe(audio_path, verbose=False)
    
    formats_tested = {}
    
    # TXT
    txt_content = result["text"]
    formats_tested["txt"] = len(txt_content) > 0
    
    # JSON
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    formats_tested["json"] = len(json_str) > 0
    
    # SRT
    with tempfile.TemporaryDirectory() as tmpdir:
        srt_writer = WriteSRT(tmpdir)
        srt_writer(result, audio_path, {})
        srt_files = [f for f in os.listdir(tmpdir) if f.endswith(".srt")]
        formats_tested["srt"] = len(srt_files) > 0
    
    # VTT
    with tempfile.TemporaryDirectory() as tmpdir:
        vtt_writer = WriteVTT(tmpdir)
        vtt_writer(result, audio_path, {})
        vtt_files = [f for f in os.listdir(tmpdir) if f.endswith(".vtt")]
        formats_tested["vtt"] = len(vtt_files) > 0
    
    for fmt, passed in formats_tested.items():
        print(f"  {'✓' if passed else '✗'} {fmt}")
    
    return {
        "test": "output_formats",
        "passed": all(formats_tested.values()),
        "formats": formats_tested,
    }


def test_language_detection(audio_path, model="tiny"):
    """Test 3: Language auto-detection."""
    print(f"\n{'='*60}")
    print(f"TEST 3: Language detection")
    print(f"{'='*60}")
    
    import whisper
    
    model_obj = whisper.load_model(model)
    result = model_obj.transcribe(audio_path, verbose=False)
    
    language = result.get("language", "")
    passed = len(language) == 2  # ISO 639-1 code
    
    print(f"Detected language: {language}")
    print(f"{'✓' if passed else '✗'} Valid language code")
    
    return {
        "test": "language_detection",
        "passed": passed,
        "language": language,
    }


def test_specified_language(audio_path, model="tiny"):
    """Test 4: Transcription with specified language."""
    print(f"\n{'='*60}")
    print(f"TEST 4: Specified language")
    print(f"{'='*60}")
    
    import whisper
    
    # First detect language
    model_obj = whisper.load_model(model)
    auto_result = model_obj.transcribe(audio_path, verbose=False)
    detected_lang = auto_result.get("language", "en")
    
    # Transcribe with explicit language
    result = model_obj.transcribe(audio_path, language=detected_lang, verbose=False)
    text = result.get("text", "").strip()
    
    passed = len(text) > 0
    print(f"Specified language: {detected_lang}")
    print(f"Transcript length: {len(text)} chars")
    print(f"{'✓' if passed else '✗'} Transcription with specified language")
    
    return {
        "test": "specified_language",
        "passed": passed,
        "language": detected_lang,
    }


def test_translation(audio_path, model="small"):
    """Test 5: Translation to English (requires non-English audio)."""
    print(f"\n{'='*60}")
    print(f"TEST 5: Translation to English")
    print(f"{'='*60}")
    
    import whisper
    
    # Note: turbo doesn't support translation
    if model == "turbo":
        print("  ⊘ Skipped — turbo doesn't support translation")
        return {"test": "translation", "passed": None, "note": "skipped"}
    
    model_obj = whisper.load_model(model)
    result = model_obj.transcribe(audio_path, task="translate", verbose=False)
    
    text = result.get("text", "").strip()
    passed = len(text) > 0
    
    print(f"Translated text length: {len(text)} chars")
    print(f"First 200 chars: {text[:200]}...")
    print(f"{'✓' if passed else '✗'} Translation")
    
    return {
        "test": "translation",
        "passed": passed,
    }


def test_word_timestamps(audio_path, model="base"):
    """Test 6: Word-level timestamps."""
    print(f"\n{'='*60}")
    print(f"TEST 6: Word-level timestamps")
    print(f"{'='*60}")
    
    import whisper
    
    # Word timestamps require non-tiny model
    if model == "tiny":
        model = "base"
        print(f"  ↑ Upgraded to base (word timestamps need non-tiny model)")
    
    model_obj = whisper.load_model(model)
    result = model_obj.transcribe(audio_path, word_timestamps=True, verbose=False)
    
    has_words = False
    for segment in result.get("segments", []):
        if "words" in segment and len(segment["words"]) > 0:
            has_words = True
            sample = segment["words"][0]
            print(f"  Sample word: {sample.get('word', '?')} start={sample.get('start', '?')} end={sample.get('end', '?')}")
            break
    
    print(f"{'✓' if has_words else '✗'} Word-level timestamps")
    
    return {
        "test": "word_timestamps",
        "passed": has_words,
    }


def generate_test_audio(output_path="test_audio.wav"):
    """Generate a simple test audio file using Python's built-in wave module."""
    import wave
    import struct
    import math
    
    print(f"Generating test audio: {output_path}")
    
    # Generate a simple tone with some silence
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz
    
    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Simple sine wave with fade in/out
        envelope = min(1.0, t * 10) * min(1.0, (duration - t) * 10)
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t) * envelope)
        samples.append(sample)
    
    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for sample in samples:
            wav_file.writeframes(struct.pack("<h", sample))
    
    print(f"✓ Generated {output_path} ({duration}s, {sample_rate}Hz)")
    return output_path


def run_all_tests(audio_path=None, model="tiny"):
    """Run the full test suite."""
    print("=" * 60)
    print("WHISPER TRANSCRIPTION — TEST SUITE")
    print("=" * 60)
    
    # Check environment
    env_ok, issues = check_env()
    if not env_ok:
        print(f"\n✗ Environment issues found:")
        for issue in issues:
            print(f"  ✗ {issue}")
        print("\nFix these before running tests:")
        if any("ffmpeg" in i for i in issues):
            print("  sudo apt install ffmpeg  (Ubuntu/Debian)")
            print("  brew install ffmpeg      (macOS)")
        if any("whisper" in i for i in issues):
            print("  pip install openai-whisper")
        if any("torch" in i for i in issues):
            print("  pip install torch torchaudio")
        sys.exit(1)
    
    # Generate test audio if needed
    if audio_path is None:
        audio_path = generate_test_audio()
    elif not os.path.isfile(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)
    
    # Run tests
    results = []
    
    results.append(test_basic_transcription(audio_path, model))
    results.append(test_output_formats(audio_path, model))
    results.append(test_language_detection(audio_path, model))
    results.append(test_specified_language(audio_path, model))
    results.append(test_word_timestamps(audio_path, model))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r["passed"] is True)
    failed = sum(1 for r in results if r["passed"] is False)
    skipped = sum(1 for r in results if r["passed"] is None)
    
    for r in results:
        status = "✓" if r["passed"] is True else ("✗" if r["passed"] is False else "⊘")
        print(f"  {status} {r['test']}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to test_results.json")
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test whisper-transcription skill")
    parser.add_argument("--model", default="tiny", help="Whisper model to use for testing")
    parser.add_argument("--audio", default=None, help="Path to test audio file")
    args = parser.parse_args()
    
    success = run_all_tests(audio_path=args.audio, model=args.model)
    sys.exit(0 if success else 1)
