"""
Generate a test WAV file for whisper_qcm evaluation.

Creates synthetic audio with known properties:
- Silent audio (for "no speech" tests)
- Short speech-like audio (for duration tests)
- Language-specific placeholder audio (requires real audio in practice)

Usage:
    python generate_test_audio.py --output test_silence.wav --type silence
    python generate_test_audio.py --output test_short.wav --type short_speech
    python generate_test_audio.py --output test_french.wav --type french_speech
"""

import argparse
import numpy as np
import wave
import struct


def generate_silence(duration_sec=5.0, sample_rate=16000):
    """Generate silent audio."""
    samples = int(duration_sec * sample_rate)
    return np.zeros(samples, dtype=np.float32)


def generate_tone(duration_sec=2.0, sample_rate=16000, freq=440.0):
    """Generate a simple tone (simulates speech-like content)."""
    samples = int(duration_sec * sample_rate)
    t = np.arange(samples) / sample_rate
    # Add some harmonics to make it more speech-like
    signal = (
        0.5 * np.sin(2 * np.pi * freq * t) +
        0.3 * np.sin(2 * np.pi * freq * 2 * t) +
        0.2 * np.sin(2 * np.pi * freq * 3 * t)
    )
    # Apply envelope to avoid clicks
    envelope = np.ones_like(signal)
    fade_len = int(0.05 * sample_rate)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    return signal * envelope


def generate_noise(duration_sec=5.0, sample_rate=16000):
    """Generate white noise (simulates background noise without speech)."""
    samples = int(duration_sec * sample_rate)
    return np.random.randn(samples).astype(np.float32) * 0.1


def save_wav(filepath, signal, sample_rate=16000):
    """Save numpy array as WAV file."""
    # Normalize to int16 range
    signal = np.clip(signal, -1.0, 1.0)
    signal_int16 = (signal * 32767).astype(np.int16)

    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_int16.tobytes())

    print(f"Saved {filepath}: {len(signal)/sample_rate:.1f}s at {sample_rate}Hz")


def main():
    parser = argparse.ArgumentParser(description="Generate test WAV files for whisper_qcm")
    parser.add_argument("--output", required=True, help="Output WAV file path")
    parser.add_argument("--type", choices=["silence", "short_speech", "long_speech", "noise", "french_speech", "english_speech"],
                        default="silence", help="Type of audio to generate")
    parser.add_argument("--duration", type=float, default=None, help="Override duration in seconds")
    args = parser.parse_args()

    if args.type == "silence":
        dur = args.duration or 5.0
        signal = generate_silence(dur)
    elif args.type == "short_speech":
        dur = args.duration or 10.0
        # 1s tone + silence (simulates short speech)
        tone = generate_tone(1.0)
        silence = generate_silence(dur - 1.0)
        signal = np.concatenate([tone, silence])
    elif args.type == "long_speech":
        dur = args.duration or 45.0
        # Multiple tones spaced out (simulates longer speech)
        parts = []
        for i in range(5):
            tone = generate_tone(2.0, freq=440 + i * 100)
            parts.append(tone)
            if i < 4:
                parts.append(generate_silence(8.0))
        signal = np.concatenate(parts)
    elif args.type == "noise":
        dur = args.duration or 5.0
        signal = generate_noise(dur)
    elif args.type in ["french_speech", "english_speech"]:
        # NOTE: These generate tones as placeholders
        # Real evaluation requires actual speech audio files
        dur = args.duration or 10.0
        freq = 440 if args.type == "french_speech" else 520
        signal = generate_tone(dur, freq=freq)
        print(f"  ⚠ Placeholder tone generated. Use real speech audio for '{args.type}' evaluation.")
    else:
        raise ValueError(f"Unknown type: {args.type}")

    save_wav(args.output, signal)


if __name__ == "__main__":
    main()
