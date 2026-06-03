#!/usr/bin/env python3
"""Generate test WAV files with known audio characteristics for librosa_qcm evaluation."""

import numpy as np
import sys

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Install dependencies: pip install librosa soundfile numpy")
    sys.exit(1)

sr = 22050
duration = 3.0  # seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)


def save_wav(y, filename, sr=sr):
    """Normalize and save as WAV."""
    y = y / (np.max(np.abs(y)) + 1e-8) * 0.8
    sf.write(filename, y, sr)
    print(f"Generated: {filename} (duration={len(y)/sr:.1f}s, sr={sr})")


# 1. Speech-like: modulated noise with formant-like structure
# Simulate speech with amplitude modulation at ~4 Hz (syllable rate)
speech = np.random.randn(len(t)) * 0.3
# Apply bandpass filter effect (simulate vocal formants around 1-3 kHz)
from scipy.signal import butter, lfilter
b, a = butter(4, [800/(sr/2), 3500/(sr/2)], btype='band')
speech = lfilter(b, a, speech)
# Amplitude modulation at syllable rate
modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4 Hz = typical syllable rate
speech = speech * modulator
# Add some silence gaps
speech[int(sr*1.2):int(sr*1.5)] *= 0.01
speech[int(sr*2.2):int(sr*2.5)] *= 0.01
save_wav(speech, "speech_test.wav")


# 2. Music-like: harmonic tones with clear rhythm
music = np.zeros_like(t)
# Add a melody: alternating notes at 120 BPM (2 beats/sec)
bpm = 120
beat_duration = 60.0 / bpm  # seconds per beat
notes = [261.63, 329.63, 392.00, 329.63, 261.63, 392.00, 349.23, 293.66]  # C4, E4, G4, etc.
for i, freq in enumerate(notes):
    start = int(i * beat_duration * sr)
    end = int((i + 0.8) * beat_duration * sr)
    if start < len(t):
        note_t = t[start:end] - t[start]
        music[start:end] += np.sin(2 * np.pi * freq * note_t) * 0.5
        # Add harmonics
        music[start:end] += np.sin(2 * np.pi * freq * 2 * note_t) * 0.2
        music[start:end] += np.sin(2 * np.pi * freq * 3 * note_t) * 0.1
save_wav(music, "music_test.wav")


# 3. Silence: very low amplitude noise
silence = np.random.randn(len(t)) * 0.001
save_wav(silence, "silence_test.wav")


# 4. Loud audio: full-amplitude square wave (clipping simulation)
loud = np.sign(np.sin(2 * np.pi * 440 * t)) * 0.95
save_wav(loud, "loud_test.wav")


# 5. Quiet audio: low-amplitude sine wave
quiet = np.sin(2 * np.pi * 440 * t) * 0.05
save_wav(quiet, "quiet_test.wav")


# 6. Fast tempo: rapid beats at 180 BPM
fast = np.zeros_like(t)
fast_bpm = 180
fast_beat = 60.0 / fast_bpm
for i in range(int(duration / fast_beat)):
    start = int(i * fast_beat * sr)
    end = start + int(0.05 * sr)  # 50ms clicks
    if start < len(t):
        fast[start:end] += np.sin(2 * np.pi * 1000 * (t[start:end] - t[start])) * 0.8
save_wav(fast, "fast_tempo_test.wav")


# 7. Slow tempo: slow beats at 60 BPM
slow = np.zeros_like(t)
slow_bpm = 60
slow_beat = 60.0 / slow_bpm
for i in range(int(duration / slow_beat)):
    start = int(i * slow_beat * sr)
    end = start + int(0.1 * sr)  # 100ms clicks
    if start < len(t):
        slow[start:end] += np.sin(2 * np.pi * 200 * (t[start:end] - t[start])) * 0.8
save_wav(slow, "slow_tempo_test.wav")


# 8. Noisy/bright: white noise
noisy = np.random.randn(len(t)) * 0.5
# Boost high frequencies
b, a = butter(2, 3000/(sr/2), btype='high')
noisy = lfilter(b, a, noisy)
save_wav(noisy, "noisy_test.wav")


print("\nAll test files generated. Run the QCM script against these files to evaluate.")
