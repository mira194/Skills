#!/usr/bin/env python3
"""
Splits an audio file into 3 equal segments and saves them as WAV files, 
plus a full copy. Outputs a JSON summary.
Uses only the Python standard library (wave module) — no external
dependencies (librosa/soundfile), to avoid environment install issues.

Usage:
    python split_audio.py <input_audio_path> <output_dir>
"""
import sys
import os
import json
import wave

def main():
    if len(sys.argv) != 3:
        print("Usage: python split_audio.py <input_audio_path> <output_dir>")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(audio_path):
        print(json.dumps({"error": f"File not found: {audio_path}"}))
        sys.exit(1)

    try:
        with wave.open(audio_path, 'rb') as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            frames = w.readframes(nframes)

        duration_seconds = nframes / float(framerate)
        frame_size = sampwidth * nchannels  # bytes per frame (all channels)
        samples_per_segment = nframes // 3

        os.makedirs(output_dir, exist_ok=True)

        def write_wav(path, frame_start, frame_end):
            byte_start = frame_start * frame_size
            byte_end = frame_end * frame_size
            with wave.open(path, 'wb') as out:
                out.setnchannels(nchannels)
                out.setsampwidth(sampwidth)
                out.setframerate(framerate)
                out.writeframes(frames[byte_start:byte_end])

        full_path = os.path.join(output_dir, "audio_full.wav")
        write_wav(full_path, 0, nframes)

        seg1_path = os.path.join(output_dir, "audio_seg1.wav")
        write_wav(seg1_path, 0, samples_per_segment)

        seg2_path = os.path.join(output_dir, "audio_seg2.wav")
        write_wav(seg2_path, samples_per_segment, 2 * samples_per_segment)

        seg3_path = os.path.join(output_dir, "audio_seg3.wav")
        write_wav(seg3_path, 2 * samples_per_segment, nframes)

        result = {
            "full": full_path,
            "seg1": seg1_path,
            "seg2": seg2_path,
            "seg3": seg3_path,
            "duration_seconds": duration_seconds,
            "segment_duration": duration_seconds / 3.0
        }
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
