#!/usr/bin/env python3
"""
Splits an audio file into 3 equal segments and saves them as WAV files, 
plus a full copy. Outputs a JSON summary.

Usage:
    python split_audio.py <input_audio_path> <output_dir>
"""
import sys
import os
import json
import librosa
import soundfile as sf

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
        # Load audio (sr=None preserves native sample rate)
        y, sr = librosa.load(audio_path, sr=None)
        duration_seconds = float(librosa.get_duration(y=y, sr=sr))
        
        total_samples = len(y)
        samples_per_segment = total_samples // 3
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full audio
        full_path = os.path.join(output_dir, "audio_full.wav")
        sf.write(full_path, y, sr)
        
        # Save segments
        seg1_path = os.path.join(output_dir, "audio_seg1.wav")
        sf.write(seg1_path, y[:samples_per_segment], sr)
        
        seg2_path = os.path.join(output_dir, "audio_seg2.wav")
        sf.write(seg2_path, y[samples_per_segment:2*samples_per_segment], sr)
        
        seg3_path = os.path.join(output_dir, "audio_seg3.wav")
        sf.write(seg3_path, y[2*samples_per_segment:], sr)
        
        # Output JSON
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