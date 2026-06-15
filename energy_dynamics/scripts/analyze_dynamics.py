#!/usr/bin/env python3
"""
Analyze the energy and dynamic range of an audio file using Librosa.
Outputs results as JSON.
"""
import librosa
import numpy as np
import json
import sys
import os

def analyze_audio_dynamics(audio_path, frame_length=2048, hop_length=512):
    if not os.path.exists(audio_path):
        return {"error": f"File not found: {audio_path}"}
    
    try:
        # Load audio at native sample rate
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        return {"error": f"Failed to load audio file: {str(e)}"}
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB (using max RMS as reference to keep values relative)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Compute peak amplitude
    peak_amp = np.max(np.abs(y))
    peak_db = librosa.amplitude_to_db(peak_amp, ref=np.max)
    
    # Crest factor (Peak to mean RMS ratio in dB)
    mean_rms_db = np.mean(rms_db)
    crest_factor_db = peak_db - mean_rms_db
    
    # Dynamic range (99th percentile of RMS vs 1st percentile of RMS)
    rms_percentile_high = np.percentile(rms_db, 99)
    rms_percentile_low = np.percentile(rms_db, 1)
    dynamic_range_db = rms_percentile_high - rms_percentile_low
    
    return {
        "file": os.path.basename(audio_path),
        "duration_seconds": float(len(y) / sr),
        "mean_rms_db": float(np.mean(rms_db)),
        "max_rms_db": float(np.max(rms_db)),
        "min_rms_db": float(np.min(rms_db)),
        "peak_db": float(peak_db),
        "crest_factor_db": float(crest_factor_db),
        "dynamic_range_db": float(dynamic_range_db)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Please provide an audio file path as an argument"}))
        sys.exit(1)
    
    audio_path = sys.argv[1]
    result = analyze_audio_dynamics(audio_path)
    print(json.dumps(result, indent=2))
    
    if "error" in result:
        sys.exit(1)