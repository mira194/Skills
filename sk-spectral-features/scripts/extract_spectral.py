#!/usr/bin/env python3
"""
Extract spectral features from an audio file using Librosa.
"""
import argparse
import json
import numpy as np
import librosa

def extract_spectral_features(audio_path, n_mfcc=13):
    """
    Extract spectral features from an audio file.
    
    Args:
        audio_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC coefficients to extract.
        
    Returns:
        dict: Dictionary containing extracted features and their statistics.
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    # Helper function to compute stats
    def get_stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "var": float(np.var(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
    
    # Compute MFCC stats per coefficient
    mfcc_stats = {}
    for i in range(n_mfcc):
        mfcc_stats[f"mfcc_{i}"] = get_stats(mfccs[i])
    
    features = {
        "file": audio_path,
        "duration_seconds": float(duration),
        "sample_rate": int(sr),
        "features": {
            "spectral_centroid": get_stats(spec_cent),
            "spectral_rolloff": get_stats(spec_rolloff),
            "spectral_bandwidth": get_stats(spec_bw),
            "mfcc": mfcc_stats
        }
    }
    
    return features

def main():
    parser = argparse.ArgumentParser(description="Extract spectral features from an audio file.")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--output", default="features.json", help="Output JSON file path")
    parser.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCC coefficients")
    
    args = parser.parse_args()
    
    try:
        features = extract_spectral_features(args.audio, n_mfcc=args.n_mfcc)
        with open(args.output, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"Features successfully extracted and saved to {args.output}")
    except Exception as e:
        print(f"Error extracting features: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()