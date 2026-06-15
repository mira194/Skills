#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Classify acoustic scene using Librosa and PANNs")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(json.dumps({"error": "Audio file not found"}))
        sys.exit(1)

    try:
        import librosa
    except ImportError:
        print(json.dumps({"error": "librosa is not installed. Please install it via 'pip install librosa'"}))
        sys.exit(1)

    try:
        import torch
        from panns_inference import AudioTagging, labels
    except ImportError:
        print(json.dumps({"error": "panns_inference or torch is not installed. Please install via 'pip install torch panns-inference'"}))
        sys.exit(1)

    # 1. Extract Librosa features (low-level)
    y, sr = librosa.load(args.audio_path, sr=22050, mono=True)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    librosa_features = {
        "mfcc_mean": mfcc.tolist(),
        "spectral_centroid_mean": spectral_centroid,
        "zero_crossing_rate_mean": zero_crossing_rate
    }

    # 2. Extract PANNs features (high-level AudioSet tags)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    at = AudioTagging(checkpoint_path=None, device=device)
    
    # PANNs expects 32000 Hz
    audio_panns, _ = librosa.load(args.audio_path, sr=32000, mono=True)
    audio_tensor = torch.Tensor(audio_panns).unsqueeze(0).to(device)
    
    clipwise_output, _ = at.inference(audio_tensor)
    clipwise_output = clipwise_output.detach().cpu().numpy()[0]
    
    # Get top 5 labels
    top_indices = np.argsort(clipwise_output)[-5:][::-1]
    top_labels = [{"label": labels[idx], "score": float(clipwise_output[idx])} for idx in top_indices]

    result = {
        "audio_path": args.audio_path,
        "librosa_features": librosa_features,
        "panns_top_labels": top_labels
    }
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()