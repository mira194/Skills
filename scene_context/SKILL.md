# scene_context

Classify the acoustic context or scene of an audio file by combining Librosa feature extraction and PANNs CNN14 (AudioSet) audio classification.

## Description
This skill analyzes an audio file to determine its acoustic scene (e.g., urban, nature, indoor, office, street, park) by extracting low-level audio features using Librosa and high-level scene classifications using PANNs CNN14.

## Usage
Use this skill when the user wants to:
- Classify the acoustic scene or environment of an audio file.
- Understand the context of an audio recording (e.g., "Is this recorded in a cafe or a park?").
- Combine low-level signal processing features with deep learning-based audio tagging for robust scene classification.

## Dependencies
- `librosa`
- `numpy`
- `torch`
- `panns_inference`

## Script
Run `python scripts/classify.py --audio_path <path_to_audio>`