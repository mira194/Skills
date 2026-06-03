---
name: audio_qcm
description: >
  Answer multiple-choice questions (QCM) about audio content using PANNs CNN14 audio classification.
  Use this skill whenever the user provides a WAV file along with a question and choices and wants to
  automatically determine the correct answer based on audio content analysis. Also use when the user
  mentions audio classification, audio QCM, sound detection, AudioSet tagging, PANNs, or wants to
  classify audio into categories. This skill handles any audio classification task where the choices
  can be mapped to AudioSet sound classes (speech, music, animals, vehicles, environmental sounds, etc.).
---

# Audio QCM – PANNs CNN14 Audio Classification

Answer multiple-choice questions about audio files using the PANNs CNN14 model trained on AudioSet's 527 sound classes.

## How it works

1. Load the WAV file and run PANNs CNN14 inference to get scores for all 527 AudioSet classes
2. Map each QCM choice to relevant AudioSet class indices
3. Average the scores for each choice's classes
4. Return the choice with the highest average score and its confidence

## Dependencies

```
torch
librosa
numpy
pandas
```

Install with: `pip install torch librosa numpy pandas`

## Model weights

Download the pretrained CNN14 model from Zenodo:
```
https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
```

Place it in `~/.cache/panns/Cnn14_mAP=0.431.pth` or pass the path via `--model-path`.

## Files

- `scripts/audio_qcm_inference.py` – Main inference script
- `references/class_labels_indices.csv` – AudioSet class labels (527 classes)

## Usage

### Basic usage (CLI)

```bash
python scripts/audio_qcm_inference.py \
    --wav path/to/audio.wav \
    --class-map '{"A": [0,1,2,3,4], "B": [500]}'
```

Output:
```json
{
  "answer": "A",
  "confidence": 0.8234,
  "details": {
    "choice_scores": {"A": 0.8234, "B": 0.1245},
    "top10_detected": [...]
  }
}
```

### As a Python function

```python
from scripts.audio_qcm_inference import run_inference, load_class_labels, get_classes_for_keyword

# Load labels
labels = load_class_labels()  # {0: "Speech", 1: "Male speech, ...", ...}

# Map choices to AudioSet class indices
# For binary yes/no questions about speech:
class_map = {
    "A": [0, 1, 2, 3, 4, 5],  # Speech-related classes → "Oui"
    "B": [500]                  # Silence class → "Non"
}

result = run_inference(
    wav_path="audio.wav",
    class_map=class_map,
    # model_path="path/to/Cnn14_mAP=0.431.pth",  # optional
    # device="cuda"  # optional
)
print(result["answer"], result["confidence"])
```

## Mapping choices to AudioSet classes

The key step is mapping each choice to relevant AudioSet class indices. Use these strategies:

### 1. Keyword lookup (built-in)

The inference script includes `_KEYWORD_MAP` for common concepts:
- `speech`/`parole` → indices 0–15 (all speech variants)
- `music`/`musique` → indices 27–281 (all music variants)
- `dog`/`chien` → indices 74–80
- `bird`/`oiseau` → indices 111–121
- `rain`/`pluie` → indices 288–291
- `silence` → index 500
- `noise`/`bruit` → indices 512–522

### 2. Fuzzy label matching

If a keyword isn't in the built-in map, the script searches all 527 class labels for substring matches:
```python
from scripts.audio_qcm_inference import get_classes_for_keyword, load_class_labels
labels = load_class_labels()
indices = get_classes_for_keyword("helicopter", labels)  # Returns [338, 339, ...]
```

### 3. Manual mapping via the CSV

Read `references/class_labels_indices.csv` to find exact indices:
```
index,mid,display_name
0,/m/09x0r,"Speech"
338,/m/02l6bg,"Propeller, airscrew"
339,/m/09ct_,"Helicopter"
...
```

### 4. Binary yes/no questions

For questions like "Y a-t-il de la parole ?":
- "Oui" → classes relevant to the question keyword (e.g., speech classes 0–15)
- "Non" → empty list `[]` (the script computes `1 - mean(speech_scores)`)

## Class label reference

The AudioSet ontology covers these broad categories:
- **Human sounds** (0–71): speech, laughter, crying, singing, etc.
- **Animal sounds** (72–136): dogs, cats, birds, insects, etc.
- **Music** (137–281): instruments, genres, vocal music
- **Natural sounds** (282–298): wind, rain, water, fire
- **Vehicles** (299–352): cars, trains, aircraft, engines
- **Household sounds** (353–424): doors, appliances, tools
- **Sound effects** (425–527): explosions, impacts, electronic sounds

See `references/class_labels_indices.csv` for the complete list of 527 classes with indices.

## Limitations

- Audio must be in WAV format (convert MP3/other formats first)
- Model expects 32kHz mono audio (the script handles resampling)
- Confidence scores are sigmoid probabilities, not calibrated probabilities
- The model was trained on 10-second clips; very short or very long audio may give less reliable results
