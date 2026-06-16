---
name: audio_split
description: Splits an audio file into 3 equal segments and saves them as WAV files, plus a full copy. Use this skill whenever the user provides an audio file path and wants to divide it into equal chunks, extract segments for processing, or needs the duration and split paths of an audio file. Also use when the user mentions splitting audio, segmenting audio, or dividing audio into parts using librosa and soundfile.
---

# Audio Split Skill

This skill takes an audio file path, calculates its duration, splits it into 3 equal segments, and saves all outputs as WAV files.

## Requirements
- `librosa` for audio loading and duration calculation.
- `soundfile` for saving WAV files.
- Ensure a virtual environment with these packages is active, or install them if needed (`pip install librosa soundfile`).

## Workflow

You can use the bundled `scripts/split_audio.py` to handle this task directly, or implement the logic inline.

### Using the bundled script (Recommended)
```bash
python skills/audio_split/scripts/split_audio.py <input_audio_path> <output_dir>
```

### Inline Implementation
1. **Load Audio and Get Duration**:
   Use `librosa` to load the audio file and get its duration in seconds. Preserve the native sample rate.
   ```python
   import librosa
   import soundfile as sf
   import os
   import json

   # Load audio (sr=None preserves native sample rate)
   y, sr = librosa.load(audio_path, sr=None)
   duration_seconds = float(librosa.get_duration(y=y, sr=sr))
   ```

2. **Determine Segment Length**:
   Calculate the number of samples per segment.
   ```python
   total_samples = len(y)
   samples_per_segment = total_samples // 3
   ```

3. **Split and Save**:
   Slice the audio array and save each segment, plus the full audio, as `.wav` files in the specified output directory.
   ```python
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
   ```

4. **Output JSON**:
   Return a JSON object with the exact format requested:
   ```json
   {
     "full": "<output_dir>/audio_full.wav",
     "seg1": "<output_dir>/audio_seg1.wav",
     "seg2": "<output_dir>/audio_seg2.wav",
     "seg3": "<output_dir>/audio_seg3.wav",
     "duration_seconds": <float>,
     "segment_duration": <float>
   }
   ```
   Note: `segment_duration` should be calculated as `duration_seconds / 3.0`.

## Important Notes
- Always use `sr=None` when loading with `librosa.load()` to preserve the original sample rate and avoid unnecessary resampling artifacts.
- Ensure the output directory exists before writing files.
- Handle potential exceptions (e.g., file not found, unsupported format) gracefully and report them clearly.
- If the audio is shorter than expected or has edge cases, ensure the slicing logic does not throw index errors (Python slicing is safe, but good to keep in mind).