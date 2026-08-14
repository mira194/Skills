---
name: audio_split
description: Splits an audio file into 3 equal segments and saves them as WAV files, plus a full copy. Use this skill whenever the user provides an audio file path and wants to divide it into equal chunks, extract segments for processing, or needs the duration and split paths of an audio file. Also use when the user mentions splitting audio, segmenting audio, or dividing audio into parts.
---

# Audio Split Skill

This skill takes an audio file path, calculates its duration, splits it into 3 equal segments, and saves all outputs as WAV files.

## Requirements
- Python standard library only (`wave` module). No external dependencies 
  (no librosa, no soundfile) — nothing to install.
- Input files must be standard WAV (PCM) format.

## Workflow

ALWAYS use the bundled `scripts/split_audio.py` script to handle this task.
Do NOT implement the logic inline (do not write your own audio-processing 
code) — the script is the only supported execution path, to keep token 
usage and behavior consistent across runs.

```bash
python skills/audio_split/scripts/split_audio.py <input_audio_path> <output_dir>
```

The script returns a JSON object in this exact format:
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
Use this output directly — do not re-derive or recompute any of these values yourself.

## Important Notes
- Call the script exactly once per audio file. Do not retry or call it multiple times for the same input.
- If the script fails (file not found, unsupported format, etc.), report the error clearly and stop — do not attempt to reimplement the split as a workaround.
- Non-WAV inputs (mp3, flac, etc.) are NOT supported by this version of the script.
