"""
PANNs CNN14 audio QCM inference script.

Takes a WAV file and explicit AudioSet class indices per choice,
runs PANNs CNN14 inference, and returns scores to determine the best answer.

Usage:
    python audio_qcm_inference.py \
        --wav path/to/audio.wav \
        --class-map '{"A": [0,1,2,3], "B": [500]}' \
        [--model-path path/to/Cnn14_mAP=0.431.pth]

Output: JSON dict with keys: answer (str), confidence (float), details (dict)
"""

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import csv


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_class_labels(labels_path=None):
    """Load AudioSet class labels from CSV."""
    if labels_path is None:
        labels_path = Path(__file__).parent.parent / "references" / "class_labels_indices.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Class labels CSV not found at {labels_path}")

    labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['index'])
            labels[idx] = row['display_name'].strip('"')
    return labels


# ---------------------------------------------------------------------------
# PANNs CNN14 model (minimal, matching original architecture)
# ---------------------------------------------------------------------------

def init_layer(layer):
    if hasattr(layer, 'weight'):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0.0)


def init_bn(bn):
    torch.nn.init.constant_(bn.weight, 1.0)
    torch.nn.init.constant_(bn.bias, 0.0)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2)):
        x = torch.nn.functional.relu_(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu_(self.bn2(self.conv2(x)))
        x = torch.nn.functional.avg_pool2d(x, pool_size)
        return x


class Cnn14(torch.nn.Module):
    def __init__(self, classes_num=527):
        super().__init__()
        # Mel spectrogram parameters
        self.sample_rate = 32000
        self.n_fft = 1024
        self.hop_length = 320
        self.n_mels = 64
        self.fmin = 50
        self.fmax = 14000

        # CNN backbone
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = torch.nn.Linear(2048, 2048)
        self.fc_audioset = torch.nn.Linear(2048, classes_num)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def mel_spectrogram(self, wav):
        """Compute log-mel spectrogram using librosa."""
        mel = librosa.feature.melspectrogram(
            y=wav, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax, power=1.0)
        # Convert to log scale
        log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
        # Normalize (matching PANNs training)
        log_mel = (log_mel + 50) / 50
        return log_mel

    def forward(self, wav):
        """
        Input: wav tensor of shape (batch_size, samples)
        Output: dict with 'clipwise_output' (sigmoid probabilities)
        """
        # Compute log-mel spectrogram for each sample in batch
        mel_specs = []
        for i in range(wav.shape[0]):
            mel = self.mel_spectrogram(wav[i].cpu().numpy())
            mel_specs.append(mel.T)  # (time, mel_bins)

        # Pad to same time length
        max_time = max(m.shape[0] for m in mel_specs)
        padded = []
        for m in mel_specs:
            if m.shape[0] < max_time:
                pad = np.zeros((max_time - m.shape[0], m.shape[1]))
                m = np.vstack([m, pad])
            padded.append(m)

        x = np.array(padded)  # (batch, time, mel_bins)
        x = torch.FloatTensor(x).to(wav.device)

        # Add channel dimension and normalize
        x = x.unsqueeze(1)  # (batch, 1, time, mel_bins)
        x = torch.nn.functional.batch_norm(x, training=False)

        # CNN forward
        x = self.conv_block1(x, pool_size=(2, 2))
        x = torch.nn.functional.dropout(x, p=0.2, training=False)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = torch.nn.functional.dropout(x, p=0.2, training=False)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = torch.nn.functional.dropout(x, p=0.2, training=False)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = torch.nn.functional.dropout(x, p=0.2, training=False)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = torch.nn.functional.dropout(x, p=0.2, training=False)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = torch.nn.functional.dropout(x, p=0.2, training=False)

        # Global pooling
        x = torch.mean(x, dim=3)  # (batch, 2048, time)
        x1, _ = torch.max(x, dim=2)  # max over time
        x2 = torch.mean(x, dim=2)  # mean over time
        x = x1 + x2

        # Classification head
        x = torch.nn.functional.dropout(x, p=0.5, training=False)
        x = torch.nn.functional.relu_(self.fc1(x))
        embedding = torch.nn.functional.dropout(x, p=0.5, training=False)
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))

        return {'clipwise_output': clipwise_output, 'embedding': embedding}


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def run_inference(wav_path, class_map, model_path=None, device=None):
    """
    Run PANNs CNN14 inference and score each choice.

    Args:
        wav_path: Path to WAV file
        class_map: Dict mapping choice keys to lists of AudioSet class indices
                   e.g., {"A": [0, 1, 2], "B": [500]}
        model_path: Path to pretrained weights file
        device: Torch device string

    Returns:
        Dict with 'answer', 'confidence', 'details'
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Resolve model path
    if model_path is None:
        candidates = [
            os.path.expanduser('~/.cache/panns/Cnn14_mAP=0.431.pth'),
            os.path.expanduser('~/.cache/torch/hub/checkpoints/Cnn14_mAP=0.431.pth'),
            Path(__file__).parent.parent / "models" / "Cnn14_mAP=0.431.pth",
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = str(p)
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model weights not found. Download Cnn14_mAP=0.431.pth from "
            "https://zenodo.org/record/3987831 and place it in ~/.cache/panns/ "
            "or pass --model-path explicitly."
        )

    # Build and load model
    model = Cnn14(classes_num=527)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load audio at model's expected sample rate
    audio, _ = librosa.load(wav_path, sr=model.sample_rate, mono=True)
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(audio_tensor)

    scores = output['clipwise_output'].cpu().numpy()[0]  # (527,)
    labels = load_class_labels()

    # Score each choice by averaging its relevant class scores
    choice_scores = {}
    for key, indices in class_map.items():
        if indices:
            choice_scores[key] = float(np.mean(scores[indices]))
        else:
            # Empty class list -> choice represents "absence" of target sounds
            # Score = 1 - mean of all relevant classes from OTHER choices
            all_other = [i for k, v in class_map.items() if k != key for i in v]
            if all_other:
                choice_scores[key] = 1.0 - float(np.mean(scores[all_other]))
            else:
                choice_scores[key] = 0.5  # neutral fallback

    # Pick winner
    best_key = max(choice_scores, key=choice_scores.get)
    confidence = choice_scores[best_key]

    # Top-10 detected sounds for context
    top10 = np.argsort(scores)[-10:][::-1]
    top10_details = [{'class': labels[i], 'score': round(float(scores[i]), 4)} for i in top10]

    return {
        'answer': best_key,
        'confidence': round(float(confidence), 4),
        'details': {
            'choice_scores': {k: round(v, 4) for k, v in choice_scores.items()},
            'top10_detected': top10_details,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='PANNs CNN14 Audio QCM')
    parser.add_argument('--wav', required=True, help='Path to WAV file')
    parser.add_argument('--class-map', required=True,
                        help='JSON: {"A": [0,1,2], "B": [500]}')
    parser.add_argument('--model-path', default=None, help='Path to .pth weights')
    parser.add_argument('--device', default=None, help='cpu or cuda')
    args = parser.parse_args()

    class_map = json.loads(args.class_map)
    result = run_inference(args.wav, class_map, args.model_path, args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
