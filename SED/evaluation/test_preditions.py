"""
Generate predictions for evaluation
Runs inference on test set and saves in format compatible with evaluate_sed_comprehensive.py
"""

import json
import torch
import torchaudio
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
from tqdm import tqdm

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "upset"]

CONFIG = {
    "sample_rate": 16000,
    "wav2vec2_hub": "microsoft/wavlm-base-plus",
    "output_neurons": 7,
    "dropout": 0.3,
}

class FrameLevelEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = AutoModel.from_pretrained(
            CONFIG['wav2vec2_hub'],
            cache_dir="pretrained_models/"
        )
        self.feature_dim = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(128, CONFIG['output_neurons'])
        )
    
    def forward(self, wavs):
        outputs = self.wav2vec2(wavs)
        feats = outputs.last_hidden_state
        batch_size, time_steps, feat_dim = feats.shape
        feats_flat = feats.reshape(-1, feat_dim)
        logits_flat = self.classifier(feats_flat)
        logits = logits_flat.reshape(batch_size, time_steps, -1)
        return logits


def generate_predictions(checkpoint_path, test_json_path, output_path, device='cuda', smoothing=5):
    """
    Generate predictions for all samples in test set
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_json_path: Path to test_frames.json
        output_path: Where to save predictions
        device: cuda or cpu
        smoothing: Smoothing window size
    """
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {checkpoint_path}")
    
    model = FrameLevelEmotionModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Load test data
    print(f"Loading test data from: {test_json_path}")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Found {len(test_data)} test samples\n")
    
    # Generate predictions
    predictions = {}
    
    print("Generating predictions...")
    for sample_id, sample_info in tqdm(test_data.items(), desc="Processing"):
        audio_path = sample_info['wav']
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.to(device)
            
            # Predict
            with torch.no_grad():
                logits = model(waveform)
                probs = F.softmax(logits, dim=-1)
            
            probs = probs.squeeze(0).cpu()
            pred_indices = probs.argmax(dim=-1).numpy().tolist()
            
            # Apply smoothing if requested
            if smoothing > 1:
                pred_indices = smooth_predictions(pred_indices, smoothing)
            
            # Convert indices to labels
            frame_labels = [EMOTION_LABELS[idx] for idx in pred_indices]
            
            # Store prediction
            predictions[sample_id] = {
                'frame_labels': frame_labels,
                'wav': audio_path
            }
            
        except Exception as e:
            print(f"\nError processing {sample_id}: {e}")
            continue
    
    # Save predictions
    print(f"\nSaving predictions to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✓ Saved {len(predictions)} predictions")
    print(f"\nNow run evaluation with:")
    print(f"python evaluate_sed_comprehensive.py \\")
    print(f"    --predictions {output_path} \\")
    print(f"    --ground_truth {test_json_path} \\")
    print(f"    --output_dir evaluation_results/")


def smooth_predictions(predictions, window=5):
    """Apply temporal smoothing"""
    if len(predictions) < window:
        return predictions
    
    smoothed = []
    half_window = window // 2
    
    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        window_preds = predictions[start:end]
        most_common = max(set(window_preds), key=window_preds.count)
        smoothed.append(most_common)
    
    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions for evaluation'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-json', type=str, 
                       default='datasets/processed_emotions_7class/test_frames.json',
                       help='Path to test_frames.json')
    parser.add_argument('--output', type=str, default='test_predictions.json',
                       help='Output predictions file')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--smoothing', type=int, default=5,
                       help='Smoothing window size (default: 5)')
    
    args = parser.parse_args()
    
    generate_predictions(
        args.checkpoint,
        args.test_json,
        args.output,
        args.device,
        args.smoothing
    )


if __name__ == "__main__":
    main()


# python evaluation/test_preditions.py --checkpoint results/emotion_diarization_7class/save/CKPT+epoch_50/model.ckpt --test-json data/processed_emotions_7class/test_frames.json --output evaluation/test_predictions.json --smoothing 5