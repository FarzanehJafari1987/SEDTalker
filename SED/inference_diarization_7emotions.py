"""
Speech Emotion Diarization Inference - FIXED VERSION
Uses frame-level trained model for temporal emotion detection
FIXED: Proper checkpoint loading for SpeechBrain format
"""

import os
import torch
import torchaudio
import numpy as np
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sounddevice as sd
from pathlib import Path
from typing import Dict, List, Union

# ============================================================================
# Configuration
# ============================================================================

# 7 emotions including both upset and neutral
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "upset"]

CONFIG = {
    "sample_rate": 16000,
    "wav2vec2_hub": "microsoft/wavlm-base-plus",
    "output_neurons": 7,  # 7 emotions
    "dropout": 0.3,
    "frame_shift": 0.02,  # 20ms per frame
}

# ============================================================================
# Model Definition (same as training)
# ============================================================================

class FrameLevelEmotionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.wav2vec2 = AutoModel.from_pretrained(
            config['wav2vec2_hub'],
            cache_dir="pretrained_models/"
        )
        
        self.feature_dim = self.wav2vec2.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, config['output_neurons'])
        )
    
    def forward(self, wavs):
        outputs = self.wav2vec2(wavs)
        feats = outputs.last_hidden_state  # [batch, time, feat_dim]
        
        batch_size, time_steps, feat_dim = feats.shape
        feats_flat = feats.reshape(-1, feat_dim)
        logits_flat = self.classifier(feats_flat)
        logits = logits_flat.reshape(batch_size, time_steps, -1)
        
        return logits


# ============================================================================
# Emotion Diarization Detector
# ============================================================================

class EmotionDiarizer:
    def __init__(
        self, 
        checkpoint_path, 
        device='cuda', 
        smoothing_window=3
    ):
        """
        Initialize emotion diarizer
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            smoothing_window: Number of frames to smooth predictions
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        #print(f"Using device: {self.device}")
        
        self.smoothing_window = smoothing_window
        self.sample_rate = CONFIG['sample_rate']
        self.frame_shift = CONFIG['frame_shift']
        
        # Emotion code mapping
        self.emotion_to_code = {
            'happy': 'h',
            'sad': 's',
            'angry': 'a',
            'disgust': 'd',
            'fear': 'f',
            'upset': 'u',
            'neutral': 'n'
        }
        
        # Load model - FIXED VERSION
        #print(f"Loading checkpoint from: {checkpoint_path}")
        self.model = FrameLevelEmotionModel(CONFIG)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        #print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Handle different checkpoint formats
        loaded = False
        
        # Try SpeechBrain Checkpointer format (most common)
        if 'model' in checkpoint:
            try:
                if isinstance(checkpoint['model'], dict):
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    # It's the model object itself
                    self.model = checkpoint['model']
                print("✓ Loaded with 'model' key (SpeechBrain format)")
                loaded = True
            except Exception as e:
                print(f"  Failed to load 'model' key: {e}")
        
        # Try standard PyTorch format
        if not loaded and 'state_dict' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['state_dict'])
                print("✓ Loaded with 'state_dict' key")
                loaded = True
            except Exception as e:
                print(f"  Failed to load 'state_dict' key: {e}")
        
        # Try modelparams key
        if not loaded and 'modelparams' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['modelparams'])
                print("✓ Loaded with 'modelparams' key")
                loaded = True
            except Exception as e:
                print(f"  Failed to load 'modelparams' key: {e}")
        
        # Try direct state dict
        if not loaded:
            try:
                self.model.load_state_dict(checkpoint)
                print("✓ Loaded as direct state dict")
                loaded = True
            except Exception as e:
                print(f"  Failed to load as direct state dict: {e}")
        
        if not loaded:
            print("\n❌ ERROR: Cannot load checkpoint!")
            print(f"Available keys: {list(checkpoint.keys())}")
            print("\nPlease check your checkpoint file!")
            raise ValueError("Failed to load model checkpoint")
        
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully!")
        print(f"  Emotions: {EMOTION_LABELS}")
        print(f"  Frame shift: {self.frame_shift}s")
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        sr = int(sr)
        target_sr = int(self.sample_rate)
        
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze(0)
    
    def smooth_predictions(self, predictions, window=3):
        """
        Apply moving average smoothing to predictions
        
        Args:
            predictions: List of emotion indices
            window: Smoothing window size (number of frames)
        
        Returns:
            Smoothed predictions
        """
        if len(predictions) < window:
            return predictions
        
        smoothed = []
        half_window = window // 2
        
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            
            # Get most common emotion in window
            window_preds = predictions[start:end]
            most_common = max(set(window_preds), key=window_preds.count)
            smoothed.append(most_common)
        
        return smoothed
    
    def merge_consecutive_segments(self, segments):
        """Merge consecutive segments with same emotion"""
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            if segment['emotion'] == current['emotion']:
                # Merge: extend end time
                current['end'] = segment['end']
                # Average confidence if present
                if 'confidence' in current and 'confidence' in segment:
                    # Weighted average by duration
                    dur1 = current['end'] - current['start']
                    dur2 = segment['end'] - segment['start']
                    total_dur = dur1 + dur2
                    current['confidence'] = (
                        current['confidence'] * dur1 + segment['confidence'] * dur2
                    ) / total_dur
            else:
                merged.append(current)
                current = segment.copy()
        
        merged.append(current)
        return merged
    
    def diarize_file(
        self,
        audio_path,
        use_short_codes=True,
        include_confidence=True,
        merge_consecutive=True,
        apply_smoothing=True
    ):
        """
        Perform frame-level emotion diarization
        
        Args:
            audio_path: Path to audio file
            use_short_codes: Use short codes (h/s/a/n) instead of full names
            include_confidence: Include confidence scores
            merge_consecutive: Merge consecutive segments with same emotion
            apply_smoothing: Apply temporal smoothing to predictions
            
        Returns:
            Dictionary with emotion timeline
        """
        audio_path = str(audio_path)
        
        print(f"Processing: {audio_path}")
        
        # Load audio
        waveform = self.preprocess_audio(audio_path)
        print(f"  Audio duration: {len(waveform) / self.sample_rate:.2f}s")
        
        # Predict frame-by-frame
        waveform_batch = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(waveform_batch)  # [1, time, num_classes]
            probs = F.softmax(logits, dim=-1)
        
        # Get predictions for each frame
        probs = probs.squeeze(0).cpu()  # [time, num_classes]
        confidences, predictions = torch.max(probs, dim=-1)
        
        print(f"  Model output: {probs.shape[0]} frames")
        
        # Convert to lists
        predictions = predictions.numpy().tolist()
        confidences = confidences.numpy().tolist()
        
        # Show raw prediction distribution
        pred_counts = {}
        for pred in predictions:
            emo = EMOTION_LABELS[pred]
            pred_counts[emo] = pred_counts.get(emo, 0) + 1
        
        print(f"  Raw predictions:")
        for emo, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(predictions) * 100
            print(f"    {emo}: {count} frames ({pct:.1f}%)")
        
        # Apply smoothing if requested
        if apply_smoothing and self.smoothing_window > 1:
            predictions = self.smooth_predictions(predictions, self.smoothing_window)
            print(f"  Applied smoothing with window={self.smoothing_window}")
        
        # Convert frame predictions to time segments
        segments = []
        
        for frame_idx, (pred_idx, confidence) in enumerate(zip(predictions, confidences)):
            emotion = EMOTION_LABELS[pred_idx]
            
            start_time = frame_idx * self.frame_shift
            end_time = (frame_idx + 1) * self.frame_shift
            
            segment = {
                'start': start_time,
                'end': end_time,
                'emotion': self.emotion_to_code[emotion] if use_short_codes else emotion
            }
            
            if include_confidence:
                segment['confidence'] = confidence
            
            segments.append(segment)
        
        # Merge consecutive segments
        if merge_consecutive:
            segments = self.merge_consecutive_segments(segments)
            print(f"  After merging: {len(segments)} segments")
        
        return {audio_path: segments}
    
    def predict_utterance(self, audio_input, return_probabilities=False):
        """
        Predict single emotion for entire audio (for comparison)
        Uses majority voting over frames
        """
        if isinstance(audio_input, str) or isinstance(audio_input, Path):
            waveform = self.preprocess_audio(audio_input)
        else:
            waveform = audio_input
        
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(waveform)
            probs = F.softmax(logits, dim=-1)
        
        # Average probabilities across all frames
        avg_probs = probs.mean(dim=1).squeeze(0)  # [num_classes]
        
        pred_idx = avg_probs.argmax(dim=-1).item()
        emotion = EMOTION_LABELS[pred_idx]
        
        if return_probabilities:
            probs_dict = {
                EMOTION_LABELS[i]: avg_probs[i].item() 
                for i in range(len(EMOTION_LABELS))
            }
            return emotion, probs_dict
        else:
            return emotion


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Speech Emotion Diarization - Frame-Level Temporal Detection (FIXED)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained frame-level model checkpoint'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to audio file'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Directory containing multiple audio files'
    )
    parser.add_argument(
        '--smoothing',
        type=int,
        default=5,
        help='Temporal smoothing window in frames (default: 5, use 1 for no smoothing)'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Do not merge consecutive segments with same emotion'
    )
    parser.add_argument(
        '--full-names',
        action='store_true',
        help='Use full emotion names instead of codes'
    )
    parser.add_argument(
        '--no-confidence',
        action='store_true',
        help='Do not include confidence scores'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--utterance-mode',
        action='store_true',
        help='Predict single emotion for entire audio (comparison mode)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EMOTION DIARIZATION INFERENCE (FIXED VERSION)")
    print("="*70 + "\n")
    
    # Initialize diarizer
    diarizer = EmotionDiarizer(
        args.checkpoint,
        device=args.device,
        smoothing_window=args.smoothing
    )
    
    # Single file diarization
    if args.audio:
        if args.utterance_mode:
            # Simple utterance-level prediction
            emotion, probs = diarizer.predict_utterance(args.audio, return_probabilities=True)
            print(f"\n{'='*70}")
            print(f"File: {args.audio}")
            print(f"Predicted Emotion: {emotion.upper()}")
            print(f"{'='*70}")
            print("\nFrame-averaged Probabilities:")
            for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 50)
                print(f"  {emo:10s}: {prob:.4f} {bar}")
        else:
            # Full diarization
            print(f"\nPerforming emotion diarization...\n")
            
            diary = diarizer.diarize_file(
                args.audio,
                use_short_codes=not args.full_names,
                include_confidence=not args.no_confidence,
                merge_consecutive=not args.no_merge,
                apply_smoothing=(args.smoothing > 1)
            )
            
            # Print results
            for filename, segments in diary.items():
                print(f"\n{'='*70}")
                print(f"Emotion Timeline: {filename}")
                print(f"{'='*70}\n")
                
                emoji_map = {
                    'h': '😊', 's': '😢', 'a': '😠', 'd': '🤢', 
                    'f': '😨', 'u': '😔', 'n': '😐',
                    'happy': '😊', 'sad': '😢', 'angry': '😠', 
                    'disgust': '🤢', 'fear': '😨', 'upset': '😔', 'neutral': '😐'
                }
                
                for i, seg in enumerate(segments, 1):
                    duration = seg['end'] - seg['start']
                    emoji = emoji_map.get(seg['emotion'], '?')
                    
                    if not args.no_confidence:
                        print(f"{i:4d}. {seg['start']:7.3f}s - {seg['end']:7.3f}s "
                              f"({duration:6.3f}s)  {emoji} {seg['emotion']:7s}  "
                              f"(conf: {seg['confidence']:.3f})")
                    else:
                        print(f"{i:4d}. {seg['start']:7.3f}s - {seg['end']:7.3f}s "
                              f"({duration:6.3f}s)  {emoji} {seg['emotion']}")
                
                print(f"\n{'='*70}")
                print(f"Total segments: {len(segments)}")
                print(f"Total duration: {segments[-1]['end']:.2f}s")
                
                # Emotion statistics
                emotion_durations = {}
                for seg in segments:
                    dur = seg['end'] - seg['start']
                    emo = seg['emotion']
                    emotion_durations[emo] = emotion_durations.get(emo, 0) + dur
                
                print(f"\nEmotion Distribution:")
                total_dur = segments[-1]['end']
                for emo, dur in sorted(emotion_durations.items(), key=lambda x: x[1], reverse=True):
                    percentage = (dur / total_dur) * 100
                    print(f"  {emo}: {dur:.2f}s ({percentage:.1f}%)")
    
    # Batch processing
    elif args.batch:
        audio_dir = Path(args.batch)
        audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
        
        print(f"\nProcessing {len(audio_files)} audio files with diarization...")
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for audio_file in audio_files:
            try:
                print(f"\nProcessing: {audio_file.name}")
                diary = diarizer.diarize_file(
                    str(audio_file),
                    use_short_codes=not args.full_names,
                    include_confidence=not args.no_confidence,
                    merge_consecutive=not args.no_merge,
                    apply_smoothing=(args.smoothing > 1)
                )
                all_results.update(diary)
                
                # Show summary
                for filename, segments in diary.items():
                    print(f"  → {len(segments)} emotion segments, {segments[-1]['end']:.2f}s duration")
                
            except Exception as e:
                print(f"  → ERROR: {str(e)}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# python inference_diarization_7emotions.py --checkpoint results/emotion_diarization_7class/save/CKPT+epoch_50/model.ckpt --audio datasets/JL_corpus/combined/female1/female1_c3bc5700_4.wav --smoothing 5
