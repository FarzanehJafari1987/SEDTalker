"""
Speech Emotion Diarization with Intensity Estimation - FIXED SEGMENTATION
Segments audio by emotion AND intensity changes
RUNNABLE VERSION - Can be executed directly with Run button
"""

import os
import torch
import torchaudio
import numpy as np
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
from typing import Dict, List, Union, Tuple
import json

# Configuration
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "upset"]

CONFIG = {
    "sample_rate": 16000,
    "wav2vec2_hub": "microsoft/wavlm-base-plus",
    "output_neurons": 7,
    "dropout": 0.3,
    "frame_shift": 0.02,  # 20ms per frame
}

# Intensity thresholds - 3 levels
INTENSITY_THRESHOLDS = {
    'low': 0.5,      # confidence < 0.5
    'medium': 0.75,  # 0.5 <= confidence < 0.75
    'high': 1.0,     # 0.75 <= confidence < 1.0
}

# Model Definition
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

# Diagnostic Functions
def diagnose_predictions(diarizer, audio_path, num_samples=20):
    """Check if model is actually producing varied predictions"""
    waveform = diarizer.preprocess_audio(audio_path)
    waveform_batch = waveform.unsqueeze(0).to(diarizer.device)
    
    with torch.no_grad():
        logits = diarizer.model(waveform_batch)
        probs = F.softmax(logits, dim=-1)
    
    probs = probs.squeeze(0).cpu()
    
    # Sample evenly across the audio
    indices = np.linspace(0, len(probs)-1, num_samples, dtype=int)
    
    print("\n PREDICTION DIAGNOSTICS:")
    print("="*80)
    for idx in indices:
        frame_probs = probs[idx]
        top3_probs, top3_idx = torch.topk(frame_probs, 3)
        
        print(f"Frame {idx:4d} ({idx*diarizer.frame_shift:.2f}s):")
        for i in range(3):
            emotion = EMOTION_LABELS[top3_idx[i]]
            prob = top3_probs[i].item()
            bar = '█' * int(prob*10)
            print(f"  {emotion:10s}: {prob:.4f} {bar}")
        print()
    
    # Calculate prediction diversity
    confidences, predictions = torch.max(probs, dim=-1)
    unique_predictions = len(set(predictions.numpy().tolist()))
    
    print(f"\n DIVERSITY METRICS:")
    print(f"  Total frames: {len(probs)}")
    print(f"  Unique predictions: {unique_predictions} / 7 emotions")
    print(f"  Average confidence: {confidences.mean():.4f}")
    print(f"  Confidence std: {confidences.std():.4f}")
    
    # Per-emotion statistics
    print(f"\n PER-EMOTION FRAME COUNTS:")
    pred_counts = {}
    for pred in predictions.numpy():
        emotion = EMOTION_LABELS[pred]
        pred_counts[emotion] = pred_counts.get(emotion, 0) + 1
    
    for emotion in EMOTION_LABELS:
        count = pred_counts.get(emotion, 0)
        pct = (count / len(predictions)) * 100
        bar = '█' * int(pct / 2)
        print(f"  {emotion:10s}: {count:5d} frames ({pct:5.1f}%) {bar}")
    
    print("="*80)

# Intensity Estimation Methods
class IntensityEstimator:
    """Estimate emotion intensity from model predictions"""
    
    @staticmethod
    def from_confidence(confidence: float) -> Tuple[str, float]:
        """Estimate intensity from prediction confidence"""
        if confidence >= INTENSITY_THRESHOLDS['high']:
            return 'high', confidence
        elif confidence >= INTENSITY_THRESHOLDS['medium']:
            return 'medium', confidence
        else:
            return 'low', confidence
    
    @staticmethod
    def from_probability_distribution(probs: torch.Tensor) -> Tuple[str, float]:
        """Estimate intensity from probability distribution entropy"""
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        intensity_score = 1.0 - normalized_entropy.item()
        
        if intensity_score >= 0.75:
            return 'high', intensity_score
        elif intensity_score >= 0.50:
            return 'medium', intensity_score
        else:
            return 'low', intensity_score
    
    @staticmethod
    def from_top_probabilities(probs: torch.Tensor) -> Tuple[str, float]:
        """Estimate intensity from gap between top two probabilities"""
        sorted_probs, _ = torch.sort(probs, descending=True)
        top1 = sorted_probs[0].item()
        top2 = sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0
        gap = top1 - top2
        intensity_score = (top1 + gap) / 2.0
        
        if intensity_score >= 0.75:
            return 'high', intensity_score
        elif intensity_score >= 0.50:
            return 'medium', intensity_score
        else:
            return 'low', intensity_score
    
    @staticmethod
    def aggregate_methods(
        confidence: float, 
        probs: torch.Tensor,
        method: str = 'combined'
    ) -> Tuple[str, float, Dict]:
        """Aggregate multiple intensity estimation methods"""
        conf_label, conf_score = IntensityEstimator.from_confidence(confidence)
        entropy_label, entropy_score = IntensityEstimator.from_probability_distribution(probs)
        gap_label, gap_score = IntensityEstimator.from_top_probabilities(probs)
        
        method_scores = {
            'confidence': conf_score,
            'entropy': entropy_score,
            'gap': gap_score
        }
        
        if method == 'confidence':
            return conf_label, conf_score, method_scores
        elif method == 'entropy':
            return entropy_label, entropy_score, method_scores
        elif method == 'gap':
            return gap_label, gap_score, method_scores
        else:  # combined
            combined_score = (
                0.4 * conf_score + 
                0.3 * entropy_score + 
                0.3 * gap_score
            )
            
            if combined_score >= 0.75:
                label = 'high'
            elif combined_score >= 0.50:
                label = 'medium'
            else:
                label = 'low'
            
            return label, combined_score, method_scores

# Enhanced Emotion Diarizer with Intensity
class EmotionIntensityDiarizer:
    def __init__(
        self, 
        checkpoint_path, 
        device='cuda', 
        smoothing_window=3,
        intensity_method='combined'
    ):
        """Initialize emotion diarizer with intensity estimation"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.smoothing_window = smoothing_window
        self.sample_rate = CONFIG['sample_rate']
        self.frame_shift = CONFIG['frame_shift']
        self.intensity_method = intensity_method
        
        # Emotion code mapping
        self.emotion_to_code = {
            'happy': 'h', 'sad': 's', 'angry': 'a', 'disgust': 'd',
            'fear': 'f', 'upset': 'u', 'neutral': 'n'
        }
        
        # Load model
        print(f"Loading checkpoint from: {checkpoint_path}")
        self.model = FrameLevelEmotionModel(CONFIG)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        loaded = False
        for key in ['model', 'state_dict', 'modelparams']:
            if key in checkpoint:
                try:
                    if isinstance(checkpoint[key], dict):
                        self.model.load_state_dict(checkpoint[key])
                    else:
                        self.model = checkpoint[key]
                    print(f"✓ Loaded with '{key}' key")
                    loaded = True
                    break
                except:
                    pass
        
        if not loaded:
            try:
                self.model.load_state_dict(checkpoint)
                print("✓ Loaded as direct state dict")
                loaded = True
            except:
                pass
        
        if not loaded:
            raise ValueError("Failed to load model checkpoint")
        
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully!")
        print(f"  Emotions: {EMOTION_LABELS}")
        print(f"  Intensity method: {self.intensity_method}")
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        if int(sr) != int(self.sample_rate):
            resampler = torchaudio.transforms.Resample(int(sr), int(self.sample_rate))
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze(0)
    
    def smooth_predictions(self, predictions, window=3):
        """Apply moving average smoothing to predictions"""
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
    
    def merge_consecutive_segments(self, segments, merge_by='emotion_and_intensity', 
                                   min_duration=0.1, max_duration=10.0):
        """
        Merge consecutive segments based on specified criteria
        
        Args:
            segments: List of segments
            merge_by: 'emotion_only', 'emotion_and_intensity', or 'none'
            min_duration: Minimum segment duration (seconds)
            max_duration: Maximum segment duration (seconds)
        """
        if not segments or merge_by == 'none':
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            # Determine if we should merge
            same_emotion = segment['emotion'] == current['emotion']
            same_intensity = segment['intensity'] == current['intensity']
            current_duration = current['end'] - current['start']
            
            # Decide based on merge strategy
            if merge_by == 'emotion_only':
                should_merge = same_emotion
            elif merge_by == 'emotion_and_intensity':
                should_merge = same_emotion and same_intensity  # KEY FIX!
            else:
                should_merge = False
            
            # Don't merge if current segment is already too long
            if current_duration >= max_duration:
                should_merge = False
            
            if should_merge:
                # Merge segments
                dur1 = current['end'] - current['start']
                dur2 = segment['end'] - segment['start']
                total_dur = dur1 + dur2
                
                current['end'] = segment['end']
                
                # Weighted average of confidence and intensity
                if 'confidence' in current:
                    current['confidence'] = (
                        current['confidence'] * dur1 + segment['confidence'] * dur2
                    ) / total_dur
                
                if 'intensity_score' in current:
                    current['intensity_score'] = (
                        current['intensity_score'] * dur1 + 
                        segment['intensity_score'] * dur2
                    ) / total_dur
                    
                    # Re-determine intensity level
                    score = current['intensity_score']
                    if score >= 0.75:
                        current['intensity'] = 'high'
                    elif score >= 0.50:
                        current['intensity'] = 'medium'
                    else:
                        current['intensity'] = 'low'
            else:
                # Check minimum duration before appending
                if (current['end'] - current['start']) >= min_duration:
                    merged.append(current)
                elif merged:
                    # Too short - merge with previous segment
                    merged[-1]['end'] = current['end']
                else:
                    # First segment, keep even if short
                    merged.append(current)
                
                current = segment.copy()
        
        # Handle last segment
        if (current['end'] - current['start']) >= min_duration:
            merged.append(current)
        elif merged:
            merged[-1]['end'] = current['end']
        else:
            merged.append(current)
        
        return merged
    
    def diarize_with_intensity(
        self,
        audio_path,
        use_short_codes=True,
        merge_consecutive=True,
        merge_by='emotion_and_intensity',
        apply_smoothing=True,
        include_intensity_details=True,
        min_segment_duration=0.1,
        max_segment_duration=10.0,
        run_diagnostics=False
    ):
        """
        Perform emotion diarization with intensity estimation
        
        Args:
            audio_path: Path to audio file
            use_short_codes: Use short emotion codes (h/s/a/n)
            merge_consecutive: Merge consecutive segments
            merge_by: 'emotion_only', 'emotion_and_intensity', or 'none'
            apply_smoothing: Apply temporal smoothing
            include_intensity_details: Include detailed intensity scores
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            run_diagnostics: Run diagnostic checks before processing
            
        Returns:
            Dictionary with emotion and intensity timeline
        """
        audio_path = str(audio_path)
        print(f"\nProcessing: {audio_path}")
        
        # Run diagnostics if requested
        if run_diagnostics:
            diagnose_predictions(self, audio_path)
        
        # Load audio
        waveform = self.preprocess_audio(audio_path)
        duration = len(waveform) / self.sample_rate
        print(f"  Duration: {duration:.2f}s")
        
        # Predict frame-by-frame
        waveform_batch = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(waveform_batch)
            probs = F.softmax(logits, dim=-1)
        
        # Get predictions for each frame
        probs = probs.squeeze(0).cpu()
        confidences, predictions = torch.max(probs, dim=-1)
        
        num_frames = probs.shape[0]
        print(f"  Frames: {num_frames}")
        
        # Convert to lists
        predictions = predictions.numpy().tolist()
        confidences = confidences.numpy().tolist()
        
        # Apply smoothing if requested
        if apply_smoothing and self.smoothing_window > 1:
            predictions = self.smooth_predictions(predictions, self.smoothing_window)
            print(f"  Applied smoothing: window={self.smoothing_window}")
        
        # Convert frame predictions to segments with intensity
        segments = []
        
        for frame_idx, (pred_idx, confidence) in enumerate(zip(predictions, confidences)):
            emotion = EMOTION_LABELS[pred_idx]
            frame_probs = probs[frame_idx]
            
            # Estimate intensity
            intensity_label, intensity_score, method_scores = \
                IntensityEstimator.aggregate_methods(
                    confidence, 
                    frame_probs, 
                    self.intensity_method
                )
            
            start_time = frame_idx * self.frame_shift
            end_time = (frame_idx + 1) * self.frame_shift
            
            segment = {
                'start': start_time,
                'end': end_time,
                'emotion': self.emotion_to_code[emotion] if use_short_codes else emotion,
                'intensity': intensity_label,
                'confidence': confidence,
                'intensity_score': intensity_score
            }
            
            if include_intensity_details:
                segment['intensity_details'] = method_scores
            
            segments.append(segment)
        
        print(f"  Raw segments: {len(segments)}")
        
        # Merge consecutive segments
        if merge_consecutive:
            segments = self.merge_consecutive_segments(
                segments, 
                merge_by=merge_by,
                min_duration=min_segment_duration,
                max_duration=max_segment_duration
            )
            print(f"  After merging ({merge_by}): {len(segments)} segments")
        
        # Calculate statistics
        stats = self._calculate_statistics(segments)
        
        return {
            'file': audio_path,
            'duration': duration,
            'segments': segments,
            'statistics': stats
        }
    
    def _calculate_statistics(self, segments):
        """Calculate emotion and intensity statistics"""
        emotion_stats = {}
        intensity_stats = {}
        emotion_intensity_matrix = {}
        
        for seg in segments:
            duration = seg['end'] - seg['start']
            emotion = seg['emotion']
            intensity = seg['intensity']
            
            # Emotion durations
            if emotion not in emotion_stats:
                emotion_stats[emotion] = {
                    'duration': 0,
                    'count': 0,
                    'avg_intensity': 0,
                    'intensities': []
                }
            emotion_stats[emotion]['duration'] += duration
            emotion_stats[emotion]['count'] += 1
            emotion_stats[emotion]['intensities'].append(seg['intensity_score'])
            
            # Intensity durations
            if intensity not in intensity_stats:
                intensity_stats[intensity] = {
                    'duration': 0,
                    'count': 0
                }
            intensity_stats[intensity]['duration'] += duration
            intensity_stats[intensity]['count'] += 1
            
            # Emotion-Intensity matrix
            key = f"{emotion}_{intensity}"
            if key not in emotion_intensity_matrix:
                emotion_intensity_matrix[key] = {
                    'duration': 0,
                    'count': 0
                }
            emotion_intensity_matrix[key]['duration'] += duration
            emotion_intensity_matrix[key]['count'] += 1
        
        # Calculate averages
        for emotion in emotion_stats:
            intensities = emotion_stats[emotion]['intensities']
            emotion_stats[emotion]['avg_intensity'] = np.mean(intensities)
            del emotion_stats[emotion]['intensities']
        
        return {
            'emotions': emotion_stats,
            'intensities': intensity_stats,
            'emotion_intensity_matrix': emotion_intensity_matrix
        }

# Visualization and Export
def print_segments(result, show_details=True):
    """Pretty print emotion-intensity segments"""
    
    emoji_map = {
        'h': '😊', 's': '😢', 'a': '😠', 'd': '🤢', 
        'f': '😨', 'u': '😔', 'n': '😐',
        'happy': '😊', 'sad': '😢', 'angry': '😠', 
        'disgust': '🤢', 'fear': '😨', 'upset': '😔', 'neutral': '😐'
    }
    
    intensity_symbols = {
        'low': '▁',
        'medium': '▄',
        'high': '█',
    }
    
    print(f"\n{'='*80}")
    print(f"EMOTION-INTENSITY TIMELINE")
    print(f"File: {result['file']}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"{'='*80}\n")
    
    segments = result['segments']
    
    for i, seg in enumerate(segments, 1):
        duration = seg['end'] - seg['start']
        emoji = emoji_map.get(seg['emotion'], '?')
        intensity_sym = intensity_symbols.get(seg['intensity'], '?')
        
        # Base info
        info = (f"{i:4d}. {seg['start']:7.3f}s - {seg['end']:7.3f}s "
                f"({duration:6.3f}s)  {emoji} {seg['emotion']:7s}  "
                f"{intensity_sym*4} {seg['intensity']:10s}")
        
        if show_details:
            info += f"  (conf: {seg['confidence']:.3f}, score: {seg['intensity_score']:.3f})"
        
        print(info)
    
    print(f"\n{'='*80}")
    print_statistics(result['statistics'], result['duration'])

def print_statistics(stats, total_duration):
    """Print emotion and intensity statistics"""
    
    print(f"\n EMOTION DISTRIBUTION:")
    print("-" * 80)
    for emotion, data in sorted(stats['emotions'].items(), 
                                key=lambda x: x[1]['duration'], 
                                reverse=True):
        pct = (data['duration'] / total_duration) * 100
        bar = '█' * int(pct / 2)
        print(f"  {emotion:7s}: {data['duration']:6.2f}s ({pct:5.1f}%)  {bar}")
        print(f"           Avg intensity: {data['avg_intensity']:.3f}  "
              f"({data['count']} segments)")
    
    print(f"\n INTENSITY DISTRIBUTION:")
    print("-" * 80)
    intensity_order = ['high', 'medium', 'low']
    for intensity in intensity_order:
        if intensity in stats['intensities']:
            data = stats['intensities'][intensity]
            pct = (data['duration'] / total_duration) * 100
            bar = '█' * int(pct / 2)
            print(f"  {intensity:10s}: {data['duration']:6.2f}s ({pct:5.1f}%)  {bar}")
    
    print(f"\n EMOTION-INTENSITY MATRIX:")
    print("-" * 80)
    for key, data in sorted(stats['emotion_intensity_matrix'].items(),
                           key=lambda x: x[1]['duration'],
                           reverse=True):
        emotion, intensity = key.split('_')
        pct = (data['duration'] / total_duration) * 100
        print(f"  {emotion:7s} + {intensity:10s}: "
              f"{data['duration']:5.2f}s ({pct:4.1f}%)  "
              f"({data['count']} segments)")


def export_to_json(result, output_path):
    """Export results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Exported to: {output_path}")


def export_to_label_file(result, output_path):
    """Export in label file format (for Audacity/other tools)"""
    with open(output_path, 'w') as f:
        for seg in result['segments']:
            f.write(f"{seg['start']:.3f}\t{seg['end']:.3f}\t"
                   f"{seg['emotion']}\t{seg['intensity']}\n")
    print(f"✓ Exported label file: {output_path}")

# Main CLI
def main():
    parser = argparse.ArgumentParser(
        description='Speech Emotion Diarization with Intensity - RUNNABLE VERSION'
    )
    # NOTE: removed 'required=True' to enable Run button execution
    parser.add_argument('--checkpoint', type=str, 
                       default="results/emotion_diarization_7class_1/save/CKPT+epoch_50/model.ckpt",
                       help='Path to trained model checkpoint')
    parser.add_argument('--audio', type=str, 
                       default="wav/mixed_test.wav", 
                       help='Path to audio file')
    parser.add_argument('--batch', type=str, help='Directory containing audio files')
    parser.add_argument('--output-dir', type=str, default='diarization_results',
                       help='Output directory for results')
    parser.add_argument('--smoothing', type=int, default=3,
                       help='Temporal smoothing window (default: 3)')
    parser.add_argument('--intensity-method', type=str, default='combined',
                       choices=['confidence', 'entropy', 'gap', 'combined'],
                       help='Intensity estimation method')
    parser.add_argument('--merge-by', type=str, default='emotion_and_intensity',
                       choices=['emotion_only', 'emotion_and_intensity', 'none'],
                       help='Merge strategy: emotion_only, emotion_and_intensity, or none')
    parser.add_argument('--min-duration', type=float, default=0.2,
                       help='Minimum segment duration in seconds (default: 0.2)')
    parser.add_argument('--max-duration', type=float, default=5.0,
                       help='Maximum segment duration in seconds (default: 5.0)')
    parser.add_argument('--no-merge', action='store_true',
                       help='Do not merge consecutive segments (same as --merge-by none)')
    parser.add_argument('--full-names', action='store_true',
                       help='Use full emotion names')
    parser.add_argument('--export-json', action='store_true', default=True,
                       help='Export results to JSON')
    parser.add_argument('--export-labels', action='store_true',
                       help='Export in label file format')
    parser.add_argument('--diagnostics', action='store_true',
                       help='Run diagnostic checks before processing')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("EMOTION DIARIZATION WITH INTENSITY - RUNNABLE VERSION")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine merge strategy
    merge_by = 'none' if args.no_merge else args.merge_by
    
    # Initialize diarizer
    diarizer = EmotionIntensityDiarizer(
        args.checkpoint,
        device=args.device,
        smoothing_window=args.smoothing,
        intensity_method=args.intensity_method
    )
    
    # Process single file
    if args.audio and not args.batch:
        result = diarizer.diarize_with_intensity(
            args.audio,
            use_short_codes=not args.full_names,
            merge_consecutive=not args.no_merge,
            merge_by=merge_by,
            apply_smoothing=(args.smoothing > 1),
            min_segment_duration=args.min_duration,
            max_segment_duration=args.max_duration,
            run_diagnostics=args.diagnostics
        )
        
        print_segments(result, show_details=True)
        
        # Export if requested
        audio_name = Path(args.audio).stem
        if args.export_json:
            json_path = output_dir / f"{audio_name}_diarization.json"
            export_to_json(result, json_path)
        
        if args.export_labels:
            label_path = output_dir / f"{audio_name}_labels.txt"
            export_to_label_file(result, label_path)
    
    # Batch processing
    elif args.batch:
        audio_dir = Path(args.batch)
        audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
        
        print(f"\nProcessing {len(audio_files)} files...")
        
        all_results = []
        
        for audio_file in audio_files:
            try:
                result = diarizer.diarize_with_intensity(
                    str(audio_file),
                    use_short_codes=not args.full_names,
                    merge_consecutive=not args.no_merge,
                    merge_by=merge_by,
                    apply_smoothing=(args.smoothing > 1),
                    min_segment_duration=args.min_duration,
                    max_segment_duration=args.max_duration
                )
                
                all_results.append(result)
                
                # Quick summary
                stats = result['statistics']
                top_emotion = max(stats['emotions'].items(), 
                                key=lambda x: x[1]['duration'])[0]
                top_intensity = max(stats['intensities'].items(),
                                  key=lambda x: x[1]['duration'])[0]
                
                print(f"  ✓ {audio_file.name}: {len(result['segments'])} segments, "
                      f"dominant: {top_emotion} ({top_intensity})")
                
                # Export individual results
                audio_name = audio_file.stem
                if args.export_json:
                    json_path = output_dir / f"{audio_name}_diarization.json"
                    export_to_json(result, json_path)
                
                if args.export_labels:
                    label_path = output_dir / f"{audio_name}_labels.txt"
                    export_to_label_file(result, label_path)
                    
            except Exception as e:
                print(f"  ✗ {audio_file.name}: ERROR - {str(e)}")
        
        # Save batch summary
        if args.export_json and all_results:
            summary_path = output_dir / "batch_summary.json"
            with open(summary_path, 'w') as f:
                json.dump({
                    'num_files': len(all_results),
                    'total_duration': sum(r['duration'] for r in all_results),
                    'results': all_results
                }, f, indent=2)
            print(f"\n✓ Batch summary saved: {summary_path}")
    
    else:
        print("\n⚠️  No audio file or batch directory specified!")
        print("Using defaults: checkpoint={}, audio={}".format(args.checkpoint, args.audio))
        parser.print_help()


if __name__ == "__main__":
    main()


# python inference_diarization_7emotions_multiple.py --checkpoint results/emotion_diarization_7class_1/save/CKPT+epoch_50/model.ckpt --audio wav/test.wav --merge-by emotion_and_intensity --smoothing 3 --min-duration 0.2 --max-duration 5.0 --export-json

# python inference_diarization_7emotions_multiple.py --diagnostics
