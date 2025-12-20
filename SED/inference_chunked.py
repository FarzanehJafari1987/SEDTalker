"""
Chunk-Based Emotion Diarization
Process audio in small chunks to get frame-level predictions
WORKAROUND for models that learn file-level features
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from inference_diarization_7emotions_multiple import (
    FrameLevelEmotionModel, CONFIG, EMOTION_LABELS,
    IntensityEstimator, print_segments, export_to_json
)

# Make sure numpy is imported for statistics
import numpy as np

class ChunkedEmotionDiarizer:
    """
    Process audio in overlapping chunks to force frame-level predictions
    This works around models that learn file-level context
    """
    
    def __init__(self, checkpoint_path, device='cuda', chunk_duration=2.0, overlap=0.5):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: cuda or cpu
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap ratio (0.5 = 50% overlap)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sample_rate = CONFIG['sample_rate']
        self.frame_shift = CONFIG['frame_shift']
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        
        self.emotion_to_code = {
            'happy': 'h', 'sad': 's', 'angry': 'a', 'disgust': 'd',
            'fear': 'f', 'upset': 'u', 'neutral': 'n'
        }
        
        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        self.model = FrameLevelEmotionModel(CONFIG)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load state dict
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded for chunked processing")
    
    def diarize_chunked(self, audio_path, use_short_codes=True):
        """
        Process audio in chunks to get frame-level predictions
        """
        print(f"\n{'='*80}")
        print(f"CHUNKED EMOTION DIARIZATION")
        print(f"  Chunk size: {self.chunk_duration}s")
        print(f"  Overlap: {self.overlap*100:.0f}%")
        print(f"{'='*80}\n")
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        total_duration = len(waveform) / self.sample_rate
        
        print(f"Processing: {audio_path}")
        print(f"Duration: {total_duration:.2f}s\n")
        
        # Calculate chunk parameters
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        hop_samples = int(chunk_samples * (1 - self.overlap))
        
        # Process chunks
        all_predictions = []
        all_confidences = []
        all_probs = []
        
        num_chunks = 0
        for start in range(0, len(waveform), hop_samples):
            end = min(start + chunk_samples, len(waveform))
            
            # Pad if necessary
            chunk = waveform[start:end]
            if len(chunk) < chunk_samples:
                padding = chunk_samples - len(chunk)
                chunk = F.pad(chunk, (0, padding))
            
            # Predict on chunk
            chunk_batch = chunk.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(chunk_batch)
                probs = F.softmax(logits, dim=-1)
            
            probs = probs.squeeze(0).cpu()
            confidences, predictions = torch.max(probs, dim=-1)
            
            # Store predictions for this chunk
            all_predictions.append(predictions)
            all_confidences.append(confidences)
            all_probs.append(probs)
            
            num_chunks += 1
            
            if end >= len(waveform):
                break
        
        print(f"Processed {num_chunks} chunks")
        
        # Merge overlapping predictions
        frame_predictions, frame_confidences, frame_probs = self._merge_chunks(
            all_predictions, all_confidences, all_probs,
            hop_samples, len(waveform)
        )
        
        print(f"Total frames: {len(frame_predictions)}")
        
        # Check diversity
        unique_preds = len(set(frame_predictions.numpy().tolist()))
        print(f"Unique emotions detected: {unique_preds} / 7\n")
        
        # Create segments
        segments = self._create_segments(
            frame_predictions, frame_confidences, frame_probs, use_short_codes
        )
        
        # Merge consecutive
        segments = self._merge_consecutive(segments)
        
        # Calculate stats
        stats = self._calculate_stats(segments)
        
        return {
            'file': audio_path,
            'duration': total_duration,
            'segments': segments,
            'statistics': stats
        }
    
    def _merge_chunks(self, predictions_list, confidences_list, probs_list, hop_samples, total_samples):
        """Merge overlapping chunk predictions"""
        
        # Calculate actual number of frames from audio duration
        actual_frames = int(total_samples / self.sample_rate / self.frame_shift)
        
        # Initialize with actual frame count
        frame_probs_sum = torch.zeros(actual_frames, 7)
        frame_counts = torch.zeros(actual_frames)
        
        chunk_frames = predictions_list[0].shape[0]
        hop_frames = int(hop_samples / self.sample_rate / self.frame_shift)
        
        for i, (preds, confs, probs) in enumerate(zip(predictions_list, confidences_list, probs_list)):
            start_frame = i * hop_frames
            end_frame = min(start_frame + chunk_frames, actual_frames)
            actual_chunk_frames = end_frame - start_frame
            
            if actual_chunk_frames > 0:
                frame_probs_sum[start_frame:end_frame] += probs[:actual_chunk_frames]
                frame_counts[start_frame:end_frame] += 1
        
        # Average (avoid division by zero)
        frame_probs = frame_probs_sum / frame_counts.unsqueeze(1).clamp(min=1)
        frame_confidences, frame_predictions = torch.max(frame_probs, dim=-1)
        
        return frame_predictions, frame_confidences, frame_probs
    
    def _create_segments(self, predictions, confidences, probs, use_short_codes):
        """Convert frame predictions to segments"""
        segments = []
        
        for frame_idx, (pred_idx, confidence) in enumerate(zip(predictions, confidences)):
            emotion = EMOTION_LABELS[pred_idx]
            frame_probs = probs[frame_idx]
            
            intensity_label, intensity_score, method_scores = \
                IntensityEstimator.aggregate_methods(
                    confidence.item(), frame_probs, 'combined'
                )
            
            start_time = frame_idx * self.frame_shift
            end_time = (frame_idx + 1) * self.frame_shift
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'emotion': self.emotion_to_code[emotion] if use_short_codes else emotion,
                'intensity': intensity_label,
                'confidence': confidence.item(),
                'intensity_score': intensity_score
            })
        
        return segments
    
    def _merge_consecutive(self, segments, min_duration=0.2, max_duration=5.0):
        """Merge consecutive segments with same emotion"""
        if not segments:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            if (segment['emotion'] == current['emotion'] and 
                segment['intensity'] == current['intensity'] and
                (current['end'] - current['start']) < max_duration):
                
                # Merge
                dur1 = current['end'] - current['start']
                dur2 = segment['end'] - segment['start']
                total_dur = dur1 + dur2
                
                current['end'] = segment['end']
                current['confidence'] = (
                    current['confidence'] * dur1 + segment['confidence'] * dur2
                ) / total_dur
                current['intensity_score'] = (
                    current['intensity_score'] * dur1 + segment['intensity_score'] * dur2
                ) / total_dur
            else:
                if (current['end'] - current['start']) >= min_duration:
                    merged.append(current)
                current = segment.copy()
        
        if (current['end'] - current['start']) >= min_duration:
            merged.append(current)
        
        return merged
    
    def _calculate_stats(self, segments):
        """Calculate statistics"""
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chunked Emotion Diarization')
    parser.add_argument('--checkpoint', type=str,
                       default="results/emotion_diarization_7class_1/save/CKPT+epoch_50/model.ckpt")
    parser.add_argument('--audio', type=str, default="wav/mixed_test.wav")
    parser.add_argument('--chunk-duration', type=float, default=2.0,
                       help='Chunk duration in seconds (default: 2.0)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio (default: 0.5)')
    parser.add_argument('--export-json', action='store_true', default=True)
    parser.add_argument('--output-dir', type=str, default='diarization_results',
                       help='Output directory for JSON files (default: diarization_results)')
    
    args = parser.parse_args()
    
    diarizer = ChunkedEmotionDiarizer(
        args.checkpoint,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap
    )
    
    result = diarizer.diarize_chunked(args.audio)
    
    print_segments(result, show_details=True)
    
    if args.export_json:
        output_path = Path(args.output_dir) / f"{Path(args.audio).stem}_chunked.json"
        # Create directory if it doesn't exist
        output_path.parent.mkdir(exist_ok=True, parents=True)
        export_to_json(result, output_path)
