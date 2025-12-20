"""
Create Mixed-Emotion Test File
Concatenates different emotion files to test diarization
"""

import torch
import torchaudio
from pathlib import Path
import argparse

def create_mixed_emotion_file(input_files, output_path='wav/mixed_test.wav', 
                               segment_duration=5.0, target_sr=16000):
    """
    Create a mixed-emotion test file by concatenating audio files
    
    Args:
        input_files: List of (filepath, emotion_label) tuples
        output_path: Where to save the mixed file
        segment_duration: Duration of each segment in seconds
        target_sr: Target sample rate
    """
    
    print("\n" + "="*80)
    print("CREATING MIXED-EMOTION TEST FILE")
    print("="*80 + "\n")
    
    segments = []
    segment_info = []
    current_time = 0.0
    
    for filepath, emotion in input_files:
        if not Path(filepath).exists():
            print(f"⚠️  Skipping {filepath} (not found)")
            continue
        
        # Load audio
        waveform, sr = torchaudio.load(filepath)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Calculate segment samples
        segment_samples = int(segment_duration * target_sr)
        
        # Take first N seconds (or all if shorter)
        actual_samples = min(segment_samples, waveform.shape[1])
        segment = waveform[:, :actual_samples]
        
        # Pad if shorter than desired duration
        if actual_samples < segment_samples:
            padding = segment_samples - actual_samples
            segment = torch.nn.functional.pad(segment, (0, padding))
        
        segments.append(segment)
        
        # Track segment info
        end_time = current_time + segment_duration
        segment_info.append({
            'emotion': emotion,
            'start': current_time,
            'end': end_time,
            'file': filepath
        })
        current_time = end_time
        
        print(f"✓ Added {segment_duration:.1f}s of '{emotion}' from {Path(filepath).name}")
    
    if not segments:
        print("\n❌ No valid audio files found!")
        return None
    
    # Concatenate all segments
    mixed_waveform = torch.cat(segments, dim=1)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    torchaudio.save(str(output_path), mixed_waveform, target_sr)
    
    total_duration = mixed_waveform.shape[1] / target_sr
    
    print(f"\n{'='*80}")
    print(f"✓ Created mixed-emotion file: {output_path}")
    print(f"  Total duration: {total_duration:.2f}s")
    print(f"  Sample rate: {target_sr} Hz")
    print(f"  Segments: {len(segments)}")
    
    print(f"\n📋 EXPECTED EMOTION TIMELINE:")
    print("-"*80)
    for i, info in enumerate(segment_info, 1):
        print(f"  {i}. {info['start']:6.2f}s - {info['end']:6.2f}s: {info['emotion']:10s} "
              f"(from {Path(info['file']).name})")
    
    print(f"\n{'='*80}")
    print(f"\n🧪 TEST THIS FILE:")
    print(f"python inference_diarization_7emotions_runnable.py \\")
    print(f"  --audio {output_path} --diagnostics\n")
    
    return str(output_path), segment_info


def main():
    parser = argparse.ArgumentParser(description='Create mixed-emotion test file')
    parser.add_argument('--files', nargs='+', required=False,
                       help='Audio files to mix (emotion:filepath format)')
    parser.add_argument('--output', type=str, default='wav/mixed_test.wav',
                       help='Output file path')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duration of each segment in seconds')
    
    args = parser.parse_args()
    
    # Default files if none specified
    if args.files:
        # Parse emotion:filepath format
        input_files = []
        for f in args.files:
            if ':' in f:
                emotion, filepath = f.split(':', 1)
                input_files.append((filepath, emotion))
            else:
                # Guess emotion from filename
                emotion = Path(f).stem
                input_files.append((f, emotion))
    else:
        # Use default test files
        input_files = [
            ('wav/happy.wav', 'happy'),
            ('wav/sad.wav', 'sad'),
            ('wav/angry.wav', 'angry'),
            ('wav/neutral.wav', 'neutral'),
        ]
        
        print("No files specified, using default pattern:")
        print("  Looking for: happy.wav, sad.wav, angry.wav, neutral.wav in wav/")
        print("\nTo specify custom files:")
        print("  python create_mixed_test.py --files emotion1:file1.wav emotion2:file2.wav")
    
    create_mixed_emotion_file(input_files, args.output, args.duration)


if __name__ == "__main__":
    main()
