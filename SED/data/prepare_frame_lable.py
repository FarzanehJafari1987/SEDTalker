"""
Prepare Frame-Level Dataset for Speech Emotion Diarization
Converts utterance-level labels to frame-level labels
"""

import json
import torchaudio
import torch
import os
from pathlib import Path
from tqdm import tqdm

def create_frame_level_dataset(input_json, output_json, sample_rate=16000, frame_shift=0.02):
    """
    Convert utterance-level to frame-level labels
    
    Args:
        input_json: Path to original JSON file
        output_json: Path to output JSON file
        sample_rate: Audio sample rate
        frame_shift: Frame shift in seconds (20ms = 0.02s is standard for WavLM)
    """
    
    print(f"\nProcessing: {input_json}")
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    frame_data = {}
    errors = []
    
    for key, item in tqdm(data.items(), desc="Converting to frame-level"):
        try:
            wav_path = item['wav']
            emotion = item['emotion']
            
            # Check if file exists
            if not os.path.exists(wav_path):
                errors.append(f"{key}: File not found - {wav_path}")
                continue
            
            # Load audio to get duration
            try:
                info = torchaudio.info(wav_path)
                duration = info.num_frames / info.sample_rate
            except Exception as e:
                errors.append(f"{key}: Cannot load audio - {e}")
                continue
            
            # Calculate number of frames
            # WavLM produces 1 frame per 20ms of audio
            num_frames = int(duration / frame_shift)
            
            if num_frames <= 0:
                errors.append(f"{key}: Audio too short ({duration}s)")
                continue
            
            # Create frame-level labels (all frames have the same emotion for single-emotion utterances)
            frame_labels = [emotion] * num_frames
            
            frame_data[key] = {
                'wav': wav_path,
                'emotion': emotion,  # Keep original utterance-level label
                'frame_labels': frame_labels,  # Add frame-level labels
                'duration': duration,
                'num_frames': num_frames
            }
            
        except Exception as e:
            errors.append(f"{key}: Unexpected error - {e}")
            continue
    
    # Save frame-level dataset
    with open(output_json, 'w') as f:
        json.dump(frame_data, f, indent=2)
    
    print(f"✓ Created {output_json}")
    print(f"  Total samples: {len(frame_data)}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        error_file = output_json.replace('.json', '_errors.txt')
        with open(error_file, 'w') as f:
            f.write('\n'.join(errors))
        print(f"  Error log: {error_file}")
    
    # Print statistics
    total_frames = sum(item['num_frames'] for item in frame_data.values())
    avg_frames = total_frames / len(frame_data) if frame_data else 0
    print(f"  Total frames: {total_frames:,}")
    print(f"  Avg frames per sample: {avg_frames:.1f}")
    print(f"  Avg duration: {avg_frames * frame_shift:.2f}s")
    
    return frame_data


def analyze_frame_distribution(frame_data):
    """ Analyze emotion distribution at the frame level."""
    
    emotion_frame_counts = {}
    
    for item in frame_data.values():
        for label in item['frame_labels']:
            emotion_frame_counts[label] = emotion_frame_counts.get(label, 0) + 1
    
    total_frames = sum(emotion_frame_counts.values())
    
    print("\nFrame-Level Distribution:")
    print("="*50)
    for emotion, count in sorted(emotion_frame_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_frames) * 100
        print(f"  {emotion:10s}: {count:8,} frames ({percentage:5.2f}%)")
    print("="*50)
    print(f"  Total:       {total_frames:8,} frames")
    
    return emotion_frame_counts


def main():
    # Configuration
    # For the 7-emotion system with neutral:
    data_folder = "data/processed_emotions_7class"
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"\n{'='*70}")
        print("ERROR: Data folder not found!")
        print(f"{'='*70}")
        print(f"Looking for: {data_folder}")
        print()
        print("Did you run data_preparation_7emotions.py first?")
        print()
        print("Steps to fix:")
        print("  1. Make sure you ran: python data_preparation_7emotions.py")
        print("  2. Check that the output folder exists")
        print("  3. Update data_folder in this script if using different path")
        print(f"{'='*70}")
        return
    
    print("="*70)
    print("Frame-Level Data Preparation for Emotion Diarization")
    print("="*70)
    print(f"Data folder: {data_folder}")
    
    # Check which files exist
    for split in ['train', 'valid', 'test']:
        input_file = os.path.join(data_folder, f'{split}.json')
        if os.path.exists(input_file):
            print(f"  ✓ Found {split}.json")
        else:
            print(f"  ✗ Missing {split}.json")
    
    # Process all splits
    all_frame_data = {}
    
    for split in ['train', 'valid', 'test']:
        input_file = os.path.join(data_folder, f'{split}.json')
        output_file = os.path.join(data_folder, f'{split}_frames.json')
        
        if not os.path.exists(input_file):
            print(f"\n⚠️  Skipping {split}: {input_file} not found")
            continue
        
        frame_data = create_frame_level_dataset(input_file, output_file)
        all_frame_data[split] = frame_data
    
    # Analyze distribution for training set
    if 'train' in all_frame_data:
        print("\n" + "="*70)
        print("Training Set Frame Distribution")
        print("="*70)
        analyze_frame_distribution(all_frame_data['train'])
    
    print("\n" + "="*70)
    print("✓ Frame-level data preparation complete!")
    print("="*70)
    print("\nOutput files:")
    print(f"  {data_folder}/train_frames.json")
    print(f"  {data_folder}/valid_frames.json")
    print(f"  {data_folder}/test_frames.json")
    print("\nNext steps:")
    print("  1. Run: python train_frame_level_7emotions.py")
    print("  2. Use trained model with: python inference_diarization_7emotions.py")


if __name__ == "__main__":
    main()
