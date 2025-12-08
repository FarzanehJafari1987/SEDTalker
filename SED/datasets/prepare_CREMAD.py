#!/usr/bin/env python3
"""
CREMA-D Dataset Preparation - Simple and Complete

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- 7,442 audio clips
- 91 actors (48 male, 43 female)
- 6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad
- File format: 1001_DFA_ANG_XX.wav

Dataset: https://github.com/CheyneyComputerScience/CREMA-D
Or Kaggle: https://www.kaggle.com/datasets/ejlok1/cremad

Author: 2024
"""

import os
import json
from collections import Counter

# CREMA-D emotion codes
EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def parse_cremad_filename(filename):
    """
    Parse CREMA-D filename: 1001_DFA_ANG_XX.wav
    
    Returns:
        dict: {actor, sentence, emotion, intensity} or None
    
    Format:
        1001 = Actor ID (1001-1091)
        DFA  = Sentence ID
        ANG  = Emotion code
        XX   = Intensity level (LO, MD, HI, XX)
    """
    
    basename = filename.replace('.wav', '')
    parts = basename.split('_')
    
    if len(parts) < 4:
        return None
    
    actor_id = parts[0]
    sentence_id = parts[1]
    emotion_code = parts[2]
    intensity = parts[3]
    
    # Map emotion
    emotion = EMOTION_MAP.get(emotion_code)
    
    if emotion is None:
        return None
    
    return {
        'actor': actor_id,
        'sentence': sentence_id,
        'emotion': emotion,
        'emotion_code': emotion_code,
        'intensity': intensity
    }


def prepare_cremad(data_folder, output_json):
    """
    Prepare CREMA-D dataset
    
    Args:
        data_folder: Path to CREMA-D folder (should contain AudioWAV/ or .wav files)
        output_json: Output JSON file path
    """
    
    print("="*70)
    print("CREMA-D DATASET PREPARATION")
    print("="*70)
    print(f"Data folder: {data_folder}")
    print(f"Output JSON: {output_json}")
    print()
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"❌ ERROR: Folder not found: {data_folder}")
        print("\nPlease download CREMA-D from:")
        print("  https://www.kaggle.com/datasets/ejlok1/cremad")
        print("\nOr:")
        print("  https://github.com/CheyneyComputerScience/CREMA-D")
        return None
    
    # Find audio folder
    # CREMA-D typically has AudioWAV/ subdirectory
    audio_folder = data_folder
    
    if os.path.exists(os.path.join(data_folder, "AudioWAV")):
        audio_folder = os.path.join(data_folder, "AudioWAV")
        print(f"✓ Found AudioWAV folder")
    elif os.path.exists(os.path.join(data_folder, "audio")):
        audio_folder = os.path.join(data_folder, "audio")
        print(f"✓ Found audio folder")
    
    print(f"Audio folder: {audio_folder}")
    
    # Check for .wav files
    wav_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"\n❌ ERROR: No .wav files found in {audio_folder}")
        print("\nExpected structure:")
        print("  datasets/CREMA-D/")
        print("    └── AudioWAV/")
        print("        ├── 1001_DFA_ANG_XX.wav")
        print("        ├── 1001_DFA_DIS_XX.wav")
        print("        └── ...")
        return None
    
    print(f"✓ Found {len(wav_files)} .wav files")
    print()
    
    # Process files
    data_json = {}
    skipped = 0
    
    print("Processing files...")
    
    for i, filename in enumerate(wav_files, 1):
        # Parse filename
        metadata = parse_cremad_filename(filename)
        
        if metadata is None:
            skipped += 1
            continue
        
        wav_path = os.path.join(audio_folder, filename)
        file_id = filename.replace('.wav', '')
        
        data_json[file_id] = {
            "wav": os.path.abspath(wav_path),
            "emotion": [{
                "emo": metadata['emotion'],
                "start": 0,
                "end": 2.0  # Approximate duration
            }],
            "speaker": metadata['actor'],
            "sentence": metadata['sentence'],
            "intensity": metadata['intensity'],
            "dataset": "CREMA-D"
        }
        
        # Show progress
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(wav_files)} files...")
    
    print(f"\n✓ Processed {len(data_json)} files")
    
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} files (could not parse)")
    
    # Show statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    # Emotion distribution
    emotion_counts = Counter()
    for item in data_json.values():
        emotion_counts[item['emotion'][0]['emo']] += 1
    
    print("\nEmotion distribution:")
    for emotion in ["angry", "disgust", "fear", "happy", "neutral", "sad"]:
        count = emotion_counts.get(emotion, 0)
        percentage = (count / len(data_json) * 100) if len(data_json) > 0 else 0
        print(f"  {emotion:12s}: {count:4d} ({percentage:5.1f}%)")
    
    # Intensity distribution
    intensity_counts = Counter()
    for item in data_json.values():
        intensity_counts[item['intensity']] += 1
    
    print("\nIntensity distribution:")
    for intensity, count in sorted(intensity_counts.items()):
        percentage = (count / len(data_json) * 100) if len(data_json) > 0 else 0
        print(f"  {intensity:5s}: {count:4d} ({percentage:5.1f}%)")
    
    # Speaker count
    speakers = set(item['speaker'] for item in data_json.values())
    print(f"\nTotal speakers: {len(speakers)}")
    
    # Save JSON
    print("\n" + "="*70)
    print("SAVING JSON")
    print("="*70)
    
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(data_json, f, indent=2)
    
    # Verify
    if os.path.exists(output_json):
        file_size = os.path.getsize(output_json)
        print(f"\n✓ JSON file created successfully!")
        print(f"  Location: {output_json}")
        print(f"  Size: {file_size / 1024:.1f} KB ({file_size / 1024 / 1024:.1f} MB)")
        print(f"  Samples: {len(data_json)}")
        
        # Show sample entries
        print("\nSample entries:")
        for i, (key, data) in enumerate(list(data_json.items())[:3]):
            print(f"  {i+1}. {key}")
            print(f"     Emotion: {data['emotion'][0]['emo']}")
            print(f"     Actor: {data['speaker']}")
            print(f"     Intensity: {data['intensity']}")
    else:
        print("\n❌ ERROR: Failed to create JSON file!")
        return None
    
    print("\n" + "="*70)
    print("SUCCESS! ✓")
    print("="*70)
    
    return data_json


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare CREMA-D dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CREMA-D Dataset Information:
  - 7,442 audio clips
  - 91 actors (48 male, 43 female)
  - 6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad
  - 4 intensity levels: LO, MD, HI, XX
  
File naming: 1001_DFA_ANG_XX.wav
  - 1001 = Actor ID
  - DFA  = Sentence ID
  - ANG  = Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
  - XX   = Intensity (LO, MD, HI, XX)

Download from:
  https://www.kaggle.com/datasets/ejlok1/cremad
  https://github.com/CheyneyComputerScience/CREMA-D

Examples:
  python prepare_cremad.py
  python prepare_cremad.py --data_folder datasets/CREMA-D
  python prepare_cremad.py --output datasets/CREMA-D/CREMA-D.json
        """
    )
    
    parser.add_argument(
        "--data_folder",
        type=str,
        default="datasets/CREMA-D",
        help="Path to CREMA-D dataset folder"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: data_folder/CREMA-D.json)"
    )
    
    args = parser.parse_args()
    
    # Set default output
    if args.output is None:
        args.output = os.path.join(args.data_folder, "CREMA-D.json")
    
    # Prepare dataset
    result = prepare_cremad(args.data_folder, args.output)
    
    if result:
        print("\nNext steps:")
        print("  1. Verify JSON: ls -lh", args.output)
        print("  2. Add to prepare_enhanced_3emotions.py:")
        print("     'datasets/CREMA-D/CREMA-D.json'")
        print("  3. Combine: python prepare_enhanced_3emotions.py")
        print("  4. Train: python train_wav2vec2_3emotions.py")
        print("\n" + "="*70)