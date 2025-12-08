"""
Data preparation for MELD (Multimodal EmotionLines Dataset).

MELD is a multimodal emotion dataset with utterances from Friends TV show.

Dataset info:
- Language: English
- Speakers: ~304 speakers from Friends TV show
- Emotions: neutral, joy, surprise, anger, sadness, disgust, fear (7 emotions)
- Samples: ~13,000 utterances across 1,433 dialogues
- Splits: train (~10K), dev (~1K), test (~2.6K)

Dataset structure:
  MELD/
    ├── train/
    │   ├── dia0_utt0.wav
    │   ├── dia0_utt1.wav
    │   └── ...
    ├── dev/
    │   └── ...
    ├── test/
    │   └── ...
    ├── train_sent_emo.csv
    ├── dev_sent_emo.csv
    └── test_sent_emo.csv

Download from: https://affective-meca.github.io/MELD.Sharp/

Author
------
Adapted from AESDD preparation script 2024
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

# MELD emotion mapping to standard labels
EMOTION_MAP = {
    "neutral": "neutral",
    "joy": "happy",
    "surprise": "surprise",
    "anger": "angry",
    "sadness": "sad",
    "disgust": "disgust",
    "fear": "fear",
}

# Map to 3-class system (optional)
EMOTION_MAP_3CLASS = {
    "neutral": "neutral",
    "joy": "happy",
    "surprise": "happy",
    "anger": "angry",
    "sadness": "sad",
    "disgust": "angry",
    "fear": "sad",
}


def prepare_meld(
    data_folder,
    save_json,
    seed=12
):
    """
    Prepares the json files for the MELD dataset.
    
    Arguments
    ---------
    data_folder : str
        Path to the folder where MELD dataset is stored.
        Should contain train/, dev/, test/ folders and CSV files.
    save_json : str
        Path where the data specification file will be saved.
    seed : int
        Seed for reproducibility
    """
    
    # Check if already done
    if skip(save_json):
        logger.info("Preparation completed in previous run, skipping.")
        return
    
    logger.info("Starting MELD dataset preparation...")
    
    if not os.path.exists(data_folder):
        raise ValueError(f"MELD folder not found: {data_folder}")
    
    logger.info(f"MELD folder: {data_folder}")
    
    # Create JSON
    data_json = create_meld_json(data_folder, save_json)
    
    logger.info(f"MELD preparation complete! Total samples: {len(data_json)}")
    
    return data_json


def load_meld_csv(csv_path):
    """Load MELD CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def get_audio_path(data_folder, split, dialogue_id, utterance_id):
    """
    Get audio file path for a given utterance
    Handles multiple naming conventions and finds actual files
    
    Args:
        data_folder: Root MELD folder
        split: 'train', 'dev', or 'test'
        dialogue_id: Dialogue ID
        utterance_id: Utterance ID
        
    Returns:
        Path to audio file or None if not found
    """
    # Try multiple possible filenames
    possible_names = [
        f"dia{dialogue_id}_utt{utterance_id}.wav",
        f"dia{dialogue_id}_utt{utterance_id}.mp4",
        f"._dia{dialogue_id}_utt{utterance_id}.wav",  # macOS metadata
    ]
    
    split_dir = os.path.join(data_folder, split)
    
    # Try each possible filename
    for filename in possible_names:
        audio_path = os.path.join(split_dir, filename)
        if os.path.exists(audio_path) and not filename.startswith('._'):
            return audio_path
    
    # If not found, try to find any matching file in the directory
    if os.path.exists(split_dir):
        for filename in os.listdir(split_dir):
            # Skip macOS metadata files
            if filename.startswith('._'):
                continue
            
            # Check if filename matches pattern
            if f"dia{dialogue_id}_utt{utterance_id}" in filename:
                return os.path.join(split_dir, filename)
    
    return None


def create_meld_json(data_folder, save_json):
    """
    Create JSON for MELD dataset
    """
    
    emo_map = EMOTION_MAP
    
    print(f"\n{'='*70}")
    print("MELD DATASET PREPARATION")
    print(f"{'='*70}")
    print(f"Scanning: {data_folder}")
    
    # Check if audio directories exist and list actual files
    print("\nChecking audio directories...")
    for split in ['train', 'dev', 'test']:
        split_dir = os.path.join(data_folder, split)
        if os.path.exists(split_dir):
            files = [f for f in os.listdir(split_dir) if f.endswith('.wav') and not f.startswith('._')]
            print(f"  {split}: {len(files)} .wav files found")
        else:
            print(f"  {split}: directory not found!")
    
    data_json = {}
    total_files = 0
    skipped_files = 0
    audio_not_found = 0
    
    emotion_counts = Counter()
    speaker_counts = Counter()
    
    # Process each split
    for split in ['train', 'dev', 'test']:
        csv_path = os.path.join(data_folder, f"{split}_sent_emo.csv")
        
        if not os.path.exists(csv_path):
            print(f"\n⚠ Warning: {split} CSV not found, skipping: {csv_path}")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Load CSV
        df = load_meld_csv(csv_path)
        
        # Process each utterance
        for idx, row in df.iterrows():
            # Get emotion
            original_emotion = row['Emotion'].lower()
            
            # Map emotion
            if original_emotion not in emo_map:
                skipped_files += 1
                continue
            
            target_emotion = emo_map[original_emotion]
            
            # Get audio path
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            audio_path = get_audio_path(data_folder, split, dialogue_id, utterance_id)
            
            # Check if audio exists
            if audio_path is None or not os.path.exists(audio_path):
                # Only log first 10 missing files to avoid spam
                if audio_not_found < 10:
                    logger.warning(f"Audio not found: dia{dialogue_id}_utt{utterance_id} in {split}")
                audio_not_found += 1
                continue
            
            # Create entry ID
            file_id = f"{split}_dia{dialogue_id}_utt{utterance_id}"
            
            # Create entry (matching AESDD format)
            data_json[file_id] = {
                "wav": os.path.abspath(audio_path),
                "emotion": [{
                    "emo": target_emotion,
                    "start": 0,
                    "end": 5.0  # Approximate duration
                }],
                "speaker": row['Speaker'],
                "dataset": "MELD",
                "language": "English",
                "split": split,
                "dialogue_id": int(dialogue_id),
                "utterance_id": int(utterance_id),
                "text": row['Utterance'],
                "sentiment": row['Sentiment'],
                "original_emotion": original_emotion
            }
            
            total_files += 1
            emotion_counts[target_emotion] += 1
            speaker_counts[row['Speaker']] += 1
    
    print(f"\n✓ Processed {total_files} files")
    if skipped_files > 0:
        print(f"⚠ Skipped {skipped_files} files (emotion filtering)")
    if audio_not_found > 0:
        print(f"⚠ Audio not found for {audio_not_found} utterances")
    
    if total_files == 0:
        print("\n❌ No files found!")
        print("\n" + "="*70)
        print("TROUBLESHOOTING")
        print("="*70)
        print("\nPossible issues:")
        print("1. Audio files have wrong extension (.mp4 instead of .wav)")
        print("2. Audio files are in wrong location")
        print("3. MELD dataset not properly extracted")
        print("\nExpected directory structure:")
        print("  MELD/")
        print("    ├── train/")
        print("    │   ├── dia0_utt0.wav")
        print("    │   ├── dia0_utt1.wav")
        print("    │   └── ...")
        print("    ├── dev/")
        print("    │   └── ...")
        print("    ├── test/")
        print("    │   └── ...")
        print("    ├── train_sent_emo.csv")
        print("    ├── dev_sent_emo.csv")
        print("    └── test_sent_emo.csv")
        print("\nTo check your structure:")
        print(f"  ls -la {data_folder}/train/ | head -20")
        print(f"  ls -la {data_folder}/*.csv")
        print("\nDownload from: https://affective-meca.github.io/MELD.Sharp/")
        print("Or: https://github.com/declare-lab/MELD")
        return {}
    
    # Show statistics
    print(f"\nEmotion distribution:")
    for emo, count in sorted(emotion_counts.items()):
        percentage = (count / total_files * 100)
        print(f"  {emo:12s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nTop 10 speakers:")
    for speaker, count in speaker_counts.most_common(10):
        percentage = (count / total_files * 100)
        print(f"  {speaker:20s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nTotal unique speakers: {len(speaker_counts)}")
    
    # Save JSON
    print(f"\nSaving to: {save_json}")
    
    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    
    with open(save_json, 'w') as f:
        json.dump(data_json, f, indent=2)
    
    # Verify
    if os.path.exists(save_json):
        file_size = os.path.getsize(save_json)
        print(f"✓ JSON file created!")
        print(f"  Location: {save_json}")
        print(f"  Size: {file_size / 1024:.1f} KB")
        print(f"  Samples: {len(data_json)}")
    
    return data_json


def skip(save_json):
    """Check if preparation already done"""
    return os.path.isfile(save_json)


# ====================================================
# Main entry
# ====================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare MELD dataset")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="datasets/MELD",
        help="Path to MELD dataset folder"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to output JSON file (default: data_folder/MELD.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_json is None:
        args.output_json = os.path.join(args.data_folder, f"MELD.json")
    
    # Prepare dataset
    prepare_meld(
        data_folder=args.data_folder,
        save_json=args.output_json,
        seed=args.seed
    )
    
    print(f"\n{'='*70}")
    print("SUCCESS! ✓")
    print(f"{'='*70}")
    print(f"\nMELD.json is ready at: {args.output_json}")
    print("\nUsage:")
    print("  python prepare_meld.py")
    print("  python prepare_meld.py --data_folder path/to/MELD")
    print("\nNext steps:")
    print("  1. Add to prepare_enhanced_3emotions.py:")
    print(f"     '{args.output_json}'")
    print("  2. Run: python prepare_enhanced_3emotions.py")
    print("  3. Train with enhanced dataset!")
    print(f"{'='*70}")