"""
Data preparation for SAVEE (Surrey Audio-Visual Expressed Emotion) Dataset.

SAVEE is a British English emotional speech database recorded from 4 male speakers.

Dataset info:
- Language: British English
- Speakers: 4 male speakers (DC, JE, JK, KL)
- Emotions: anger, disgust, fear, happiness, sadness, surprise, neutral (7 emotions)
- Samples: 480 utterances (120 per speaker)
- Recording: High-quality studio recordings

IMPORTANT: SAVEE has TWO common directory structures:
  
  Structure 1 (Speaker folders):
    SAVEE/
      ├── DC/
      │   ├── a01.wav
      │   ├── d01.wav
      │   └── ...
      ├── JE/
      ├── JK/
      └── KL/
  
  Structure 2 (Flat with prefix):
    SAVEE/
      ├── DC_a01.wav
      ├── DC_d01.wav
      ├── JE_a01.wav
      └── ...

This script handles BOTH structures automatically!

File naming:
- With folders: a01.wav, d01.wav, n01.wav, etc.
- Flat structure: DC_a01.wav, JE_d01.wav, KL_n01.wav, etc.

Emotion codes:
- a = anger
- d = disgust
- f = fear
- h = happiness
- n = neutral
- sa = sadness
- su = surprise

Download from:
- Official: http://kahlan.eps.surrey.ac.uk/savee/
- Kaggle: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
"""

import os
import json
import logging
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

# SAVEE emotion codes (case-insensitive)
EMOTION_MAP = {
    "a": "angry",      # anger
    "d": "disgust",    # disgust
    "f": "fear",       # fear
    "h": "happy",      # happiness
    "n": "neutral",    # neutral ← INCLUDED
    "sa": "sad",       # sadness
    "su": "surprise",  # surprise
}

# SAVEE speakers
SPEAKERS = ["DC", "JE", "JK", "KL"]


def parse_savee_filename(filename):
    """
    Parse the SAVEE filename to extract the speaker and emotion
    
    Handles two formats:
    1. Flat: DC_a01.wav, JE_d02.wav, KL_n01.wav
    2. Folder: a01.wav, d02.wav, n01.wav (speaker from folder name)
    
    Args:
        filename: Audio filename (e.g., "DC_a01.wav" or "a01.wav")
        
    Returns:
        dict: {emotion, speaker, utterance_id, emotion_code} or None if parsing fails
    """
    
    basename = filename.replace('.wav', '').replace('.WAV', '')
    
    # Check for speaker prefix (flat structure)
    speaker = None
    if '_' in basename:
        parts = basename.split('_', 1)
        if parts[0].upper() in SPEAKERS:
            speaker = parts[0].upper()
            basename = parts[1]  # Rest is emotion code + number
    
    basename = basename.lower()
    
    # Try two-letter codes first (sa, su)
    if basename[:2] in EMOTION_MAP:
        emotion_code = basename[:2]
        utterance_id = basename[2:]
    # Then single-letter codes
    elif len(basename) > 0 and basename[0] in EMOTION_MAP:
        emotion_code = basename[0]
        utterance_id = basename[1:]
    else:
        logger.warning(f"Cannot parse emotion from filename: {filename}")
        return None
    
    emotion = EMOTION_MAP[emotion_code]
    
    return {
        'emotion': emotion,
        'speaker': speaker,  # May be None if no prefix
        'utterance_id': utterance_id,
        'emotion_code': emotion_code
    }


def detect_structure(data_folder):
    """
    Detect SAVEE directory structure
    
    Returns:
        str: 'folders' if speaker folders exist, 'flat' if flat with prefix, 'unknown' otherwise
    """
    
    # Check for speaker folders
    has_folders = False
    for speaker in SPEAKERS:
        speaker_dir = os.path.join(data_folder, speaker)
        speaker_dir_lower = os.path.join(data_folder, speaker.lower())
        if os.path.exists(speaker_dir) or os.path.exists(speaker_dir_lower):
            has_folders = True
            break
    
    if has_folders:
        return 'folders'
    
    # Check for flat structure with speaker prefix
    wav_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.wav')]
    if wav_files:
        # Check if files have speaker prefix
        for f in wav_files[:5]:  # Check first 5 files
            if '_' in f:
                prefix = f.split('_')[0].upper()
                if prefix in SPEAKERS:
                    return 'flat'
    
    return 'unknown'


def prepare_savee(
    data_folder,
    save_json,
    seed=12
):
    """
    Prepares the json files for the SAVEE dataset.
    
    Arguments
    ---------
    data_folder : str
        Path to the folder where the SAVEE dataset is stored.
        Handles both folder structure and flat structure.
    save_json : str
        Path where the data specification file will be saved.
    seed: int
        Seed for reproducibility
    """
    
    # Check if already done
    if skip(save_json):
        logger.info("Preparation completed in previous run, skipping.")
        return
    
    logger.info("Starting SAVEE dataset preparation...")
    
    if not os.path.exists(data_folder):
        raise ValueError(f"SAVEE folder not found: {data_folder}")
    
    logger.info(f"SAVEE folder: {data_folder}")
    
    # Create JSON
    data_json = create_savee_json(data_folder, save_json)
    
    logger.info(f"SAVEE preparation complete! Total samples: {len(data_json)}")
    
    return data_json


def create_savee_json(data_folder, save_json):
    """
    Create JSON for SAVEE dataset
    Automatically detects and handles both directory structures
    """
    
    data_json = {}
    total_files = 0
    skipped_files = 0
    
    print(f"\n{'='*70}")
    print("SAVEE DATASET PREPARATION")
    print(f"{'='*70}")
    print(f"Scanning: {data_folder}")
    
    # Detect structure
    structure = detect_structure(data_folder)
    print(f"Detected structure: {structure}")
    print()
    
    # Show emotion mapping
    print("Emotion codes:")
    for code, emotion in sorted(EMOTION_MAP.items()):
        print(f"  {code:4s} → {emotion}")
    print()
    
    emotion_counts = Counter()
    speaker_counts = Counter()
    
    if structure == 'folders':
        # Process speaker folders
        print("Processing speaker folders...")
        
        for speaker in SPEAKERS:
            speaker_dir = os.path.join(data_folder, speaker)
            speaker_dir_lower = os.path.join(data_folder, speaker.lower())
            
            # Try both cases
            if os.path.exists(speaker_dir):
                actual_dir = speaker_dir
            elif os.path.exists(speaker_dir_lower):
                actual_dir = speaker_dir_lower
            else:
                continue
            
            print(f"  Processing {speaker}...")
            
            wav_files = [f for f in os.listdir(actual_dir) if f.lower().endswith('.wav')]
            
            for filename in wav_files:
                wav_path = os.path.join(actual_dir, filename)
                
                # Parse filename
                metadata = parse_savee_filename(filename)
                
                if metadata is None:
                    skipped_files += 1
                    continue
                
                # Override speaker from folder
                metadata['speaker'] = speaker
                
                # Create entry ID
                file_id = f"SAVEE_{speaker}_{filename.replace('.wav', '').replace('.WAV', '')}"
                
                # Create entry
                data_json[file_id] = {
                    "wav": os.path.abspath(wav_path),
                    "emotion": [{
                        "emo": metadata['emotion'],
                        "start": 0,
                        "end": 3.0
                    }],
                    "speaker": speaker,
                    "utterance_id": metadata['utterance_id'],
                    "emotion_code": metadata['emotion_code'],
                    "dataset": "SAVEE",
                    "language": "British English"
                }
                
                total_files += 1
                emotion_counts[metadata['emotion']] += 1
                speaker_counts[speaker] += 1
            
            print(f"    ✓ {speaker_counts[speaker]} files")
    
    elif structure == 'flat':
        # Process flat structure with speaker prefix
        print("Processing flat structure with speaker prefixes...")
        
        wav_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.wav')]
        
        for filename in wav_files:
            wav_path = os.path.join(data_folder, filename)
            
            # Parse filename
            metadata = parse_savee_filename(filename)
            
            if metadata is None or metadata['speaker'] is None:
                skipped_files += 1
                continue
            
            speaker = metadata['speaker']
            
            # Create entry ID
            file_id = f"SAVEE_{filename.replace('.wav', '').replace('.WAV', '')}"
            
            # Create entry
            data_json[file_id] = {
                "wav": os.path.abspath(wav_path),
                "emotion": [{
                    "emo": metadata['emotion'],
                    "start": 0,
                    "end": 3.0
                }],
                "speaker": speaker,
                "utterance_id": metadata['utterance_id'],
                "emotion_code": metadata['emotion_code'],
                "dataset": "SAVEE",
                "language": "British English"
            }
            
            total_files += 1
            emotion_counts[metadata['emotion']] += 1
            speaker_counts[speaker] += 1
        
        print(f"  ✓ Processed {total_files} files")
    
    else:
        print(f"\nERROR: Cannot detect SAVEE structure!")
        print(f"\nExpected structures:")
        print(f"\nOption 1 (Speaker folders):")
        print(f"  {data_folder}/")
        print(f"    ├── DC/")
        print(f"    │   ├── a01.wav")
        print(f"    │   └── ...")
        print(f"    ├── JE/")
        print(f"    ├── JK/")
        print(f"    └── KL/")
        print(f"\nOption 2 (Flat with prefix):")
        print(f"  {data_folder}/")
        print(f"    ├── DC_a01.wav")
        print(f"    ├── DC_d01.wav")
        print(f"    ├── JE_a01.wav")
        print(f"    └── ...")
        print()
        print("Download from:")
        print("  http://kahlan.eps.surrey.ac.uk/savee/")
        print("  https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee")
        return {}
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"✓ Processed {total_files} files")
    
    if skipped_files > 0:
        print(f"⚠️  Skipped {skipped_files} files (could not parse)")
    
    if total_files == 0:
        print("\nNo files processed!")
        return {}
    
    # Show statistics
    print(f"\nEmotion distribution:")
    for emo in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
        count = emotion_counts.get(emo, 0)
        percentage = (count / total_files * 100) if total_files > 0 else 0
        icon = "✓" if count > 0 else "✗"
        print(f"  {icon} {emo:12s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nSpeaker distribution:")
    for speaker in SPEAKERS:
        count = speaker_counts.get(speaker, 0)
        percentage = (count / total_files * 100) if total_files > 0 else 0
        icon = "✓" if count > 0 else "✗"
        print(f"  {icon} {speaker:4s}: {count:4d} ({percentage:5.1f}%)")
    
    # Save JSON
    print(f"\n{'='*70}")
    print("SAVING JSON")
    print(f"{'='*70}")
    print(f"Output: {save_json}")
    
    os.makedirs(os.path.dirname(save_json) if os.path.dirname(save_json) else '.', exist_ok=True)
    
    with open(save_json, 'w') as f:
        json.dump(data_json, f, indent=2)
    
    # Verify
    if os.path.exists(save_json):
        file_size = os.path.getsize(save_json)
        print(f"\n✓ JSON file created successfully!")
        print(f"  Location: {save_json}")
        print(f"  Size: {file_size / 1024:.1f} KB")
        print(f"  Samples: {len(data_json)}")
        
        # Show sample entries
        print(f"\nSample entries:")
        for i, (key, data) in enumerate(list(data_json.items())[:5]):
            print(f"  {i+1}. {key}")
            print(f"     Emotion: {data['emotion'][0]['emo']}")
            print(f"     Speaker: {data['speaker']}")
            print(f"     File: {os.path.basename(data['wav'])}")
    else:
        print(f"\nERROR: Failed to create JSON file!")
        return {}
    
    return data_json


def skip(save_json):
    """ Check if preparation already done."""
    return os.path.isfile(save_json)


# ====================================================
# Main entry
# ====================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare SAVEE dataset (handles both folder and flat structures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SAVEE Dataset Information:
  - 480 utterances from 4 male speakers (DC, JE, JK, KL)
  - 7 emotions: anger, disgust, fear, happiness, neutral, sadness, surprise
  - 120 utterances per speaker (15-20 per emotion)
  - British English
  - High-quality studio recordings
  
Directory Structure Options:
  
  Option 1 (Speaker folders):
    datasets/SAVEE/
      ├── DC/
      │   ├── a01.wav
      │   ├── d01.wav
      │   ├── n01.wav
      │   └── ...
      ├── JE/
      ├── JK/
      └── KL/
  
  Option 2 (Flat with speaker prefix):
    datasets/SAVEE/
      ├── DC_a01.wav
      ├── DC_d01.wav
      ├── DC_n01.wav
      ├── JE_a01.wav
      └── ...

This script automatically detects which structure you have!

File naming:
  - With folders: a01.wav, d01.wav, n01.wav
  - Flat: DC_a01.wav, JE_d01.wav, KL_n01.wav

Emotion codes:
  a = anger, d = disgust, f = fear, h = happiness,
  n = neutral, sa = sadness, su = surprise

Download from:
  Official: http://kahlan.eps.surrey.ac.uk/savee/
  Kaggle: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee

Examples:
  python prepare_SAVEE.py
  python prepare_SAVEE.py --data_folder datasets/SAVEE
  python prepare_SAVEE.py --output datasets/SAVEE/SAVEE.json
        """
    )
    
    parser.add_argument(
        "--data_folder",
        type=str,
        default="datasets/SAVEE",
        help="Path to SAVEE dataset folder."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path JSON file (default: data_folder/SAVEE.json)."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = os.path.join(args.data_folder, "SAVEE.json")
    
    # Prepare dataset
    result = prepare_savee(
        data_folder=args.data_folder,
        save_json=args.output,
        seed=args.seed
    )
    
    if result:
        print("\n" + "="*70)
        print("SUCCESS! ✓")
        print("="*70)
        print(f"\nSAVEE.json is ready at: {args.output}")
    else:
        print("\n" + "="*70)
        print("FAILED ✗")
        print("="*70)
        print("Please check:")
        print("  1. Dataset downloaded and extracted")
        print("  2. Either speaker folders (DC/, JE/, etc.) exist")
        print("  3. OR flat files with prefix (DC_a01.wav, etc.) exist")
        print("  4. .wav files exist")
        print("="*70)
