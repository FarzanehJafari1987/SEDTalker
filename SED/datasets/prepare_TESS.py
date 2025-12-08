#!/usr/bin/env python3
"""
Flexible TESS JSON Creator - Handles Different Directory Structures
"""

import os
import json

TESS_FOLDER = "datasets/TESS"

print("="*70)
print("FLEXIBLE TESS JSON CREATOR")
print("="*70)

if not os.path.exists(TESS_FOLDER):
    print(f"❌ TESS folder not found: {TESS_FOLDER}")
    exit(1)

print(f"\nTESS folder: {os.path.abspath(TESS_FOLDER)}")

# First, explore the structure
print("\nExploring TESS directory structure...")

def explore_directory(path, level=0, max_level=2):
    """Recursively explore directory"""
    items = []
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            indent = "  " * level
            
            if os.path.isdir(item_path):
                wav_count = sum(1 for f in os.listdir(item_path) if f.endswith('.wav'))
                print(f"{indent}📁 {item}/ ({wav_count} .wav files)")
                items.append((item, item_path, wav_count))
                
                if level < max_level:
                    explore_directory(item_path, level + 1, max_level)
            elif item.endswith('.wav'):
                if level == 0:
                    print(f"{indent}🎵 {item}")
    except PermissionError:
        pass
    
    return items

folders = explore_directory(TESS_FOLDER, 0, 2)

# Strategy 1: Check for emotion-named folders (OAF_angry, YAF_happy, etc.)
print("\n" + "="*70)
print("SEARCHING FOR EMOTION FOLDERS...")
print("="*70)

EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    #"ps": "surprise",
    "sad": "sad",
    "neutral": "neutral"
}

data_json = {}
total_files = 0

# Strategy 1: Look for OAF_emotion, YAF_emotion pattern
emotion_folders = []
for item in os.listdir(TESS_FOLDER):
    item_path = os.path.join(TESS_FOLDER, item)
    if not os.path.isdir(item_path):
        continue
    
    parts = item.split('_')
    if len(parts) >= 2 and parts[0] in ['OAF', 'YAF']:
        emotion_folders.append((item, item_path))

if emotion_folders:
    print(f"✓ Found {len(emotion_folders)} emotion folders (OAF_*/YAF_* pattern)")
    
    for folder_name, folder_path in emotion_folders:
        parts = folder_name.split('_')
        speaker = parts[0]
        emotion_code = parts[1]
        emotion = EMOTION_MAP.get(emotion_code, emotion_code)
        
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"  {folder_name:20s}: {len(wav_files):4d} files → {emotion}")
        
        for filename in wav_files:
            wav_path = os.path.join(folder_path, filename)
            file_id = filename.replace('.wav', '')
            
            data_json[file_id] = {
                "wav": os.path.abspath(wav_path),
                "emotion": [{
                    "emo": emotion,
                    "start": 0,
                    "end": 2.0
                }],
                "speaker": speaker,
                "dataset": "TESS"
            }
            total_files += 1

# Strategy 2: Look in subdirectories (TESS/combined/, TESS/audio/, etc.)
else:
    print("⚠️  Standard emotion folders not found")
    print("Searching in subdirectories...")
    
    # Check common subdirectory names
    subdirs_to_check = ['combined', 'audio', 'data', 'wav', 'files']
    
    for subdir in subdirs_to_check:
        subdir_path = os.path.join(TESS_FOLDER, subdir)
        
        if not os.path.exists(subdir_path):
            continue
        
        print(f"\nChecking: {subdir}/")
        
        # Look for emotion folders inside this subdirectory
        for item in os.listdir(subdir_path):
            item_path = os.path.join(subdir_path, item)
            
            if not os.path.isdir(item_path):
                continue
            
            parts = item.split('_')
            
            # Check if it matches OAF_emotion or YAF_emotion pattern
            if len(parts) >= 2 and parts[0] in ['OAF', 'YAF']:
                speaker = parts[0]
                emotion_code = parts[1]
                emotion = EMOTION_MAP.get(emotion_code, emotion_code)
                
                wav_files = [f for f in os.listdir(item_path) if f.endswith('.wav')]
                print(f"  ✓ {item:20s}: {len(wav_files):4d} files → {emotion}")
                
                for filename in wav_files:
                    wav_path = os.path.join(item_path, filename)
                    file_id = filename.replace('.wav', '')
                    
                    data_json[file_id] = {
                        "wav": os.path.abspath(wav_path),
                        "emotion": [{
                            "emo": emotion,
                            "start": 0,
                            "end": 2.0
                        }],
                        "speaker": speaker,
                        "dataset": "TESS"
                    }
                    total_files += 1

# Strategy 3: Parse from filenames directly if .wav files are in root or subdir
if total_files == 0:
    print("\n⚠️  No emotion folders found")
    print("Trying to parse from filenames...")
    
    # Look for .wav files recursively
    for root, dirs, files in os.walk(TESS_FOLDER):
        for filename in files:
            if not filename.endswith('.wav'):
                continue
            
            # TESS files are named like: OAF_back_angry.wav or YAF_dog_happy.wav
            # Format: SPEAKER_WORD_EMOTION.wav
            parts = filename.replace('.wav', '').split('_')
            
            if len(parts) >= 3:
                speaker = parts[0]  # OAF or YAF
                word = parts[1]     # back, dog, etc.
                emotion_code = parts[2]  # angry, happy, etc.
                
                if speaker in ['OAF', 'YAF'] and emotion_code in EMOTION_MAP:
                    emotion = EMOTION_MAP[emotion_code]
                    
                    wav_path = os.path.join(root, filename)
                    file_id = filename.replace('.wav', '')
                    
                    data_json[file_id] = {
                        "wav": os.path.abspath(wav_path),
                        "emotion": [{
                            "emo": emotion,
                            "start": 0,
                            "end": 2.0
                        }],
                        "speaker": speaker,
                        "dataset": "TESS"
                    }
                    total_files += 1

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Total files processed: {total_files}")

if total_files == 0:
    print("\n❌ No TESS audio files found!")
    print("\nExpected directory structure (one of these):")
    print("\nOption 1 (Standard):")
    print("  datasets/TESS/")
    print("    ├── OAF_angry/")
    print("    │   ├── OAF_back_angry.wav")
    print("    │   └── ...")
    print("    ├── OAF_happy/")
    print("    ├── YAF_angry/")
    print("    └── ...")
    print("\nOption 2 (With subdirectory):")
    print("  datasets/TESS/")
    print("    └── TESS/")
    print("        ├── OAF_angry/")
    print("        ├── OAF_happy/")
    print("        └── ...")
    print("\nOption 3 (Flat structure):")
    print("  datasets/TESS/")
    print("    ├── OAF_back_angry.wav")
    print("    ├── OAF_dog_happy.wav")
    print("    └── ...")
    print("\nPlease check your TESS download and extraction!")
    exit(1)

# Save JSON
output_file = os.path.join(TESS_FOLDER, "TESS.json")

print(f"\nSaving to: {output_file}")

with open(output_file, 'w') as f:
    json.dump(data_json, f, indent=2)

# Verify and show stats
if os.path.exists(output_file) and total_files > 0:
    file_size = os.path.getsize(output_file)
    print(f"✓ JSON file created!")
    print(f"  Location: {output_file}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    print(f"  Samples: {len(data_json)}")
    
    # Emotion distribution
    emotion_counts = {}
    speaker_counts = {}
    
    for item in data_json.values():
        emo = item['emotion'][0]['emo']
        speaker = item['speaker']
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    print(f"\nEmotion distribution:")
    for emo, count in sorted(emotion_counts.items()):
        percentage = (count / len(data_json) * 100)
        print(f"  {emo:12s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nSpeaker distribution:")
    for speaker, count in sorted(speaker_counts.items()):
        percentage = (count / len(data_json) * 100)
        print(f"  {speaker:12s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\n" + "="*70)
    print("SUCCESS! ✓")
    print("="*70)
    print(f"TESS.json is ready!")
    print("\nNext: Run prepare_enhanced_3emotions.py to combine all datasets")
else:
    print("\n❌ Failed to create useful JSON!")

print("="*70)