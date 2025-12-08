"""
Data Preparation
Loads from ALL dataset JSON files and creates balanced train/valid/test splits
Includes the 7 emotions ["happy", "sad", "angry", "upset", "disgust", "fear", "neutral"]
"""

import json
import os
from collections import Counter, defaultdict
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

# 7-EMOTION SYSTEM
SELECTED_EMOTIONS = ["happy", "sad", "angry", "upset", "disgust", "fear", "neutral"]

# All dataset JSON files
DATASET_JSON_FILES = [
    "datasets/IEMOCAP/IEMOCAP.json",
    "datasets/RAVDESS/RAVDESS.json",
    "datasets/ESD/ESD.json",
    "datasets/EmoV-DB/EMOV-DB.json",
    "datasets/JL_corpus/JL_CORPUS.json",
    "datasets/TESS/TESS.json",
    "datasets/CREMA-D/CREMA-D.json",
    "datasets/MELD/MELD.json",
    "datasets/SAVEE/SAVEE.json",
]

# Output directory
OUTPUT_DIR = "data/processed_emotions_7class"

# Split ratios
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

print("="*70)
print(f"PREPARING 7-EMOTION DATASET (WITH NEUTRAL)")
print("="*70)
print(f"Selected emotions: {', '.join(SELECTED_EMOTIONS)}")
print(f"Output directory: {OUTPUT_DIR}")
print()


# ============================================================================
# Emotion Mapping (for consistency)
# ============================================================================

EMOTION_MAPPING = {
    # Core emotions
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "anger": "angry",
    
    # Additional emotions
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "neutral",
    "upset": "upset",
    
    # Aliases
    "happiness": "happy",
    "sadness": "sad",
    "frustrated": "upset",
    "frustration": "upset",
    "excited": "happy",
    "exc": "happy",
}


def get_emotion_from_entry(entry):
    """
    Extract emotion from dataset entry (handles different formats)
    """
    # Format 1: Direct emotion key
    if "emotion" in entry and isinstance(entry["emotion"], str):
        emotion = entry["emotion"]
        return EMOTION_MAPPING.get(emotion.lower(), emotion.lower())
    
    # Format 2: Emotion as a list of dicts
    if "emotion" in entry and isinstance(entry["emotion"], list):
        if len(entry["emotion"]) > 0:
            emo_dict = entry["emotion"][0]
            if isinstance(emo_dict, dict) and "emo" in emo_dict:
                emotion = emo_dict["emo"]
                return EMOTION_MAPPING.get(emotion.lower(), emotion.lower())
    
    return None


# ============================================================================
# Load All Datasets
# ============================================================================

def load_all_datasets(json_files, selected_emotions):
    """
    Load all dataset JSON files and filter to selected emotions
    """
    
    all_data = {}
    dataset_stats = {}
    
    print("Loading datasets...")
    print("-" * 70)
    
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"⚠️  Not found: {json_file}")
            continue
        
        dataset_name = json_file.split('/')[1] if '/' in json_file else json_file
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Filter and count
            filtered = 0
            emotion_counts = Counter()
            
            for key, entry in data.items():
                emotion = get_emotion_from_entry(entry)
                
                if emotion in selected_emotions:
                    # Create a unique key with dataset prefix
                    unique_key = f"{dataset_name}_{key}"
                    
                    # Standardize entry format
                    all_data[unique_key] = {
                        "wav": entry["wav"],
                        "emotion": emotion,
                        "dataset": dataset_name,
                        "speaker": entry.get("speaker", "unknown"),
                        "original_key": key
                    }
                    
                    filtered += 1
                    emotion_counts[emotion] += 1
            
            if filtered > 0:
                dataset_stats[dataset_name] = {
                    "total": len(data),
                    "filtered": filtered,
                    "emotions": dict(emotion_counts)
                }
                
                print(f"✓ {dataset_name:20s}: {filtered:5d}/{len(data):5d} samples")
                for emo, count in emotion_counts.items():
                    print(f"    {emo:12s}: {count:4d}")
            else:
                print(f"○ {dataset_name:20s}: No samples for selected emotions")
        
        except Exception as e:
            print(f"✗ {dataset_name:20s}: Error - {e}")
    
    print("-" * 70)
    print(f"Total samples loaded: {len(all_data)}")
    
    return all_data, dataset_stats


# ============================================================================
# Create Balanced Train/Valid/Test Splits
# ============================================================================

def create_balanced_splits(
    all_data,
    selected_emotions,
    train_ratio=0.7,
    valid_ratio=0.15,
    random_seed=42
):
    """
    Create balanced train/valid/test splits
    Stratified by emotion to maintain distribution
    """
    
    random.seed(random_seed)
    
    # Group by emotion
    emotion_samples = defaultdict(list)
    
    for key, entry in all_data.items():
        emotion = entry["emotion"]
        emotion_samples[emotion].append((key, entry))
    
    # Show emotion distribution
    print("\n" + "="*70)
    print("EMOTION DISTRIBUTION")
    print("="*70)
    
    total = len(all_data)
    for emotion in selected_emotions:
        count = len(emotion_samples[emotion])
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {emotion:12s}: {count:5d} samples ({percentage:5.1f}%)")
    
    # Create splits
    train_data = {}
    valid_data = {}
    test_data = {}
    
    print("\n" + "="*70)
    print("CREATING STRATIFIED SPLITS")
    print("="*70)
    print(f"Train: {train_ratio*100:.0f}%, Valid: {valid_ratio*100:.0f}%, Test: {(1-train_ratio-valid_ratio)*100:.0f}%")
    print()
    
    for emotion in selected_emotions:
        samples = emotion_samples[emotion]
        random.shuffle(samples)
        
        n = len(samples)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        
        train_samples = samples[:n_train]
        valid_samples = samples[n_train:n_train + n_valid]
        test_samples = samples[n_train + n_valid:]
        
        print(f"  {emotion:12s}: train={len(train_samples):5d}, valid={len(valid_samples):5d}, test={len(test_samples):5d}")
        
        for key, entry in train_samples:
            train_data[key] = entry
        for key, entry in valid_samples:
            valid_data[key] = entry
        for key, entry in test_samples:
            test_data[key] = entry
    
    return train_data, valid_data, test_data


# ============================================================================
# Save Datasets
# ============================================================================

def save_datasets(train_data, valid_data, test_data, output_dir, selected_emotions, dataset_stats):
    """
    Save train/valid/test JSON files and statistics
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING DATASETS")
    print("="*70)
    
    # Save JSON files
    train_path = os.path.join(output_dir, "train.json")
    valid_path = os.path.join(output_dir, "valid.json")
    test_path = os.path.join(output_dir, "test.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"✓ train.json: {len(train_data):5d} samples")
    
    with open(valid_path, 'w') as f:
        json.dump(valid_data, f, indent=2)
    print(f"✓ valid.json: {len(valid_data):5d} samples")
    
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"✓ test.json:  {len(test_data):5d} samples")
    
    # Save emotion labels
    labels_path = os.path.join(output_dir, "emotion_labels.txt")
    with open(labels_path, 'w') as f:
        for emotion in selected_emotions:
            f.write(f"{emotion}\n")
    print(f"✓ emotion_labels.txt")
    
    # Count by dataset in the train set
    train_dataset_dist = Counter()
    for entry in train_data.values():
        train_dataset_dist[entry["dataset"]] += 1
    
    # Save detailed statistics
    stats = {
        "emotions": selected_emotions,
        "num_emotions": len(selected_emotions),
        "total_samples": len(train_data) + len(valid_data) + len(test_data),
        "train_samples": len(train_data),
        "valid_samples": len(valid_data),
        "test_samples": len(test_data),
        "source_datasets": dataset_stats,
        "train_dataset_distribution": dict(train_dataset_dist)
    }
    
    stats_path = os.path.join(output_dir, "dataset_info.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ dataset_info.json")
    
    # Show train set composition by dataset
    print("\n" + "="*70)
    print("TRAINING SET COMPOSITION BY DATASET")
    print("="*70)
    
    for dataset, count in sorted(train_dataset_dist.items(), key=lambda x: -x[1]):
        percentage = (count / len(train_data) * 100)
        print(f"  {dataset:20s}: {count:5d} ({percentage:5.1f}%)")
    
    print("\n✓ All files saved to:", output_dir)


# ============================================================================
# Calculate Class Weights
# ============================================================================

def calculate_class_weights(train_data, selected_emotions):
    """
    Calculate balanced class weights for training
    """
    
    emotion_counts = Counter()
    for entry in train_data.values():
        emotion_counts[entry["emotion"]] += 1
    
    total = len(train_data)
    n_classes = len(selected_emotions)
    
    print("\n" + "="*70)
    print("CLASS WEIGHTS FOR BALANCED TRAINING")
    print("="*70)
    
    print(f"\nTraining set distribution:")
    weights = []
    for emotion in selected_emotions:
        count = emotion_counts.get(emotion, 1)
        weight = total / (n_classes * count)
        weights.append(weight)
        percentage = (count / total * 100)
        print(f"  {emotion:12s}: {count:5d} samples ({percentage:5.1f}%) → weight: {weight:.4f}")
    
    # Normalize weights
    import torch
    import numpy as np
    weights = torch.FloatTensor(weights)
    weights = weights * n_classes / weights.sum()
    
    print(f"\nNormalized class weights:")
    for emotion, weight in zip(selected_emotions, weights):
        print(f"  {emotion:12s}: {weight:.4f}")
    
    # Save weights
    weights_path = os.path.join(OUTPUT_DIR, "class_weights.pt")
    torch.save(weights, weights_path)
    print(f"✓ class_weights.pt")
    
    weights_txt = os.path.join(OUTPUT_DIR, "class_weights.txt")
    with open(weights_txt, 'w') as f:
        for emotion, weight in zip(selected_emotions, weights):
            f.write(f"{emotion}: {weight:.4f}\n")
    print(f"✓ class_weights.txt")
    
    # Also save as JSON
    weights_json = os.path.join(OUTPUT_DIR, "class_weights.json")
    weights_dict = {emotion: float(weight) for emotion, weight in zip(selected_emotions, weights)}
    with open(weights_json, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    print(f"✓ class_weights.json")
    
    return weights


# ============================================================================
# Main Execution
# ============================================================================

def main():
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "7-EMOTION DATA PREPARATION (WITH NEUTRAL)" + " "*17 + "║")
    print("║" + " "*15 + "Loading from ALL Datasets" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Step 1: Load all datasets
    all_data, dataset_stats = load_all_datasets(
        DATASET_JSON_FILES,
        SELECTED_EMOTIONS
    )
    
    if len(all_data) == 0:
        print("\n ERROR: No samples loaded!")
        print("Check:")
        print("  1. JSON files exist")
        print("  2. Selected emotions match dataset emotions")
        print("  3. File paths are correct")
        return
    
    # Step 2: Create balanced splits
    train_data, valid_data, test_data = create_balanced_splits(
        all_data,
        SELECTED_EMOTIONS,
        TRAIN_RATIO,
        VALID_RATIO,
        RANDOM_SEED
    )
    
    # Step 3: Save datasets
    save_datasets(
        train_data,
        valid_data,
        test_data,
        OUTPUT_DIR,
        SELECTED_EMOTIONS,
        dataset_stats
    )
    
    # Step 4: Calculate class weights
    try:
        import torch
        weights = calculate_class_weights(train_data, SELECTED_EMOTIONS)
    except ImportError:
        print("\n⚠️  PyTorch not found, skipping class weights calculation")
        print("   Install with: pip install torch")
    
    # Final summary
    print("\n" + "="*70)
    print("✓ PREPARATION COMPLETE!")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Emotions: {', '.join(SELECTED_EMOTIONS)}")
    print(f"Total samples: {len(all_data)}")
    print(f"  - Train: {len(train_data)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  - Valid: {len(valid_data)} ({VALID_RATIO*100:.0f}%)")
    print(f"  - Test:  {len(test_data)} ({(1-TRAIN_RATIO-VALID_RATIO)*100:.0f}%)")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Update your training script:")
    print(f'   CONFIG["data_folder"] = "{OUTPUT_DIR}"')
    print(f'   CONFIG["output_neurons"] = {len(SELECTED_EMOTIONS)}')
    print()
    print("2. Update emotion labels:")
    print(f'   EMOTION_LABELS = {SELECTED_EMOTIONS}')
    print()
    print("3. Use class weights:")
    print(f'   weights = torch.load("{OUTPUT_DIR}/class_weights.pt")')
    print()
    print("4. Prepare frame-level data:")
    print("   python prepare_frame_level_data.py")
    print("   (Update data_folder in script to use '{OUTPUT_DIR}')")
    print()
    print("5. Train your model:")
    print("   python train_frame_level_7emotions.py")
    print("="*70)


if __name__ == "__main__":
    main()
