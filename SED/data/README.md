# Data Preparation Pipeline for Speech Emotion Recognition

---

## Overview

This pipeline converts multiple emotion speech datasets into a unified format for training frame-level emotion diarization systems. The process consists of two main steps:

1. **Dataset Preparation** - Combine multiple datasets, create balanced splits
2. **Frame-Level Conversion** - Convert utterance labels to frame-level annotations (20ms resolution)

**Output**: Training-ready data with ~58,000 utterances → ~12 million frames for temporal emotion modeling

---

## Files

| File | Purpose | Runtime |
|------|---------|---------|
| `data_preparation_7emotions.py` | Load & combine 9 datasets, create train/valid/test splits | ~5 minutes |
| `prepare_frame_lable.py` | Convert utterances to frame-level labels (20ms) | ~15 minutes |

---

## Quick Start

### Prerequisites

```bash
# Required packages
pip install torch torchaudio tqdm

# Verify datasets are downloaded
ls datasets/IEMOCAP/
ls datasets/RAVDESS/
# ... (see DATASETS_COMPLETE_GUIDE.md for download links)
```

### Step 1: Prepare Datasets

```bash
python data_preparation_7emotions.py
```

**What it does:**
- Loads 9 English emotion datasets (IEMOCAP, RAVDESS, CREMA-D, etc.)
- Maps emotions to 7 classes: `angry, disgust, fear, happy, neutral, sad, upset`
- Creates stratified 70/15/15 train/valid/test splits
- Calculates class weights for balanced training
- Saves unified JSON files

**Output files:**
```
data/processed_emotions_7class/
├── train.json              # 41,181 utterances (70%)
├── valid.json              # 8,822 utterances (15%)
├── test.json               # 8,831 utterances (15%)
├── emotion_labels.txt      # List of 7 emotions
├── class_weights.pt        # PyTorch tensor of weights
├── class_weights.txt       # Human-readable weights
├── class_weights.json      # JSON format weights
└── dataset_info.json       # Dataset statistics
```

### Step 2: Create Frame-Level Labels

```bash
python prepare_frame_lable.py
```

**What it does:**
- Reads utterance-level labels from Step 1
- Converts to frame-level (20ms per frame, 50 FPS)
- Propagates the emotion label to all frames in the utterance
- Generates ~12 million frame-level annotations

**Output files:**
```
data/processed_emotions_7class/
├── train_frames.json       # 32,780 samples, 8.7M frames
├── valid_frames.json       # 6,977 samples, 1.9M frames
├── test_frames.json        # 7,007 samples, 1.9M frames
├── train_frames_errors.txt # Error log (failed conversions)
├── valid_frames_errors.txt
└── test_frames_errors.txt
```

---

## Detailed Usage

### Step 1: Dataset Preparation

#### Configuration

Edit `data_preparation_7emotions.py` to customize:

```python
# Select which datasets to include
DATASET_JSON_FILES = [
    "datasets/IEMOCAP/IEMOCAP.json",
    "datasets/RAVDESS/RAVDESS.json",
    "datasets/CREMA-D/CREMA-D.json",
    # ... add or remove datasets
]

# Choose emotion set
SELECTED_EMOTIONS = ["happy", "sad", "angry", "upset", "disgust", "fear", "neutral"]

# Output directory
OUTPUT_DIR = "data/processed_emotions_7class"

# Split ratios
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15
```

#### Run Preparation

```bash
python data_preparation_7emotions.py
```

#### Expected Output

```
======================================================================
PREPARING 7-EMOTION DATASET (WITH NEUTRAL)
======================================================================
Selected emotions: happy, sad, angry, upset, disgust, fear, neutral
Output directory: data/processed_emotions_7class

Loading datasets...
----------------------------------------------------------------------
✓ IEMOCAP             : 11032/11032 samples
    angry       : 2147
    happy       : 3165
    sad         : 2064
    upset       : 3573
    fear        :   79
    disgust     :    4
    neutral     : XXXX
✓ RAVDESS             :   960/  960 samples
    disgust     :  192
    angry       :  192
    fear        :  192
    happy       :  192
    sad         :  192
    neutral     :  192
...
----------------------------------------------------------------------
Total samples loaded: 58834

======================================================================
EMOTION DISTRIBUTION
======================================================================
  happy       : 16133 samples ( 27.4%)
  sad         : 10558 samples ( 17.9%)
  angry       : 15115 samples ( 25.7%)
  upset       :  3573 samples (  6.1%)
  disgust     :  3054 samples (  5.2%)
  fear        :  2360 samples (  4.0%)
  neutral     :  8041 samples ( 13.7%)

======================================================================
CREATING STRATIFIED SPLITS
======================================================================
Train: 70%, Valid: 15%, Test: 15%

  happy       : train=11293, valid= 2419, test= 2421
  sad         : train= 7390, valid= 1583, test= 1585
  angry       : train=10580, valid= 2267, test= 2268
  upset       : train= 2501, valid=  535, test=  537
  disgust     : train= 2137, valid=  458, test=  459
  fear        : train= 1652, valid=  354, test=  354
  neutral     : train= 5628, valid= 1206, test= 1207

======================================================================
SAVING DATASETS
======================================================================
✓ train.json: 41181 samples
✓ valid.json:  8822 samples
✓ test.json:   8831 samples
✓ emotion_labels.txt
✓ dataset_info.json

======================================================================
TRAINING SET COMPOSITION BY DATASET
======================================================================
  MELD                :  8401 ( 20.4%)
  IEMOCAP             :  7723 ( 18.8%)
  JL_corpus           :  7470 ( 18.1%)
  ESD                 :  7322 ( 17.8%)
  CREMA-D             :  5253 ( 12.8%)
  EmoV-DB             :  2346 (  5.7%)
  TESS                :  1674 (  4.1%)
  RAVDESS             :   681 (  1.7%)
  SAVEE               :   311 (  0.8%)

======================================================================
CLASS WEIGHTS FOR BALANCED TRAINING
======================================================================

Training set distribution:
  happy       : 11293 samples ( 27.4%) → weight: 0.5209
  sad         :  7390 samples ( 17.9%) → weight: 0.7961
  angry       : 10580 samples ( 25.7%) → weight: 0.5560
  upset       :  2501 samples (  6.1%) → weight: 2.3523
  disgust     :  2137 samples (  5.2%) → weight: 2.7529
  fear        :  1652 samples (  4.0%) → weight: 3.5611
  neutral     :  5628 samples ( 13.7%) → weight: 1.0453

Normalized class weights:
  happy       : 0.3148
  sad         : 0.4810
  angry       : 0.3360
  upset       : 1.4213
  disgust     : 1.6634
  fear        : 2.1518
  neutral     : 0.6316

✓ class_weights.pt
✓ class_weights.txt
✓ class_weights.json

✓ All files saved to: data/processed_emotions_7class
```

---

### Step 2: Frame-Level Conversion

#### Configuration

Edit `prepare_frame_lable.py` to customize:

```python
# Input data folder (from Step 1)
data_folder = "data/processed_emotions_7class"

# Frame settings (match WavLM)
sample_rate = 16000  # Audio sample rate
frame_shift = 0.02   # 20ms per frame = 50 FPS
```

#### Run Conversion

```bash
python prepare_frame_lable.py
```

#### Expected Output

```
======================================================================
Frame-Level Data Preparation for Emotion Diarization
======================================================================
Data folder: data/processed_emotions_7class
  ✓ Found train.json
  ✓ Found valid.json
  ✓ Found test.json

Processing: data/processed_emotions_7class/train.json
Converting to frame-level: 100%|████████████| 41181/41181 [00:11<00:00, 3596.67it/s]
✓ Created data/processed_emotions_7class/train_frames.json
  Total samples: 32780
  Errors: 8401
  Error log: data/processed_emotions_7class/train_frames_errors.txt
  Total frames: 8,740,095
  Avg frames per sample: 266.6
  Avg duration: 5.33s

Processing: data/processed_emotions_7class/valid.json
Converting to frame-level: 100%|██████████████| 8822/8822 [00:02<00:00, 3714.52it/s]
✓ Created data/processed_emotions_7class/valid_frames.json
  Total samples: 6977
  Errors: 1845
  Error log: data/processed_emotions_7class/valid_frames_errors.txt
  Total frames: 1,860,206
  Avg frames per sample: 266.6
  Avg duration: 5.33s

Processing: data/processed_emotions_7class/test.json
Converting to frame-level: 100%|██████████████| 8831/8831 [00:01<00:00, 5154.03it/s]
✓ Created data/processed_emotions_7class/test_frames.json
  Total samples: 7007
  Errors: 1824
  Error log: data/processed_emotions_7class/test_frames_errors.txt
  Total frames: 1,852,126
  Avg frames per sample: 264.3
  Avg duration: 5.29s

======================================================================
Training Set Frame Distribution
======================================================================

Frame-Level Distribution:
==================================================
  angry     : 2,835,076 frames (32.44%)
  happy     : 2,807,923 frames (32.13%)
  sad       : 1,614,534 frames (18.47%)
  upset     :  790,085 frames ( 9.04%)
  disgust   :  379,907 frames ( 4.35%)
  fear      :  171,185 frames ( 1.96%)
  neutral   :  141,385 frames ( 1.62%)
==================================================
  Total:       8,740,095 frames

---

## Output Format

### Utterance-Level JSON (Step 1)

```json
{
  "IEMOCAP_Ses01F_impro01_F000": {
    "wav": "datasets/IEMOCAP/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav",
    "emotion": "happy",
    "dataset": "IEMOCAP",
    "speaker": "Ses01F",
    "original_key": "Ses01F_impro01_F000"
  },
  ...
}
```

### Frame-Level JSON (Step 2)

```json
{
  "IEMOCAP_Ses01F_impro01_F000": {
    "wav": "datasets/IEMOCAP/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav",
    "emotion": "happy",
    "frame_labels": ["happy", "happy", "happy", ..., "happy"],
    "duration": 3.45,
    "num_frames": 172
  },
  ...
}
```

**Key fields:**
- `wav`: Path to audio file
- `emotion`: Utterance-level label
- `frame_labels`: Array of emotions, one per 20ms frame
- `duration`: Audio duration in seconds
- `num_frames`: Number of 20ms frames

---

## 🔍 Understanding Frame-Level Conversion

### Why 20ms Frames?

1. **WavLM Output**: WavLM-base-plus produces 1 feature vector per 20ms of audio
2. **Temporal Resolution**: Fine enough to capture emotion transitions
3. **Animation Compatibility**: Matches video frame rates (24-60 FPS)
4. **Real-time Capable**: Enables streaming emotion detection

### Frame Calculation

```
Duration: 3.45 seconds
Frame shift: 0.02 seconds (20ms)
Number of frames: ⌊3.45 / 0.02⌋ = 172 frames

Frame 0: 0.00s - 0.02s → "happy"
Frame 1: 0.02s - 0.04s → "happy"
Frame 2: 0.04s - 0.06s → "happy"
...
Frame 171: 3.42s - 3.44s → "happy"
```

### Utterance vs Frame Distribution

**Utterance-level** (Step 1 output):
- Happy: 27.4% of utterances
- Angry: 25.7% of utterances

**Frame-level** (Step 2 output):
- Angry: 32.4% of frames (longer utterances)
- Happy: 32.1% of frames (longer utterances)
- Neutral: 1.6% of frames (shorter utterances)

**Duration bias**: Emotions with longer average utterances get more frames!

---

## Statistics & Metrics

### Dataset Scale

| Metric | Value |
|--------|-------|
| **Utterances** | 58,834 total |
| **Train** | 41,181 (70%) |
| **Valid** | 8,822 (15%) |
| **Test** | 8,831 (15%) |
| **Frames** | 12,452,427 total |
| **Duration** | ~69.2 hours |
| **Avg Utterance** | 5.32 seconds |
| **Emotions** | 7 classes |

### Success Rate

| Split | Input | Valid | Errors | Success Rate |
|-------|-------|-------|--------|--------------|
| Train | 41,181 | 32,780 | 8,401 | 79.6% |
| Valid | 8,822 | 6,977 | 1,845 | 79.1% |
| Test | 8,831 | 7,007 | 1,824 | 79.4% |

**Errors** (~20%) primarily from:
- MELD conversational audio (background noise, crosstalk)
- Corrupted/truncated audio files
- Extremely short utterances (<100ms)
- Audio format incompatibilities

---

### Dataset Preparation Scripts

Each dataset has its own preparation script:
```
prepare_IEMOCAP.py
prepare_RAVDESS.py
prepare_CREMA_D.py
prepare_TESS.py
prepare_SAVEE.py
prepare_ESD.py
prepare_EMOVDB.py
prepare_JL_CORPUS.py
prepare_MELD.py
```

Run these **before** Step 1 to create individual `{DATASET}.json` files.

---

## Expected Results

After completing both steps, you should have:

```
data/processed_emotions_7class/
├── train.json              ✅ 41,181 utterances
├── valid.json              ✅ 8,822 utterances
├── test.json               ✅ 8,831 utterances
├── train_frames.json       ✅ 8.7M frames
├── valid_frames.json       ✅ 1.9M frames
├── test_frames.json        ✅ 1.9M frames
├── emotion_labels.txt      ✅ 7 emotions
├── class_weights.pt        ✅ Training weights
├── class_weights.json      ✅ JSON weights
└── dataset_info.json       ✅ Statistics

Total: 58,834 utterances → 12.45M frames → Ready for training!
```

---

## Citation

If you use this data preparation pipeline, please cite the original datasets:

- **IEMOCAP**: Busso et al. (2008)
- **RAVDESS**: Livingstone & Russo (2018)
- **CREMA-D**: Cao et al. (2014)
- **TESS**: Dupuis & Pichora-Fuller (2010)
- **SAVEE**: Haq & Jackson (2002)
- **ESD**: Zhou et al. (2021)
- **EmoV-DB**: Adigwe et al. (2018)
- **JL-Corpus**: James & Lech (2014)
- **MELD**: Poria et al. (2019)

See the datasts folder `README.md` for full citations.
