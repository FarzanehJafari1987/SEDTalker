# Speech-Driven Frame-Level Emotion Recognition

A deep learning system for frame-level emotion classification from speech, achieving **78.92% test accuracy** across 7 emotion classes using WavLM-based temporal modeling.

---

## Project Overview

This project implements a state-of-the-art frame-level emotion recognition system that classifies speech into seven distinct emotional categories at 20ms temporal resolution. The model processes speech data frame-by-frame (50 FPS), making it ideal for real-time applications and integration with 3D facial animation pipelines like JambaTalk.

### Emotion Classes

The system recognizes 7 emotions with hierarchical difficulty:

- **Happy** - High arousal, positive valence
- **Sad** - Low arousal, negative valence
- **Angry** - High arousal, negative valence
- **Disgust** - Moderate arousal, negative valence
- **Fear** - High arousal, negative valence
- **Upset** - Moderate arousal, negative valence
- **Neutral** - Low arousal, neutral valence (baseline state)

---

## Performance Summary (Test Set Evaluation)

### Overall Metrics
- **Test Accuracy**: **78.92%** (Exceeds 77.5% target by +1.4%)
- **Weighted F1-Score**: **78.85%**
- **Macro F1-Score**: **77.49%**
- **Test Frames Evaluated**: 1,849,663 frames (~10.3 hours of speech)
- **Temporal Jitter**: 0.0013 (excellent stability - ~1 switch per 20 seconds)
- **Segment Purity**: 45.7% (moderate coherence)

### Per-Emotion Performance (Test Set)

| Emotion | Precision | Recall | F1-Score | Support (Frames) | % of Test |
|---------|-----------|--------|----------|------------------|-----------|
| **Disgust** | **0.837** | **0.915** | **0.874** | 77,918 | 4.2% |
| **Neutral** | **0.766** | **0.879** | **0.819** | 24,052 | 1.3% |
| **Angry** | **0.797** | **0.824** | **0.810** | 594,488 | 32.1% |
| **Happy** | **0.822** | **0.773** | **0.797** | 595,723 | 32.2% |
| **Upset** | **0.694** | **0.828** | **0.755** | 169,368 | 9.2% |
| **Sad** | **0.766** | **0.736** | **0.750** | 351,115 | 19.0% |
| **Fear** | **0.824** | **0.496** | **0.620** | 36,999 | 2.0% |

### Performance Highlights

**Major Achievements**:
- **Disgust**: 87.4% F1 (exceptional, despite only 4.2% of test data)
- **Neutral**: 81.9% F1 (outstanding, 87.9% recall)
- **Angry**: 81.0% F1 (excellent, balanced precision/recall)
- **Happy**: 79.7% F1 (excellent, 82.2% precision)
- **6 out of 7 emotions**: F1 > 75%
- **Temporal stability**: Jitter = 0.0013 (outstanding - minimal flickering)

**Known Challenges**:
- **Fear**: 62.0% F1, only 49.6% recall (model misses half of fear frames)
  - Root cause: Severe data scarcity (2.0% of training frames)
  - Confusion: 25% of fear frames misclassified as sad
  - High precision (82.4%): When the model predicts fear, it's usually correct
- **Happy↔Angry confusion**: 14% of happy frames misclassified as angry (high arousal overlap)
- **Segment purity**: 45.7% (moderate - some brief spurious emotion switches)

**Advantages**: 
- **+5.3%** accuracy over WavLM-EmoDiarization (78.92% vs 73.6%)
- **2.5× finer** temporal resolution (20ms vs 50ms)
- **More emotions** than most baselines (7 vs 4-6)
- **Public datasets** (9 sources, fully reproducible)

---

## Architecture

### Model Design

**Backbone**: WavLM-Base-Plus (Microsoft, 94K hours pre-training)
- **Parameters**: 95M total, 12.4M trainable (encoder layers 0-5 frozen)
- **Input**: 16kHz raw audio waveforms
- **Features**: 768-dim hidden states per 20ms frame
- **Frame Rate**: 50 FPS (20ms temporal resolution)

**Architecture**:
```
Input Audio (16kHz)
       ↓
WavLM Feature Extractor (frozen)
       ↓
WavLM Transformer Encoder (layers 0-5 frozen, 6-11 trainable)
       ↓
Frame-Level Features [batch, time, 768]
       ↓
Shared Encoder: Linear(768→512→256) + LayerNorm + ReLU + Dropout(0.3)
       ↓
Emotion Head: Linear(256→128→7) + Softmax
       ↓
Frame-Level Predictions [batch, time, 7]
```

**Training Configuration**:
- **Loss**: Weighted Cross-Entropy (class weights: fear 2.15, happy 0.31)
- **Optimizer**: AdamW (lr=2e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (T_max=100)
- **Batch Size**: 16 (effective 64 with 4× gradient accumulation)
- **Mixed Precision**: BFloat16 (RTX 4090 optimized)
- **Regularization**: Dropout 0.3, early stopping at epoch 40
- **Training Time**: ~30 hours (RTX 4090)

---

## Installation

### System Requirements

- **GPU**: NVIDIA RTX 3090/4090 recommended (24GB VRAM)
- **CPU**: 16+ cores for data loading (8 workers)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB (datasets + processed files)

### Dependencies
```bash
# Core dependencies
pip install torch==2.0.1 torchaudio==2.0.2
pip install transformers==4.30.0
pip install speechbrain==0.5.13

# Data processing
pip install numpy pandas tqdm
pip install librosa soundfile

# Evaluation & visualization
pip install scikit-learn matplotlib seaborn
pip install tensorboard wandb
```

### Quick Setup
```bash
# Clone repository
git clone https://github.com/yourusername/sedtalker.git
cd sedtalker

# Install requirements
pip install -r requirements.txt

# Download datasets (see DATASETS_COMPLETE_GUIDE.md)
python download_datasets.py

# Prepare data
python data_preparation_7emotions.py
python prepare_frame_lable.py

# Train model
python SED/train_frame_level_7emotions.py
```

---

## Usage

### Training

**Training** (batch_size=4):
```bash
python SED/train_frame_level_7emotions.py \
    --data_folder data/processed_emotions_7class \
    --output_folder results/emotion_7class \
    --epochs 100 \
    --batch_size 4 \
    --lr 0.0001
```

### Evaluation

**Generate Test Predictions**:
```bash
python SED/evaluation/test_preditions.py \
    --checkpoint results/emotion_7class/save/CKPT+epoch_40/model.ckpt \
    --test-json data/processed_emotions_7class/test_frames.json \
    --output evaluation/test_predictions.json \
    --smoothing 5
```

**Run Comprehensive Evaluation**:
```bash
python evaluate_sed_comprehensive.py \
    --predictions evaluation/test_predictions.json \
    --ground_truth data/processed_emotions_7class/test_frames.json \
    --output_dir evaluation_results/
```

**Output**: Detailed metrics + confusion matrix + per-class visualizations + emotion timeline plots

### Inference - Frame-Level Diarization

**Single File**:
```bash
python inference_diarization_7emotions.py \
    --checkpoint results/emotion_7class/save/CKPT+epoch_40/model.ckpt \
    --audio test.wav \
    --smoothing 5
```

**Output**:
```
================================================================================
EMOTION TIMELINE: test.wav
================================================================================

   1.   0.000s -   2.340s ( 2.340s)  😊 happy     (conf: 0.92)
   2.   2.340s -   4.560s ( 2.220s)  😢 sad       (conf: 0.88)
   3.   4.560s -   6.780s ( 2.220s)  😠 angry     (conf: 0.85)

Emotion Distribution:
  😊 happy   :   2.34s ( 33.8%)
  😢 sad     :   2.22s ( 32.1%)
  😠 angry   :   2.22s ( 32.1%)
```

**Batch Processing**:
```bash
python inference_diarization_7emotions.py \
    --checkpoint results/emotion_7class/save/CKPT+epoch_40/model.ckpt \
    --batch audio_folder/ \
    --smoothing 5 \
    --no-merge  # Keep all 20ms frames
```

### Python API
```python
import torch
from emotion_model import FrameLevelEmotionModel

# Load trained model
model = FrameLevelEmotionModel.load_from_checkpoint(
    'results/emotion_7class/save/CKPT+epoch_40/model.ckpt'
)
model.eval()
model.to('cuda')

# Process audio file
import torchaudio
waveform, sr = torchaudio.load('audio.wav')
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

# Get frame-level predictions (20ms resolution)
with torch.no_grad():
    logits = model(waveform.unsqueeze(0).cuda())
    probs = torch.softmax(logits, dim=-1)
    emotions = probs.argmax(dim=-1).squeeze(0).cpu().numpy()

# Map to emotion names
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "upset"]
emotion_timeline = [EMOTIONS[idx] for idx in emotions]

# Print timeline (50 FPS)
for i, emotion in enumerate(emotion_timeline):
    time = i * 0.02  # 20ms per frame
    print(f"{time:.2f}s: {emotion}")
```

---

## Confusion Analysis

### Primary Confusion Patterns

From test set evaluation (1.85M frames):

**High-Arousal Confusion** (14% of errors):
- **Happy → Angry**: 14% of happy frames misclassified
- **Angry → Happy**: 8% of angry frames misclassified
- **Cause**: Both high-arousal, opposite valence (arousal prioritized over valence)

**Fear Recognition Issue** (50% missed):
- **Fear → Sad**: 25% (low energy similarity)
- **Fear → Disgust**: 9% (negative valence overlap)
- **Fear → Happy**: 5% (unexpected, needs investigation)
- **Cause**: Only 2.0% training frames, insufficient acoustic patterns learned

**Low-Arousal Overlap**:
- **Sad → Happy**: 11% (subtle valence cues)
- **Sad → Upset**: 6% (semantic similarity)
- **Neutral → Sad**: 6% (low arousal baseline confusion)

### Diagonal Dominance

Correct classification rates per emotion:
- Disgust: 91.5% ⭐⭐⭐
- Neutral: 87.9% ⭐⭐
- Angry: 82.4% ⭐⭐
- Happy: 77.3% ⭐
- Upset: 82.8% ⭐
- Sad: 73.6% ⭐
- Fear: 49.6% ⚠️

---

## Datasets Used

### Training Corpus

| Dataset | Samples | Contribution | Emotions | Quality |
|---------|---------|--------------|----------|---------|
| **MELD** | 12,070 | 20.5% | 7→7 | ⭐⭐ Conversational |
| **IEMOCAP** | 11,032 | 18.8% | 10→7 | ⭐⭐⭐ Scripted+Improv |
| **JL-Corpus** | 10,661 | 18.1% | 3→3 | ⭐⭐ New Zealand English |
| **ESD** | 10,500 | 17.8% | 5→5 | ⭐⭐⭐ Professional |
| **CREMA-D** | 7,442 | 12.6% | 6→6 | ⭐⭐⭐ Multi-ethnic |
| **EmoV-DB** | 6,000 | 5.7% | 5→4 | ⭐⭐ Varied contexts |
| **TESS** | 2,800 | 4.1% | 7→7 | ⭐⭐⭐ Studio quality |
| **RAVDESS** | 1,440 | 1.7% | 8→7 | ⭐⭐⭐ Studio quality |
| **SAVEE** | 480 | 0.8% | 7→7 | ⭐⭐ British English |

**Total**: 58,834 utterances → 46,764 valid samples → 12.45M frames (69.2 hours)

**Test Set**: 7,007 utterances → 1.85M frames (10.3 hours)
