# Speech Emotion Diarization (SED) - Evaluation Guide

## Metrics Overview

Your SED system should be evaluated on **three levels**:

### 1️⃣ **Frame-Level Metrics** (Basic)
- **Accuracy**: Overall correctness (70-76% is good for 7 emotions)
- **Per-class F1**: How well each emotion is recognized
- **Confusion Matrix**: Which emotions are confused

### 2️⃣ **Segment-Level Metrics** (Important!)
- **Segment Purity**: How "clean" are emotion segments?
- **Segment F1**: Detection quality (like object detection)
- **DER (Diarization Error Rate)**: Standard metric (lower is better)

### 3️⃣ **Temporal Metrics** (Critical for Diarization!)
- **Temporal Jitter**: Prediction stability (lower is better)
- **Boundary Detection**: How well transitions are detected
- **Segment Duration**: Are segments realistic length?

---

## 🚀 Quick Start

### **Step 1: Run Inference**
```bash
python inference_diarization_7emotions.py \
    --checkpoint results/emotion_diarization_7class/save/CKPT+*/model.pt \
    --audio test_audio.wav \
    --output predictions.json
```

### **Step 2: Run Evaluation**
```bash
python evaluate_sed_comprehensive.py \
    --predictions predictions.json \
    --ground_truth datasets/processed_emotions_7class/test_frames.json \
    --output_dir evaluation_results/
```

### **Step 3: Check Results**
```bash
cat evaluation_results/evaluation_report.txt
```
