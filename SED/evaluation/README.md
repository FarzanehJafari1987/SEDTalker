# Evaluation Guide

Quick guide to evaluate trained emotion diarization models

## Overview

A comprehensive evaluation pipeline for **frame-level speech emotion 
diarization**.

**Includes:** 
- Metrics: Accuracy, Precision, Recall, F1-score per emotion
- Visualizations: Confusion matrix, per-class metrics, emotion timeline
- Input: Model checkpoint + test dataset - Output: Detailed report + figures

## Quick Start

### Prerequisites

```
pip install torch torchaudio transformers scikit-learn matplotlib seaborn tqdm
```

## Step 1: Generate Predictions

```
python test_preditions.py --checkpoint results/emotion_7class/save/CKPT+epoch_40/model.ckpt --test-json data/processed_emotions_7class/test_frames.json --output evaluation/test_predictions.json --smoothing 5
```

### Output

``` json
{
  "sample_id": {
    "frame_labels": ["happy", "happy", "..."],
    "wav": "path/to/audio.wav"
  }
}
```

## Step 2: Run Comprehensive Evaluation

```
python evaluate_sed_comprehensive.py --predictions evaluation/test_predictions.json --ground_truth data/processed_emotions_7class/test_frames.json --output_dir evaluation_results/
```

### Output Files

    evaluation_results/
    ├── evaluation_results.json
    ├── evaluation_report.txt
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    └── emotion_timeline_sample.png

## Expected Results

Overall accuracy, F1 scores, jitter, and more.

## Understanding Outputs

-   Confusion matrix analysis
-   Per-class metric interpretation
-   Emotion timeline alignment

## Metrics

Formulas for accuracy, F1, weighted F1, jitter, and purity.

## Customization

Adjust smoothing, evaluate subsets, and generate ROC/PR curves.

## File Formats

JSON + text report structure.
