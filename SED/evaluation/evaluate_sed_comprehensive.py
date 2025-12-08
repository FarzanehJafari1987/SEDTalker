"""
Comprehensive Evaluation for Speech Emotion Diarization (SED)

This script provides multiple evaluation metrics:
1. Frame-level metrics (accuracy, F1, precision, recall)
2. Segment-level metrics (purity, DER, segment F1)
3. Temporal consistency metrics (jitter, duration stats)
4. Boundary detection metrics
5. Visualizations (confusion matrix, timeline plots)

Usage:
    python evaluate_sed_comprehensive.py \
        --predictions results/predictions.json \
        --ground_truth datasets/test_frames.json \
        --output_dir evaluation_results/
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import argparse
import os


EMOTION_LABELS = ["happy", "sad", "angry", "disgust", "fear", "upset", "neutral"]
EMOTION_TO_IDX = {emo: idx for idx, emo in enumerate(EMOTION_LABELS)}


# ============================================================================
# 1. FRAME-LEVEL METRICS
# ============================================================================

def compute_frame_metrics(predictions, ground_truth):
    """
    Compute frame-level accuracy, precision, recall, F1
    
    Args:
        predictions: [N] array of predicted emotion labels
        ground_truth: [N] array of ground truth labels
    
    Returns:
        Dictionary with metrics
    """
    # Overall accuracy
    accuracy = (predictions == ground_truth).mean()
    
    # Per-class metrics
    report = classification_report(
        ground_truth,
        predictions,
        target_names=EMOTION_LABELS,
        output_dict=True,
        zero_division=0
    )
    
    # Weighted F1 (accounts for class imbalance)
    weighted_f1 = f1_score(ground_truth, predictions, average='weighted')
    
    # Macro F1 (treats all classes equally)
    macro_f1 = f1_score(ground_truth, predictions, average='macro')
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'per_class': report,
        'confusion_matrix': cm,
        'confusion_matrix_norm': cm_norm
    }


def analyze_confusions(cm_norm, threshold=0.1):
    """Find most common emotion confusions"""
    confusions = []
    
    for i, true_emo in enumerate(EMOTION_LABELS):
        for j, pred_emo in enumerate(EMOTION_LABELS):
            if i != j and cm_norm[i, j] > threshold:
                confusions.append({
                    'true': true_emo,
                    'predicted': pred_emo,
                    'rate': cm_norm[i, j]
                })
    
    confusions.sort(key=lambda x: x['rate'], reverse=True)
    return confusions


# ============================================================================
# 2. SEGMENT-LEVEL METRICS
# ============================================================================

def extract_segments(frame_labels, frame_rate=50):
    """
    Convert frame-level labels to segments
    
    Args:
        frame_labels: [T] frame-level emotion labels
        frame_rate: Frames per second (50 for 20ms frames)
    
    Returns:
        List of segments: [{'start': s, 'end': e, 'emotion': emo}, ...]
    """
    if len(frame_labels) == 0:
        return []
    
    segments = []
    current_emotion = frame_labels[0]
    start_frame = 0
    
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != current_emotion:
            # Segment ended
            segments.append({
                'start': start_frame / frame_rate,
                'end': i / frame_rate,
                'emotion': current_emotion,
                'duration': (i - start_frame) / frame_rate
            })
            current_emotion = frame_labels[i]
            start_frame = i
    
    # Add last segment
    segments.append({
        'start': start_frame / frame_rate,
        'end': len(frame_labels) / frame_rate,
        'emotion': current_emotion,
        'duration': (len(frame_labels) - start_frame) / frame_rate
    })
    
    return segments


def segment_purity(pred_labels, gt_labels):
    """
    Measure purity of predicted segments
    
    Returns:
        Average purity and rate of pure segments
    """
    pred_segments = extract_segments(pred_labels)
    
    purities = []
    frame_rate = 50
    
    for seg in pred_segments:
        start_frame = int(seg['start'] * frame_rate)
        end_frame = int(seg['end'] * frame_rate)
        
        # Get ground truth in this segment
        gt_in_segment = gt_labels[start_frame:end_frame]
        
        # Count matches
        matches = (gt_in_segment == seg['emotion']).sum()
        purity = matches / len(gt_in_segment) if len(gt_in_segment) > 0 else 0
        
        purities.append(purity)
    
    avg_purity = np.mean(purities) if purities else 0
    pure_segments = sum(p > 0.8 for p in purities)
    pure_rate = pure_segments / len(purities) if purities else 0
    
    return {
        'average_purity': avg_purity,
        'pure_segment_rate': pure_rate,
        'num_segments': len(pred_segments)
    }


def compute_iou(seg1, seg2):
    """Intersection over Union for temporal segments"""
    start1, end1 = seg1['start'], seg1['end']
    start2, end2 = seg2['start'], seg2['end']
    
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    
    return intersection / union if union > 0 else 0


def segment_f1_score(pred_labels, gt_labels, iou_threshold=0.5):
    """
    F1 score for segment detection (like object detection)
    """
    pred_segments = extract_segments(pred_labels)
    gt_segments = extract_segments(gt_labels)
    
    true_positives = 0
    matched_gt = set()
    
    for pred_seg in pred_segments:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_seg in enumerate(gt_segments):
            if gt_seg['emotion'] != pred_seg['emotion']:
                continue
            
            iou = compute_iou(pred_seg, gt_seg)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    false_positives = len(pred_segments) - true_positives
    false_negatives = len(gt_segments) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'segment_precision': precision,
        'segment_recall': recall,
        'segment_f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


# ============================================================================
# 3. TEMPORAL CONSISTENCY METRICS
# ============================================================================

def temporal_jitter(predictions, window=5):
    """
    Measure prediction stability (lower is better)
    """
    if len(predictions) < window:
        return 0
    
    changes = []
    for i in range(len(predictions) - window):
        window_preds = predictions[i:i+window]
        num_changes = sum(window_preds[j] != window_preds[j+1] 
                         for j in range(len(window_preds)-1))
        changes.append(num_changes)
    
    avg_jitter = np.mean(changes)
    jitter_rate = avg_jitter / (window - 1)
    
    return jitter_rate


def segment_duration_stats(predictions):
    """
    Analyze duration distribution of emotion segments
    """
    segments = extract_segments(predictions)
    
    if not segments:
        return {
            'mean': 0, 'median': 0, 'std': 0,
            'min': 0, 'max': 0, 'num_segments': 0,
            'very_short_rate': 0
        }
    
    durations = [seg['duration'] for seg in segments]
    
    very_short = sum(d < 0.2 for d in durations)
    
    return {
        'mean': np.mean(durations),
        'median': np.median(durations),
        'std': np.std(durations),
        'min': np.min(durations),
        'max': np.max(durations),
        'num_segments': len(segments),
        'very_short_rate': very_short / len(segments)
    }


def boundary_detection_metrics(pred_labels, gt_labels, tolerance=0.5):
    """
    Evaluate emotion boundary detection
    
    Args:
        tolerance: Time tolerance (seconds) for boundary match
    """
    pred_segments = extract_segments(pred_labels)
    gt_segments = extract_segments(gt_labels)
    
    # Extract boundaries (transition points)
    pred_boundaries = [seg['start'] for seg in pred_segments[1:]]
    gt_boundaries = [seg['start'] for seg in gt_segments[1:]]
    
    if not gt_boundaries:
        return {'boundary_precision': 0, 'boundary_recall': 0, 'boundary_f1': 0}
    
    true_positives = 0
    for gt_bound in gt_boundaries:
        if any(abs(pred_bound - gt_bound) <= tolerance 
               for pred_bound in pred_boundaries):
            true_positives += 1
    
    precision = true_positives / len(pred_boundaries) if pred_boundaries else 0
    recall = true_positives / len(gt_boundaries) if gt_boundaries else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'boundary_precision': precision,
        'boundary_recall': recall,
        'boundary_f1': f1,
        'num_pred_boundaries': len(pred_boundaries),
        'num_gt_boundaries': len(gt_boundaries)
    }


# ============================================================================
# 4. VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm_norm, save_path):
    """Plot normalized confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cbar_kws={'label': 'Percentage'},
        vmin=0,
        vmax=1
    )
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.title('Emotion Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {save_path}")


def plot_per_class_metrics(metrics, save_path):
    """Plot precision, recall, F1 for each emotion"""
    emotions = EMOTION_LABELS
    precision = [metrics['per_class'][emo]['precision'] for emo in emotions]
    recall = [metrics['per_class'][emo]['recall'] for emo in emotions]
    f1 = [metrics['per_class'][emo]['f1-score'] for emo in emotions]
    
    x = np.arange(len(emotions))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Emotion Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class metrics: {save_path}")


def plot_emotion_timeline(pred_labels, gt_labels, audio_duration, save_path):
    """
    Plot predicted vs ground truth emotion timeline
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    # Create colormap - support both numeric indices and string labels
    colors = plt.cm.Set3(np.linspace(0, 1, len(EMOTION_LABELS)))
    emotion_colors = {emo: colors[i] for i, emo in enumerate(EMOTION_LABELS)}
    # Also map numeric indices to colors
    for i in range(len(EMOTION_LABELS)):
        emotion_colors[i] = colors[i]
        emotion_colors[np.int64(i)] = colors[i]  # Handle numpy int64
    
    # Plot ground truth
    gt_segments = extract_segments(gt_labels)
    for seg in gt_segments:
        # Get emotion label (handle both numeric and string)
        emotion = seg['emotion']
        if isinstance(emotion, (int, np.integer)):
            emotion_label = EMOTION_LABELS[emotion]
        else:
            emotion_label = emotion
        
        ax1.barh(0, seg['duration'], left=seg['start'], 
                height=0.8, color=emotion_colors[emotion], 
                edgecolor='black', linewidth=0.5)
        if seg['duration'] > 0.5:  # Only label if segment is wide enough
            ax1.text(seg['start'] + seg['duration']/2, 0, emotion_label[:3],
                    ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Ground\nTruth', fontsize=10, fontweight='bold')
    ax1.set_ylim([-0.5, 0.5])
    ax1.set_yticks([])
    ax1.set_title('Emotion Timeline Comparison', fontsize=14, fontweight='bold')
    
    # Plot predictions
    pred_segments = extract_segments(pred_labels)
    for seg in pred_segments:
        # Get emotion label (handle both numeric and string)
        emotion = seg['emotion']
        if isinstance(emotion, (int, np.integer)):
            emotion_label = EMOTION_LABELS[emotion]
        else:
            emotion_label = emotion
        
        ax2.barh(0, seg['duration'], left=seg['start'], 
                height=0.8, color=emotion_colors[emotion], 
                edgecolor='black', linewidth=0.5)
        if seg['duration'] > 0.5:
            ax2.text(seg['start'] + seg['duration']/2, 0, emotion_label[:3],
                    ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Prediction', fontsize=10, fontweight='bold')
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_yticks([])
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_xlim([0, audio_duration])
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, fc=emotion_colors[emo], 
                                    edgecolor='black', label=emo.capitalize())
                      for emo in EMOTION_LABELS]
    fig.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(1.12, 0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved timeline: {save_path}")


# ============================================================================
# 5. MAIN EVALUATION
# ============================================================================

def evaluate_all(predictions, ground_truth, output_dir):
    """
    Run complete evaluation and generate report
    
    Args:
        predictions: Dict {sample_id: {'frame_labels': [...]}}
        ground_truth: Dict {sample_id: {'frame_labels': [...]}}
        output_dir: Where to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SPEECH EMOTION DIARIZATION - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Collect all predictions and ground truth
    all_preds = []
    all_gt = []
    
    skipped_samples = 0
    length_mismatches = 0
    
    for sample_id in predictions.keys():
        if sample_id not in ground_truth:
            print(f"Warning: {sample_id} not in ground truth, skipping")
            skipped_samples += 1
            continue
        
        pred = predictions[sample_id]['frame_labels']
        gt = ground_truth[sample_id]['frame_labels']
        
        # Convert to indices if strings
        if isinstance(pred[0], str):
            pred = [EMOTION_TO_IDX[e] for e in pred]
        if isinstance(gt[0], str):
            gt = [EMOTION_TO_IDX[e] for e in gt]
        
        # Handle length mismatch (due to model downsampling)
        if len(pred) != len(gt):
            length_mismatches += 1
            # Truncate to shorter length
            min_len = min(len(pred), len(gt))
            pred = pred[:min_len]
            gt = gt[:min_len]
        
        all_preds.extend(pred)
        all_gt.extend(gt)
    
    if skipped_samples > 0:
        print(f"Skipped {skipped_samples} samples not in ground truth")
    if length_mismatches > 0:
        print(f"Fixed {length_mismatches} samples with length mismatch")
    
    all_preds = np.array(all_preds)
    all_gt = np.array(all_gt)
    
    print(f"\nTotal frames evaluated: {len(all_preds):,}")
    
    # 1. Frame-level metrics
    print("\n" + "-"*70)
    print("1. FRAME-LEVEL METRICS")
    print("-"*70)
    
    frame_metrics = compute_frame_metrics(all_preds, all_gt)
    
    print(f"\nOverall Accuracy: {frame_metrics['accuracy']:.4f} ({frame_metrics['accuracy']*100:.2f}%)")
    print(f"Weighted F1: {frame_metrics['weighted_f1']:.4f}")
    print(f"Macro F1: {frame_metrics['macro_f1']:.4f}")
    
    print("\nPer-Emotion Results:")
    print(f"{'Emotion':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for emo in EMOTION_LABELS:
        metrics = frame_metrics['per_class'][emo]
        print(f"{emo:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}")
    
    # Confusion analysis
    confusions = analyze_confusions(frame_metrics['confusion_matrix_norm'])
    print("\nMost Common Confusions (>10%):")
    for conf in confusions[:10]:
        print(f"  {conf['true']:8s} → {conf['predicted']:8s}: {conf['rate']:.2%}")
    
    # 2. Segment-level metrics
    print("\n" + "-"*70)
    print("2. SEGMENT-LEVEL METRICS")
    print("-"*70)
    
    purity_metrics = segment_purity(all_preds, all_gt)
    print(f"\nSegment Purity:")
    print(f"  Average purity: {purity_metrics['average_purity']:.4f} ({purity_metrics['average_purity']*100:.2f}%)")
    print(f"  Pure segments (>80%): {purity_metrics['pure_segment_rate']:.4f} ({purity_metrics['pure_segment_rate']*100:.2f}%)")
    print(f"  Total segments: {purity_metrics['num_segments']}")
    
    seg_f1_metrics = segment_f1_score(all_preds, all_gt, iou_threshold=0.5)
    print(f"\nSegment Detection (IoU>0.5):")
    print(f"  Precision: {seg_f1_metrics['segment_precision']:.4f}")
    print(f"  Recall: {seg_f1_metrics['segment_recall']:.4f}")
    print(f"  F1-Score: {seg_f1_metrics['segment_f1']:.4f}")
    
    # 3. Temporal consistency
    print("\n" + "-"*70)
    print("3. TEMPORAL CONSISTENCY METRICS")
    print("-"*70)
    
    jitter = temporal_jitter(all_preds, window=5)
    print(f"\nTemporal Jitter: {jitter:.4f}")
    if jitter < 0.2:
        print("  ✓ Very stable predictions")
    elif jitter < 0.4:
        print("  ✓ Moderate stability")
    else:
        print("  ⚠ High jitter - consider smoothing")
    
    duration_stats = segment_duration_stats(all_preds)
    print(f"\nSegment Duration Statistics:")
    print(f"  Mean: {duration_stats['mean']:.3f}s")
    print(f"  Median: {duration_stats['median']:.3f}s")
    print(f"  Std: {duration_stats['std']:.3f}s")
    print(f"  Range: [{duration_stats['min']:.3f}s, {duration_stats['max']:.3f}s]")
    print(f"  Very short (<200ms): {duration_stats['very_short_rate']:.2%}")
    
    boundary_metrics = boundary_detection_metrics(all_preds, all_gt, tolerance=0.5)
    print(f"\nBoundary Detection (±0.5s tolerance):")
    print(f"  Precision: {boundary_metrics['boundary_precision']:.4f}")
    print(f"  Recall: {boundary_metrics['boundary_recall']:.4f}")
    print(f"  F1-Score: {boundary_metrics['boundary_f1']:.4f}")
    
    # 4. Visualizations
    print("\n" + "-"*70)
    print("4. GENERATING VISUALIZATIONS")
    print("-"*70)
    
    plot_confusion_matrix(
        frame_metrics['confusion_matrix_norm'],
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    plot_per_class_metrics(
        frame_metrics,
        os.path.join(output_dir, 'per_class_metrics.png')
    )
    
    # Plot timeline for first sample
    first_sample = list(predictions.keys())[0]
    audio_duration = len(all_preds) * 0.02  # 20ms per frame
    plot_emotion_timeline(
        all_preds[:1000],  # First 20 seconds
        all_gt[:1000],
        20.0,
        os.path.join(output_dir, 'emotion_timeline_sample.png')
    )
    
    # 5. Save summary report
    print("\n" + "-"*70)
    print("5. SAVING RESULTS")
    print("-"*70)
    
    results = {
        'frame_metrics': {
            'accuracy': float(frame_metrics['accuracy']),
            'weighted_f1': float(frame_metrics['weighted_f1']),
            'macro_f1': float(frame_metrics['macro_f1']),
            'per_class': {
                emo: {
                    'precision': float(frame_metrics['per_class'][emo]['precision']),
                    'recall': float(frame_metrics['per_class'][emo]['recall']),
                    'f1': float(frame_metrics['per_class'][emo]['f1-score']),
                    'support': int(frame_metrics['per_class'][emo]['support'])
                }
                for emo in EMOTION_LABELS
            }
        },
        'segment_metrics': {
            'purity': float(purity_metrics['average_purity']),
            'pure_rate': float(purity_metrics['pure_segment_rate']),
            'segment_f1': float(seg_f1_metrics['segment_f1'])
        },
        'temporal_metrics': {
            'jitter': float(jitter),
            'mean_duration': float(duration_stats['mean']),
            'boundary_f1': float(boundary_metrics['boundary_f1'])
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results: {output_dir}/evaluation_results.json")
    
    # Generate text report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SPEECH EMOTION DIARIZATION - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Frames: {len(all_preds):,}\n")
        f.write(f"Overall Accuracy: {frame_metrics['accuracy']:.4f} ({frame_metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Weighted F1: {frame_metrics['weighted_f1']:.4f}\n")
        f.write(f"Segment Purity: {purity_metrics['average_purity']:.4f}\n")
        f.write(f"Temporal Jitter: {jitter:.4f}\n")
        f.write("\n" + "-"*70 + "\n")
        f.write("Per-Emotion Results:\n")
        f.write("-"*70 + "\n")
        
        for emo in EMOTION_LABELS:
            metrics = frame_metrics['per_class'][emo]
            f.write(f"{emo:<10}: P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
                   f"F1={metrics['f1-score']:.3f}  Support={int(metrics['support'])}\n")
    
    print(f"✓ Saved report: {report_path}")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("  - evaluation_results.json (detailed metrics)")
    print("  - evaluation_report.txt (summary)")
    print("  - confusion_matrix.png")
    print("  - per_class_metrics.png")
    print("  - emotion_timeline_sample.png")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive SED Evaluation')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--ground_truth', type=str, required=True,
                       help='Path to ground truth JSON file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loading ground truth from: {args.ground_truth}")
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    # Run evaluation
    results = evaluate_all(predictions, ground_truth, args.output_dir)


if __name__ == "__main__":
    main()



# python evaluation/evaluation_results/evaluate_sed_comprehensive.py --predictions evaluation/test_predictions.json --ground_truth data/processed_emotions_7class/test_frames.json --output_dir evaluation/evaluation_results