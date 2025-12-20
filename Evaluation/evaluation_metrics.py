import numpy as np
import argparse
import os
from collections import defaultdict

def fourier_frequency_error(M: np.ndarray, M_hat: np.ndarray) -> float:
    """
    Compute Fourier Frequency Error (FFE) between two motion sequences.
    """
    assert M.shape == M_hat.shape, "Input sequences must have the same shape"
    T, V, C = M.shape
    assert C == 3, "FFE expects 3 channels"

    F_M = np.fft.rfft(M, axis=0)
    F_M_hat = np.fft.rfft(M_hat, axis=0)

    diff_squared = np.abs(F_M - F_M_hat) ** 2
    ffe = np.sum(diff_squared) / (T * V * C)
    return float(ffe)

def evaluate(pred_seq, gt_seq, mouth_mask, upper_mask, emotion_mask):
    """
    Compute all evaluation metrics for one sequence.
    pred_seq and gt_seq must be (T, V, 3)
    """
    errors = {}

    min_len = min(pred_seq.shape[0], gt_seq.shape[0])
    pred_seq = pred_seq[:min_len]
    gt_seq = gt_seq[:min_len]

    # ----- MVE, LVE, EVE -----
    errors['mve'] = np.linalg.norm(pred_seq - gt_seq, axis=2).mean()
    errors['lve'] = np.linalg.norm(pred_seq[:, mouth_mask, :] - gt_seq[:, mouth_mask, :], axis=2).mean()
    errors['eve'] = np.linalg.norm(pred_seq[:, emotion_mask, :] - gt_seq[:, emotion_mask, :], axis=2).mean()

    # ----- MOD & VE -----
    pred_offset = pred_seq[1:] - pred_seq[:-1]
    gt_offset = gt_seq[1:] - gt_seq[:-1]
    diff_offset = np.linalg.norm(pred_offset - gt_offset, axis=2).mean(axis=1)
    errors['mod'] = diff_offset.mean()
    errors['ve'] = diff_offset.mean()  # same as mod

    # ----- Acceleration Error -----
    pred_acc = pred_offset[1:] - pred_offset[:-1]
    gt_acc = gt_offset[1:] - gt_offset[:-1]
    errors['ae'] = np.mean(np.linalg.norm(pred_acc - gt_acc, axis=2).mean(axis=1))

    # ----- Temporal Consistency -----
    errors['tc'] = np.mean(np.linalg.norm(pred_offset, axis=2).mean(axis=1))

    # ----- FDD & ABS FDD -----
    def motion_std(seq, mask):
        selected = seq[:, mask, :]
        L2 = np.sum(selected ** 2, axis=2)
        return np.std(L2, axis=0)

    gt_motion_std = motion_std(gt_seq, upper_mask)
    pred_motion_std = motion_std(pred_seq, upper_mask)
    errors['fdd'] = gt_motion_std - pred_motion_std
    errors['abs_fdd'] = abs(gt_motion_std - pred_motion_std)

    # ----- Fourier Frequency Error -----
    if pred_seq.shape[2] >= 3:
        errors['ffe'] = fourier_frequency_error(gt_seq, pred_seq)
    else:
        errors['ffe'] = np.nan

    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="EmoVOCA")
    parser.add_argument("--pred_path", type=str, default="EmoVOCA/result/")
    parser.add_argument("--gt_path", type=str, default="EmoVOCA/sequences/")
    args = parser.parse_args()

    # ---- Define vertex masks ----
    mouth_mask = list(range(94, 114)) + list(range(146, 178)) + list(range(183, 192))
    upper_mask = [x for x in range(192) if x not in mouth_mask]
    emotion_mask = list(range(36, 94))  # adjust based on your dataset

    # ---- Metrics containers ----
    metrics = defaultdict(list)
    metrics_emotion = defaultdict(lambda: defaultdict(list))

    total_frames = 0
    num_seq = 0

    for file in os.listdir(args.pred_path):
        if not file.endswith('.npy'):
            continue

        pred_file = os.path.join(args.pred_path, file)
        pred_seq = np.load(pred_file)

        # ---- Parse prediction filename to match GT ----
        tokens = os.path.splitext(file)[0].split('_')
        target_id = "_".join(tokens[-4:])         # target speaker
        sentence = tokens[4]
        emotion = tokens[5]
        index = tokens[6]
        seq_name = f"{target_id}_{sentence}_{emotion}_{index}"
        gt_file = os.path.join(args.gt_path, seq_name + ".npy")

        if not os.path.exists(gt_file):
            print(f"[WARNING] GT file not found: {gt_file}")
            continue

        gt_seq = np.load(gt_file)

        # ---- Ensure both sequences are (T, V, 3) ----
        if gt_seq.ndim == 2:
            gt_seq = gt_seq.reshape(gt_seq.shape[0], -1, 3)
        if pred_seq.ndim == 2:
            pred_seq = pred_seq.reshape(pred_seq.shape[0], gt_seq.shape[1], 3)

        if gt_seq.shape[0] < 3 or pred_seq.shape[0] < 3:
            continue

        errs = evaluate(pred_seq, gt_seq, mouth_mask, upper_mask, emotion_mask)

        # ---- Store metrics ----
        for key, val in errs.items():
            metrics[key].append(val)
            metrics_emotion[emotion][key].append(val)

        total_frames += min(pred_seq.shape[0], gt_seq.shape[0])
        num_seq += 1

    # ---- Print overall metrics ----
    if num_seq == 0:
        print("No valid sequences found.")
        return

    print(f'Total Frames: {total_frames}')
    print(f'Total Sequences: {num_seq}')
    print('-'*40)
    print("=== Overall Metrics ===")
    for key in ['mve','lve','eve','fdd','abs_fdd','mod','ve','ae','tc','ffe']:
        print(f"{key.upper()}: {np.mean(metrics[key]):.4e}")

    # ---- Metrics per emotion ----
    print("\n=== Metrics by Emotion ===")
    for emotion, emo_metrics in metrics_emotion.items():
        print(f"\nEmotion: {emotion}")
        for key in ['mve','lve','eve','fdd','abs_fdd','mod','ve','ae','tc','ffe']:
            print(f"  {key.upper()}: {np.mean(emo_metrics[key]):.4e}")

if __name__ == "__main__":
    main()
