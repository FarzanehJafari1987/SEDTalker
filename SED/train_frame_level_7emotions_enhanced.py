"""
Complete Frame-Level Training Script for Emotion Diarization
ENHANCED VERSION - Better model architecture to prevent file-level overfitting
Compatible with SpeechBrain 0.5.13
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import make_dataloader
from transformers import AutoModel
import json
import torchaudio
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

EMOTION_LABELS = ["happy", "sad", "angry", "disgust", "fear", "upset", "neutral"]  # 7 emotions

try:
    CLASS_WEIGHTS = torch.load("data/processed_emotions_7class/class_weights.pt")
    print(f"✓ Loaded class weights: {CLASS_WEIGHTS}")
except FileNotFoundError:
    CLASS_WEIGHTS = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    print("⚠️  Using equal class weights")

CONFIG = {
    "seed": 1234,
    "data_folder": "data/processed_emotions_7class",
    "output_folder": "results/emotion_diarization_7class_enhanced",  # New folder for enhanced model
    "save_folder": "results/emotion_diarization_7class_enhanced/save",
    "batch_size": 4,
    "grad_accumulation_factor": 8,
    "number_of_epochs": 100,
    "lr": 0.0001,
    "output_neurons": 7,  # 7 emotions
    "sample_rate": 16000,
    "wav2vec2_hub": "microsoft/wavlm-base-plus",
    "freeze_feature_extractor": True,
    "freeze_encoder_layers": 8,  # ENHANCED: More aggressive freezing
    "dropout": 0.3,
    "frame_shift": 0.02,  # 20ms per frame (WavLM default)
}

# ============================================================================
# ENHANCED Frame-Level Model (prevents file-level overfitting)
# ============================================================================

class FrameLevelEmotionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        print(f"Loading Enhanced Model: {config['wav2vec2_hub']}")
        self.wav2vec2 = AutoModel.from_pretrained(
            config['wav2vec2_hub'],
            cache_dir="pretrained_models/"
        )
        
        # ENHANCED: More aggressive freezing to prevent file-level learning
        
        # 1. Freeze feature extractor completely
        if hasattr(self.wav2vec2, 'feature_extractor'):
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
            print("  ✓ Froze feature extractor")
        
        # 2. Freeze more encoder layers (keep only last 4 trainable)
        if hasattr(self.wav2vec2, 'encoder') and hasattr(self.wav2vec2.encoder, 'layers'):
            num_layers = len(self.wav2vec2.encoder.layers)
            freeze_layers = max(config.get('freeze_encoder_layers', 6), num_layers - 4)
            
            for i in range(min(freeze_layers, num_layers)):
                for param in self.wav2vec2.encoder.layers[i].parameters():
                    param.requires_grad = False
            
            print(f"  ✓ Froze {freeze_layers}/{num_layers} encoder layers (keeping last {num_layers - freeze_layers} trainable)")
        
        self.feature_dim = self.wav2vec2.config.hidden_size
        
        # ENHANCED: Add layer normalization for stability
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        
        # ENHANCED: Deeper classifier with better regularization
        self.classifier = nn.Sequential(
            # First block
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),  # GELU instead of ReLU for better gradients
            nn.Dropout(config['dropout']),
            
            # Second block
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            
            # Third block
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            
            # Output
            nn.Linear(128, config['output_neurons'])
        )
        
        # Print trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Model: {trainable_params:,} / {total_params:,} trainable params ({trainable_params/total_params*100:.1f}%)")
    
    def forward(self, wavs, wav_lens=None):
        """
        Args:
            wavs: [batch, samples]
            wav_lens: Optional relative lengths
            
        Returns:
            logits: [batch, time_frames, num_classes]
        """
        # Extract WavLM features
        outputs = self.wav2vec2(wavs)
        feats = outputs.last_hidden_state  # [batch, time, feat_dim]
        
        # ENHANCED: Normalize features
        feats = self.feature_norm(feats)
        
        # NO POOLING - keep all time frames
        batch_size, time_steps, feat_dim = feats.shape
        
        # Frame-level classification
        feats_flat = feats.reshape(-1, feat_dim)  # [batch*time, feat_dim]
        logits_flat = self.classifier(feats_flat)  # [batch*time, num_classes]
        
        # Reshape back to [batch, time, num_classes]
        logits = logits_flat.reshape(batch_size, time_steps, -1)
        
        return logits


# ============================================================================
# Frame-Level Brain
# ============================================================================

class FrameLevelEmotionBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # Get frame-level logits
        logits = self.modules.model(wavs, wav_lens)  # [batch, time, num_classes]
        
        return logits
    
    def compute_objectives(self, logits, batch, stage):
        """
        Compute loss for frame-level predictions
        
        Args:
            logits: [batch, time, num_classes]
            batch: Batch containing frame_labels
        """
        # Get frame labels from batch
        frame_labels = batch.frame_labels_encoded
        
        # Get dimensions
        batch_size, time_steps, num_classes = logits.shape
        
        # Convert frame labels to tensor if needed
        if isinstance(frame_labels, list):
            # Handle list of lists
            max_len = time_steps  # Use model output length
            frame_labels_tensor = torch.full(
                (batch_size, max_len), 
                -1,  # Padding value (will be ignored in loss)
                dtype=torch.long, 
                device=logits.device
            )
            
            for i, labels in enumerate(frame_labels):
                if isinstance(labels, list):
                    # Truncate or pad to match time_steps
                    labels_len = min(len(labels), time_steps)
                    frame_labels_tensor[i, :labels_len] = torch.tensor(
                        labels[:labels_len], 
                        device=logits.device
                    )
                else:
                    frame_labels_tensor[i, 0] = labels
        else:
            frame_labels_tensor = frame_labels.to(logits.device)
        
        # Ensure frame_labels match time dimension
        if frame_labels_tensor.shape[1] != time_steps:
            # Pad or truncate
            if frame_labels_tensor.shape[1] < time_steps:
                padding = torch.full(
                    (batch_size, time_steps - frame_labels_tensor.shape[1]),
                    -1,
                    dtype=torch.long,
                    device=logits.device
                )
                frame_labels_tensor = torch.cat([frame_labels_tensor, padding], dim=1)
            else:
                frame_labels_tensor = frame_labels_tensor[:, :time_steps]
        
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, num_classes)  # [batch*time, num_classes]
        labels_flat = frame_labels_tensor.reshape(-1)  # [batch*time]
        
        # Create mask for valid labels (ignore padding)
        valid_mask = labels_flat >= 0
        
        if valid_mask.sum() == 0:
            # No valid labels
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Filter valid frames
        logits_valid = logits_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        # Compute loss (with class weights)
        log_probs = F.log_softmax(logits_valid, dim=-1)
        loss = self.compute_cost(log_probs, labels_valid)
        
        # Compute accuracy for validation/test
        if stage != sb.Stage.TRAIN:
            preds = logits_valid.argmax(dim=-1)
            
            for pred, target in zip(preds, labels_valid):
                emotion_name = EMOTION_LABELS[target.item()]
                
                if emotion_name not in self.class_correct:
                    self.class_correct[emotion_name] = 0
                    self.class_total[emotion_name] = 0
                
                self.class_total[emotion_name] += 1
                if pred == target:
                    self.class_correct[emotion_name] += 1
        
        return loss
    
    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.class_correct = {}
            self.class_total = {}
    
    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            print(f"Epoch {epoch}: Train Loss = {stage_loss:.4f}")
        else:
            # Calculate overall accuracy
            total_correct = sum(self.class_correct.values())
            total_samples = sum(self.class_total.values())
            acc = total_correct / total_samples if total_samples > 0 else 0
            
            print(f"\n{stage} - Loss: {stage_loss:.4f}, Frame Accuracy: {acc:.4f}")
            
            print(f"\n{stage} Per-Class Frame Accuracy:")
            for emotion in EMOTION_LABELS:
                if emotion in self.class_total and self.class_total[emotion] > 0:
                    class_acc = self.class_correct[emotion] / self.class_total[emotion]
                    print(f"  {emotion:10s}: {class_acc:.4f} ({self.class_correct[emotion]}/{self.class_total[emotion]})")
            
            if stage == sb.Stage.VALID:
                if acc > self.best_acc:
                    self.best_acc = acc
                    print(f"✓ New best frame accuracy: {acc:.4f}")


# ============================================================================
# Data Loading
# ============================================================================

def load_json_data(data_dir):
    """Load frame-level JSON data"""
    with open(os.path.join(data_dir, "train_frames.json"), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, "valid_frames.json"), 'r') as f:
        valid_data = json.load(f)
    with open(os.path.join(data_dir, "test_frames.json"), 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded frame-level datasets:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Valid: {len(valid_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    return train_data, valid_data, test_data


def create_label_encoder(train_data, valid_data, test_data):
    encoder = CategoricalEncoder()
    
    # Collect all emotions from frame labels
    all_emotions = set()
    for data in [train_data, valid_data, test_data]:
        for item in data.values():
            for label in item['frame_labels']:
                all_emotions.add(label)
    
    # Fit encoder
    encoder.update_from_iterable(sorted(all_emotions))
    
    print(f"\nLabel encoder: {len(encoder)} classes")
    print(f"Classes: {list(encoder.lab2ind.keys())}")
    
    return encoder


def create_datasets(train_data, valid_data, test_data, label_encoder, config):
    
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        try:
            sig, sr = torchaudio.load(wav)
            if sr != config['sample_rate']:
                resampler = torchaudio.transforms.Resample(sr, config['sample_rate'])
                sig = resampler(sig)
            if sig.shape[0] > 1:
                sig = sig.mean(dim=0, keepdim=True)
            return sig.squeeze(0)
        except Exception as e:
            print(f"⚠️  Error loading {wav}: {e}")
            return torch.zeros(config['sample_rate'])
    
    @sb.utils.data_pipeline.takes("emotion")
    @sb.utils.data_pipeline.provides("emo_encoded")
    def emotion_pipeline(emotion):
        emotion_encoded = label_encoder.encode_label(emotion)
        return emotion_encoded
    
    @sb.utils.data_pipeline.takes("frame_labels")
    @sb.utils.data_pipeline.provides("frame_labels_encoded")
    def frame_label_pipeline(frame_labels):
        """Encode frame-level labels"""
        encoded_labels = [label_encoder.encode_label(label) for label in frame_labels]
        return encoded_labels
    
    train_set = DynamicItemDataset(
        data=train_data,
        dynamic_items=[audio_pipeline, emotion_pipeline, frame_label_pipeline],
        output_keys=["id", "sig", "emotion", "emo_encoded", "frame_labels_encoded"],
    )
    
    valid_set = DynamicItemDataset(
        data=valid_data,
        dynamic_items=[audio_pipeline, emotion_pipeline, frame_label_pipeline],
        output_keys=["id", "sig", "emotion", "emo_encoded", "frame_labels_encoded"],
    )
    
    test_set = DynamicItemDataset(
        data=test_data,
        dynamic_items=[audio_pipeline, emotion_pipeline, frame_label_pipeline],
        output_keys=["id", "sig", "emotion", "emo_encoded", "frame_labels_encoded"],
    )
    
    return train_set, valid_set, test_set


# ============================================================================
# Training
# ============================================================================

def train():
    # Set seed
    torch.manual_seed(CONFIG["seed"])
    os.makedirs(CONFIG["save_folder"], exist_ok=True)
    
    # Load data
    train_data, valid_data, test_data = load_json_data(CONFIG["data_folder"])
    
    # Create label encoder
    label_encoder = create_label_encoder(train_data, valid_data, test_data)
    
    # Create datasets
    train_set, valid_set, test_set = create_datasets(
        train_data, valid_data, test_data, label_encoder, CONFIG
    )
    
    # Create model
    model = FrameLevelEmotionModel(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} params ({trainable_params:,} trainable)")
    
    # Class weights
    weight_tensor = CLASS_WEIGHTS.to(device)
    print(f"\nClass weights: {CLASS_WEIGHTS}")
    
    # Loss function
    def weighted_nll_loss(log_probs, targets):
        return nn.functional.nll_loss(log_probs, targets, weight=weight_tensor)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=0.0001
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Checkpointer
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=CONFIG["save_folder"],
        recoverables={"model": model, "optimizer": optimizer}
    )
    
    # Create brain
    brain = FrameLevelEmotionBrain(
        modules={"model": model},
        hparams={},
        run_opts={"device": str(device)},
        checkpointer=checkpointer,
    )
    
    # Assign functions and variables
    brain.optimizer = optimizer
    brain.compute_cost = weighted_nll_loss
    brain.label_encoder = label_encoder
    brain.best_acc = 0.0
    
    # Mixed precision scaler
    use_amp = torch.cuda.is_available()
    if use_amp:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Training loop
    print("\n" + "="*60)
    print("Starting ENHANCED Frame-Level Training for Emotion Diarization")
    print("="*60 + "\n")
    
    for epoch in range(1, CONFIG["number_of_epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{CONFIG['number_of_epochs']}")
        print('='*60)
        
        # Train
        model.train()
        brain.on_stage_start(sb.Stage.TRAIN, epoch)
        train_loader = make_dataloader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
        
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            # Forward pass with mixed precision
            if use_amp:
                try:
                    with torch.amp.autocast('cuda'):
                        logits = brain.compute_forward(batch, sb.Stage.TRAIN)
                        loss = brain.compute_objectives(logits, batch, sb.Stage.TRAIN)
                        loss = loss / CONFIG["grad_accumulation_factor"]
                except:
                    with torch.cuda.amp.autocast():
                        logits = brain.compute_forward(batch, sb.Stage.TRAIN)
                        loss = brain.compute_objectives(logits, batch, sb.Stage.TRAIN)
                        loss = loss / CONFIG["grad_accumulation_factor"]
            else:
                logits = brain.compute_forward(batch, sb.Stage.TRAIN)
                loss = brain.compute_objectives(logits, batch, sb.Stage.TRAIN)
                loss = loss / CONFIG["grad_accumulation_factor"]
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item() * CONFIG["grad_accumulation_factor"]
            
            # Optimizer step with gradient accumulation
            if (i + 1) % CONFIG["grad_accumulation_factor"] == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {train_loss/(i+1):.4f}")
        
        train_loss /= len(train_loader)
        brain.on_stage_end(sb.Stage.TRAIN, train_loss, epoch)
        
        # Validate
        model.eval()
        brain.on_stage_start(sb.Stage.VALID, epoch)
        valid_loader = make_dataloader(valid_set, batch_size=CONFIG["batch_size"])
        
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                logits = brain.compute_forward(batch, sb.Stage.VALID)
                loss = brain.compute_objectives(logits, batch, sb.Stage.VALID)
                valid_loss += loss.item()
        
        valid_loss /= len(valid_loader)
        brain.on_stage_end(sb.Stage.VALID, valid_loss, epoch)
        
        scheduler.step(valid_loss)
        
        # Save checkpoint
        if epoch % 5 == 0:
            checkpointer.save_checkpoint(name=f"epoch_{epoch}")
    
    print("\n" + "="*60)
    print("Training Complete! Evaluating on test set...")
    print("="*60 + "\n")
    
    # Test
    model.eval()
    brain.on_stage_start(sb.Stage.TEST)
    test_loader = make_dataloader(test_set, batch_size=CONFIG["batch_size"])
    
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            logits = brain.compute_forward(batch, sb.Stage.TEST)
            loss = brain.compute_objectives(logits, batch, sb.Stage.TEST)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    brain.on_stage_end(sb.Stage.TEST, test_loss)
    
    print("\n✓ ENHANCED frame-level training and evaluation complete!")
    print(f"Best validation frame accuracy: {brain.best_acc:.4f}")


if __name__ == "__main__":
    train()
