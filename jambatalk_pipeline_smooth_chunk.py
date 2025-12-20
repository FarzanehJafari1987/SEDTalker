#!/usr/bin/env python3
"""
Emotion-conditioned 3D animation with BETTER SMOOTHING
Uses temporal filtering on vertices instead of emotion blending
"""

import os
import sys
import argparse
import pickle
import subprocess
import json
import warnings
warnings.filterwarnings('ignore')

# Check dependencies
try:
    import torch
    import numpy as np
    import librosa
    import cv2
    from transformers import Wav2Vec2Processor
    import trimesh
    import pyrender
    from scipy.ndimage import gaussian_filter1d
    print("✓ All dependencies loaded")
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install: pip install scipy")
    sys.exit(1)

try:
    from psbody.mesh import Mesh
    HAVE_PSBODY = True
except ImportError:
    HAVE_PSBODY = False
    print("Using trimesh (psbody.mesh not available)")

from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "wav_path": "SED/wav/test.wav",
    "dataset": "EmoVOCA",
    "model_name": "save_512_12_10_22_42/50_model",
    "template_path": "templates.pkl",
    "subject": "FaceTalk_170809_00138_TA",
    "output_dir": "demo/output",
    "sed_checkpoint": "SED/results/emotion_diarization_7class_1/save/CKPT+epoch_50/model.ckpt",
    "chunk_duration": 2.0,
    "overlap": 0.5,
    "skip_sed": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "feature_dim": 512,
    "vertice_dim": 5023*3,
    "period": 30,
    "save_meshes": False,
    
    # BETTER SMOOTHING - Temporal filtering on vertices
    "smooth_vertices": True,      # Apply temporal smoothing
    "smooth_sigma": 1.0,           # Gaussian smoothing strength (0.5-2.0)
    "smooth_window": 5,            # Window size for smoothing
    
    # CHUNK REDUCTION - Fewer, longer emotion segments
    "min_segment_duration": 0.0,  # Minimum segment length (0=keep all, 2.0=recommended)
    "merge_same_emotions": False, # Merge consecutive same emotions
}

print("\n" + "="*70)
print("🎬 EMOTION-CONDITIONED 3D ANIMATION (IMPROVED SMOOTHING)")
print("="*70)
print("\nConfiguration:")
for key, value in DEFAULT_CONFIG.items():
    print(f"  {key}: {value}")
print("="*70 + "\n")

# ============================================================================
# Mesh Wrapper
# ============================================================================

class MeshWrapper:
    def __init__(self, vertices=None, faces=None, filename=None):
        if filename:
            if HAVE_PSBODY:
                self.mesh = Mesh(filename=filename)
                self.v = self.mesh.v
                self.f = self.mesh.f
            else:
                self.mesh = trimesh.load(filename, process=False)
                self.v = np.array(self.mesh.vertices)
                self.f = np.array(self.mesh.faces)
        else:
            self.v = vertices
            self.f = faces
            if HAVE_PSBODY:
                self.mesh = Mesh(v=vertices, f=faces)
            else:
                self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def write_obj(self, filename):
        if HAVE_PSBODY:
            self.mesh.write_obj(filename)
        else:
            self.mesh.export(filename)

# ============================================================================
# Emotion Mappings
# ============================================================================

EMO_CHAR_TO_NAME = {
    "h": "happy", "a": "angry", "d": "disgust",
    "f": "fear", "n": "neutral", "s": "sad", "u": "upset"
}

SED_TO_JAMBATALK = {
    "angry": "angry", "disgust": "disgust", "fear": "fear",
    "happy": "happy", "sad": "sad", "upset": "upset", "neutral": "neutral"
}

EMO_NAME_TO_ID = {"happy": 0, "sad": 1, "angry": 2, "upset": 3, "disgust": 4, "fear": 5}
INTENSITY_STR_TO_INT = {"low": 1, "mid": 2, "high": 3}
INTENSITY_TO_ID = {1: 0, 2: 1, 3: 2}

# Emotion emojis! 😊
EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "upset": "😔",
    "neutral": "😐"
}

INTENSITY_EMOJIS = {
    1: "⚪",  # Low
    2: "🔵",  # Mid
    3: "🔴"   # High
}

# ============================================================================
# SED Inference (same as before)
# ============================================================================

def run_sed_inference(audio_path, sed_checkpoint, output_dir, chunk_duration=2.0, overlap=0.5):
    print("\n" + "="*70)
    print("🎵 STEP 1: EMOTION DIARIZATION")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_basename = Path(audio_path).stem
    emotion_json = os.path.join(output_dir, f"{audio_basename}_emotions.json")
    
    if os.path.exists(emotion_json):
        print(f"✓ Using existing: {emotion_json}")
        return emotion_json
    
    sed_scripts = ["SED/inference_chunked.py", "inference_chunked.py"]
    sed_script = None
    for script in sed_scripts:
        if os.path.exists(script):
            sed_script = script
            break
    
    if not sed_script:
        print("⚠️ SED script not found")
        return None
    
    cmd = [
        sys.executable, sed_script,
        "--checkpoint", sed_checkpoint,
        "--audio", audio_path,
        "--chunk-duration", str(chunk_duration),
        "--overlap", str(overlap),
        "--output-dir", output_dir,
        "--export-json"
    ]
    
    print(f"🔍 Running emotion diarization...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"⚠️ SED failed")
            return None
    except Exception as e:
        print(f"⚠️ SED error: {e}")
        return None
    
    default_json = os.path.join(output_dir, f"{audio_basename}_chunked.json")
    if os.path.exists(default_json):
        if default_json != emotion_json:
            os.rename(default_json, emotion_json)
        print(f"✅ Generated: {emotion_json}")
        return emotion_json
    
    return None

def load_emotion_segments(emotion_json_path):
    with open(emotion_json_path, 'r') as f:
        data = json.load(f)
    
    segments = []
    for seg in data.get('segments', []):
        emotion_char = seg.get("emotion", "n")
        emotion_name = EMO_CHAR_TO_NAME.get(emotion_char, "neutral")
        jambatalk_emotion = SED_TO_JAMBATALK.get(emotion_name, "neutral")
        intensity_str = seg.get("intensity", "mid")
        intensity_int = INTENSITY_STR_TO_INT.get(intensity_str, 2)
        
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "emotion": jambatalk_emotion,
            "intensity": intensity_int,
        })
    
    segments.sort(key=lambda x: x["start"])
    
    print(f"\n✅ Loaded {len(segments)} emotion segments")
    emotion_counts = {}
    for seg in segments:
        emotion_counts[seg['emotion']] = emotion_counts.get(seg['emotion'], 0) + 1
    
    print("\n📊 Emotion distribution:")
    for emotion in sorted(emotion_counts.keys()):
        emoji = EMOTION_EMOJIS.get(emotion, "❓")
        print(f"  {emoji} {emotion:8s}: {emotion_counts[emotion]:3d} segments")
    
    return segments

# ============================================================================
# CHUNK REDUCTION FUNCTIONS
# ============================================================================

def filter_short_segments(segments, min_duration=1.5):
    """Remove segments shorter than min_duration."""
    if min_duration <= 0:
        return segments
    
    print(f"\n✂️ Filtering segments shorter than {min_duration}s...")
    original_count = len(segments)
    
    filtered = [seg for seg in segments if (seg['end'] - seg['start']) >= min_duration]
    
    removed = original_count - len(filtered)
    print(f"  ❌ Removed {removed} short segments")
    print(f"  ✅ Kept {len(filtered)} segments")
    
    return filtered

def merge_consecutive_same_emotions(segments, gap_threshold=0.5):
    """Merge consecutive segments with same emotion, even across small gaps."""
    if not segments:
        return segments
    
    print(f"\n🔗 Merging consecutive same emotions (gap < {gap_threshold}s)...")
    original_count = len(segments)
    
    merged = []
    current = segments[0].copy()
    
    for seg in segments[1:]:
        gap = seg['start'] - current['end']
        
        # Merge if same emotion and small gap
        if (seg['emotion'] == current['emotion'] and 
            seg['intensity'] == current['intensity'] and 
            gap <= gap_threshold):
            # Extend current segment
            current['end'] = seg['end']
        else:
            # Save current and start new
            merged.append(current)
            current = seg.copy()
    
    # Add last segment
    merged.append(current)
    
    print(f"  🔀 Merged {original_count} → {len(merged)} segments")
    
    return merged

def reduce_chunks(segments, min_segment_duration=0.0, merge_same=False, gap_threshold=0.5):
    """
    Apply chunk reduction strategies.
    
    Args:
        segments: List of emotion segments
        min_segment_duration: Minimum duration to keep (seconds)
        merge_same: Whether to merge consecutive same emotions
        gap_threshold: Maximum gap for merging (seconds)
    
    Returns:
        Reduced segment list
    """
    original_count = len(segments)
    
    if original_count == 0:
        return segments
    
    print("\n" + "="*70)
    print("📉 CHUNK REDUCTION")
    print("="*70)
    print(f"📥 Original segments: {original_count}")
    
    # Step 1: Merge consecutive same emotions (if enabled)
    if merge_same:
        segments = merge_consecutive_same_emotions(segments, gap_threshold)
    
    # Step 2: Filter short segments
    if min_segment_duration > 0:
        segments = filter_short_segments(segments, min_segment_duration)
    
    final_count = len(segments)
    reduction_pct = (original_count - final_count) / original_count * 100 if original_count > 0 else 0
    
    print(f"\n📤 Final: {final_count} segments ({reduction_pct:.1f}% reduction)")
    
    # Show statistics
    if segments:
        durations = [seg['end'] - seg['start'] for seg in segments]
        avg_duration = np.mean(durations)
        print(f"⏱️  Average segment duration: {avg_duration:.2f}s")
    
    print("="*70)
    
    return segments

# ============================================================================
# IMPROVED SMOOTHING - Temporal Filtering on Vertices
# ============================================================================

def smooth_vertices_temporal(vertices, sigma=1.0, window=5):
    """
    Apply temporal Gaussian smoothing to vertex sequences.
    This smooths the motion over time while preserving emotion changes.
    
    Args:
        vertices: (T, V, 3) array of vertices over time
        sigma: Gaussian kernel standard deviation (higher = more smoothing)
        window: Number of frames to consider for smoothing
        
    Returns:
        Smoothed vertices array
    """
    print("\n" + "="*70)
    print("✨ APPLYING TEMPORAL VERTEX SMOOTHING")
    print("="*70)
    print(f"  🎯 Method: Gaussian filter")
    print(f"  📊 Sigma: {sigma}")
    print(f"  🪟 Window: {window} frames")
    
    T, V, _ = vertices.shape
    smoothed = np.copy(vertices)
    
    # Apply Gaussian filter along time axis for each vertex and coordinate
    print(f"  🔄 Smoothing {V} vertices across {T} frames...")
    
    for v in range(V):
        for c in range(3):  # x, y, z
            smoothed[:, v, c] = gaussian_filter1d(
                vertices[:, v, c],
                sigma=sigma,
                mode='nearest'  # Handle boundaries
            )
    
    # Calculate smoothing effect
    diff = np.abs(smoothed - vertices).mean()
    print(f"  📏 Average vertex displacement: {diff:.6f}")
    print("✅ Temporal smoothing applied")
    
    return smoothed

def smooth_vertices_savgol(vertices, window=5, order=2):
    """
    Apply Savitzky-Golay filter for smoother motion preservation.
    Better preserves sharp features while smoothing.
    """
    from scipy.signal import savgol_filter
    
    print("\n" + "="*70)
    print("✨ APPLYING SAVITZKY-GOLAY SMOOTHING")
    print("="*70)
    print(f"  🪟 Window: {window} frames")
    print(f"  📐 Polynomial order: {order}")
    
    T, V, _ = vertices.shape
    smoothed = np.copy(vertices)
    
    # Window must be odd
    if window % 2 == 0:
        window += 1
    
    print(f"  🔄 Smoothing {V} vertices across {T} frames...")
    
    for v in range(V):
        for c in range(3):
            smoothed[:, v, c] = savgol_filter(
                vertices[:, v, c],
                window_length=min(window, T),
                polyorder=min(order, window-1),
                mode='nearest'
            )
    
    print("✅ Savitzky-Golay smoothing applied")
    
    return smoothed

# ============================================================================
# Rendering (same as before)
# ============================================================================

def render_mesh_helper(mesh, t_center):
    camera_params = {
        'c': np.array([400, 400]),
        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])
    }
    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    tri_mesh = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f, process=False)
    
    material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )
    
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=True)
    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    
    camera = pyrender.IntrinsicsCamera(
        fx=camera_params['f'][0], fy=camera_params['f'][1],
        cx=camera_params['c'][0], cy=camera_params['c'][1],
        znear=frustum['near'], zfar=frustum['far']
    )

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0])
    scene.add(camera, pose=camera_pose)
    scene.add(render_mesh, pose=np.eye(4))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    angle = np.pi / 6.0
    
    for rot_axis in [[angle, 0, 0], [-angle, 0, 0], [0, angle, 0], [0, -angle, 0]]:
        light_pose = np.eye(4)
        light_pose[:3, 3] = cv2.Rodrigues(np.array(rot_axis))[0].dot(camera_pose[:3, 3])
        scene.add(light, pose=light_pose)

    r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
    color, _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    r.delete()

    return color[..., ::-1]

def render_video(audio_path, vertices, faces, output_dir):
    print("\n" + "="*70)
    print("🎬 STEP 7: RENDER VIDEO")
    print("="*70)
    
    temp_avi = os.path.join(output_dir, "temp.avi")
    output_mp4 = os.path.join(output_dir, "video.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(temp_avi, fourcc, 30, (800, 800))
    
    if not writer.isOpened():
        print("❌ ERROR: Could not open video writer")
        return None
    
    center = np.mean(vertices[0], axis=0)
    
    print(f"🎨 Rendering {len(vertices)} frames...")
    for i in range(len(vertices)):
        mesh = MeshWrapper(vertices=vertices[i], faces=faces)
        img = render_mesh_helper(mesh, center)
        writer.write(img)
        
        if (i + 1) % 100 == 0:
            print(f"  ⏳ {i + 1}/{len(vertices)}")
    
    writer.release()
    print(f"✅ Rendered frames")
    
    print("🎵 Adding audio...")
    cmd = ['ffmpeg', '-y', '-i', temp_avi, '-i', audio_path,
           '-c:v', 'libx264', '-c:a', 'aac', '-shortest', output_mp4]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and os.path.exists(output_mp4):
            print(f"✅ Video saved: {output_mp4}")
            try:
                os.remove(temp_avi)
            except:
                pass
            return output_mp4
    except:
        pass
    
    print(f"⚠️ Video without audio: {temp_avi}")
    return temp_avi

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Emotion-conditioned animation with improved smoothing')
    
    # All optional with defaults
    parser.add_argument("--wav_path", default=DEFAULT_CONFIG["wav_path"])
    parser.add_argument("--dataset", default=DEFAULT_CONFIG["dataset"])
    parser.add_argument("--model_name", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--template_path", default=DEFAULT_CONFIG["template_path"])
    parser.add_argument("--subject", default=DEFAULT_CONFIG["subject"])
    parser.add_argument("--output_dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--sed_checkpoint", default=DEFAULT_CONFIG["sed_checkpoint"])
    parser.add_argument("--chunk_duration", type=float, default=DEFAULT_CONFIG["chunk_duration"])
    parser.add_argument("--overlap", type=float, default=DEFAULT_CONFIG["overlap"])
    parser.add_argument("--skip_sed", action="store_true", default=DEFAULT_CONFIG["skip_sed"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--feature_dim", type=int, default=DEFAULT_CONFIG["feature_dim"])
    parser.add_argument("--vertice_dim", type=int, default=DEFAULT_CONFIG["vertice_dim"])
    parser.add_argument("--period", type=int, default=DEFAULT_CONFIG["period"])
    parser.add_argument("--save_meshes", action="store_true", default=DEFAULT_CONFIG["save_meshes"])
    
    # Smoothing options
    parser.add_argument("--smooth_sigma", type=float, default=DEFAULT_CONFIG["smooth_sigma"],
                       help="Gaussian smoothing strength (0.5=light, 1.0=medium, 2.0=heavy)")
    parser.add_argument("--smooth_window", type=int, default=DEFAULT_CONFIG["smooth_window"],
                       help="Smoothing window size in frames")
    parser.add_argument("--no_smooth", action="store_true", help="Disable vertex smoothing")
    parser.add_argument("--smooth_method", default="gaussian", choices=["gaussian", "savgol"],
                       help="Smoothing method")
    
    # Chunk reduction options
    parser.add_argument("--min_segment_duration", type=float, default=DEFAULT_CONFIG["min_segment_duration"],
                       help="Minimum segment duration in seconds (e.g., 2.0 to remove short segments)")
    parser.add_argument("--merge_same_emotions", action="store_true", default=DEFAULT_CONFIG["merge_same_emotions"],
                       help="Merge consecutive segments with same emotion")
    parser.add_argument("--merge_gap_threshold", type=float, default=0.5,
                       help="Maximum gap for merging same emotions (seconds)")
    
    # Legacy
    parser.add_argument("--emotions", nargs='+', default=["neutral", "happy", "angry", "sad", "upset", "fear", "disgust"])
    parser.add_argument("--wav_path_arg", default="wav")
    parser.add_argument("--vertices_path", default="sequences")
    
    args = parser.parse_args()
    
    # Validate
    print("Validating inputs...")
    
    if not os.path.exists(args.wav_path):
        print(f"❌ Audio not found: {args.wav_path}")
        return
    print(f"✓ Audio: {args.wav_path}")
    
    flame_template = os.path.join(args.dataset, "FLAME_sample.ply")
    if not os.path.exists(flame_template):
        print(f"❌ FLAME template not found: {flame_template}")
        return
    print(f"✓ FLAME: {flame_template}")
    
    model_path = os.path.join(args.dataset, f"{args.model_name}.pth")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    print(f"✓ Model: {model_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # SED
    segments = []
    if not args.skip_sed:
        emotion_json = run_sed_inference(
            args.wav_path, args.sed_checkpoint, args.output_dir,
            args.chunk_duration, args.overlap
        )
        if emotion_json:
            segments = load_emotion_segments(emotion_json)
            
            # Apply chunk reduction if requested
            if args.min_segment_duration > 0 or args.merge_same_emotions:
                segments = reduce_chunks(
                    segments,
                    min_segment_duration=args.min_segment_duration,
                    merge_same=args.merge_same_emotions,
                    gap_threshold=args.merge_gap_threshold
                )
    
    # Load audio
    print("\n" + "="*70)
    print("STEP 2: LOAD AUDIO")
    print("="*70)
    
    audio, sr = librosa.load(args.wav_path, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"✓ Audio: {duration:.2f}s @ {sr}Hz")
    
    if not segments:
        print("  Using neutral emotion")
        segments = [{"start": 0.0, "end": duration, "emotion": "neutral", "intensity": 2}]
    
    # Load FLAME
    print("\nLoading FLAME template...")
    template_mesh = MeshWrapper(filename=flame_template)
    print(f"✓ FLAME: {template_mesh.v.shape[0]} vertices, {template_mesh.f.shape[0]} faces")
    
    # Load model
    print("\n" + "="*70)
    print("STEP 3: LOAD JAMBATALK")
    print("="*70)
    
    try:
        from jambatalk import JambaTalk
    except ImportError:
        print("❌ Cannot import JambaTalk")
        return
    
    model = JambaTalk(args).to(args.device).eval()
    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint, strict=False)
    print("✓ Model loaded")
    
    with open(os.path.join(args.dataset, args.template_path), "rb") as f:
        templates = pickle.load(f, encoding="latin1")
    template = torch.FloatTensor(templates[args.subject].reshape(1, -1)).to(args.device)
    print(f"✓ Template: {args.subject}")
    
    # Features
    print("\n" + "="*70)
    print("STEP 4: AUDIO FEATURES")
    print("="*70)
    
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    proc_out = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        audio_features = model.audio_encoder(proc_out["input_values"].to(args.device)).last_hidden_state
        vertice_input = model.audio_feature_map(audio_features)
    
    seq_len = vertice_input.shape[1]
    print(f"✓ Features: {vertice_input.shape}")
    
    # Emotion conditioning (NO BLENDING - sharp transitions)
    print("\n" + "="*70)
    print("STEP 5: EMOTION CONDITIONING (SHARP TRANSITIONS)")
    print("="*70)
    
    cond_vec = torch.zeros_like(vertice_input)
    
    # Pre-compute embeddings
    JAMBATALK_EMOTIONS = ["happy", "angry", "sad", "upset", "fear", "disgust"]
    emotion_embeddings = {}
    
    for emo in JAMBATALK_EMOTIONS:
        for intensity in [1, 2, 3]:
            emo_id = torch.tensor([EMO_NAME_TO_ID[emo]], device=args.device)
            int_id = torch.tensor([[INTENSITY_TO_ID[intensity]]], device=args.device, dtype=torch.float32)
            dummy = torch.zeros(1, 1, model.feature_dim, device=args.device)
            emb = model._condition_features(dummy, emo_id, int_id)
            emotion_embeddings[(emo, intensity)] = emb.squeeze(1)
    
    # Apply emotions sharply (no blending)
    intensity_names = {1: "LOW", 2: "MID", 3: "HIGH"}
    print("\nApplying emotions (sharp transitions - smoothing will be on vertices):")
    
    for i, seg in enumerate(segments, 1):
        if seg["emotion"] == "neutral":
            print(f"  Segment {i}: NEUTRAL (no conditioning)")
            continue
        
        start_frame = int(seg["start"] * seq_len / duration)
        end_frame = int(seg["end"] * seq_len / duration)
        emb = emotion_embeddings[(seg["emotion"], seg["intensity"])]
        cond_vec[:, start_frame:end_frame, :] = emb
        
        print(f"  Segment {i}: frames {start_frame:4d}-{end_frame:4d} | "
              f"{seg['emotion'].upper():8s} {intensity_names[seg['intensity']]:4s}")
    
    vertice_input = vertice_input + cond_vec
    print("✓ Emotion conditioning applied (sharp)")
    
    # Generate
    print("\n" + "="*70)
    print("STEP 6: GENERATE ANIMATION")
    print("="*70)
    
    with torch.no_grad():
        vertice_out = model._run_sequence_backbone(vertice_input)
        vertice_out = model.vertice_map_r(vertice_out)
        vertice_out = vertice_out + template
    
    vertices = vertice_out.cpu().numpy().reshape(-1, 5023, 3)
    print(f"✓ Generated {vertices.shape[0]} frames (before smoothing)")
    
    # Apply temporal smoothing on VERTICES
    if not args.no_smooth:
        if args.smooth_method == "gaussian":
            vertices = smooth_vertices_temporal(vertices, sigma=args.smooth_sigma, window=args.smooth_window)
        else:
            vertices = smooth_vertices_savgol(vertices, window=args.smooth_window)
    else:
        print("\n⚠ Vertex smoothing disabled - may have sharp transitions")
    
    # Render
    video_path = render_video(args.wav_path, vertices, template_mesh.f, args.output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("✅ COMPLETED!")
    print("="*70)
    print(f"\n📁 Output: {video_path}")
    if not args.no_smooth:
        print(f"✨ Temporal smoothing applied (sigma={args.smooth_sigma})")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()