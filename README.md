# SEDTalker: Speech-Driven 3D Facial Animation with Emotion Conditioning
### International Conference on Pattern Recognition (ICPR 2026)

Farzaneh Jafari, Stefano Berretti, Anup Basu

[[Paper]]()|[[Project Page]](https://farzanehjafari1987.github.io/SEDTalker.github.io/)|[[License]](https://github.com/FarzanehJafari1987/SEDTalker/blob/main/LICENSE)

<p align="center">
  <img src="SEDTalker.png" alt="SEDTalker Overview" width="100%">
</p>

## Key Features

### 1. Emotion-Conditioned Animation
- **6 Emotions**: (Happy 😊, Sad 😢, Angry 😠, Disgust 🤢, Fear 😨, Upset 😔) + Neutral 😐
- **3 Intensity Levels**: Low (⚪), Medium (🔵), High (🔴)
- **19 Unique Combinations**: (Each emotion × intensity pair creates distinct expressions) + Neutral

### 2. Speech Emotion Diarization
- Automatic emotion detection from audio
- Temporal segmentation with configurable chunk sizes
- Intensity estimation (low/medium/high)
- Chunk reduction for smoother, longer segments

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg (for video rendering with audio)

### Environment Setup

1. **Set up the JambaTalk environment:**

   Follow the installation instructions from the official JambaTalk repository: [JambaTalk GitHub](https://github.com/FarzanehJafari1987/JambaTalk)

2. **Clone this repository:**

   ```bash
   git clone https://github.com/your-repo/SEDTalker.git
   cd SEDTalker
   ```

3. **Install additional dependencies:**

   ```bash
   pip install scipy pyrender opencv-python
   ```

### Download Pre-trained Models

Download the pre-trained [JambaTalk](https://drive.google.com/drive/folders/1LWAjL14QiSdh0RNs_z8L03_oTTQvY-H6) and [SED](https://drive.google.com/drive/folders/1nVlLI1VJ0cFvvA2MEmAUYQa2CN6jnuDw) models.

**Extract and organize:**

```bash
# Extract the downloaded models,
unzip models.zip

# Expected structure:
# SEDTalker/
# ├── EmoVOCA/
# │   ├── save/
# │   │   └── 50_model.pth          # JambaTalk model trained on EmoVOCA
# │   ├── templates.pkl
# │   └── FLAME_sample.ply
# └── SED/
#     └── results/
#         └── emotion_diarization_7class/
#             └── save/CKPT+epoch_50/
#                 └── model.ckpt    # SED model
```

## Training Your Own Model

### Training

```bash
python train.py \
  --dataset EmoVOCA \
  --lr 0.0001 \
  --max_epoch 100 \
  --feature_dim 512 \
  --device cuda
```

**Training Features:**
- Added Emotion-conditioned generation
- Added 3 intensity levels per emotion

### Testing

```bash
python test.py \
  --dataset EmoVOCA \
  --save_path save \
  --max_epoch 50 \
  --test_emotion Smile2 \
  --test_intensity 3
```

---

## Emotion System

### Emotion Mappings

| SED Output | JambaTalk | Emoji | Description |
|------------|-----------|-------|-------------|
| h | happy | 😊 | Joyful, smiling |
| s | sad | 😢 | Sorrowful, downcast |
| a | angry | 😠 | Frustrated, tense |
| d | disgust | 🤢 | Repulsed, negative |
| f | fear | 😨 | Afraid, anxious |
| u | upset | 😔 | Disappointed, troubled |
| n | neutral | 😐 | Baseline, calm |

### Intensity Visualization

- **████** High (3) 🔴 - Maximum expression strength
- **▄▄▄▄** Medium (2) 🔵 - Moderate expression
- **▁▁▁▁** Low (1) ⚪ - Subtle expression

---

## Citation

If you use SEDTalker in your research, please cite:

```bibtex
@article{sedtalker2025jafari,
  title={SEDTalker: Speech-Driven 3D Facial Animation with Emotion Conditioning},
  author={Farzaneh Jafari, Stefano Berretti, Anup Basu},
  journal={arXiv preprint},
  year={2026}
}

@misc{jambatalk2026jafari,
 title={JambaTalk: Speech-driven 3D Talking Head Generation based on a Hybrid Transformer-Mamba Model},
 author={Farzaneh Jafari, Stefano Berretti, Anup Basu},
 note={Transactions on Multimedia Computing, Communications, and Applications},
 doi={10.1145/3793196},
 year={2026}
}
```

---

## Acknowledgments

- [**JambaTalk**](https://github.com/FarzanehJafari1987/JambaTalk): Hybrid Transformer-Mamba architecture for facial animation
- [**EmoVOCA**](https://arxiv.org/abs/2403.12886): Emotional speech dataset

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
