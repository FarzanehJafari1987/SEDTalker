# Emotion Speech Datasets - Complete Guide

This document provides detailed information about all 11 emotion speech datasets used for training the 7-emotion diarization system with neutral support.

---

## 📊 Dataset Overview

| Dataset | Samples | Languages | Emotions | Neutral | Download Size |
|---------|---------|-----------|----------|---------|---------------|
| **IEMOCAP** | ~9,000 | English (US) | 8 | ✓ Yes | ~12 GB |
| **RAVDESS** | ~850 | English (US) | 8 | ✓ Yes | ~1 GB |
| **ESD** | ~8,500 | English, Chinese | 4 | ✓ Yes | ~3 GB |
| **EmoV-DB** | ~2,800 | English (US) | 4 | ✓ Yes | ~2 GB |
| **JL-Corpus** | ~7,500 | English (NZ) | 4 | ✗ No | ~5 GB |
| **TESS** | ~1,800 | English (US) | 7 | ✓ Yes | ~500 MB |
| **AESDD** | ~470 | Greek | 5 | ✗ No | ~200 MB |
| **CREMA-D** | ~5,800 | English (US) | 6 | ✓ Yes | ~2 GB |
| **MELD** | ~4,500 | English (US) | 7 | ✓ Yes | ~3 GB |
| **ZED** | varies | French | 7 | ✓ Yes | ~1 GB |
| **SAVEE** | 480 | English (UK) | 7 | ✓ Yes | ~200 MB |
| **TOTAL** | **~51,500** | Multiple | Various | **9/11** | **~30 GB** |

**Neutral Support:** 9 out of 11 datasets include neutral emotion (~6,350 samples)

---

## 📥 Dataset Download Links & Instructions

### 1. IEMOCAP (Interactive Emotional Dyadic Motion Capture)

**Details:**
- **Samples:** ~9,000 utterances
- **Speakers:** 10 actors (5 male, 5 female)
- **Emotions:** neutral, happiness, sadness, anger, surprise, fear, disgust, frustration (upset)
- **Language:** English (US)
- **Sessions:** 5 sessions with scripted and improvised dialogs
- **Neutral:** ✓ Yes (~1,800 samples)

**Download:**
- **Official:** https://sail.usc.edu/iemocap/iemocap_release.htm
- **Access:** Requires academic registration and approval (1-2 days)
- **Size:** ~12 GB

**Directory Structure:**
```
IEMOCAP/
├── Session1/
│   ├── dialog/
│   │   └── EmoEvaluation/*.txt
│   └── sentences/wav/
├── Session2/
├── Session3/
├── Session4/
└── Session5/
```

**Preparation:**
```bash
python prepare_IEMOCAP.py --data_folder datasets/IEMOCAP
```

---

### 2. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

**Details:**
- **Samples:** ~850 utterances (speech only)
- **Speakers:** 24 actors (12 male, 12 female)
- **Emotions:** neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Language:** English (US - North American)
- **Neutral:** ✓ Yes (~380 samples with codes 01 and 02)

**Download:**
- **Zenodo:** https://zenodo.org/record/1188976
- **Direct Link:** https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
- **Size:** ~1 GB

**File Format:** `03-01-06-01-02-01-12.wav`
- Modality: 03 = audio-visual, 01 = audio-only
- Vocal channel: 01 = speech, 02 = song
- Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprised
- Emotional intensity: 01=normal, 02=strong
- Statement: 01 or 02
- Repetition: 01 = 1st, 02 = 2nd
- Actor: 01 to 24 (odd=male, even=female)

**Directory Structure:**
```
RAVDESS/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav  (neutral)
│   ├── 03-01-03-01-01-01-01.wav  (happy)
│   └── ...
├── Actor_02/
└── ... (up to Actor_24)
```

**Preparation:**
```bash
python prepare_RAVDESS_FIXED.py --data_folder datasets/RAVDESS
# Use FIXED version to extract neutral emotions (codes 01 and 02)
```

---

### 3. ESD (Emotional Speech Dataset)

**Details:**
- **Samples:** ~8,500 utterances
- **Speakers:** 10 English, 10 Chinese
- **Emotions:** neutral, happy, angry, sad
- **Languages:** English (US), Mandarin Chinese
- **Neutral:** ✓ Yes (~1,200 samples)

**Download:**
- **GitHub:** https://github.com/HLTSingapore/Emotional-Speech-Data
- **Google Drive:** https://drive.google.com/file/d/1_0qvBnrrDPWqFiGi4JThm0X5fYTu-Fm4/view
- **Size:** ~3 GB

**Directory Structure:**
```
ESD/
├── 0011/ (English speakers: 0011-0020)
│   ├── Angry/
│   │   ├── train/
│   │   ├── evaluation/
│   │   └── test/
│   ├── Happy/
│   ├── Neutral/
│   └── Sad/
├── 0012/
└── ... (up to 0020)
```

**Preparation:**
```bash
python prepare_ESD.py --data_folder datasets/ESD
```

---

### 4. EmoV-DB (Emotional Voices Database)

**Details:**
- **Samples:** ~2,800 utterances
- **Speakers:** 4 (bea, jenie, josh, sam)
- **Emotions:** neutral, amused (happy), angry, disgusted
- **Language:** English (US)
- **Neutral:** ✓ Yes (~500 samples)

**Download:**
- **OpenSLR:** https://openslr.org/115/
- **Direct Links:**
  - bea_Amused.tar.gz, bea_Angry.tar.gz, bea_Neutral.tar.gz, bea_Disgusted.tar.gz
  - jenie_Amused.tar.gz, jenie_Angry.tar.gz, jenie_Neutral.tar.gz, jenie_Disgusted.tar.gz
  - josh_Amused.tar.gz, josh_Neutral.tar.gz
  - sam_Amused.tar.gz, sam_Angry.tar.gz, sam_Neutral.tar.gz, sam_Disgusted.tar.gz
- **Size:** ~2 GB (all tar.gz files)

**Directory Structure:**
```
EmoV-DB/
├── bea_Amused/
│   └── *.wav
├── bea_Angry/
├── bea_Neutral/
├── bea_Disgusted/
├── jenie_Amused/
├── jenie_Angry/
├── jenie_Neutral/
├── jenie_Disgusted/
├── josh_Amused/
├── josh_Neutral/
├── sam_Amused/
├── sam_Angry/
├── sam_Neutral/
└── sam_Disgusted/
```

**Preparation:**
```bash
python prepare_EMOVDB.py --data_folder datasets/EmoV-DB
```

---

### 5. JL-Corpus

**Details:**
- **Samples:** ~7,500 combined utterances
- **Speakers:** 4 (female1, female2, male1, male2)
- **Emotions:** neutral (in combinations), happy, angry, sad
- **Language:** English (New Zealand)
- **Neutral:** ℹ️ Used in combinations (not standalone)

**Download:**
- **Kaggle:** https://www.kaggle.com/datasets/tli725/jl-corpus
- **Size:** ~5 GB

**Directory Structure:**
```
JL_corpus/
├── female1_angry_*.wav
├── female1_happy_*.wav
├── female1_neutral_*.wav
├── female1_sad_*.wav
├── female2_*/
├── male1_*/
└── male2_*/
```

**Note:** This dataset creates synthetic combinations (neu_emo, emo_neu, neu_emo_neu, emo_emo)

**Preparation:**
```bash
python prepare_JLCORPUS.py --data_folder datasets/JL_corpus
```

---

### 6. TESS (Toronto Emotional Speech Set)

**Details:**
- **Samples:** ~1,800 utterances
- **Speakers:** 2 actresses (OAF - older, YAF - younger)
- **Emotions:** angry, disgust, fear, happy, pleasant surprise (ps), sad, neutral
- **Language:** English (Canadian)
- **Neutral:** ✓ Yes (~400 samples)

**Download:**
- **Official:** https://tspace.library.utoronto.ca/handle/1807/24487
- **Dataverse:** https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF
- **Size:** ~500 MB

**Directory Structure:**
```
TESS/
├── OAF_angry/
│   ├── OAF_back_angry.wav
│   └── ...
├── OAF_disgust/
├── OAF_fear/
├── OAF_happy/
├── OAF_ps/
├── OAF_sad/
├── OAF_neutral/
├── YAF_angry/
├── YAF_disgust/
├── YAF_fear/
├── YAF_happy/
├── YAF_ps/
├── YAF_sad/
└── YAF_neutral/
```

**Preparation:**
```bash
python prepare_TESS.py --data_folder datasets/TESS
```

---

### 7. AESDD (Acted Emotional Speech Dynamic Database)

**Details:**
- **Samples:** ~470 utterances
- **Speakers:** 5 actors (3 male, 2 female)
- **Emotions:** anger, disgust, fear, happiness, sadness
- **Language:** Greek
- **Neutral:** ✗ No

**Download:**
- **Official:** https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/
- **Direct:** Contact authors for access
- **Size:** ~200 MB

**File Format:**
- `a01.wav` = anger
- `d01.wav` = disgust
- `f01.wav` = fear
- `h01.wav` = happiness
- `s01.wav` = sadness

**Directory Structure:**
```
AESDD/
├── a01.wav
├── a02.wav
├── d01.wav
├── f01.wav
├── h01.wav
├── s01.wav
└── ...
```

**Preparation:**
```bash
python prepare_AESDD.py --data_folder datasets/AESDD
```

---

### 8. CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

**Details:**
- **Samples:** ~7,442 utterances
- **Speakers:** 91 actors (48 male, 43 female)
- **Emotions:** anger (ANG), disgust (DIS), fear (FEA), happy (HAP), neutral (NEU), sad (SAD)
- **Language:** English (US)
- **Neutral:** ✓ Yes (~1,300 samples)

**Download:**
- **GitHub:** https://github.com/CheyneyComputerScience/CREMA-D
- **Kaggle:** https://www.kaggle.com/datasets/ejlok1/cremad
- **Direct:** https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV.zip
- **Size:** ~2 GB

**File Format:** `1001_DFA_ANG_XX.wav`
- 1001 = Actor ID
- DFA = Sentence ID
- ANG = Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
- XX = Intensity (LO, MD, HI, XX)

**Directory Structure:**
```
CREMA-D/
└── AudioWAV/
    ├── 1001_DFA_ANG_XX.wav
    ├── 1001_DFA_NEU_XX.wav
    └── ...
```

**Preparation:**
```bash
python prepare_CREMAD.py --data_folder datasets/CREMA-D
```

---

### 9. MELD (Multimodal EmotionLines Dataset)

**Details:**
- **Samples:** ~13,000 utterances
- **Speakers:** ~304 from Friends TV show
- **Emotions:** neutral, joy (happy), surprise, anger, sadness, disgust, fear
- **Language:** English (US)
- **Neutral:** ✓ Yes (~600 samples)

**Download:**
- **Official:** https://affective-meld.github.io/
- **GitHub:** https://github.com/declare-lab/MELD
- **Size:** ~3 GB

**Directory Structure:**
```
MELD/
├── train/
│   ├── dia0_utt0.wav
│   └── ...
├── dev/
├── test/
├── train_sent_emo.csv
├── dev_sent_emo.csv
└── test_sent_emo.csv
```

**Preparation:**
```bash
python prepare_MELD.py --data_folder datasets/MELD
```

---

### 10. ZED (Zurich Emotional Database)

**Details:**
- **Samples:** Varies by version
- **Speakers:** Multiple
- **Emotions:** 7 emotions including neutral
- **Language:** French/multilingual
- **Neutral:** ✓ Yes (code 7)

**Download:**
- **Contact:** Usually requires academic request
- **Alternative sources:** Check academic repositories
- **Size:** ~1 GB

**File Format:** `spk1_1.wav`
- spk1 = Speaker ID
- 1 = Emotion code (1=angry, 2=disgust, 3=fear, 4=happy, 5=sad, 7=neutral)

**Directory Structure:**
```
ZED/
├── spk1_1.wav
├── spk1_4.wav
├── spk1_7.wav  (neutral)
└── ...
```

**Preparation:**
```bash
python prepare_ZED.py --data_folder datasets/ZED
```

---

### 11. SAVEE (Surrey Audio-Visual Expressed Emotion)

**Details:**
- **Samples:** 480 utterances
- **Speakers:** 4 male (DC, JE, JK, KL)
- **Emotions:** anger, disgust, fear, happiness, neutral, sadness, surprise
- **Language:** English (British)
- **Neutral:** ✓ Yes (~60-70 samples)

**Download:**
- **Official:** http://kahlan.eps.surrey.ac.uk/savee/
- **Kaggle:** https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
- **Size:** ~200 MB

**File Format:**
- Flat structure: `DC_a01.wav`, `DC_n01.wav`
- Or folders: `DC/a01.wav`, `DC/n01.wav`

**Emotion Codes:**
- a = anger
- d = disgust
- f = fear
- h = happiness
- n = neutral
- sa = sadness
- su = surprise

**Directory Structure (Option 1):**
```
SAVEE/
├── DC/
│   ├── a01.wav
│   ├── n01.wav
│   └── ...
├── JE/
├── JK/
└── KL/
```

**Directory Structure (Option 2 - Flat):**
```
SAVEE/
├── DC_a01.wav
├── DC_n01.wav
├── JE_a01.wav
└── ...
```

**Preparation:**
```bash
python prepare_SAVEE.py --data_folder datasets/SAVEE
# Automatically detects structure!
```

---

## 🎯 Complete Dataset Setup

### Step 1: Create Directory Structure

```bash
mkdir -p datasets
cd datasets

# Create folders for each dataset
mkdir IEMOCAP RAVDESS ESD EmoV-DB JL_corpus TESS AESDD CREMA-D MELD ZED SAVEE
```

### Step 2: Download All Datasets

Follow the download links above for each dataset. Estimated total download: **~30 GB**

### Step 3: Organize Files

Extract all datasets to their respective folders:

```
datasets/
├── IEMOCAP/        (~12 GB)
├── RAVDESS/        (~1 GB)
├── ESD/            (~3 GB)
├── EmoV-DB/        (~2 GB)
├── JL_corpus/      (~5 GB)
├── TESS/           (~500 MB)
├── AESDD/          (~200 MB)
├── CREMA-D/        (~2 GB)
├── MELD/           (~3 GB)
├── ZED/            (~1 GB)
└── SAVEE/          (~200 MB)
```

### Step 4: Prepare All Datasets

Run preparation scripts to generate JSON files:

```bash
# Datasets that already handle neutral correctly:
python prepare_IEMOCAP.py
python prepare_ESD.py
python prepare_EMOVDB.py
python prepare_CREMAD.py
python prepare_MELD.py
python prepare_TESS.py
python prepare_JLCORPUS.py
python prepare_ZED.py
python prepare_SAVEE.py

# RAVDESS needs FIXED version for neutral:
python prepare_RAVDESS_FIXED.py

# AESDD has no neutral:
python prepare_AESDD.py
```

### Step 5: Combine All Datasets

```bash
# Combines all 11 datasets with 7 emotions including neutral
python data_preparation_7emotions.py
```

**Output:** `datasets/processed_emotions_7class/`
- `train.json` (~40,250 samples)
- `valid.json` (~8,625 samples)
- `test.json` (~8,625 samples)
- `class_weights.pt`

### Step 6: Prepare Frame-Level Data

```bash
# Edit prepare_frame_level_data.py line 54:
# data_folder = "datasets/processed_emotions_7class"

python prepare_frame_level_data.py
```

### Step 7: Train Model

```bash
python train_frame_level_7emotions.py
```

---

## 📊 Final Dataset Statistics

After combining all datasets:

| Emotion | Samples | Percentage |
|---------|---------|------------|
| **happy** | ~16,200 | 28.2% |
| **sad** | ~10,800 | 18.8% |
| **angry** | ~15,400 | 26.8% |
| **disgust** | ~3,100 | 5.4% |
| **fear** | ~2,500 | 4.3% |
| **upset** | ~3,600 | 6.3% |
| **neutral** | ~6,350 | 11.0% |
| **TOTAL** | **~57,500** | **100%** |

**Neutral sources breakdown:**
- IEMOCAP: ~1,800
- ESD: ~1,200
- CREMA-D: ~1,300
- MELD: ~600
- EmoV-DB: ~500
- TESS: ~400
- RAVDESS: ~380
- ZED: ~100
- SAVEE: ~70

---

## 🔍 Dataset Quality Notes

**Highest Quality (Studio recordings):**
- RAVDESS, TESS, SAVEE, ESD

**Natural Conversations:**
- IEMOCAP, MELD

**Diverse Speakers:**
- CREMA-D (91 speakers)
- IEMOCAP (10 speakers, improvisations)

**Multiple Languages:**
- ESD (English + Chinese)
- AESDD (Greek)
- ZED (French)
- Others (English variants: US, UK, Canadian, NZ)

**Best for Neutral:**
- IEMOCAP, ESD, CREMA-D (large neutral samples)

---

## ⚠️ Important Notes

1. **IEMOCAP Access:** Requires academic email and approval (1-2 days wait)
2. **RAVDESS:** Use `prepare_RAVDESS_FIXED.py` to extract neutral (codes 01, 02)
3. **ESD:** Use the old version linked above (new version has different structure)
4. **SAVEE:** Script auto-detects flat or folder structure
5. **AESDD:** No neutral emotion available in this dataset
6. **ZED:** May require academic request for access

---

## 📚 Citation

If you use these datasets, please cite the original papers:

**IEMOCAP:** Busso et al. (2008)
**RAVDESS:** Livingstone & Russo (2018)
**ESD:** Zhou et al. (2021)
**EmoV-DB:** Adigwe et al. (2018)
**JL-Corpus:** James & Zhang (2019)
**TESS:** Dupuis & Pichora-Fuller (2010)
**AESDD:** Vrysis et al. (2019)
**CREMA-D:** Cao et al. (2014)
**MELD:** Poria et al. (2019)
**SAVEE:** Haq & Jackson (2011)

---

## 🆘 Troubleshooting

**Problem:** Download links not working
- Try alternative sources (Kaggle, GitHub)
- Contact dataset authors

**Problem:** Files won't extract
- Check file integrity (MD5/SHA)
- Use different extraction tool
- Re-download if corrupted

**Problem:** Preparation script fails
- Check directory structure matches expected format
- Verify all required files are present
- Check file permissions

**Problem:** Not enough disk space
- ~30 GB needed for raw datasets
- Additional ~5 GB for processed data
- Total: ~35 GB recommended

---

## ✅ Quick Start Checklist

- [ ] Create `datasets/` directory
- [ ] Download all 11 datasets (~30 GB)
- [ ] Extract to correct folders
- [ ] Run all preparation scripts
- [ ] Verify JSON files created
- [ ] Run `data_preparation_7emotions.py`
- [ ] Check total: ~57,500 samples
- [ ] Verify neutral: ~6,350 samples
- [ ] Run `prepare_frame_level_data.py`
- [ ] Start training!

---

**Total Time to Setup:** 4-6 hours (depending on download speed)
**Total Storage Needed:** ~35 GB
**Expected Combined Samples:** ~57,500
**Neutral Samples:** ~6,350 (11%)

Good luck! 🚀
