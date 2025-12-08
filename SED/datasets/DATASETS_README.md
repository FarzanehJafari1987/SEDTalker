# 📚 Speech Emotion Recognition Datasets - Complete Guide

**9 English emotion speech datasets for training robust SER systems**

---

## 🎯 Dataset Overview

| Dataset | Size | Samples | Emotions | Neutral | Language | Access |
|---------|------|---------|----------|---------|----------|--------|
| **IEMOCAP** | 12 GB | ~11,000 | 10 → 7 | ✅ 1,800 | 🇺🇸 English | Registration |
| **RAVDESS** | 1 GB | 1,440 | 8 → 7 | ✅ 380 | 🇺🇸 English | Instant |
| **CREMA-D** | 2 GB | 7,442 | 6 → 6 | ✅ 1,300 | 🇺🇸 English (Multi-ethnic) | Instant |
| **TESS** | 500 MB | 2,800 | 7 → 7 | ✅ 400 | 🇨🇦 English | Instant |
| **SAVEE** | 200 MB | 480 | 7 → 7 | ✅ 70 | 🇬🇧 English | Instant |
| **ESD** | 3 GB | 10,500 | 5 → 5 | ✅ 1,200 | 🇺🇸/🇨🇳 English+Chinese | Instant |
| **EmoV-DB** | 2 GB | 6,000 | 5 → 4 | ✅ 500 | 🇺🇸 English | Instant |
| **JL-Corpus** | 5 GB | 10,661 | 3 → 3 | ❌ None | 🇳🇿 English | Instant |
| **MELD** | 3 GB | 13,706 | 7 → 7 | ✅ 600 | 🇺🇸 English | Instant |

**Total: ~30 GB, ~64,000 utterances, 7 emotions (angry, disgust, fear, happy, neutral, sad, upset)**

---

## 📥 Download Links

### ✅ Instant Access (No Registration)

**1. RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Link**: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
- **Size**: 1 GB
- **Samples**: 1,440 (24 actors × 60 utterances)
- **Emotions**: calm, happy, sad, angry, fearful, surprise, disgust (+ neutral via script fix)
- **Quality**: Professional studio, acted emotions, sentences
- **Notes**: High-quality, balanced gender, North American English

**2. CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Link**: https://www.kaggle.com/datasets/ejlok1/cremad
- **Size**: 2 GB
- **Samples**: 7,442 utterances
- **Emotions**: anger, disgust, fear, happy, neutral, sad
- **Quality**: Professional actors, multi-ethnic speakers
- **Notes**: Best for diversity, excellent neutral coverage

**3. TESS** (Toronto Emotional Speech Set)
- **Link**: https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF
- **Size**: 500 MB
- **Samples**: 2,800 utterances
- **Emotions**: anger, disgust, fear, happiness, pleasant surprise, sadness, neutral
- **Quality**: Professional recording, 2 female speakers (young & old)
- **Notes**: Clear, high-quality, good for testing age variation

**4. SAVEE** (Surrey Audio-Visual Expressed Emotion)
- **Link**: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
- **Size**: 200 MB
- **Samples**: 480 utterances
- **Emotions**: anger, disgust, fear, happiness, sadness, surprise, neutral
- **Quality**: 4 male speakers, studio recording
- **Notes**: British English, compact dataset

**5. ESD** (Emotional Speech Dataset)
- **Link**: https://github.com/HLTSingapore/Emotional-Speech-Data
- **Size**: 3 GB
- **Samples**: 10,500 (English portion)
- **Emotions**: neutral, happy, angry, sad, surprise
- **Quality**: Professional, parallel Chinese-English
- **Notes**: Download English (0011-0020), filter English utterances only

**6. EmoV-DB** (Emotional Voices Database)
- **Link**: https://openslr.org/115/
- **Size**: 2 GB
- **Samples**: 6,000 utterances
- **Emotions**: amused, anger, disgust, neutral, sleepiness
- **Quality**: 4 speakers, varied emotional contexts
- **Notes**: Download all `.tar.gz` files, unique "sleepiness" emotion (map to neutral/sad)

**7. JL-Corpus** (New Zealand English Emotional Speech)
- **Link**: https://www.kaggle.com/datasets/tli725/jl-corpus
- **Size**: 5 GB
- **Samples**: 10,661 utterances
- **Emotions**: angry, sad, happy (combinations)
- **Quality**: New Zealand English accent
- **Notes**: No neutral, but a large angry/sad/happy dataset

**8. MELD** (Multimodal EmotionLines Dataset)
- **Link**: https://affective-meld.github.io/
- **Size**: 3 GB
- **Samples**: 13,706 utterances
- **Emotions**: neutral, surprise, fear, sadness, joy, disgust, anger
- **Quality**: TV show dialogues (Friends), conversational
- **Notes**: Natural emotions, challenging (background noise), download train/dev/test splits

---

### 🔐 Registration Required (1-2 Days)

**9. IEMOCAP** (Interactive Emotional Dyadic Motion Capture)
- **Link**: https://sail.usc.edu/iemocap/iemocap_release.htm
- **Size**: 12 GB
- **Samples**: ~11,000 utterances
- **Emotions**: neutral, happiness, sadness, anger, frustration, excitement, fear, surprise, disgust, other
- **Quality**: Scripted + improvised, dyadic conversations
- **Registration**: 
  1. Fill form on the website
  2. Academic email required
  3. Approval: 1-2 business days
  4. Check the spam folder for the download link
- **Notes**: ⭐ MUST-HAVE dataset, best quality, extensive annotations, ~1,800 neutral samples

---

## 🚀 Quick Start: Fastest Path to 50K+ Samples

### Option 1: Core 5 (Minimum, ~18 GB, 1 day)
Start training immediately with high-quality datasets:

```bash
# Day 1 Morning: Download (4-6 hours)
1. RAVDESS (1 GB) - https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
2. CREMA-D (2 GB) - https://www.kaggle.com/datasets/ejlok1/cremad  
3. TESS (500 MB) - https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF
4. ESD (3 GB) - https://github.com/HLTSingapore/Emotional-Speech-Data

# While downloading:
5. Register for IEMOCAP - https://sail.usc.edu/iemocap/iemocap_release.htm

# Day 2-3: IEMOCAP approved
6. Download IEMOCAP (12 GB)

Result: ~24,000 samples, ~4,300 neutral ✅
```

### Option 2: Complete 9 (Recommended, ~30 GB, 3-4 days)
Maximum diversity and coverage:

```bash
# Day 1: Instant downloads (6-8 hours)
1-4: Same as Option 1 (RAVDESS, CREMA-D, TESS, ESD)
5. SAVEE (200 MB) - https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
6. EmoV-DB (2 GB) - https://openslr.org/115/
7. JL-Corpus (5 GB) - https://www.kaggle.com/datasets/tli725/jl-corpus
8. MELD (3 GB) - https://affective-meld.github.io/

# Day 1 also: Register IEMOCAP

# Day 2-3: IEMOCAP download
9. IEMOCAP (12 GB)

Result: ~64,000 samples, ~6,200 neutral ✅✅✅
```

---

## 📊 Emotion Coverage by Dataset

### Emotions Available

| Emotion | Datasets | Total Samples | Notes |
|---------|----------|---------------|-------|
| **Neutral** | 7/9 | ~6,200 | IEMOCAP, CREMA-D, ESD best sources |
| **Happy** | 9/9 | ~16,000 | Universal coverage |
| **Sad** | 9/9 | ~10,500 | Universal coverage |
| **Angry** | 9/9 | ~15,000 | Universal coverage |
| **Fear** | 6/9 | ~2,400 | Limited (IEMOCAP, RAVDESS, TESS, CREMA-D, SAVEE) |
| **Disgust** | 5/9 | ~3,000 | Moderate (RAVDESS, CREMA-D, TESS, EmoV-DB, MELD) |
| **Upset/Frustration** | 1/9 | ~3,600 | Only IEMOCAP |

### Emotion Mapping for Your 7-Class System

```python
# Target: [angry, disgust, fear, happy, neutral, sad, upset]

IEMOCAP: anger→angry, frustration→upset, excitement→happy, happiness→happy
RAVDESS: calm→neutral, fearful→fear, surprise→happy
CREMA-D: Direct mapping (6/7 emotions)
TESS: pleasant_surprise→happy
SAVEE: surprise→happy
ESD: surprise→happy
EmoV-DB: amused→happy, sleepiness→neutral
JL-Corpus: Only angry, sad, happy (no neutral/fear/disgust/upset)
MELD: joy→happy, surprise→happy
```

---

## 💾 Storage & Processing Requirements

### Disk Space
- **Download**: ~30 GB (compressed)
- **Extracted**: ~35 GB
- **After Processing**: ~40 GB (includes prepared JSON files)
- **Total Recommended**: 50 GB free space

### Processing Time (Estimates)
- **Download**: 4-8 hours (depends on internet speed)
- **Extraction**: 30-60 minutes
- **Data Preparation**: 2-3 hours (running all scripts)
- **Frame-Level Conversion**: 15-20 minutes
- **Total**: ~1 day end-to-end

---

## 🔧 Download Instructions

### For Direct Links (RAVDESS, TESS, ESD)
```bash
# RAVDESS
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

# Extract
unzip Audio_Speech_Actors_01-24.zip -d datasets/RAVDESS/
```

### For Kaggle Datasets (CREMA-D, SAVEE, JL-Corpus)
```bash
# Option 1: Browser download (easiest)
1. Login to Kaggle
2. Click download button
3. Extract to datasets/ folder

# Option 2: Kaggle API
pip install kaggle
kaggle datasets download -d ejlok1/cremad
kaggle datasets download -d ejlok1/surrey-audiovisual-expressed-emotion-savee
kaggle datasets download -d tli725/jl-corpus
```

### For GitHub Repos (ESD)
```bash
git clone https://github.com/HLTSingapore/Emotional-Speech-Data.git
cd Emotional-Speech-Data
# Download English portion (0011-0020 speakers)
```

### For OpenSLR (EmoV-DB)
```bash
# Download all emotion tar files
wget -r -np -nH --cut-dirs=2 -R "index.html*" https://openslr.org/resources/115/
```

### For MELD
```bash
# Visit: https://affective-meld.github.io/
# Download: train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
# Download: train.tar.gz, dev.tar.gz, test.tar.gz
```

---

## ✅ Verification After Download

**Expected Directory Structure:**
```
datasets/
├── RAVDESS/
│   └── Audio_Speech_Actors_01-24/
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ... (24 folders)
├── CREMA-D/
│   └── AudioWAV/
│       └── *.wav (7,442 files)
├── TESS/
│   ├── OAF_*.wav
│   └── YAF_*.wav (2,800 files)
├── SAVEE/
│   └── *.wav (480 files)
├── IEMOCAP/
│   ├── Session1/
│   ├── Session2/
│   └── ... (5 sessions)
├── ESD/
│   ├── 0011/
│   ├── 0012/
│   └── ... (10 speakers)
├── EmoV-DB/
│   ├── bea/
│   ├── jenie/
│   ├── josh/
│   └── sam/
├── JL_corpus/
│   ├── female1/
│   ├── female2/
│   ├── male1/
│   └── male2/
└── MELD/
    ├── train/
    ├── dev/
    └── test/
```

**File Count Verification:**
```bash
find datasets/RAVDESS -name "*.wav" | wc -l  # Should be ~1440
find datasets/CREMA-D -name "*.wav" | wc -l  # Should be 7442
find datasets/TESS -name "*.wav" | wc -l     # Should be 2800
find datasets/SAVEE -name "*.wav" | wc -l    # Should be 480
find datasets/IEMOCAP -name "*.wav" | wc -l  # Should be ~10000
find datasets/ESD -name "*.wav" | wc -l      # Should be ~10500
find datasets/EmoV-DB -name "*.wav" | wc -l  # Should be ~6000
find datasets/JL_corpus -name "*.wav" | wc -l # Should be ~10661
find datasets/MELD -name "*.mp4" | wc -l     # Should be ~13706
```

---

## 🎯 Recommended Download Priority

**For Training ASAP (Day 1):**
1. ✅ RAVDESS - Quick, high-quality
2. ✅ CREMA-D - Large, diverse
3. ✅ TESS - Clean, balanced
4. ✅ Register IEMOCAP - Start approval process

**For Better Coverage (Day 2-3):**
5. ✅ ESD - Large, neutral-rich
6. ✅ IEMOCAP - Once approved (MUST HAVE)
7. ✅ SAVEE - Quick addition

**For Maximum Performance (Day 3-4):**
8. ✅ JL-Corpus - Large, NZ English
9. ✅ MELD - Natural, conversational
10. ✅ EmoV-DB - Unique emotions

---

## 📞 Support & Troubleshooting

### Common Issues

**Kaggle Login Required:**
- Create free account at kaggle.com
- Accept dataset terms before downloading

**IEMOCAP Delayed Approval:**
- Check spam folder
- Usually 1-2 business days
- Contact: sail@usc.edu if >3 days

**Download Speed Slow:**
- Use download manager (wget, aria2c)
- Try different time of day
- Consider university/institutional network

**Extraction Errors:**
- Verify file integrity (checksum if available)
- Re-download corrupted files
- Use 7-Zip or similar for problematic archives

---

## 🎓 Citation Information

If you use these datasets in your research, please cite:

**IEMOCAP**: Busso et al. (2008) IEMOCAP: Interactive emotional dyadic motion capture database
**RAVDESS**: Livingstone & Russo (2018) Ryerson Audio-Visual Database of Emotional Speech and Song  
**CREMA-D**: Cao et al. (2014) CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
**TESS**: Dupuis & Pichora-Fuller (2010) Toronto Emotional Speech Set
**SAVEE**: Haq & Jackson (2002) Surrey Audio-Visual Expressed Emotion database
**ESD**: Zhou et al. (2021) Emotional Speech Dataset
**EmoV-DB**: Adigwe et al. (2018) Emotional Voices Database
**JL-Corpus**: James & Lech (2014) JL Corpus of emotional speech
**MELD**: Poria et al. (2019) MELD: Multimodal EmotionLines Dataset

---

## 📊 Quick Stats Summary

| Metric | Value |
|--------|-------|
| **Total Datasets** | 9 |
| **Total Size** | ~30 GB |
| **Total Samples** | ~64,000 utterances |
| **Emotions** | 7 (angry, disgust, fear, happy, neutral, sad, upset) |
| **Languages** | English (US, UK, CA, NZ variants) |
| **Neutral Samples** | ~6,200 (9.7% of total) |
| **Training Time** | ~2 hours (RTX 4090) |
| **Expected Accuracy** | 77-79% (7 emotions) |

---

**Ready to download? Start with RAVDESS + CREMA-D + TESS, then register for IEMOCAP! 🚀**

**Questions? Issues? Check the preparation scripts or open an issue on GitHub.**
