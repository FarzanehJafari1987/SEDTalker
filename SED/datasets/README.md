# 📚 Speech Emotion Recognition Datasets - Complete Guide

**9 English emotion speech datasets for training robust SER systems**

---

## 🎯 Dataset Overview

| Dataset | Size | Samples | Language | Access |
|---------|------|---------|----------|---------|
| **IEMOCAP** | 12 GB | ~11,000 | 🇺🇸 English | Registration |
| **RAVDESS** | 1 GB | 1,440 | 🇺🇸 English | Instant |
| **CREMA-D** | 2 GB | 7,442 | 🇺🇸 English (Multi-ethnic) | Instant |
| **TESS** | 500 MB | 2,800 | 🇨🇦 English | Instant |
| **SAVEE** | 200 MB | 480 | 🇬🇧 English | Instant |
| **ESD** | 3 GB | 10,500 | 🇺🇸/🇨🇳 English+Chinese | Instant |
| **EmoV-DB** | 2 GB | 6,000 | 🇺🇸 English | Instant |
| **JL-Corpus** | 5 GB | 10,661 | 🇳🇿 English | Instant |
| **MELD** | 3 GB | 13,706 | 🇺🇸 English | Instant |

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

## 💾 Storage & Processing Requirements

### Disk Space
- **Download**: ~30 GB (compressed)
- **Extracted**: ~35 GB
- **After Processing**: ~40 GB (includes prepared JSON files)
- **Total Recommended**: 50 GB free space

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
