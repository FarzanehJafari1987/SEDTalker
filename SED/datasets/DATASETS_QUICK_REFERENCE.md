# Dataset Quick Reference Card

**Quick links for all 11 emotion speech datasets**

## 📥 Instant Download Links

| # | Dataset | Size | Direct Link | Neutral |
|---|---------|------|-------------|---------|
| 1 | **RAVDESS** | 1 GB | https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip | ✓ |
| 2 | **TESS** | 500 MB | https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF | ✓ |
| 3 | **CREMA-D** | 2 GB | https://www.kaggle.com/datasets/ejlok1/cremad | ✓ |
| 4 | **SAVEE** | 200 MB | https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee | ✓ |
| 5 | **EmoV-DB** | 2 GB | https://openslr.org/115/ | ✓ |
| 6 | **JL-Corpus** | 5 GB | https://www.kaggle.com/datasets/tli725/jl-corpus | ✗ |
| 7 | **MELD** | 3 GB | https://affective-meld.github.io/ | ✓ |
| 8 | **ESD** | 3 GB | https://github.com/HLTSingapore/Emotional-Speech-Data | ✓ |

## 🔐 Requires Registration

| # | Dataset | Size | Registration Link | Wait Time | Neutral |
|---|---------|------|-------------------|-----------|---------|
| 9 | **IEMOCAP** | 12 GB | https://sail.usc.edu/iemocap/iemocap_release.htm | 1-2 days | ✓ |

## 📧 Contact Required

| # | Dataset | Size | Contact Method | Neutral |
|---|---------|------|----------------|---------|
| 10 | **AESDD** | 200 MB | https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/ | ✗ |
| 11 | **ZED** | 1 GB | Academic request | ✓ |

---

## 🚀 Quick Download Order

**Start with these (no registration, instant download):**

1. RAVDESS ✓ (1 GB) - https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
2. SAVEE ✓ (200 MB) - https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
3. CREMA-D ✓ (2 GB) - https://www.kaggle.com/datasets/ejlok1/cremad
4. TESS ✓ (500 MB) - https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF

**While downloading, register for:**

5. IEMOCAP (12 GB) - https://sail.usc.edu/iemocap/iemocap_release.htm
   ⏰ Approval takes 1-2 days

**Then continue with:**

6. EmoV-DB ✓ (2 GB) - https://openslr.org/115/
7. JL-Corpus (5 GB) - https://www.kaggle.com/datasets/tli725/jl-corpus
8. MELD ✓ (3 GB) - https://affective-meld.github.io/
9. ESD ✓ (3 GB) - https://github.com/HLTSingapore/Emotional-Speech-Data

**Optional (if accessible):**

10. AESDD (200 MB) - Contact authors
11. ZED (1 GB) - Academic request

---

## 📊 Prioritization by Neutral Samples

**Highest Neutral Contribution:**

1. **IEMOCAP** - ~1,800 neutral ⭐ (Must have!)
2. **CREMA-D** - ~1,300 neutral ⭐ (Instant download)
3. **ESD** - ~1,200 neutral ⭐ (Good quality)
4. **MELD** - ~600 neutral
5. **EmoV-DB** - ~500 neutral
6. **TESS** - ~400 neutral
7. **RAVDESS** - ~380 neutral (needs FIXED script)
8. **ZED** - ~100 neutral
9. **SAVEE** - ~70 neutral

**No Neutral:**
- AESDD (470 samples, no neutral)
- JL-Corpus (7,500 samples, combinations only)

---

## 💾 Storage Planning

**Minimum Setup (Core Datasets):**
- RAVDESS + CREMA-D + ESD + IEMOCAP = ~18 GB
- Provides: ~24,000 samples with ~4,300 neutral
- Training: Possible but limited

**Recommended Setup (9 datasets):**
- All except AESDD, ZED = ~28 GB
- Provides: ~50,000 samples with ~6,200 neutral
- Training: Excellent coverage

**Complete Setup (All 11):**
- All datasets = ~30 GB
- Provides: ~57,500 samples with ~6,350 neutral
- Training: Maximum diversity

---

## 🎯 Download Strategy

**Day 1 (0-2 hours):**
```bash
# Instant downloads - Start these first!
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
# Download CREMA-D from Kaggle (requires Kaggle account)
# Download SAVEE from Kaggle
# Download TESS from Dataverse

# Register for IEMOCAP while downloading
# Go to: https://sail.usc.edu/iemocap/iemocap_release.htm
```

**Day 2-3 (After IEMOCAP approval):**
```bash
# IEMOCAP download link received
# Download IEMOCAP (12 GB)

# Continue with other datasets
# EmoV-DB, JL-Corpus, MELD, ESD
```

**Day 3-4 (Preparation):**
```bash
# Extract all datasets
# Run preparation scripts
# Combine datasets
```

---

## 🔗 All Download Links (Copy-Paste Ready)

```
# Instant downloads:
RAVDESS: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
CREMA-D: https://www.kaggle.com/datasets/ejlok1/cremad
SAVEE: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
TESS: https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF
EmoV-DB: https://openslr.org/115/
JL-Corpus: https://www.kaggle.com/datasets/tli725/jl-corpus
MELD: https://affective-meld.github.io/
ESD: https://github.com/HLTSingapore/Emotional-Speech-Data

# Requires registration:
IEMOCAP: https://sail.usc.edu/iemocap/iemocap_release.htm

# Contact required:
AESDD: https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/
ZED: Academic request
```

---

## ✅ Verification Checklist

After downloading each dataset, verify:

- [ ] RAVDESS: 24 Actor folders, ~850 .wav files
- [ ] CREMA-D: AudioWAV folder, ~7,442 files
- [ ] SAVEE: 4 speaker folders OR flat files, 480 files
- [ ] TESS: OAF/YAF emotion folders, ~1,800 files
- [ ] IEMOCAP: 5 Session folders, ~9,000 files
- [ ] ESD: Speaker folders (0011-0020), ~8,500 files
- [ ] EmoV-DB: Speaker emotion folders, ~2,800 files
- [ ] JL-Corpus: female/male wav files, ~7,500 files
- [ ] MELD: train/dev/test folders + CSV files, ~13,000 files
- [ ] AESDD: .wav files with emotion codes, ~470 files
- [ ] ZED: spk*.wav files with emotion codes

---

## 🆘 Troubleshooting Downloads

**Kaggle datasets require login:**
1. Create free Kaggle account
2. Accept dataset terms
3. Download via browser or Kaggle API

**Zenodo slow download:**
- Try different time of day
- Use download manager (wget, aria2c)
- Direct link: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

**OpenSLR multiple files:**
```bash
# Download all EmoV-DB files:
wget https://openslr.org/resources/115/bea_Amused.tar.gz
wget https://openslr.org/resources/115/bea_Angry.tar.gz
wget https://openslr.org/resources/115/bea_Neutral.tar.gz
# ... etc (14 files total)
```

**IEMOCAP not approved yet:**
- Check spam folder for approval email
- Usually takes 1-2 business days
- Contact: sail@usc.edu if delayed

---

**Total Download Time:** 4-8 hours (depending on speed)
**Total Size:** ~30 GB
**Result:** ~57,500 samples with 7 emotions including neutral! 🎉
