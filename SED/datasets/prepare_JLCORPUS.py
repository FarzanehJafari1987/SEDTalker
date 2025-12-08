"""
Data preparation for JL-Corpus.
Dataset link: https://www.kaggle.com/datasets/tli725/jl-corpus?resource=download
extra dependencies: pathlib, pydub, webrtcvad
Author: Yingzhi Wang 2023
"""

import numpy as np
import os
import random
import json
import uuid
import shutil
import logging
from pathlib import Path
from pydub import AudioSegment
from vad import vad_for_folder

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

repos = ["female1", "female2", "male1", "male2"]
combinations = ["neu_emo", "emo_neu", "neu_emo_neu", "emo_emo"]
probabilities = np.array([0.25, 0.25, 0.25, 0.25])


# -------------------------------------------------
# Main preparation function
# -------------------------------------------------
def prepare_jlcorpus(data_folder, save_json, seed=12):
    """
    Prepares the json files for the JL-CORPUS dataset.
    """
    random.seed(seed)

    if skip(save_json):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # Step 1: VAD and resampling
    logger.info("Applying VAD and resampling ...")

    for repo in repos:
        files = Path(data_folder).rglob(f"{repo}_*.wav")
        destin_folder = os.path.join(data_folder, "processed", repo)
        os.makedirs(destin_folder, exist_ok=True)

        # Copy raw .wav files (excluding processed folder)
        for file in files:
            if file.is_file() and "processed" not in str(file):
                dest_file = os.path.join(destin_folder, os.path.basename(file))
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copyfile(str(file), dest_file)

        # Apply resampling + VAD
        resampling_for_folder(destin_folder, destin_folder)
        vad_for_folder_safe(destin_folder, destin_folder)

    logger.info("VAD and resampling finished")
    logger.info("Start JL-CORPUS concatenation ...")

    data_json = concat_wavs(data_folder, save_json)
    logger.info("JL-CORPUS concatenation finished.")
    return data_json


# -------------------------------------------------
# Audio resampling
# -------------------------------------------------
def resampling_for_folder(in_folder, out_folder):
    """
    Resamples all audios from a folder to 16kHz.
    Skips directories or non-wav files.
    """
    for file_name in os.listdir(in_folder):
        file_path = os.path.join(in_folder, file_name)
        if os.path.isdir(file_path):
            logger.warning(f"Skipped directory during resampling: {file_path}")
            continue

        try:
            sound = AudioSegment.from_file(file_path, format="wav")
            sound = sound.set_frame_rate(16000)
            sound.export(os.path.join(out_folder, file_name), format="wav")
        except Exception as e:
            logger.warning(f"Failed resampling {file_name}: {e}")


# -------------------------------------------------
# Safe VAD wrapper (skips directories)
# -------------------------------------------------
def vad_for_folder_safe(in_path, out_path):
    """
    Wrapper for vad_for_folder that skips directories.
    """
    try:
        vad_for_folder(in_path, out_path)
    except IsADirectoryError as e:
        logger.warning(f"Skipped directory during VAD: {e}")
    except Exception as e:
        logger.warning(f"VAD failed in {in_path}: {e}")


# -------------------------------------------------
# Emotion extraction helper
# -------------------------------------------------
def get_emotion(wav_path):
    """
    Get the emotion of an audio from its filepath
    """
    if "angry" in wav_path:
        return "angry"
    elif "happy" in wav_path or "excited" in wav_path:
        return "happy"
    elif "sad" in wav_path:
        return "sad"

# -------------------------------------------------
# Audio concatenation
# -------------------------------------------------
def concat_wavs(data_folder, save_json):
    """
    Concatenates audios from the same speaker with a randomized structure.
    Uses short UUID filenames to prevent path overflow.
    """
    data_json = {}

    for repo in repos:
        emotion_wavs, neutral_wavs = [], []

        paths = Path(os.path.join(data_folder, "processed", repo))
        for pattern in ["*angry*.wav", "*sad*.wav", "*happy.wav*", "*excited*.wav"]:
            emotion_wavs.extend(map(str, paths.rglob(pattern)))
        neutral_wavs.extend(map(str, paths.rglob("*neutral*.wav")))

        random.shuffle(emotion_wavs)
        random.shuffle(neutral_wavs)
        neutral_wavs *= 10  # duplicate neutral samples for availability

        combine_path = os.path.join(data_folder, "combined", repo)
        os.makedirs(combine_path, exist_ok=True)

        counter = 0
        while emotion_wavs:
            combination = np.random.choice(combinations, p=probabilities.ravel())
            counter += 1
            unique_id = f"{repo}_{uuid.uuid4().hex[:8]}_{counter}"
            out_name = os.path.join(combine_path, f"{unique_id}.wav")

            try:
                if combination == "neu_emo":
                    if len(neutral_wavs) < 1:
                        break
                    neutral_input = AudioSegment.from_wav(neutral_wavs[0])
                    emotion_input = AudioSegment.from_wav(emotion_wavs[0])
                    emotion_input += neutral_input.dBFS - emotion_input.dBFS
                    combined_input = neutral_input + emotion_input

                    combined_input.export(out_name, format="wav")
                    data_json[unique_id] = {
                        "wav": out_name,
                        "duration": len(combined_input) / 1000,
                        "emotion": [{
                            "emo": get_emotion(emotion_wavs[0]),
                            "start": len(neutral_input) / 1000,
                            "end": len(combined_input) / 1000,
                        }],
                    }
                    neutral_wavs = neutral_wavs[1:]
                    emotion_wavs = emotion_wavs[1:]

                elif combination == "emo_neu":
                    if len(neutral_wavs) < 1:
                        break
                    neutral_input = AudioSegment.from_wav(neutral_wavs[0])
                    emotion_input = AudioSegment.from_wav(emotion_wavs[0])
                    neutral_input += emotion_input.dBFS - neutral_input.dBFS
                    combined_input = emotion_input + neutral_input

                    combined_input.export(out_name, format="wav")
                    data_json[unique_id] = {
                        "wav": out_name,
                        "duration": len(combined_input) / 1000,
                        "emotion": [{
                            "emo": get_emotion(emotion_wavs[0]),
                            "start": 0,
                            "end": len(emotion_input) / 1000,
                        }],
                    }
                    neutral_wavs = neutral_wavs[1:]
                    emotion_wavs = emotion_wavs[1:]

                elif combination == "neu_emo_neu":
                    if len(neutral_wavs) < 2:
                        break
                    neutral_input_1 = AudioSegment.from_wav(neutral_wavs[0])
                    neutral_input_2 = AudioSegment.from_wav(neutral_wavs[1])
                    emotion_input = AudioSegment.from_wav(emotion_wavs[0])

                    emotion_input += neutral_input_1.dBFS - emotion_input.dBFS
                    neutral_input_2 += neutral_input_1.dBFS - neutral_input_2.dBFS
                    combined_input = neutral_input_1 + emotion_input + neutral_input_2

                    combined_input.export(out_name, format="wav")
                    data_json[unique_id] = {
                        "wav": out_name,
                        "duration": len(combined_input) / 1000,
                        "emotion": [{
                            "emo": get_emotion(emotion_wavs[0]),
                            "start": len(neutral_input_1) / 1000,
                            "end": (len(neutral_input_1) + len(emotion_input)) / 1000,
                        }],
                    }
                    neutral_wavs = neutral_wavs[2:]
                    emotion_wavs = emotion_wavs[1:]

                else:  # emo_emo
                    emotion_input_1 = AudioSegment.from_wav(emotion_wavs[0])
                    emotion_input_1.export(out_name, format="wav")
                    data_json[unique_id] = {
                        "wav": out_name,
                        "duration": len(emotion_input_1) / 1000,
                        "emotion": [{
                            "emo": get_emotion(emotion_wavs[0]),
                            "start": 0,
                            "end": len(emotion_input_1) / 1000,
                        }],
                    }
                    emotion_wavs = emotion_wavs[1:]

            except Exception as e:
                logger.warning(f"Failed concatenation for {repo}: {e}")
                break

    # Save JSON
    with open(save_json, "w") as outfile:
        json.dump(data_json, outfile, indent=4)
    return data_json


# -------------------------------------------------
# Skip helper
# -------------------------------------------------
def skip(save_json):
    return os.path.isfile(save_json)


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    data_folder = "datasets/JL_corpus"
    save_json = os.path.join(data_folder, "JL_CORPUS.json")
    prepare_jlcorpus(data_folder, save_json)
