import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa

# Emotion/Intensity mappings
EMO_NAME_TO_ID = {"happy": 0, "sad": 1, "angry": 2, "upset": 3, "disgust": 4, "fear": 5}
INTENSITY_TO_ID = {1: 0, 2: 1, 3: 2} 


class Dataset(data.Dataset):
    """Custom Dataset with lazy loading and audio caching."""

    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.audio_cache = {}  # Cache for loaded audio files

    def __getitem__(self, index):
        file_name = self.data[index]["name"]
        
        # Get base audio path (shared across emotions)
        audio_path = self.data[index]["audio_path"]
        
        # Check if audio is already cached
        if audio_path not in self.audio_cache:
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
            audio = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)
            self.audio_cache[audio_path] = audio
        else:
            audio = self.audio_cache[audio_path]
        
        # Lazy load vertices
        vertice_path = self.data[index]["vertice_path"]
        vertice = np.load(vertice_path, allow_pickle=True)[::2, :]
        
        template = self.data[index]["template"]
        emotion = self.data[index].get("emotion", "happy")
        intensity = int(self.data[index].get("intensity", 2))

        # Map to IDs
        emotion_id = EMO_NAME_TO_ID.get(emotion, 0)
        intensity_id = INTENSITY_TO_ID.get(intensity, 2)

        return (torch.FloatTensor(audio),
                torch.FloatTensor(vertice),
                torch.FloatTensor(template),
                torch.tensor(emotion_id, dtype=torch.long),
                torch.tensor(intensity_id, dtype=torch.float32),
                file_name)

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data, valid_data, test_data = [], [], []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)

    emotions = args.emotions
    intensities = args.intensities
    template_file = os.path.join(args.dataset, args.template_file)

    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    # Dictionary to map base sentence ID to the first audio file found
    # Key: base_id (e.g., "FaceTalk_170725_00137_TA_sentence01")
    # Value: path to one audio file for that sentence
    base_audio_map = {}
    
    valid_files = []
    
    # First pass: scan audio files and build base audio map
    print("Scanning audio files...")
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs, desc="Processing audio"):
            if f.endswith("wav"):
                parts = f.replace(".wav", "").split("_")
                # Extract base ID (everything except last 2 parts: emotion and intensity)
                base_id = "_".join(parts[:-2])
                
                # Store the first audio file found for this base sentence
                if base_id not in base_audio_map:
                    base_audio_map[base_id] = os.path.join(r, f)
    
    print(f"Found {len(base_audio_map)} unique base sentences")

    # Second pass: scan vertices and match with base audio
    print("Scanning vertices and matching with audio...")
    for r, ds, fs in os.walk(vertices_path):
        for f in tqdm(fs, desc="Processing vertices"):
            if f.endswith("npy"):
                key = f
                
                # Parse emotion and intensity from filename
                parts = f.replace(".npy", "").split("_")
                emotion = parts[-2]
                intensity = int(parts[-1])
                
                # Get base ID for this file
                base_id = "_".join(parts[:-2])

                if args.dataset == "EmoVOCA" and emotion in emotions and str(intensity) in intensities:
                    # Check if we have a base audio for this sentence
                    if base_id in base_audio_map:
                        vertice_path = os.path.join(r, f)
                        base_audio_path = base_audio_map[base_id]
                        
                        valid_files.append((f.replace("npy", "wav"), base_audio_path, vertice_path, emotion, intensity, key))

    # Third pass: build data dict
    print(f"Found {len(valid_files)} valid samples. Building dataset...")
    for f, base_audio_path, vertice_path, emotion, intensity, key in tqdm(valid_files, desc="Building dataset"):
        # Store base audio path (shared across all emotions of same sentence)
        data[key]["audio_path"] = base_audio_path
        data[key]["vertice_path"] = vertice_path
        
        subject_id = "_".join(key.split("_")[:-3])
        temp = templates[subject_id]

        data[key]["name"] = f
        data[key]["template"] = temp.reshape((-1))
        data[key]["intensity"] = intensity

        # Normalize emotion names
        if emotion == "Irritated1":
            data[key]["emotion"] = "angry"
        elif emotion == "Smile2":
            data[key]["emotion"] = "happy"
        elif emotion == "Sad1":
            data[key]["emotion"] = "sad"
        elif emotion == "Upset":
            data[key]["emotion"] = "upset"
        elif emotion == "Afraid":
            data[key]["emotion"] = "fear"
        elif emotion == "Disgust":
            data[key]["emotion"] = "disgust"

    subjects_dict = {
        "train": [i for i in args.train_subjects.split(" ")],
        "val": [i for i in args.val_subjects.split(" ")],
        "test": [i for i in args.test_subjects.split(" ")]
    }

    splits = {'EmoVOCA': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)}}

    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-3])
        sentence_id = int((k.split(".")[0]).split("_")[-3][-2:])

        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    
    # Use num_workers=2 for parallel data loading
    dataset["train"] = data.DataLoader(
        Dataset(train_data, subjects_dict, "train"), 
        batch_size=1, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    dataset["valid"] = data.DataLoader(
        Dataset(valid_data, subjects_dict, "val"), 
        batch_size=1, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    dataset["test"] = data.DataLoader(
        Dataset(test_data, subjects_dict, "test"), 
        batch_size=1, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return dataset