import os
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Path dataset
DATASET_PATH = "./dataset"
WAV_PATH = os.path.join(DATASET_PATH, "wav_dataset")
TRANSCRIPT_PATH = os.path.join(DATASET_PATH, "transcripts")

# Output paths
OUTPUT_PATH = "./processed"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def load_dataset():
    data = []
    for wav_file in tqdm(os.listdir(WAV_PATH)):
        if wav_file.endswith(".wav"):
            transcript_file = wav_file.replace(".wav", ".txt")
            transcript_path = os.path.join(TRANSCRIPT_PATH, transcript_file)
            if os.path.exists(transcript_path):
                with open(transcript_path, "r") as f:
                    transcript = f.read().strip()
                if transcript:  
                    data.append((os.path.join(WAV_PATH, wav_file), transcript))
    return pd.DataFrame(data, columns=["audio_path", "transcript"])

def preprocess_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def save_processed_data(df):
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    train.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_PATH, "val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False)

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Found {len(dataset)} samples with non-empty transcripts.")
    print("Saving splits...")
    save_processed_data(dataset)
    print("Preprocessing completed.")
