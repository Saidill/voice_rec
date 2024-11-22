import os
import librosa
import tensorflow as tf
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from jiwer import wer, cer  

# Paths
PROCESSED_PATH = "./processed"
MODEL_PATH = "./models/final_model"

# Function to load dataset
def load_data(split):
    data_path = os.path.join(PROCESSED_PATH, f"{split}.csv")
    df = pd.read_csv(data_path)
    return df["audio_path"].tolist(), df["transcript"].tolist()

# Prepare dataset
def prepare_dataset(audio_paths, transcripts, processor, target_sr=16000, max_length=448):
    input_features = []
    labels = []
    
    for audio_path, transcript in tqdm(zip(audio_paths, transcripts), total=len(transcripts)):
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=None)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Convert audio to features
            input_feature = processor.feature_extractor(
                audio, 
                sampling_rate=target_sr, 
                return_tensors="tf"
            ).input_features[0]

            # Prepare label with special tokens
            label = processor.tokenizer(
                transcript, 
                return_tensors="tf", 
                max_length=max_length, 
                padding="max_length", 
                truncation=True
            ).input_ids[0]

            input_features.append(input_feature)
            labels.append(label)
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    # Convert to TensorFlow tensors
    input_features = tf.stack(input_features)
    labels = tf.stack(labels)
    
    return input_features, labels

# Plot training history
def plot_history(history, output_path="training_history.png"):
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')

    # Add labels, legend, and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save plot to file
    plt.savefig(output_path)
    print(f"Training history saved to {output_path}")

# Custom callback to calculate WER and CER
class WerCerCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, processor):
        self.val_dataset = val_dataset
        self.processor = processor

    def on_epoch_end(self, epoch, logs=None):
        decoded_preds = []
        decoded_labels = []

        # Loop over the validation dataset
        for batch in self.val_dataset:
            input_features = batch["input_features"]
            labels = batch["labels"]

            # Predict
            preds = self.model.generate(input_features)

            # Decode predictions and labels
            decoded_preds.extend(self.processor.tokenizer.batch_decode(preds, skip_special_tokens=True))
            decoded_labels.extend(self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True))

        # Compute WER and CER
        wer_score = wer(decoded_labels, decoded_preds)
        cer_score = cer(decoded_labels, decoded_preds)

        print(f"Epoch {epoch + 1} - WER: {wer_score:.4f}, CER: {cer_score:.4f}")
        logs['wer'] = wer_score
        logs['cer'] = cer_score

# Training function
def train():
    # Ensure directories exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    print("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    print("Loading dataset...")
    train_audio, train_text = load_data("train")
    val_audio, val_text = load_data("val")

    print("Preparing datasets...")
    train_features, train_labels = prepare_dataset(train_audio, train_text, processor)
    val_features, val_labels = prepare_dataset(val_audio, val_text, processor)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_features": train_features, 
            "labels": train_labels
        }
    )).batch(8)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_features": val_features, 
            "labels": val_labels
        }
    )).batch(4)

    # Compile model
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, 'best_model'),
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=2
        ),
        WerCerCallback(val_dataset, processor)  # Add custom WER/CER callback
    ]

    # Training
    print("Training model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        callbacks=callbacks
    )

    # Save model and processor
    print("Saving model...")
    model.save_pretrained(MODEL_PATH)
    processor.save_pretrained(MODEL_PATH)
    plot_history(history, output_path=os.path.join(MODEL_PATH, "training_history.png"))
    
    return history

if __name__ == "__main__":
    train()
