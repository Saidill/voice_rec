import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration

# Load pretrained model and processor
MODEL_PATH = "./models/whisper_multilingual"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = TFWhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

def record_audio(duration=5, sr=16000):
    """
    Record audio using the microphone for a specific duration.
    """
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is done
    audio = np.squeeze(audio)  # Remove extra dimensions
    print("Recording finished!")
    return audio, sr

def speech_to_text(audio, processor, model, sr=16000):
    """
    Perform Speech-to-Text using the Whisper model.
    """
    # Convert audio to features
    input_features = processor.feature_extractor(
        audio, 
        sampling_rate=sr, 
        return_tensors="tf"
    ).input_features

    # Predict transcription
    predicted_ids = model.generate(input_features)
    predicted_text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return predicted_text

def main():
    # Record audio
    audio, sr = record_audio(duration=5, sr=16000)

    # Perform STT
    predicted_text = speech_to_text(audio, processor, model, sr)

    # Display result
    print("\n===== Transcription =====")
    print(f"Predicted Text: {predicted_text}")

if __name__ == "__main__":
    main()
