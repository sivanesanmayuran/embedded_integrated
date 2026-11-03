import os
import time
from gtts import gTTS
import pygame
import speech_recognition as sr
import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.models import Sequential
import tf2onnx
import onnxruntime as ort
import cv2
import torch

def detect_objects(
    image_source, model_name="yolov5s", show=True, save_path=None
):
    """Performs object detection on an image using YOLOv5 via PyTorch Hub."""
    # 1. Load the requested YOLOv5 model variant
    model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)

    # 2. Run object detection inference
    results = model(image_source)

    # FIX 1: Access the first element [0] of the pandas xyxy list
    predictions_df = results.pandas().xyxy[0]

    # 3. Generate the image containing rendered bounding boxes
    results.render()  # This modifies results.ims in-place

    # FIX 2: Pull out the actual raw NumPy image array from the internal list container
    annotated_image = results.ims[0]

    # 4. Handle optional storage parameters
    if save_path:
        # Note: YOLOv5 works in RGB; OpenCV write expects BGR format.
        # Converting it ensures colors aren't inverted when saving!
        bgr_annotated = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr_annotated)
        print(f"[INFO] Annotated image successfully saved to: {save_path}")

    # 5. Handle user interface rendering
    if show:
        # Convert RGB back to BGR just for the live local preview window
        bgr_preview = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"YOLOv5 Detection - {model_name}", bgr_preview)
        print("[INFO] Press any key on the image window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return predictions_df, annotated_image

def extract_mfcc_features(audio_data, sample_rate, n_mfcc=40, max_time_steps=50):
    """
    Converts raw audio arrays into fixed-shape MFCC time-series matrices.
    """
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc = mfcc.T
    
    if mfcc.shape[0] < max_time_steps:
        pad_width = max_time_steps - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_time_steps, :]
        
    return mfcc

def plot_voice_features(audio_data, sample_rate, mfcc_features):
    """
    Plots the raw audio waveform and the extracted MFCC feature matrix side by side.
    """
    plt.figure(figsize=(12, 4))
    
    # 1. Plot Raw Audio Waveform
    plt.subplot(1, 2, 1)
    time_axis = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    plt.plot(time_axis, audio_data, color='teal')
    plt.title("Raw Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # 2. Plot MFCC Heatmap
    plt.subplot(1, 2, 2)
    # Transpose back to (features, time) for a standard spectrogram view
    plt.imshow(mfcc_features.T, cmap='viridis', origin='lower', aspect='auto')
    plt.title("Extracted MFCC Heatmap")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar(label="Intensity")
    
    plt.tight_layout()
    plt.show()
    
    
def record_audio(duration=2.0, sample_rate=16000):
    """
    Captures live audio from the microphone for a specified duration.
    """
    print(f"\n[Record] Speak now! Recording for {duration} seconds...")
    # Record mono audio (channels=1)
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("[Record] Recording stopped.")
    return audio_data.flatten(), sample_rate

def extract_mfcc_features(audio_data, sample_rate, n_mfcc=40, max_time_steps=50):
    """
    Converts raw audio arrays into fixed-shape MFCC time-series matrices.
    """
    # Extract Mel-Frequency Cepstral Coefficients
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Transpose to get shape: (time_steps, features)
    mfcc = mfcc.T
    
    # Pad or truncate to ensure strict, consistent time-step sizing for the LSTM
    if mfcc.shape[0] < max_time_steps:
        pad_width = max_time_steps - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_time_steps, :]
        
    return mfcc

def collect_voice_dataset(num_samples_per_class=3, duration=2.0, sample_rate=16000, n_mfcc=40, max_time_steps=50):
    """
    Prompts the user to record voice samples to dynamically build a training dataset.
    Class 0 = Your Custom Wake Word/Command, Class 1 = Random Background Noise.
    """
    X_list = []
    y_list = []
    
    classes = {0: "YOUR CUSTOM COMMAND", 1: "BACKGROUND NOISE / SILENCE"}
    
    print("\n=== STARTING DATASET COLLECTION ===")
    print("We will collect samples to train the model to recognize your specific command.")
    
    for class_id, class_name in classes.items():
        print(f"\n--- Preparing to collect class: {class_name} ---")
        for i in range(num_samples_per_class):
            input(f"Press Enter when ready to record sample {i+1}/{num_samples_per_class}...")
            
            # 1. Capture the live audio
            raw_audio, sr = record_audio(duration, sample_rate)
            
            # 2. Convert to the specific structure the LSTM requires
            features = extract_mfcc_features(raw_audio, sr, n_mfcc, max_time_steps)
            
            X_list.append(features)
            y_list.append(class_id)
            time.sleep(0.5)

    # Convert collected datasets to NumPy arrays
    X = np.array(X_list).astype(np.float32)
    y_raw = np.array(y_list)
    
    # Convert integer targets to categorical one-hot encoding matrices
    num_classes = len(classes)
    y = tf.keras.utils.to_categorical(y_raw, num_classes)
    
    return X, y, num_classes

def build_lstm_model(time_steps, num_features, num_classes):
    """
    Creates and compiles a Bidirectional LSTM network for voice command matching.
    """
    model = Sequential([
        Input(shape=(time_steps, num_features)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, epochs=15, batch_size=4):
    """
    Trains the compiled network using your custom captured audio features.
    """
    print("\n--- Training the Model on Your Voice ---")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size
    )
    return history

def live_inference(model, duration=2.0, sample_rate=16000, n_mfcc=40, max_time_steps=50):
    """
    Records a new live clip, displays its features, and runs inference.
    """
    print("\n=== LIVE TEST MODE ===")
    input("Press Enter to speak a test phrase...")
    
    # 1. Capture and Process Audio
    raw_audio, sr = record_audio(duration, sample_rate)
    features = extract_mfcc_features(raw_audio, sr, n_mfcc, max_time_steps)
    
    # 2. Plot Visualizations
    print("[Visualization] Generating feature plots...")
    plot_voice_features(raw_audio, sr, features)
    
    # 3. Model Prediction
    input_sample = np.expand_dims(features, axis=0)
    prediction = model.predict(input_sample)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    print(f"\n[Result] Predicted Class ID: {predicted_class}")
    print(f"[Result] Match Confidence: {confidence:.2%}")
    
    if predicted_class == 0:
        print("🎉 Success! The model recognized your voice command.")
    else:
        print("💤 The model classified this as background noise or structural silence.")


def save_trained_model(model, file_path="voice_recognition_model.keras"):
    """
    Saves the entire model architecture, weights, and compilation configuration 
    to a permanent file on your local storage drive.
    """
    print(f"\n[Storage] Saving model to local drive at: {file_path}")
    model.save(file_path)
    print("[Storage] Save complete.")

def load_existing_model(file_path="voice_recognition_model.keras"):
    """
    Loads a previously trained and saved model from your local storage drive 
    so you can run immediate inference without retraining.
    """
    if os.path.exists(file_path):
        print(f"\n[Storage] Loading existing model from: {file_path}")
        model = tf.keras.models.load_model(file_path)
        print("[Storage] Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"No saved model found at {file_path}. Please train a model first.")



  
def main_voice_recon():
    DURATION = 2.0         
    SAMPLE_RATE = 16000    
    N_MFCC = 40            
    MAX_TIME_STEPS = 50    
    MODEL_FILE = "my_voice_model.keras"
    
    # Check if the model has already been trained and saved from a previous session
    if os.path.exists(MODEL_FILE):
        print(f"\n[🔍 Found Saved Model] '{MODEL_FILE}' already exists on disk.")
        print("Skipping training phase and launching directly into voice recognition mode...")
        
        # Load the existing model directly
        model = load_existing_model(MODEL_FILE)
    
    else:
        print(f"\n[ℹ️ No Model Found] '{MODEL_FILE}' not found on disk.")
        print("Starting data collection to train a new model...")
        
        # 1. Collect live dataset samples from your microphone
        X_train, y_train, num_classes = collect_voice_dataset(
            num_samples_per_class=3, 
            duration=DURATION,
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            max_time_steps=MAX_TIME_STEPS
        )
        
        # 2. Build the neural network architecture
        model = build_lstm_model(
            time_steps=MAX_TIME_STEPS, 
            num_features=N_MFCC, 
            num_classes=num_classes
        )
        
        # 3. Train the model using the captured data
        train_model(model, X_train, y_train, epochs=20, batch_size=2)
        
        # 4. Save the model so it can be auto-detected next time
        save_trained_model(model, MODEL_FILE)
    
    # --- RUN THE VOICE RECOGNITION TEST ---
    # This runs whether the model was just trained OR loaded from disk
    live_inference(
        model, 
        duration=DURATION, 
        sample_rate=SAMPLE_RATE, 
        n_mfcc=N_MFCC, 
        max_time_steps=MAX_TIME_STEPS
    )

def speak(text):
    """Converts text to an MP3 file and plays it out loud."""
    print(f"Assistant: {text}")
    
    # 1. Generate the audio file using Google TTS
    tts = gTTS(text=text, lang='en')
    filename = "response.mp3"
    tts.save(filename)
    
    # 2. Initialize pygame mixer and play audio
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    # 3. Keep script alive until the audio finishes playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
        
    # 4. Clean up audio resources and file to allow reuse
    pygame.mixer.quit()
    if os.path.exists(filename):
        os.remove(filename)

def process_question(question):
    """Processes the text question and returns a text answer."""
    question = question.lower()
    
    # Simple rule-based logic (Replace this with an LLM/API for smart answers)
    if "hello" in question or "hi" in question:
        return "Hello! How can I help you today?"
    elif "your name" in question:
        return "I am your Python voice assistant running on your laptop."
    elif "time" in question:
        current_time = time.strftime("%I:%M %p")
        return f"The current time is {current_time}."
    elif "how are you" in question:
        return "I am functioning perfectly, thank you for asking!"
    else:
        return "I heard you, but I do not know the answer to that question yet."

def listen_and_answer():
    """Listens to the microphone, converts to text, and triggers the response."""
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("\nListening... Speak your question clearly into the microphone.")
        # Adjust for background noise in your room
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        
    try:
        print("Processing voice input...")
        # Convert speech to text using Google's free web service
        user_text = recognizer.recognize_google(audio)
        print(f"You said: {user_text}")
        
        # Process text and speak the result
        answer = process_question(user_text)
        speak(answer)
        
    except sr.UnknownValueError:
        speak("Sorry, I could not understand what you said.")
    except sr.RequestError:
        speak("Sorry, my speech recognition service is currently unavailable.")

if __name__ == "__main__":
    
    img_array = cv2.imread("zidane.jpg")
    if img_array is not None:
        df_results, output_img = detect_objects(
            image_source=img_array,
            model_name="yolov5m",  # Using medium model for better accuracy
            show=True,
            save_path="output_zidane.jpg",
        )

        # Inspect the structured detections in code
        print("\n--- Detection Results Summary ---")
        print(df_results[["name", "confidence"]])
    else:
        print(
            "Could not load the example image. Please check your path configuration."
        )
    main_voice_recon()
    # Run the assistant in a loop until you exit manually (Ctrl+C)
    speak("Voice assistant activated.")
    while True:
        listen_and_answer()
