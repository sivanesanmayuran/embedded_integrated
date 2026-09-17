import os
import time
from gtts import gTTS
import pygame
import speech_recognition as sr

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
    # Run the assistant in a loop until you exit manually (Ctrl+C)
    speak("Voice assistant activated.")
    while True:
        listen_and_answer()
