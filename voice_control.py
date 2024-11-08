import json
import pyaudio
import vosk
import pyautogui

class VoiceControl:
    def __init__(self, vosk_model_path):
        try:
            self.model = vosk.Model(vosk_model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
            self.stream.start_stream()
            self.initialized = True
        except Exception as e:
            print(f"Failed to initialize voice control: {e}")
            self.initialized = False

    def process_voice_commands(self, main_instance):
        while main_instance.running:
            audio_data = self.stream.read(8000, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(audio_data):
                result_json = json.loads(self.recognizer.Result())
                command_text = result_json.get("text", "").lower()

                if "click" in command_text:
                    pyautogui.click()
                elif "scroll up" in command_text:
                    pyautogui.scroll(300)
                elif "scroll down" in command_text:
                    pyautogui.scroll(-300)
                elif "quit" in command_text or "exit" in command_text:
                    print("Exiting program by voice command.")
                    main_instance.running = False
