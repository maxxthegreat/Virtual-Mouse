import cv2
import mediapipe as mp
import pyautogui
import threading
import math
import sys
import time

# For speech recognition with Vosk
import vosk
import pyaudio
import json

# Conditional imports for Windows-specific functionality
if sys.platform == "win32":
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except ImportError:
        print("pycaw and comtypes are not installed. Volume control will be disabled.")
    try:
        import screen_brightness_control as sbc
    except ImportError:
        print("screen_brightness_control is not installed. Brightness control will be disabled.")
else:
    try:
        import screen_brightness_control as sbc
    except ImportError:
        print("screen_brightness_control is not installed. Brightness control will be disabled.")

class MultimodalVirtualMouse:
    def __init__(self, vosk_model_path="vosk-model-small-en-in-0.4"):
        
        # Initialize MediaPipe solutions for hands and face mesh
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        # Drawing utilities for MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Hand tracking and face mesh
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7
        )

        # Initialize Vosk model
        try:
            self.model = vosk.Model(vosk_model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            print("Vosk model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Vosk model from '{vosk_model_path}': {e}")
            self.model = None
            self.recognizer = None

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(format=pyaudio.paInt16,
                                          channels=1,
                                          rate=16000,
                                          input=True,
                                          frames_per_buffer=8000)
            self.stream.start_stream()
            print("Audio stream initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize audio stream: {e}")
            self.stream = None

        # Webcam feed
        self.cap = cv2.VideoCapture(0)

        # Flag to stop voice thread and main loop
        self.running = True

        # Initialize volume control if on Windows and dependencies are available
        if sys.platform == "win32":
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = interface.QueryInterface(IAudioEndpointVolume)
                print("Volume control initialized.")
            except Exception as e:
                print(f"Volume control initialization failed: {e}")
                self.volume = None
        else:
            self.volume = None

        # For blink detection
        self.blink_threshold = 0.2  # EAR threshold for blink
        self.blink_consec_frames = 3  # Consecutive frames required to detect a blink
        self.blink_counter = 0
        self.blinked = False

    def calculate_distance(self, lm1, lm2):
        return math.sqrt(
            (lm1.x - lm2.x) ** 2 +
            (lm1.y - lm2.y) ** 2 +
            (lm1.z - lm2.z) ** 2
        )

    def process_hand_gestures(self, frame):
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand gestures
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Use landmarks to control mouse
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

                screen_width, screen_height = pyautogui.size()
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(x, y)

                # Detect pinch gesture for left click
                distance = self.calculate_distance(index_finger_tip, thumb_tip)
                if distance < 0.05:
                    pyautogui.click()
                    print("Left Click")

                # Detect fist gesture for right click
                folded_fingers = 0
                finger_tips = [
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    self.mp_hands.HandLandmark.RING_FINGER_TIP,
                    self.mp_hands.HandLandmark.PINKY_TIP
                ]
                for tip in finger_tips:
                    finger_tip = hand_landmarks.landmark[tip]
                    finger_pip = hand_landmarks.landmark[tip - 2]  # PIP joint
                    if finger_tip.y > finger_pip.y:
                        folded_fingers += 1

                if folded_fingers == 4:
                    pyautogui.rightClick()
                    print("Right Click")

    def process_eye_movement(self, frame, face_landmarks):
        # Define indices for left and right eye
        # MediaPipe Face Mesh landmarks:
        # Left eye: 33, 160, 158, 133, 153, 144
        # Right eye: 362, 385, 387, 263, 373, 380

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]

        left_eye = [face_landmarks.landmark[i] for i in left_eye_indices]
        right_eye = [face_landmarks.landmark[i] for i in right_eye_indices]

        def compute_ear(eye):
            # Compute Eye Aspect Ratio (EAR)
            A = math.sqrt((eye[1].x - eye[5].x)**2 + (eye[1].y - eye[5].y)**2)
            B = math.sqrt((eye[2].x - eye[4].x)**2 + (eye[2].y - eye[4].y)**2)
            C = math.sqrt((eye[0].x - eye[3].x)**2 + (eye[0].y - eye[3].y)**2)
            ear = (A + B) / (2.0 * C)
            return ear

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # Check if EAR is below the blink threshold
        if ear < self.blink_threshold:
            self.blink_counter += 1
            if self.blink_counter >= self.blink_consec_frames:
                if not self.blinked:
                    pyautogui.click()
                    print("Blink Detected: Left Click")
                    self.blinked = True
        else:
            if self.blinked:
                pyautogui.rightClick()
                print("Eyes Opened: Right Click")
            self.blink_counter = 0
            self.blinked = False

    def process_voice_commands(self):
        def callback(audio_data):
            print("Debug: Callback function triggered.")  # Debug statement
            try:
                # Recognize speech using the Vosk recognizer
                if self.recognizer.AcceptWaveform(audio_data):
                    result = self.recognizer.Result()
                    result_json = json.loads(result)
                    command_text = result_json.get("text", "").lower()

                    print(f"Debug: Raw voice command received: {command_text}")  # Debug statement
                    print(f"Voice Command: {command_text}")

                    # Implement voice command controls
                    if "click" in command_text:
                        pyautogui.click()
                        print("Mouse clicked.")
                    elif "right click" in command_text:
                        pyautogui.rightClick()
                        print("Right mouse button clicked.")
                    elif "scroll up" in command_text:
                        pyautogui.scroll(300)  # Scroll up
                        print("Scrolled up.")
                    elif "scroll down" in command_text:
                        pyautogui.scroll(-300)  # Scroll down
                        print("Scrolled down.")
                    elif "increase volume" in command_text:
                        self.increase_volume()
                        print("Volume increased.")
                    elif "decrease volume" in command_text:
                        self.decrease_volume()
                        print("Volume decreased.")
                    elif "increase brightness" in command_text:
                        self.increase_brightness()
                        print("Brightness increased.")
                    elif "decrease brightness" in command_text:
                        self.decrease_brightness()
                        print("Brightness decreased.")
                    elif "quit" in command_text or "exit" in command_text:
                        print("Exiting program by voice command.")
                        self.running = False  # Stop the main loop
                else:
                    print("Debug: Could not understand the command.")  # Debug statement
            except Exception as e:
                print(f"Debug: Error in processing voice command: {e}")  # Debug statement

        # Open the audio stream manually
        try:
            self.audio_stream = self.audio.open(format=pyaudio.paInt16,
                                               channels=1,
                                               rate=16000,
                                               input=True,
                                               frames_per_buffer=8000)
            print("Debug: Microphone stream opened successfully.")  # Debug statement
        except Exception as e:
            print(f"Debug: Failed to open audio stream: {e}")  # Debug statement
            self.running = False
            return

        print("Debug: Microphone is now active.")  # Debug statement
        print("Listening for voice commands...")

        while self.running:
            try:
                audio_data = self.audio_stream.read(8000, exception_on_overflow=False)  # Read audio data
                callback(audio_data)  # Process the audio data
            except Exception as e:
                print(f"Debug: Error reading audio data: {e}")  # Debug statement
                continue

        # Stop and close the stream after exiting the loop
        try:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            print("Debug: Stopped listening and closed audio stream.")  # Debug statement
        except Exception as e:
            print(f"Debug: Error closing audio stream: {e}")  # Debug statement

    def increase_volume(self):
        if sys.platform == "win32" and self.volume:
            try:
                current_volume = self.volume.GetMasterVolumeLevelScalar()
                new_volume = min(current_volume + 0.05, 1.0)
                self.volume.SetMasterVolumeLevelScalar(new_volume, None)
                print(f"Volume set to {new_volume * 100:.0f}%")
            except Exception as e:
                print(f"Debug: Error increasing volume: {e}")

    def decrease_volume(self):
        if sys.platform == "win32" and self.volume:
            try:
                current_volume = self.volume.GetMasterVolumeLevelScalar()
                new_volume = max(current_volume - 0.05, 0.0)
                self.volume.SetMasterVolumeLevelScalar(new_volume, None)
                print(f"Volume set to {new_volume * 100:.0f}%")
            except Exception as e:
                print(f"Debug: Error decreasing volume: {e}")

    def increase_brightness(self):
        try:
            current_brightness = sbc.get_brightness(display=0)[0]
            new_brightness = min(current_brightness + 10, 100)
            sbc.set_brightness(new_brightness, display=0)
            print(f"Brightness set to {new_brightness}%")
        except Exception as e:
            print(f"Brightness control not supported or failed: {e}")

    def decrease_brightness(self):
        try:
            current_brightness = sbc.get_brightness(display=0)[0]
            new_brightness = max(current_brightness - 10, 0)
            sbc.set_brightness(new_brightness, display=0)
            print(f"Brightness set to {new_brightness}%")
        except Exception as e:
            print(f"Brightness control not supported or failed: {e}")

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from webcam.")
                break

            # Flip the frame horizontally to correct the mirroring issue
            frame = cv2.flip(frame, 1)

            # Process hand gestures
            self.process_hand_gestures(frame)

            # Process eye movements
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

                    # Process eye blinks
                    self.process_eye_movement(frame, face_landmarks)

            # Display the frame
            cv2.imshow('Multimodal Virtual Mouse', frame)

            # Press 'q' to exit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting program by key press.")
                self.running = False

        # Cleanup
        self.cap.release()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        cv2.destroyAllWindows()

    def __del__(self):
        # Ensure resources are released
        if self.cap.isOpened():
            self.cap.release()
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio_stream') and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio:
            self.audio.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to your Vosk model directory
    vosk_model_path = "vosk-model-small-en-in-0.4"  # Ensure this path is correct

    virtual_mouse = MultimodalVirtualMouse(vosk_model_path=vosk_model_path)

    if virtual_mouse.recognizer and virtual_mouse.stream:
        voice_thread = threading.Thread(target=virtual_mouse.process_voice_commands)
        voice_thread.start()
    else:
        voice_thread = None
        print("Voice command processing is disabled due to initialization failures.")

    # Run the main loop
    virtual_mouse.run()

    # Ensure the voice thread is properly terminated
    if voice_thread:
        voice_thread.join()
