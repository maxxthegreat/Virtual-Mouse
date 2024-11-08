import cv2
import pyautogui
import threading
import sys
import time

# Import utility modules
from utils.voice_control import VoiceControl
from utils.hand_gesture_control import HandGestureControl
from utils.eye_movement_control import EyeMovementControl

class MultimodalVirtualMouse:
    def __init__(self, vosk_model_path="vosk-model-small-en-in-0.4"):
        self.running = True

        # Initialize each control module
        self.voice_control = VoiceControl(vosk_model_path)
        self.hand_gesture_control = HandGestureControl()
        self.eye_movement_control = EyeMovementControl()

        # Webcam feed
        self.cap = cv2.VideoCapture(0)

    def run(self):
        if self.voice_control.initialized:
            voice_thread = threading.Thread(target=self.voice_control.process_voice_commands, args=(self,))
            voice_thread.start()
        else:
            print("Voice command processing is disabled due to initialization issues.")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            self.hand_gesture_control.process_hand_gestures(frame)
            face_landmarks = self.eye_movement_control.process_eye_movement(frame)

            if face_landmarks:
                self.eye_movement_control.process_eye_blinks(frame, face_landmarks)

            cv2.imshow('Multimodal Virtual Mouse', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting program by key press.")
                self.running = False

        self.cap.release()
        cv2.destroyAllWindows()
        if self.voice_control.stream:
            self.voice_control.stream.stop_stream()
            self.voice_control.stream.close()

if __name__ == "__main__":
    vosk_model_path = "vosk-model-small-en-in-0.4"
    virtual_mouse = MultimodalVirtualMouse(vosk_model_path=vosk_model_path)
    virtual_mouse.run()
