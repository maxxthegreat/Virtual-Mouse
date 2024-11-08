# Multimodal Virtual Mouse

This project implements a virtual mouse system using multimodal inputs such as hand gestures, eye blinks, and voice commands to control cursor movements, clicks, volume, and brightness. The system uses **MediaPipe** for hand and face tracking, **Vosk** for speech recognition, and **PyAutoGUI** for simulating mouse actions.

## Requirements

The following libraries are required:

- `opencv-python`: for accessing the webcam.
- `mediapipe`: for hand and face tracking.
- `pyautogui`: for simulating mouse actions.
- `vosk`: for speech recognition.
- `pyaudio`: for capturing audio data.
- (Windows only) `pycaw` and `comtypes`: for volume control.
- `screen_brightness_control`: for controlling screen brightness.

To install all required packages, run:

```bash
pip install opencv-python mediapipe pyautogui vosk pyaudio screen_brightness_control pycaw comtypes
