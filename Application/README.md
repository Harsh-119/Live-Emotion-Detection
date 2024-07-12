# Age, Gender, Spectacle, and Emotion Detection with Webcam

This is a Python program, [emotion_glasses_age_gender_detection.py](Application/emotion_age_gender_detection.py), that uses a webcam to detect the age, gender, and emotion of people in real-time. 

The program, [emotion_glasses_age_gender_detection.py](Application/emotion_glasses_age_gender_detection.py), also detects whether the person is wearing glasses or not.

## Features

- **Age Detection**: Estimates the age of the person.
- **Gender Detection**: Determines if the person is male or female.
- **Emotion Detection**: Recognizes and labels emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Spectacle Detection**: Detects if the person is wearing glasses. This is done by creating 2 regions one in the middle of the eyes and the other directly under the eyes.
- **Real-time Webcam Input**: Captures video from the webcam and processes each frame.

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- dlib
- TensorFlow and Keras

## Pre-trained Models

You will need to download the following pre-trained models:
- Emotion detection model (`model.json` and `model.h5`)
- Face detector (`haarcascade_frontalface_default.xml`)
- Facial landmark detector (`shape_predictor_5_face_landmarks.dat`)
- Age and Gender detection models (`opencv_face_detector_uint8.pb`, `opencv_face_detector.pbtxt`, `age_net.caffemodel`, `age_deploy.prototxt`, `gender_net.caffemodel`, `gender_deploy.prototxt`)

Place these models in the appropriate paths as specified in the code.

