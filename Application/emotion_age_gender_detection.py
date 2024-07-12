"""
slightly different approach for face detection and blob creation for age and gender prediction
no spectacle detection
"""

import os
import cv2
import numpy as np
import dlib
import tensorflow
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

# Load the pre-trained emotion detection model
model = model_from_json(open("/Users/admin/PycharmProjects/emotion_glass_gender_age/model.json", "r").read())
model.load_weights('/Users/admin/PycharmProjects/emotion_glass_gender_age/model.h5')

# Load the face detector
face_detector_path = "/Users/admin/PycharmProjects/emotion_glass_gender_age/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_detector_path)

# Load the facial landmark detector
predictor_path = "/Users/admin/PycharmProjects/emotion_glass_gender_age/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load age and gender detection models
face1 = "/Users/admin/PycharmProjects/emotion_glass_gender_age/opencv_face_detector.pbtxt"
face2 = "/Users/admin/PycharmProjects/emotion_glass_gender_age/opencv_face_detector_uint8.pb"
age1 = "/Users/admin/PycharmProjects/emotion_glass_gender_age/age_deploy.prototxt"
age2 = "/Users/admin/PycharmProjects/emotion_glass_gender_age/age_net.caffemodel"
gen1 = "/Users/admin/PycharmProjects/emotion_glass_gender_age/gender_deploy.prototxt"
gen2 = "/Users/admin/PycharmProjects/emotion_glass_gender_age/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Using models
# Face
face = cv2.dnn.readNet(face2, face1)

# Age
age_net = cv2.dnn.readNet(age2, age1)

# Gender
gender_net = cv2.dnn.readNet(gen2, gen1)

# Categories of distribution
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = roi_gray.astype(np.float32) / 255.0

        # Predict emotion
        predictions = model.predict(roi_gray)
        max_index = np.argmax(predictions[0])
        emotion_labels = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        emotion_prediction = emotion_labels[max_index]
        cv2.putText(img, "Emotion: {}".format(emotion_prediction), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (36, 255, 12), 2)

        # Face detection
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)
        face.setInput(blob)
        detections = face.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * img.shape[1])
                y1 = int(detections[0, 0, i, 4] * img.shape[0])
                x2 = int(detections[0, 0, i, 5] * img.shape[1])
                y2 = int(detections[0, 0, i, 6] * img.shape[0])
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), int(round(img.shape[0] / 150)), 8)

        # Age and gender detection
        for faceBox in faceBoxes:
            face_roi = img[max(0, faceBox[1] - 15):min(faceBox[3] + 15, img.shape[0] - 1),
                           max(0, faceBox[0] - 15):min(faceBox[2] + 15, img.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_labels[gender_preds[0].argmax()]
            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_labels[age_preds[0].argmax()]
            # Display age and gender
            cv2.putText(img, f'{gender}, {age}', (x - 150, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (217, 0, 0), 4,
                        cv2.LINE_AA)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
