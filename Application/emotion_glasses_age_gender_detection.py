"""
 This program uses the webcam and detects
 age, gender (male/female), emotion and spectacle detection

 You will need to download the following pre-trained models:
	•	Emotion detection model (model.json and model.h5)
	•	Face detector (haarcascade_frontalface_default.xml)
	•	Facial landmark detector (shape_predictor_5_face_landmarks.dat)
	•	Age and Gender detection models (opencv_face_detector_uint8.pb, opencv_face_detector.pbtxt, 
        age_net.caffemodel, age_deploy.prototxt, gender_net.caffemodel, gender_deploy.prototxt)
"""

import cv2
import numpy as np
import dlib
from tensorflow.keras.models import model_from_json

# Function to convert dlib landmarks to numpy array
def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts
    coords = np.zeros((num, 2), dtype=dtype)
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

# Function to calculate eye centers
def get_centers(img, landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (EYE_LEFT_OUTTER[0] + EYE_LEFT_INNER[0]) / 2
    x_right = (EYE_RIGHT_OUTTER[0] + EYE_RIGHT_INNER[0]) / 2
    LEFT_EYE_CENTER = np.array([np.int32(x_left), np.int32(x_left * k + b)])
    RIGHT_EYE_CENTER = np.array([np.int32(x_right), np.int32(x_right * k + b)])

    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255, 0, 0), 1)
    cv2.circle(img, (LEFT_EYE_CENTER[0], LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0], RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

# Function to align face
def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    eyescenter = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx * dx + dy * dy)
    scale = desired_dist / dist
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(eyescenter, angle, scale)

    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img, M, (desired_w, desired_h))

    return aligned_face

# Function to detect spectacles
def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    edgeness = sobel_y
    _, thresh = cv2.threshold(edgeness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    d = len(thresh) * 0.5
    x = np.int32(d * 6 / 7)
    y = np.int32(d * 3 / 4)
    w = np.int32(d * 2 / 7)
    h = np.int32(d * 2 / 4)

    x_2_1 = np.int32(d * 1 / 4)
    x_2_2 = np.int32(d * 5 / 4)
    w_2 = np.int32(d * 1 / 2)
    y_2 = np.int32(d * 8 / 7)
    h_2 = np.int32(d * 1 / 2)

    roi_1 = thresh[y:y + h, x:x + w]
    roi_2_1 = thresh[y_2:y_2 + h_2, x_2_1:x_2_1 + w_2]
    roi_2_2 = thresh[y_2:y_2 + h_2, x_2_2:x_2_2 + w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])

    measure_1 = sum(sum(roi_1 / 255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
    measure_2 = sum(sum(roi_2 / 255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
    measure = measure_1 * 0.3 + measure_2 * 0.7

    if measure > 0.15:
        judge = True
    else:
        judge = False
    return judge

# Load the pre-trained emotion detection model
model = model_from_json(open("/Users/admin/PycharmProjects/emotion_and_glass/model.json", "r").read())
model.load_weights('/Users/admin/PycharmProjects/emotion_and_glass/model.h5')

# Load the face detector
face_detector_path = "/Users/admin/PycharmProjects/emotion_and_glass/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_detector_path)

# Load the facial landmark detector
predictor_path = "/Users/admin/PycharmProjects/emotion_and_glass/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)

# Load gender and age detection models
face_model_path = "/Users/admin/PycharmProjects/emotion_and_glass/Age-Gender-Detection/opencv_face_detector_uint8.pb"
face_config_path = "/Users/admin/PycharmProjects/emotion_and_glass/Age-Gender-Detection/opencv_face_detector.pbtxt"
age_model_path = "/Users/admin/PycharmProjects/emotion_and_glass/Age-Gender-Detection/age_net.caffemodel"
age_config_path = "/Users/admin/PycharmProjects/emotion_and_glass/Age-Gender-Detection/age_deploy.prototxt"
gender_model_path = "/Users/admin/PycharmProjects/emotion_and_glass/Age-Gender-Detection/gender_net.caffemodel"
gender_config_path = "/Users/admin/PycharmProjects/emotion_and_glass/Age-Gender-Detection/gender_deploy.prototxt"

face_model = cv2.dnn.readNet(face_model_path, face_config_path)
age_model = cv2.dnn.readNet(age_model_path, age_config_path)
gender_model = cv2.dnn.readNet(gender_model_path, gender_config_path)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

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

        # Spectacle detection
        landmarks = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
        landmarks = landmarks_to_np(landmarks)
        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        judge = judge_eyeglass(aligned_face)
        if judge:
            cv2.putText(img, "With Glasses", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img, "No Glasses", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Gender and age detection
        # Convert aligned_face to RGB format
        rgb_img = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2RGB)

        # Create blob for the face detection model
        blob = cv2.dnn.blobFromImage(rgb_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)

        face_model.setInput(blob)
        detections = face_model.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                # Gender detection
                gender_blob = blob.copy()
                gender_model.setInput(gender_blob)
                gender_preds = gender_model.forward()
                gender = "Male" if gender_preds[0, 0] > 0.5 else "Female"

                # Age detection
                age_blob = blob.copy()
                age_model.setInput(age_blob)
                age_preds = age_model.forward()
                age = int(age_preds[0].argmax())

                cv2.putText(img, "Gender: {}, Age: {}".format(gender, age), (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
