import os
import logging

# ✅ Suppress TensorFlow & Mediapipe Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow info/warning messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# ✅ Suppress Mediapipe Logs
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import requests

from scipy.spatial import distance as dist

# Constants
EYE_AR_THRESH = 0.25  # Eye Aspect Ratio threshold
EYE_CLOSE_DURATION = 4  # Time in seconds before alarm starts

# ✅ Define global variables at the start
COUNTER = 0  # Counter for consecutive frames
ALARM_ON = False  # Alarm state
START_TIME = None  # Time when eyes first close

# Initialize pygame for sound alerts
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(r"C:\Users\UZAIR TAUHID\OneDrive\mixkit-facility-alarm-sound-999.wav")

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.6,  # Increased confidence
    min_tracking_confidence=0.6,
    refine_landmarks=True  # Improves accuracy
)


# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)


# Function to get the user's location (city, region, country, latitude & longitude)
def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        location_name = f"{data.get('city', 'Unknown')}, {data.get('region', 'Unknown')}, {data.get('country', 'Unknown')}"
        coordinates = data.get("loc", "Unknown Location")  # Format: "latitude,longitude"
        return location_name, coordinates
    except:
        return "Unknown Location", "Unknown Coordinates"


# Eye landmark indices for Mediapipe
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Driver Drowsiness Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Driver Drowsiness Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye landmarks
            landmarks = np.array(
                [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in face_landmarks.landmark])
            left_eye = landmarks[LEFT_EYE_INDICES]
            right_eye = landmarks[RIGHT_EYE_INDICES]

            # Compute EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # ✅ Modify global variables without redeclaring them
            if ear < EYE_AR_THRESH:
                if START_TIME is None:
                    START_TIME = time.time()  # Record the tie when eyes first close
                elapsed_time = time.time() - START_TIME  # Calculate elapsed time

                if elapsed_time >= EYE_CLOSE_DURATION and not ALARM_ON:
                    ALARM_ON = True
                    alarm_sound.play()

                    # ✅ Get exact time when drowsiness is detected
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                    # ✅ Get exact location (City, Region, Country, Latitude, Longitude)
                    location_name, coordinates = get_location()

                    print(
                        f"⚠️ DROWSINESS DETECTED! Time: {current_time} | Location: {location_name} | Coordinates: {coordinates}")

            else:
                START_TIME = None  # Reset timer if eyes open
                if ALARM_ON:
                    ALARM_ON = False
                    alarm_sound.stop()  # Stop alarm when eyes open

            # Draw a square box around both eyes
            left_x_min, left_y_min = np.min(left_eye, axis=0)
            left_x_max, left_y_max = np.max(left_eye, axis=0)
            cv2.rectangle(frame, (left_x_min - 8, left_y_min - 8), (left_x_max + 8, left_y_max + 8), (0, 255, 0), 2)

            right_x_min, right_y_min = np.min(right_eye, axis=0)
            right_x_max, right_y_max = np.max(right_eye, axis=0)
            cv2.rectangle(frame, (right_x_min - 8, right_y_min - 8), (right_x_max + 8, right_y_max + 8), (0, 255, 0), 2)

            # Display EAR on the frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame in fullscreen
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
