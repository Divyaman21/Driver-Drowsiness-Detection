import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import time
from collections import deque

MODEL_PATH = "drowsiness_model.h5"
ALARM_PATH = "alarm.wav"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ALARM_THRESHOLD = 15
THRESHOLD = 0.45
SMOOTHING_WINDOW = 8

print("🚀 Initializing Drowsiness Detection System...")

model = load_model(MODEL_PATH)
print("✅ Model loaded.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

pygame.mixer.init()
try:
    pygame.mixer.music.load(ALARM_PATH)
except Exception as e:
    print("⚠️ Alarm sound issue:", e)

def preprocess_eye(eye_img):
    eye_resized = cv2.resize(eye_img, (64, 64))
    if model.input_shape[-1] == 3:
        eye_input = eye_resized.astype("float32") / 255.0
    else:
        gray = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2GRAY)
        eye_input = np.expand_dims(gray.astype("float32") / 255.0, axis=-1)
    return np.expand_dims(eye_input, axis=0)

def predict_eye_state(eye_img):
    try:
        eye_input = preprocess_eye(eye_img)
        pred = model.predict(eye_input, verbose=0)[0][0]
        return pred
    except Exception:
        return 0.0

def draw_modern_overlay(frame, status_text, color, fps, alert_level, avg_pred):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "Driver Drowsiness Detection", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    bar_color = (0, 255, 0) if avg_pred < THRESHOLD else (0, 0, 255)
    cv2.rectangle(frame, (20, FRAME_HEIGHT - 60), (220, FRAME_HEIGHT - 30), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, FRAME_HEIGHT - 60),
                  (20 + int(avg_pred * 200), FRAME_HEIGHT - 30), bar_color, -1)
    cv2.putText(frame, f"Sleepiness: {avg_pred:.2f}", (230, FRAME_HEIGHT - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (FRAME_WIDTH - 250, FRAME_HEIGHT - 40),
                  (FRAME_WIDTH - 250 + min(alert_level * 10, 200), FRAME_HEIGHT - 20), color, -1)
    cv2.putText(frame, "ALERT LEVEL", (FRAME_WIDTH - 250, FRAME_HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (20, FRAME_HEIGHT - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

drowsy_frames = 0
predictions = deque(maxlen=SMOOTHING_WINDOW)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
    status_text = "No Face Detected"
    color = (128, 128, 128)
    avg_pred = 0.0

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4, minSize=(30, 30))

        eye_preds = []
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_img = face_roi[ey:ey+eh, ex:ex+ew]
            pred = predict_eye_state(eye_img)
            eye_preds.append(pred)
            cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 1)

        if not eye_preds:
            continue

        avg_pred = np.mean(eye_preds)
        predictions.append(avg_pred)
        smooth_pred = np.mean(predictions)

        if smooth_pred > THRESHOLD:
            status_text = f"Drowsy 😴 ({smooth_pred:.2f})"
            color = (0, 0, 255)
            drowsy_frames += 1
        else:
            status_text = f"Awake 👀 ({smooth_pred:.2f})"
            color = (0, 255, 0)
            drowsy_frames = 0

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        if drowsy_frames >= ALARM_THRESHOLD:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        elif drowsy_frames == 0 and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    draw_modern_overlay(frame, status_text, color, fps, drowsy_frames, avg_pred)
    cv2.imshow("🚗 Smart Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("✅ Clean exit. Drive safely! 🚗")
