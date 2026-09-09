"""
Multi-Person Emotion Detection - Live Webcam Demo
---------------------------------------------------
Detects every face in the webcam feed and predicts each person's emotion
using a pre-trained Keras model. Simple on purpose: one file, no extra
frameworks beyond OpenCV + TensorFlow + pyttsx3.
"""

import time

import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
MODEL_PATH = "emotion_detection_model.keras"

# Must be in the SAME order the model was trained on (alphabetical folder
# names): Angry, Happy, Neutral, Sad, Surprise.
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# What the robot says the first time it sees each emotion (won't repeat
# the same line again until the emotion changes).
REACTIONS = {
    "Happy": "I am glad to see you smiling!",
    "Sad": "Are you okay? I am here for you.",
    "Angry": "Please stay calm.",
    "Surprise": "Oh, that surprised me too!",
}

SPEAK_COOLDOWN = 3  # minimum seconds between spoken lines


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Haar cascade face detector - built into opencv-python, no extra
# download or dependency needed.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Starting voice engine...")
engine = pyttsx3.init()
engine.setProperty("rate", 150)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(
        "Could not open the webcam. Make sure it's connected, not in use "
        "by another app, and that camera permissions are granted."
    )

last_spoken_emotion = None
last_speak_time = 0.0

print("Ready. Press 'q' in the video window to quit.")


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed, stopping.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    primary_emotion = None
    primary_area = 0

    for (x, y, w, h) in faces:
        # Crop the face, resize to what the model expects, normalize
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        prediction = model.predict(roi, verbose=0)[0]
        idx = int(np.argmax(prediction))
        emotion = EMOTIONS[idx]
        confidence = prediction[idx] * 100

        # Draw box + label for this person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = f"{emotion} ({confidence:.0f}%)"
        cv2.putText(frame, label, (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Track the closest/largest face as the "primary" person to react to
        area = w * h
        if area > primary_area:
            primary_area = area
            primary_emotion = emotion

    # Speak a reaction if the primary person's emotion changed and
    # enough time has passed since the last thing said
    now = time.time()
    if (primary_emotion in REACTIONS
            and primary_emotion != last_spoken_emotion
            and now - last_speak_time > SPEAK_COOLDOWN):
        engine.say(REACTIONS[primary_emotion])
        engine.runAndWait()
        last_spoken_emotion = primary_emotion
        last_speak_time = now
    elif primary_emotion is None:
        last_spoken_emotion = None

    cv2.putText(frame, f"People detected: {len(faces)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Multi-Person Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
