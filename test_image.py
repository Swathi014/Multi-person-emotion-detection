"""
Multi-Person Emotion Detection - Single Image Test
-----------------------------------------------------
Pick a photo, see the predicted emotion for every face found in it.
"""

import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "emotion_detection_model.keras"
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# --- pick an image file ---
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Select a Face Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
)
root.destroy()

if not image_path:
    print("No image selected.")
    raise SystemExit

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

image = cv2.imread(image_path)
if image is None:
    print(f"Could not read image at '{image_path}'.")
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

if len(faces) == 0:
    print("No faces found in this image.")
else:
    print(f"Found {len(faces)} face(s).\n")

for i, (x, y, w, h) in enumerate(faces, start=1):
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 48, 48, 1)

    prediction = model.predict(roi, verbose=0)[0]
    idx = int(np.argmax(prediction))
    emotion = EMOTIONS[idx]
    confidence = prediction[idx] * 100

    print(f"Face #{i}: {emotion} ({confidence:.1f}%)")

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
    label = f"{emotion} ({confidence:.0f}%)"
    cv2.putText(image, label, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Shrink big images so they fit on screen
h_img, w_img = image.shape[:2]
if w_img > 1200 or h_img > 800:
    image = cv2.resize(image, (w_img // 2, h_img // 2))

cv2.imshow("Emotion Test", image)
print("\nPress any key on the image window to close it.")
cv2.waitKey(0)
cv2.destroyAllWindows()
