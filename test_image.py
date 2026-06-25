import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog

def test_static_image():
    # --- 1. UPLOAD YOUR IMAGE ---
    # Opens a file dialog so you can choose any photo from your computer
    root = tk.Tk()
    root.withdraw() # Hide the empty tkinter background window
    
    # Force the window to the front
    root.attributes('-topmost', True) 
    
    print("--> Opening file dialog. Please select an image...")
    IMAGE_PATH = filedialog.askopenfilename(
        title="Select a Face Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    MODEL_PATH = 'emotion_detection_model.keras'

    if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
        print("\n[INFO] No image selected. Exiting...")
        return

    print("--> Loading AI Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Suprise']

    # Initialize MediaPipe (Model selection 1 is better for static, high-res photos)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    print(f"--> Processing {IMAGE_PATH}...")
    image = cv2.imread(IMAGE_PATH)
    ih, iw, _ = image.shape
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = face_detection.process(rgb_image)

    if not results.detections:
        print("[FAILED] MediaPipe could not find a face in this image.")
        return

    print(f"[SUCCESS] Found {len(results.detections)} face(s)!\n")

    for idx, detection in enumerate(results.detections):
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        # --- NEW: ADD PADDING TO MATCH TRAINING DATA CROP ---
        # Expand the box by 20% on top/bottom and 10% on sides
        margin_y = int(h * 0.20) 
        margin_x = int(w * 0.10)
        
        x = x - margin_x
        y = y - margin_y
        w = w + (2 * margin_x)
        h = h + (2 * margin_y)

        # Safety boundary checks
        x, y = max(0, x), max(0, y)
        if x + w > iw: w = iw - x
        if y + h > ih: h = ih - y

        if w > 0 and h > 0:
            # Crop, resize, and normalize exactly like the training data
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # Get predictions
            prediction_array = model.predict(roi, verbose=0)[0]
            
            print(f"--- AI CONFIDENCE BREAKDOWN (Face #{idx+1}) ---")
            for i, emotion in enumerate(class_names):
                print(f"{emotion}: {prediction_array[i]*100:.2f}%")
            
            max_index = int(np.argmax(prediction_array))
            best_guess = class_names[max_index]
            print(f"----------------------------------------")
            print(f"AI's Final Decision: {best_guess}\n")

            # Draw on the image
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 3)
            label = f"{best_guess} ({prediction_array[max_index]*100:.1f}%)"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Shrink the image if it's too big for your laptop screen
    if iw > 1200 or ih > 800:
        image = cv2.resize(image, (int(iw/2), int(ih/2)))

    cv2.imshow("AI Static Image Test", image)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_static_image()