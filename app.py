import os
import cv2
import numpy as np
import tensorflow as tf

def main():
    model_path = 'emotion_detection_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model file 'emotion_detection_model.h5' not found. Run train.py first.")

    # Load model and define rigid class order matching directory structure
    model = tf.keras.models.load_model(model_path)
    class_names = ['anger', 'fear', 'happiness', 'neutrality', 'sadness']

    # Initialize OpenCV Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("--> Initializing video stream. Press 'q' in the window to exit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera source.")
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detectMultiScale identifies multiple bounding boxes natively
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw bounding box rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract Region of Interest (ROI), resize, and normalize
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            # Predict emotion array
            prediction = model.predict(roi, verbose=0)
            max_index = int(np.argmax(prediction))
            predicted_emotion = class_names[max_index]
            confidence = np.max(prediction)
            
            # Format annotation text
            text = f"{predicted_emotion.capitalize()} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Live Multi-Person Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--> Video stream ended cleanly.")

if __name__ == '__main__':
    main()