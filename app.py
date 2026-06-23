import os
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time

def robot_speak(text):
    """Threaded Voice Helper to prevent video feed lagging/freezing"""
    def speech_worker(words):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 145) 
            engine.say(words)
            engine.runAndWait()
        except Exception as e:
            pass 

    threading.Thread(target=speech_worker, args=(text,), daemon=True).start()

def draw_robot_face(emotion):
    """Draws a minimalist digital robot face canvas based on current emotion."""
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    color = (255, 255, 0) # Cyan
    
    if emotion == 'Happy':
        cv2.circle(canvas, (130, 150), 30, color, -1)
        cv2.circle(canvas, (270, 150), 30, color, -1)
        cv2.rectangle(canvas, (90, 150), (170, 190), (0, 0, 0), -1)
        cv2.rectangle(canvas, (230, 150), (310, 190), (0, 0, 0), -1)
        cv2.ellipse(canvas, (200, 260), (60, 40), 0, 0, 180, color, 5)
        
    elif emotion == 'Sad':
        cv2.line(canvas, (100, 160), (150, 140), color, 5)
        cv2.line(canvas, (300, 160), (250, 140), color, 5)
        cv2.ellipse(canvas, (200, 290), (50, 30), 0, 180, 360, color, 5)
        
    elif emotion == 'Angry':
        cv2.circle(canvas, (130, 160), 25, color, -1)
        cv2.circle(canvas, (270, 160), 25, color, -1)
        cv2.line(canvas, (90, 120), (160, 150), color, 6)
        cv2.line(canvas, (310, 120), (240, 150), color, 6)
        cv2.line(canvas, (150, 270), (250, 270), color, 5)
        
    elif emotion in ['Suprise', 'Surprised']:
        cv2.circle(canvas, (130, 150), 35, color, 4)
        cv2.circle(canvas, (270, 150), 35, color, 4)
        cv2.circle(canvas, (200, 270), 25, color, 4)
        
    else: # Neutral / Default state
        cv2.circle(canvas, (130, 150), 25, color, -1)
        cv2.circle(canvas, (270, 150), 25, color, -1)
        cv2.line(canvas, (160, 270), (240, 270), color, 4)
        
    return canvas

def handle_robot_reaction(emotion):
    """Returns spoken words and updates text labels based on image requirements."""
    if emotion == 'Happy':
        return "I am glad to see you smiling back!", "*Robot smiles back*"
    elif emotion == 'Sad':
        return "Are you okay? Let me offer you some comfort.", "Robot asks: 'Are you okay?' & offers comfort"
    elif emotion == 'Angry':
        return "Please remain calm.", "*Robot assumes calm expression*"
    elif emotion in ['Suprise', 'Surprised']:
        return "Oh wow! You caught me off guard!", "*Robot reacts with surprise*"
    else:
        return "", "*Robot observing*"

def main():
    model_path = 'emotion_detection_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model file not found. Run train.py first.")

    model = tf.keras.models.load_model(model_path)
    class_names = ['Angry', 'Happy', 'Sad', 'Suprise'] 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("--> Launching Interface. Press 'q' to shut down demo.")
    cap = cv2.VideoCapture(0)

    # --- STABILIZATION VARIABLES ---
    STABILITY_THRESHOLD = 1.5  # Seconds required to hold an expression
    candidate_emotion = "Neutral"
    emotion_start_time = time.time()
    
    current_stable_emotion = "Neutral"
    last_spoken_emotion = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (500, 400))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        detected_this_frame = False

        for (x, y, w, h) in faces:
            detected_this_frame = True
            
            # 1. Classify Emotion
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            prediction = model.predict(roi, verbose=0)
            predicted_emotion = class_names[int(np.argmax(prediction))]
            
            # Draw tracking box on operator view
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"Detecting: {predicted_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # --- EMOTION TIMER LOGIC ---
            if predicted_emotion == candidate_emotion:
                # If they held the face long enough, lock it in
                if time.time() - emotion_start_time >= STABILITY_THRESHOLD:
                    current_stable_emotion = candidate_emotion
            else:
                # Expression changed, restart the timer
                candidate_emotion = predicted_emotion
                emotion_start_time = time.time()
                
            break # Process only the first detected face

        if not detected_this_frame:
            candidate_emotion = "Neutral"
            current_stable_emotion = "Neutral"

        # 2. Extract Speech and Action rules based on the STABLE emotion
        speech_phrase, action_text = handle_robot_reaction(current_stable_emotion)

        # 3. Handle Voice Event Triggers (Trigger only when stable emotion shifts)
        if current_stable_emotion != last_spoken_emotion and speech_phrase != "":
            robot_speak(speech_phrase)
            last_spoken_emotion = current_stable_emotion
        elif current_stable_emotion == "Neutral":
            last_spoken_emotion = None 

        # 4. Generate the Face Element 
        robot_face_img = draw_robot_face(current_stable_emotion)

        # 5. Composite UI Panel
        combined_ui = np.hstack((frame, robot_face_img))
        
        cv2.rectangle(combined_ui, (0, 365), (900, 400), (20, 20, 20), -1)
        
        # Show what the robot has locked onto
        cv2.putText(combined_ui, f"Stable State: {current_stable_emotion.upper()}", (20, 390), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(combined_ui, f"Action: {action_text}", (520, 390), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Robot Interaction Operating Console', combined_ui)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()