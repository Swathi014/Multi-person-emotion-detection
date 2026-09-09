"""
Multi-Person Emotion Detection - Improved Live Webcam Demo
------------------------------------------------------------

Features:
- Multiple face detection
- Stable Person IDs
- Emotion smoothing over multiple frames
- Confidence smoothing
- Per-person emotion history
- Voice reactions with cooldown
- Better OpenCV UI
- Loads the newer .keras model manually so it works with
  TensorFlow 2.13 / Keras 2.13
"""

import io
import os
import time
import zipfile
from collections import Counter, deque

import cv2
import h5py
import numpy as np
import tensorflow as tf
import pyttsx3


# ======================================================================
# SETTINGS
# ======================================================================

MODEL_PATH = "emotion_detection_model.keras"

EMOTIONS = [
    "Angry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

REACTIONS = {
    "Happy": "I am glad to see you smiling!",
    "Sad": "Are you okay? I am here for you.",
    "Angry": "Please stay calm.",
    "Surprise": "Oh, that surprised me too!",
}

# --------------------------------------------------------------
# Face detection
# --------------------------------------------------------------

FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (60, 60)

# --------------------------------------------------------------
# Emotion smoothing
# --------------------------------------------------------------

# Number of predictions kept for each person.
EMOTION_HISTORY_SIZE = 12

# Minimum number of identical predictions required before
# changing the displayed emotion.
MIN_STABLE_COUNT = 7

# Minimum confidence required before accepting a prediction.
MIN_CONFIDENCE = 35.0

# --------------------------------------------------------------
# Person tracking
# --------------------------------------------------------------

# Maximum distance in pixels between the previous center and
# current center for the same person.
MAX_TRACK_DISTANCE = 180

# Number of frames a person can disappear before being removed.
MAX_MISSING_FRAMES = 15

# --------------------------------------------------------------
# Voice
# --------------------------------------------------------------

SPEAK_COOLDOWN = 5

# Set to True if you want voice reactions.
VOICE_ENABLED = True

# --------------------------------------------------------------
# Camera
# --------------------------------------------------------------

CAMERA_INDEX = 0

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

# --------------------------------------------------------------
# Performance monitoring
# --------------------------------------------------------------
FPS_SMOOTHING = 0.9


# ======================================================================
# MODEL ARCHITECTURE
# ======================================================================

def build_emotion_model():
    """
    Recreates the architecture contained in emotion_detection_model.keras.
    """

    preprocessing = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(
                input_shape=(48, 48, 1),
                name="input_layer",
            ),

            tf.keras.layers.RandomFlip(
                mode="horizontal",
                name="random_flip",
            ),

            tf.keras.layers.RandomRotation(
                factor=(-0.05, 0.05),
                fill_mode="reflect",
                interpolation="bilinear",
                name="random_rotation",
            ),
        ],
        name="sequential",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(
                input_shape=(48, 48, 1),
                name="input_layer_1",
            ),

            preprocessing,

            # ----------------------------------------------------------
            # Block 1
            # ----------------------------------------------------------

            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                name="conv2d",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid",
                name="max_pooling2d",
            ),

            tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                name="batch_normalization",
            ),

            # ----------------------------------------------------------
            # Block 2
            # ----------------------------------------------------------

            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                name="conv2d_1",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid",
                name="max_pooling2d_1",
            ),

            tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                name="batch_normalization_1",
            ),

            # ----------------------------------------------------------
            # Block 3
            # ----------------------------------------------------------

            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                name="conv2d_2",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid",
                name="max_pooling2d_2",
            ),

            tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                name="batch_normalization_2",
            ),

            # ----------------------------------------------------------
            # Classifier
            # ----------------------------------------------------------

            tf.keras.layers.Flatten(
                name="flatten",
            ),

            tf.keras.layers.Dense(
                128,
                activation="relu",
                name="dense",
            ),

            tf.keras.layers.Dropout(
                0.6,
                name="dropout",
            ),

            tf.keras.layers.Dense(
                5,
                activation="softmax",
                name="dense_1",
            ),
        ],
        name="emotion_model",
    )

    return model


# ======================================================================
# LOAD MODEL
# ======================================================================

def load_emotion_model(model_path):
    """
    Manually loads model weights from the .keras archive.

    This avoids the InputLayer compatibility problem caused by loading
    a newer Keras model with TensorFlow/Keras 2.13.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found:\n{os.path.abspath(model_path)}"
        )

    print("Building model architecture...")

    model = build_emotion_model()

    # Build variables
    dummy_input = np.zeros(
        (1, 48, 48, 1),
        dtype=np.float32,
    )

    model(dummy_input, training=False)

    print("Reading weights from model file...")

    with zipfile.ZipFile(model_path, "r") as archive:

        if "model.weights.h5" not in archive.namelist():
            raise RuntimeError(
                "model.weights.h5 was not found inside "
                + model_path
            )

        weight_data = archive.read(
            "model.weights.h5"
        )

    print("Loading model weights...")

    with h5py.File(
        io.BytesIO(weight_data),
        "r",
    ) as weights_file:

        layers = weights_file["layers"]

        # --------------------------------------------------------------
        # Conv2D 32
        # --------------------------------------------------------------

        model.get_layer("conv2d").set_weights(
            [
                np.array(
                    layers["conv2d"]["vars"]["0"]
                ),
                np.array(
                    layers["conv2d"]["vars"]["1"]
                ),
            ]
        )

        # --------------------------------------------------------------
        # BatchNorm 32
        # --------------------------------------------------------------

        model.get_layer(
            "batch_normalization"
        ).set_weights(
            [
                np.array(
                    layers["batch_normalization"]["vars"]["0"]
                ),
                np.array(
                    layers["batch_normalization"]["vars"]["1"]
                ),
                np.array(
                    layers["batch_normalization"]["vars"]["2"]
                ),
                np.array(
                    layers["batch_normalization"]["vars"]["3"]
                ),
            ]
        )

        # --------------------------------------------------------------
        # Conv2D 64
        # --------------------------------------------------------------

        model.get_layer("conv2d_1").set_weights(
            [
                np.array(
                    layers["conv2d_1"]["vars"]["0"]
                ),
                np.array(
                    layers["conv2d_1"]["vars"]["1"]
                ),
            ]
        )

        # --------------------------------------------------------------
        # BatchNorm 64
        # --------------------------------------------------------------

        model.get_layer(
            "batch_normalization_1"
        ).set_weights(
            [
                np.array(
                    layers["batch_normalization_1"]["vars"]["0"]
                ),
                np.array(
                    layers["batch_normalization_1"]["vars"]["1"]
                ),
                np.array(
                    layers["batch_normalization_1"]["vars"]["2"]
                ),
                np.array(
                    layers["batch_normalization_1"]["vars"]["3"]
                ),
            ]
        )

        # --------------------------------------------------------------
        # Conv2D 128
        # --------------------------------------------------------------

        model.get_layer("conv2d_2").set_weights(
            [
                np.array(
                    layers["conv2d_2"]["vars"]["0"]
                ),
                np.array(
                    layers["conv2d_2"]["vars"]["1"]
                ),
            ]
        )

        # --------------------------------------------------------------
        # BatchNorm 128
        # --------------------------------------------------------------

        model.get_layer(
            "batch_normalization_2"
        ).set_weights(
            [
                np.array(
                    layers["batch_normalization_2"]["vars"]["0"]
                ),
                np.array(
                    layers["batch_normalization_2"]["vars"]["1"]
                ),
                np.array(
                    layers["batch_normalization_2"]["vars"]["2"]
                ),
                np.array(
                    layers["batch_normalization_2"]["vars"]["3"]
                ),
            ]
        )

        # --------------------------------------------------------------
        # Dense 2048 -> 128
        # --------------------------------------------------------------

        model.get_layer("dense").set_weights(
            [
                np.array(
                    layers["dense"]["vars"]["0"]
                ),
                np.array(
                    layers["dense"]["vars"]["1"]
                ),
            ]
        )

        # --------------------------------------------------------------
        # Dense 128 -> 5
        # --------------------------------------------------------------

        model.get_layer("dense_1").set_weights(
            [
                np.array(
                    layers["dense_1"]["vars"]["0"]
                ),
                np.array(
                    layers["dense_1"]["vars"]["1"]
                ),
            ]
        )

    print("Model weights loaded successfully.")

    return model


# ======================================================================
# PERSON TRACK
# ======================================================================

class PersonTrack:

    def __init__(self, person_id, box):

        self.person_id = person_id

        self.x, self.y, self.w, self.h = box

        self.center = self.get_center(box)

        self.emotion_history = deque(
            maxlen=EMOTION_HISTORY_SIZE
        )

        self.confidence_history = deque(
            maxlen=EMOTION_HISTORY_SIZE
        )

        self.current_emotion = "Detecting..."

        self.current_confidence = 0.0

        self.missing_frames = 0

        self.last_seen = time.time()

    @staticmethod
    def get_center(box):

        x, y, w, h = box

        return (
            x + w // 2,
            y + h // 2,
        )

    def update(self, box):

        self.x, self.y, self.w, self.h = box

        self.center = self.get_center(box)

        self.missing_frames = 0

        self.last_seen = time.time()

    def add_prediction(
        self,
        emotion,
        confidence,
    ):

        self.emotion_history.append(
            emotion
        )

        self.confidence_history.append(
            confidence
        )

        # ----------------------------------------------------------
        # Find most common emotion
        # ----------------------------------------------------------

        counts = Counter(
            self.emotion_history
        )

        most_common_emotion, count = (
            counts.most_common(1)[0]
        )

        # ----------------------------------------------------------
        # Only change emotion after enough consistent predictions
        # ----------------------------------------------------------

        if count >= MIN_STABLE_COUNT:

            self.current_emotion = (
                most_common_emotion
            )

            matching_confidences = [
                conf
                for emo, conf in zip(
                    self.emotion_history,
                    self.confidence_history,
                )
                if emo == most_common_emotion
            ]

            if matching_confidences:

                self.current_confidence = float(
                    np.mean(
                        matching_confidences
                    )
                )

        return (
            self.current_emotion,
            self.current_confidence,
        )


# ======================================================================
# FACE TRACKER
# ======================================================================

class FaceTracker:

    def __init__(self):

        self.people = {}

        self.next_person_id = 1

    def _distance(
        self,
        point1,
        point2,
    ):

        return np.sqrt(
            (point1[0] - point2[0]) ** 2
            + (point1[1] - point2[1]) ** 2
        )

    def update(self, detected_faces):

        """
        Match newly detected faces with existing people.
        """

        if not detected_faces:

            for person in self.people.values():

                person.missing_frames += 1

            self._remove_old_people()

            return []

        current_centers = []

        for box in detected_faces:

            x, y, w, h = box

            center = (
                x + w // 2,
                y + h // 2,
            )

            current_centers.append(
                center
            )

        matched_people = set()

        # Keep a single visible person on the same track despite small
        # Haar-Cascade bounding-box jitter.
        if len(detected_faces) == 1 and len(self.people) == 1:
            person = next(iter(self.people.values()))
            person.update(detected_faces[0])
            person.missing_frames = 0
            return [person]

        results = []

        # --------------------------------------------------------------
        # Match each detected face
        # --------------------------------------------------------------

        for box, center in zip(
            detected_faces,
            current_centers,
        ):

            best_person = None

            best_distance = float("inf")

            for person_id, person in self.people.items():

                if person_id in matched_people:
                    continue

                distance = self._distance(
                    center,
                    person.center,
                )

                if (
                    distance
                    < best_distance
                    and distance
                    <= MAX_TRACK_DISTANCE
                ):

                    best_distance = distance

                    best_person = person

            # ----------------------------------------------------------
            # Existing person
            # ----------------------------------------------------------

            if best_person is not None:

                best_person.update(box)

                matched_people.add(
                    best_person.person_id
                )

                results.append(
                    best_person
                )

            # ----------------------------------------------------------
            # New person
            # ----------------------------------------------------------

            else:

                person = PersonTrack(
                    self.next_person_id,
                    box,
                )

                self.people[
                    self.next_person_id
                ] = person

                matched_people.add(
                    self.next_person_id
                )

                self.next_person_id += 1

                results.append(
                    person
                )

        # --------------------------------------------------------------
        # Mark unmatched people
        # --------------------------------------------------------------

        for person_id, person in self.people.items():

            if person_id not in matched_people:

                person.missing_frames += 1

        self._remove_old_people()

        return results

    def _remove_old_people(self):

        remove_ids = []

        for person_id, person in self.people.items():

            if (
                person.missing_frames
                > MAX_MISSING_FRAMES
            ):

                remove_ids.append(
                    person_id
                )

        for person_id in remove_ids:

            del self.people[
                person_id
            ]


# ======================================================================
# COLOR FUNCTIONS
# ======================================================================

def get_emotion_color(emotion):

    colors = {

        "Happy": (
            0,
            255,
            0,
        ),

        "Sad": (
            255,
            100,
            100,
        ),

        "Angry": (
            0,
            0,
            255,
        ),

        "Surprise": (
            0,
            165,
            255,
        ),

        "Neutral": (
            200,
            200,
            200,
        ),

    }

    return colors.get(
        emotion,
        (
            0,
            255,
            255,
        ),
    )


# ======================================================================
# DRAW PERSON
# ======================================================================

def draw_person(
    frame,
    person,
    display_emotion=None,
    display_confidence=None,
    is_primary=False,
):

    x = person.x
    y = person.y
    w = person.w
    h = person.h

    emotion = (
        display_emotion
        if display_emotion is not None
        else person.current_emotion
    )

    confidence = (
        display_confidence
        if display_confidence is not None
        else person.current_confidence
    )

    color = get_emotion_color(
        emotion
    )

    # --------------------------------------------------------------
    # Face rectangle
    # --------------------------------------------------------------

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        4 if is_primary else 3,
    )

    # Clearly mark the largest face as the primary person.
    if is_primary:
        primary_text = "PRIMARY"
        (pw, ph), _ = cv2.getTextSize(
            primary_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        py = max(ph + 8, y - 38)
        cv2.rectangle(
            frame,
            (x, py - ph - 6),
            (x + pw + 10, py + 5),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            frame,
            primary_text,
            (x + 5, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )

    # --------------------------------------------------------------
    # Person ID
    # --------------------------------------------------------------

    person_text = (
        f"Person {person.person_id}"
    )

    cv2.putText(
        frame,
        person_text,
        (x, y + h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

    # --------------------------------------------------------------
    # Emotion label
    # --------------------------------------------------------------

    if emotion != "Detecting...":

        emotion_text = (
            f"{emotion} "
            f"({confidence:.0f}%)"
        )

    else:

        emotion_text = emotion

    # Background for label
    (text_w, text_h), _ = cv2.getTextSize(
        emotion_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )

    label_y = max(
        text_h + 10,
        y - 8,
    )

    cv2.rectangle(
        frame,
        (
            x,
            label_y - text_h - 8,
        ),
        (
            x + text_w + 10,
            label_y + 5,
        ),
        color,
        -1,
    )

    # Use black text for readability
    cv2.putText(
        frame,
        emotion_text,
        (x + 5, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
    )

    # --------------------------------------------------------------
    # Confidence bar
    # --------------------------------------------------------------

    if confidence > 0:

        bar_width = w

        filled_width = int(
            bar_width
            * min(confidence, 100)
            / 100
        )

        bar_y = y + h + 35

        cv2.rectangle(
            frame,
            (x, bar_y),
            (
                x + bar_width,
                bar_y + 8,
            ),
            (50, 50, 50),
            -1,
        )

        cv2.rectangle(
            frame,
            (x, bar_y),
            (
                x + filled_width,
                bar_y + 8,
            ),
            color,
            -1,
        )


# ======================================================================
# VOICE FUNCTION
# ======================================================================

def speak_reaction(
    engine,
    emotion,
):
    """Speak the reaction reliably on Windows using SAPI/pyttsx3."""

    if emotion not in REACTIONS:
        return

    message = REACTIONS[emotion]
    print(f"Voice reaction: {message}")

    speech_engine = None

    try:
        # Reinitialize SAPI for every utterance. This avoids a common
        # Windows pyttsx3 issue where only the first utterance is heard.
        speech_engine = pyttsx3.init(driverName="sapi5")
        speech_engine.setProperty("rate", 150)
        speech_engine.setProperty("volume", 1.0)
        speech_engine.say(message)
        speech_engine.runAndWait()

    except Exception as error:
        print(f"Voice error: {error}")

    finally:
        if speech_engine is not None:
            try:
                speech_engine.stop()
            except Exception:
                pass


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 65)
    print("MULTI-PERSON EMOTION DETECTION")
    print("=" * 65)

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    print("Loading model...")

    model = load_emotion_model(
        MODEL_PATH
    )

    # --------------------------------------------------------------
    # Face detector
    # --------------------------------------------------------------

    print(
        "Loading face detector..."
    )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():

        raise RuntimeError(
            "Could not load OpenCV Haar cascade."
        )

    # --------------------------------------------------------------
    # Voice engine
    # --------------------------------------------------------------

    engine = None

    if VOICE_ENABLED:

        print(
            "Starting voice engine..."
        )

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            150,
        )

    # --------------------------------------------------------------
    # Camera
    # --------------------------------------------------------------

    print(
        "Starting webcam..."
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    # Try to use HD resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    # --------------------------------------------------------------
    # Face tracker
    # --------------------------------------------------------------

    tracker = FaceTracker()

    # --------------------------------------------------------------
    # Voice state
    # --------------------------------------------------------------

    last_spoken = {}

    last_speak_time = {}

    # Performance metrics
    fps = 0.0
    frame_count = 0
    total_detection_time = 0.0
    total_inference_time = 0.0
    total_frame_time = 0.0

    print("=" * 65)
    print("READY!")
    print(
        "Press 'q' to quit."
    )
    print("=" * 65)

    # ==================================================================
    # MAIN CAMERA LOOP
    # ==================================================================

    while True:

        frame_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret:

            print(
                "Camera read failed."
            )

            break

        # ----------------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------------

        frame = cv2.flip(
            frame,
            1,
        )

        # ----------------------------------------------------------
        # Grayscale
        # ----------------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # ----------------------------------------------------------
        # Improve contrast
        # ----------------------------------------------------------

        gray = cv2.equalizeHist(
            gray
        )

        # ----------------------------------------------------------
        # Detect faces
        # ----------------------------------------------------------

        detection_start = time.perf_counter()
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE,
        )

        detection_time = time.perf_counter() - detection_start
        total_detection_time += detection_time

        # Convert to regular list
        detected_faces = [
            (
                int(x),
                int(y),
                int(w),
                int(h),
            )
            for (
                x,
                y,
                w,
                h,
            ) in faces
        ]

        # ----------------------------------------------------------
        # Update tracker
        # ----------------------------------------------------------

        people = tracker.update(
            detected_faces
        )

        # ----------------------------------------------------------
        # Select primary person
        # ----------------------------------------------------------
        # The largest detected face is the primary person.
        primary_person = None
        if people:
            primary_person = max(
                people,
                key=lambda p: p.w * p.h,
            )

        # ----------------------------------------------------------
        # Process every person
        # ----------------------------------------------------------

        inference_start = time.perf_counter()

        for person in people:

            x = person.x
            y = person.y
            w = person.w
            h = person.h

            # ------------------------------------------------------
            # Make sure coordinates are valid
            # ------------------------------------------------------

            x1 = max(
                0,
                x,
            )

            y1 = max(
                0,
                y,
            )

            x2 = min(
                gray.shape[1],
                x + w,
            )

            y2 = min(
                gray.shape[0],
                y + h,
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):

                continue

            # ------------------------------------------------------
            # Crop face
            # ------------------------------------------------------

            roi = gray[
                y1:y2,
                x1:x2,
            ]

            if roi.size == 0:

                continue

            # ------------------------------------------------------
            # Resize
            # ------------------------------------------------------

            roi = cv2.resize(
                roi,
                (48, 48),
                interpolation=cv2.INTER_AREA,
            )

            # ------------------------------------------------------
            # Normalize
            # ------------------------------------------------------

            roi = (
                roi.astype(
                    np.float32
                )
                / 255.0
            )

            # ------------------------------------------------------
            # Model input
            # ------------------------------------------------------

            roi = roi.reshape(
                1,
                48,
                48,
                1,
            )

            # ------------------------------------------------------
            # Prediction
            # ------------------------------------------------------

            prediction = model.predict(
                roi,
                verbose=0,
            )[0]

            # ------------------------------------------------------
            # Best emotion
            # ------------------------------------------------------

            idx = int(
                np.argmax(
                    prediction
                )
            )

            emotion = EMOTIONS[
                idx
            ]

            confidence = float(
                prediction[idx]
                * 100
            )

            # ------------------------------------------------------
            # Confidence filtering
            # ------------------------------------------------------

            if confidence < MIN_CONFIDENCE:
                # Low-confidence predictions are shown as uncertain and are
                # not added to the temporal history.
                draw_person(
                    frame,
                    person,
                    display_emotion="Uncertain",
                    display_confidence=confidence,
                    is_primary=(person is primary_person),
                )
                continue

            # ------------------------------------------------------
            # Add prediction to person's history
            # ------------------------------------------------------

            old_emotion = (
                person.current_emotion
            )

            new_emotion, new_confidence = (
                person.add_prediction(
                    emotion,
                    confidence,
                )
            )

            # ------------------------------------------------------
            # Draw
            # ------------------------------------------------------

            draw_person(
                frame,
                person,
                is_primary=(person is primary_person),
            )

        inference_time = time.perf_counter() - inference_start
        total_inference_time += inference_time

        # --------------------------------------------------------------
        # Primary-person voice reaction
        # --------------------------------------------------------------
        # The largest detected face is treated as the primary person.
        # Only the primary person's stable emotion can trigger speech.
        # The primary person is always the largest detected face.
        # Voice feedback is triggered only when that person's stabilized
        # emotion has a defined reaction.
        if (
            VOICE_ENABLED
            and primary_person is not None
            and primary_person.current_emotion in REACTIONS
        ):
            primary_emotion = primary_person.current_emotion
            current_time = time.time()

            previous_spoken = last_spoken.get(
                primary_person.person_id
            )
            previous_time = last_speak_time.get(
                primary_person.person_id,
                0,
            )

            emotion_changed = (
                primary_emotion != previous_spoken
            )
            enough_time = (
                current_time - previous_time
                >= SPEAK_COOLDOWN
            )

            if emotion_changed and enough_time:
                print(
                    f"Primary Person "
                    f"{primary_person.person_id}: "
                    f"{primary_emotion}"
                )
                speak_reaction(
                    engine,
                    primary_emotion,
                )
                last_spoken[
                    primary_person.person_id
                ] = primary_emotion
                last_speak_time[
                    primary_person.person_id
                ] = current_time

        # --------------------------------------------------------------
        # Frame performance
        # --------------------------------------------------------------
        frame_time = time.perf_counter() - frame_start
        total_frame_time += frame_time
        frame_count += 1

        instant_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        if fps <= 0:
            fps = instant_fps
        else:
            fps = FPS_SMOOTHING * fps + (1.0 - FPS_SMOOTHING) * instant_fps

        # ==============================================================
        # TOP STATUS BAR
        # =============================================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                frame.shape[1],
                65,
            ),
            (20, 20, 20),
            -1,
        )

        # Slight transparency
        frame = cv2.addWeighted(
            overlay,
            0.75,
            frame,
            0.25,
            0,
        )

        # --------------------------------------------------------------
        # Title
        # --------------------------------------------------------------

        cv2.putText(
            frame,
            "MULTI-PERSON EMOTION DETECTION",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # --------------------------------------------------------------
        # People count
        # --------------------------------------------------------------

        count_text = f"People: {len(people)}"

        cv2.putText(
            frame,
            count_text,
            (frame.shape[1] - 150, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        # Primary-person status
        primary_status = "Primary: None"
        if primary_person is not None:
            primary_emotion_display = primary_person.current_emotion
            if primary_emotion_display == "Detecting...":
                primary_status = f"Primary: Person {primary_person.person_id}"
            else:
                primary_status = (
                    f"Primary: P{primary_person.person_id} - "
                    f"{primary_emotion_display}"
                )

        cv2.putText(
            frame,
            primary_status,
            (frame.shape[1] - 350, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )

        # --------------------------------------------------------------
        # Performance metrics
        # --------------------------------------------------------------
        perf_text = (
            f"FPS: {fps:.1f} | Detect: {detection_time * 1000:.0f} ms | "
            f"Inference: {inference_time * 1000:.0f} ms"
        )

        cv2.putText(
            frame,
            perf_text,
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
        )

        # --------------------------------------------------------------
        # Demo legend
        # --------------------------------------------------------------

        legend_text = "PRIMARY = largest face | Voice reacts only to PRIMARY"
        cv2.putText(
            frame,
            legend_text,
            (15, frame.shape[0] - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        # --------------------------------------------------------------
        # Instructions
        # --------------------------------------------------------------

        cv2.putText(
            frame,
            "Press Q to quit",
            (
                15,
                frame.shape[0] - 15,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        # --------------------------------------------------------------
        # Show frame
        # --------------------------------------------------------------

        cv2.imshow(
            "Multi-Person Emotion Detection",
            frame,
        )

        # --------------------------------------------------------------
        # Quit
        # --------------------------------------------------------------

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break

    # ==================================================================
    # CLEANUP
    # ==================================================================

    print(
        "Stopping..."
    )

    if frame_count > 0:
        avg_detection_ms = (total_detection_time / frame_count) * 1000
        avg_inference_ms = (total_inference_time / frame_count) * 1000
        avg_frame_ms = (total_frame_time / frame_count) * 1000
        avg_fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0

        print("Performance summary:")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Average face detection time: {avg_detection_ms:.2f} ms")
        print(f"  Average inference time: {avg_inference_ms:.2f} ms")
        print(f"  Average frame processing time: {avg_frame_ms:.2f} ms")

    cap.release()

    cv2.destroyAllWindows()

    print(
        "Program ended."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()
