# Multi-Person Emotion Detection - Simple Version

## Setup (one time)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Only 4 dependencies: tensorflow, opencv-python, numpy, pyttsx3. No mediapipe,
no extra helper files, no long dependency chains.

## Run the live webcam demo

```bash
python app.py
```

Shows a box + emotion label over every face the camera sees. Press `q` to quit.

## Test on a single photo

```bash
python test_image.py
```

Opens a file picker, then prints/shows the predicted emotion for each face
in the chosen image.

## How it works

1. **Face detection**: OpenCV's built-in Haar cascade (`haarcascade_frontalface_default.xml`,
   ships with opencv-python — nothing to download).
2. **Emotion prediction**: each detected face is cropped, resized to 48x48
   grayscale, normalized to 0-1, and passed to `emotion_detection_model.keras`.
3. **Voice reaction**: the largest/closest face is treated as the "primary"
   person; when their emotion changes, pyttsx3 speaks a short reaction line
   (at most once every 3 seconds, so it doesn't spam).

## Notes

- Haar cascades are less accurate than MediaPipe for angled/partial faces,
  but they need zero extra downloads and never cause Windows path-length
  or dependency-resolution issues — a good trade-off for getting this
  running reliably.
- If you want the more accurate MediaPipe-based version back later, it's
  a straightforward swap of the detection step; ask and I can add it once
  the basic version is confirmed working end to end.


## Important: model file was repaired

The original `emotion_detection_model.keras` you had was corrupted in a way
that's invisible until you try to load it: it was saved on Windows with a
Keras bug that writes internal weight paths using backslashes instead of
forward slashes. That silently scrambles which saved weights go to which
layer -- which is exactly the "Shape mismatch... conv2d_1... (3,3,32,64)...
(3,3,64,128)" error you kept hitting, on every TensorFlow/Keras version.

This zip contains a **repaired** model file: same architecture, same
trained weights, correctly re-saved so any standard TensorFlow/Keras
version can load it without issue. If you retrain the model on Windows in
the future, re-run it through a Linux/Colab environment (or WSL) before
shipping the `.keras` file, or verify it loads with a quick
`tf.keras.models.load_model(...)` right after saving.
