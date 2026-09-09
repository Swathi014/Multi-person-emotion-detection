# Multi-Person Facial Emotion Detection System

A real-time facial emotion detection and Human-Robot Interaction (HRI) system using OpenCV, TensorFlow/Keras, and text-to-speech feedback. The pipeline detects multiple faces, tracks individual subjects across frames, stabilizes emotion predictions, and provides vocal responses to the primary interacting user.

---

## Features

- **Multi-Face Detection & Tracking**: Detects multiple faces concurrently and tracks each subject using centroid Euclidean distance to assign stable Person IDs.
- **Temporal Prediction Smoothing**: Uses a rolling history queue ($k = 12$) with majority voting to eliminate frame-level emotion flickering.
- **Confidence Filtering**: Flags low-certainty predictions ($< 35\%$) as `Uncertain` to avoid false triggers.
- **Primary Person HRI Feedback**: Automatically selects the largest detected face as the primary subject and speaks an emotion-specific response using `pyttsx3` with a 5-second cooldown.
- **Framework Compatibility**: Custom HDF5 weight loader bypasses Keras version deserialization issues when running newer `.keras` files on TensorFlow 2.13 environments.

---

## Project Structure

```text
Multi-person-emotion-detection/
│
├── Balanced_Dataset/
│   ├── train/               # Balanced training images across 5 classes
│   └── test/                # Test split used for evaluation (8,612 images)
│
├── Emotions_Datasets/       # Original raw dataset
├── evaluation_results/      # Saved classification report and confusion matrix
├── report_figures/          # Generated experimental figures (1 to 5)
│
├── app.py                   # Main real-time multi-person webcam application
├── evaluate_model.py        # Model evaluation script (accuracy, precision, recall, F1)
├── generate_report_figures.py # Script to plot accuracy, loss, and distribution graphs
├── balance_data.py          # Script used to balance the raw dataset
├── emotion_detection_model.keras # Trained model weights archive
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation