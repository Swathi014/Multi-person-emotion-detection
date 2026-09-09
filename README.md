```markdown
# Multi-Person Facial Emotion Detection: A Real-Time, Robot-Interactive System

An end-to-end facial emotion recognition (FER) and human-robot interaction (HRI) system implemented using OpenCV, TensorFlow/Keras, and text-to-speech audio feedback[cite: 2]. The pipeline simultaneously detects multiple faces in a live video stream, maintains stable individual identifiers using centroid tracking, stabilizes emotion predictions across frames, and delivers spoken audio responses to the designated primary interacting subject[cite: 2].

---

## Core System Architecture & Features

- **Multi-Face Detection**: Ingests video at $960 \times 720$ resolution, equalizes grayscale contrast using `cv2.equalizeHist`, and localizes all visible frontal faces via OpenCV's Haar Cascade classifier (`scaleFactor=1.1`, `minNeighbors=5`, `minSize=(60, 60)`)[cite: 2].
- **Multi-Target Identity Tracking (`FaceTracker`)**: Computes Euclidean distances between detected face bounding-box centroids and active tracks to assign persistent Person IDs (`MAX_TRACK_DISTANCE = 180` px)[cite: 2]. Retains missing tracks for up to 15 frames to prevent track fragmentation[cite: 2].
- **Deep CNN Classifier**: A compact 3-block convolutional neural network operating on normalized $48 \times 48$ grayscale inputs, incorporating Batch Normalization, Dropout ($p = 0.6$), and in-graph random horizontal flips and rotations[cite: 2].
- **Temporal Prediction Smoothing**: Each tracked person maintains a rolling deque buffer ($k = 12$)[cite: 2]. The displayed emotion updates only when an expression achieves majority stability ($\ge 7$ identical predictions), preventing visual label flickering[cite: 2].
- **Confidence Gating**: Predictions with confidence below $35.0\%$ are marked as `Uncertain` and excluded from the temporal history[cite: 2].
- **Primary Person Resolution & HRI Voice Feedback**: Designates the largest detected face ($\max(w \times h)$) as the primary user[cite: 2]. Generates asynchronous text-to-speech feedback via Windows SAPI5 (`pyttsx3`) governed by a 5-second anti-chatter cooldown timer to eliminate audio collision[cite: 2].
- **HDF5 Direct Weight Extraction**: Includes a custom weight loader that reads `model.weights.h5` directly from the `.keras` zip archive to bypass Keras 2/3 cross-version deserialization issues (`ValueError: Unrecognized keyword arguments: ['batch_shape', 'optional']`)[cite: 2].

---

## Repository Structure

```text
Multi-person-emotion-detection/
│
├── Balanced_Dataset/
│   ├── train/                  # Balanced training sets (~6,700 images/class)
│   │   ├── Angry/
│   │   ├── Happy/
│   │   ├── Neutral/
│   │   ├── Sad/
│   │   └── Surprise/
│   └── test/                   # Held-out test set (8,612 images)
│       ├── Angry/
│       ├── Happy/
│       ├── Neutral/
│       ├── Sad/
│       └── Surprise/
│
├── Emotions_Datasets/          # Original raw, imbalanced dataset
├── evaluation_results/         # Generated evaluation artifacts
│   ├── classification_report.txt
│   └── confusion_matrix.png
│
├── report_figures/             # 300 DPI figures for project documentation
│   ├── Figure_1_accuracy.png
│   ├── Figure_2_loss.png
│   ├── Figure_3_confusion_matrix.png
│   ├── Figure_4_class_distribution_before.png
│   └── Figure_5_class_distribution_after.png
│
├── app.py                      # Main real-time multi-person video application
├── app_backup.py               # Application backup
├── balance_data.py             # Script for balancing class distributions
├── train.py                    # CNN model training script
├── test_image.py               # Static image inference testing script
├── evaluate_model.py           # Independent test set evaluation script
├── generate_report_figures.py  # High-resolution report figure generator
├── emotion_detection_model.keras # Serialized model weights archive
├── requirements.txt            # System dependencies
└── README.md                   # Repository documentation

```

---

## Prerequisites

* **Operating System**: Windows 10/11 (for SAPI5 vocal synthesis) or Linux/macOS


* **Python Version**: Python 3.10 or 3.11


* **Hardware**: Standard webcam ($960 \times 720$ recommended) and audio output device



---

## Installation & Setup

1. **Navigate to the Project Root**:
```cmd
cd "D:\EVOLVE ROBOTICS\Emotion Detection using Opencv\Multi-person-emotion-detection"

```


2. **Configure Virtual Environment**:
```cmd
python -m venv venv
venv\Scripts\activate

```


3. **Install Dependencies**:
Always execute `pip` as a Python module (`python -m pip`) to avoid broken launcher path errors:
```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install seaborn scikit-learn

```



---

## Execution Guide

### 1. Run Real-Time Webcam Application

Launch the live multi-person detection, tracking, and voice feedback interface:

```cmd
python app.py

```

* **HUD Interface**:
* The largest face is highlighted with a white **`PRIMARY`** badge and a thicker bounding box (thickness 4).


* Top status bar displays total person count, current primary user emotion, real-time FPS, detection latency, and CNN inference latency.




* **Exit**: Press **`q`** while focused on the video window to terminate and display performance summaries.



### 2. Run Model Evaluation

Compute quantitative classification metrics across all 8,612 test samples in `Balanced_Dataset/test`:

```cmd
python evaluate_model.py

```

Outputs saved to `evaluation_results/`:

* `classification_report.txt`: Precision, recall, and F1-scores across all 5 classes.


* `confusion_matrix.png`: Normalized confusion matrix visualization.

### 3. Generate Report Figures

Generate publication-quality (300 DPI) training curves and class distributions:

```cmd
python generate_report_figures.py

```

Outputs saved to `report_figures/`:

* `Figure_1_accuracy.png`: Training vs. Validation Accuracy over 30 epochs.
* `Figure_2_loss.png`: Training vs. Validation Cross-Entropy Loss over 30 epochs.
* `Figure_3_confusion_matrix.png`: Formatted normalized confusion matrix plot.
* `Figure_4_class_distribution_before.png`: Class breakdown of the raw dataset.
* `Figure_5_class_distribution_after.png`: Class breakdown of the balanced dataset.

---

## Experimental Benchmark Summary

Evaluated on 8,612 independent test images (`Balanced_Dataset/test`):

| Emotion Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **Angry** | 64.13% | 72.26% | 67.96% | 1,727

 |
| **Happy** | 79.38% | 79.38% | 79.38% | 1,731

 |
| **Neutral** | 99.37% | 73.72% | 84.64% | 1,716

 |
| **Sad** | 61.27% | 66.28% | 63.67% | 1,723

 |
| **Surprise** | 82.76% | 86.76% | 84.71% | 1,715

 |
| **Overall Accuracy** | \multicolumn{3}{c}{**75.67%**} | **8,612**<br> |  |  |
| **Macro Average** | **77.38%** | **75.68%** | **76.07%** | **8,612**<br> |

---

## Affective HRI Vocal Mapping

| Detected Emotion | Vocal Synthesis Response (`pyttsx3`)

 | HUD Bounding Box Color

 |
| --- | --- | --- |
| **Happy** | *"I am glad to see you smiling!"*<br> | Green `(0, 255, 0)`<br> |
| **Sad** | *"Are you okay? I am here for you."*<br> | Light Red `(255, 100, 100)`<br> |
| **Angry** | *"Please stay calm."*<br> | Red `(0, 0, 255)`<br> |
| **Surprise** | *"Oh, that surprised me too!"*<br> | Orange `(0, 165, 255)`<br> |
| **Neutral** | *(Silent observation -- no audio)* | Gray `(200, 200, 200)`<br> |

---

## Troubleshooting

* **`Fatal error in launcher: Unable to create process`**: Caused by moved or renamed virtual environment binaries. Always invoke Python tools with module execution: `python -m pip install <package>`.
* **`InputLayer / batch_shape` Keras Error**: Use `evaluate_model.py` or `app.py`, which load weights layer-by-layer directly from the internal `model.weights.h5` file using `h5py`.


* **Webcam Access Error**: If the camera fails to initialize, verify the index setting (`CAMERA_INDEX = 0` or `CAMERA_INDEX = 1`) in `app.py`.



```

```
