import io
import os
import zipfile
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= Configuration =================
TEST_DIR = os.path.join("Balanced_Dataset", "test")
MODEL_PATH = "emotion_detection_model.keras"
OUTPUT_DIR = "evaluation_results"

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
# =================================================

def build_emotion_model():
    """Recreates the architecture contained in emotion_detection_model.keras."""
    preprocessing = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(48, 48, 1), name="input_layer"),
            tf.keras.layers.RandomFlip(mode="horizontal", name="random_flip"),
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
            tf.keras.layers.InputLayer(input_shape=(48, 48, 1), name="input_layer_1"),
            preprocessing,
            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"),
            tf.keras.layers.BatchNormalization(name="batch_normalization"),
            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_1"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
            tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
            # Block 3
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_2"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"),
            tf.keras.layers.BatchNormalization(name="batch_normalization_2"),
            # Classifier
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", name="dense"),
            tf.keras.layers.Dropout(0.6, name="dropout"),
            tf.keras.layers.Dense(5, activation="softmax", name="dense_1"),
        ],
        name="emotion_model",
    )
    return model

def load_emotion_model(model_path):
    """Manually loads model weights from the .keras archive."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found:\n{os.path.abspath(model_path)}")

    print("Building model architecture...")
    model = build_emotion_model()
    dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
    model(dummy_input, training=False)

    print("Reading weights from model file...")
    with zipfile.ZipFile(model_path, "r") as archive:
        if "model.weights.h5" not in archive.namelist():
            raise RuntimeError("model.weights.h5 not found in " + model_path)
        weight_data = archive.read("model.weights.h5")

    print("Loading model weights...")
    with h5py.File(io.BytesIO(weight_data), "r") as weights_file:
        layers = weights_file["layers"]
        model.get_layer("conv2d").set_weights([
            np.array(layers["conv2d"]["vars"]["0"]),
            np.array(layers["conv2d"]["vars"]["1"])
        ])
        model.get_layer("batch_normalization").set_weights([
            np.array(layers["batch_normalization"]["vars"]["0"]),
            np.array(layers["batch_normalization"]["vars"]["1"]),
            np.array(layers["batch_normalization"]["vars"]["2"]),
            np.array(layers["batch_normalization"]["vars"]["3"])
        ])
        model.get_layer("conv2d_1").set_weights([
            np.array(layers["conv2d_1"]["vars"]["0"]),
            np.array(layers["conv2d_1"]["vars"]["1"])
        ])
        model.get_layer("batch_normalization_1").set_weights([
            np.array(layers["batch_normalization_1"]["vars"]["0"]),
            np.array(layers["batch_normalization_1"]["vars"]["1"]),
            np.array(layers["batch_normalization_1"]["vars"]["2"]),
            np.array(layers["batch_normalization_1"]["vars"]["3"])
        ])
        model.get_layer("conv2d_2").set_weights([
            np.array(layers["conv2d_2"]["vars"]["0"]),
            np.array(layers["conv2d_2"]["vars"]["1"])
        ])
        model.get_layer("batch_normalization_2").set_weights([
            np.array(layers["batch_normalization_2"]["vars"]["0"]),
            np.array(layers["batch_normalization_2"]["vars"]["1"]),
            np.array(layers["batch_normalization_2"]["vars"]["2"]),
            np.array(layers["batch_normalization_2"]["vars"]["3"])
        ])
        model.get_layer("dense").set_weights([
            np.array(layers["dense"]["vars"]["0"]),
            np.array(layers["dense"]["vars"]["1"])
        ])
        model.get_layer("dense_1").set_weights([
            np.array(layers["dense_1"]["vars"]["0"]),
            np.array(layers["dense_1"]["vars"]["1"])
        ])

    print("Model weights loaded successfully.")
    return model

def run_evaluation():
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found at: {TEST_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test data...")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    test_generator = test_datagen.flow_from_directory(
        directory=TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    class_labels = list(test_generator.class_indices.keys())
    print(f"Classes detected: {class_labels}")

    model = load_emotion_model(MODEL_PATH)

    print("Evaluating test predictions...")
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    acc = accuracy_score(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:\n")
    print(cls_report)

    report_file_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_file_path, "w") as f:
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report)
    print(f"[✓] Saved metrics report to: {report_file_path}")

    # Plot normalized confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title("Emotion Classification Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_img_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=300)
    plt.close()
    print(f"[✓] Saved confusion matrix plot to: {cm_img_path}")

if __name__ == "__main__":
    run_evaluation()