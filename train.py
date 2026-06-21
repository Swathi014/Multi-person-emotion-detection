import os
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    """Builds a regularized CNN architecture for emotion recognition."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    # Update these paths relative to your local setup
    TRAIN_DIR = r'./Emotions_Datasets/train'
    TEST_DIR = r'./Emotions_Datasets/test'
    
    IMG_HEIGHT, IMG_WIDTH = 48, 48
    BATCH_SIZE = 32
    EPOCHS = 20

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        raise FileNotFoundError("Dataset directories not found. Please check TRAIN_DIR and TEST_DIR paths.")

    print("--> Loading and preprocessing training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        color_mode='grayscale',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    print("--> Loading and preprocessing validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        color_mode='grayscale',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    # Standardize pixel values to [0, 1]
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Optimize datasets for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Instantiate and compile model
    model = create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=len(class_names))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("--> Starting model training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save('emotion_detection_model.h5')
    print("--> Model successfully trained and saved as 'emotion_detection_model.h5'")

if __name__ == '__main__':
    main()