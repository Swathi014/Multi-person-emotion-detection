import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def create_model(input_shape, num_classes):
    """Builds a regularized CNN architecture with Data Augmentation."""
    
    # 1. Data Augmentation Layers (These only run during training)
    # 1. Milder Data Augmentation Layers
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal", input_shape=input_shape),
        layers.RandomRotation(0.05), # Reduced rotation to 5%
    ])

    model = models.Sequential([
        data_augmentation,
        
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
        # Added more dropout/regularization to prevent memorization
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.6), 
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    TRAIN_DIR = r'./Balanced_Dataset/train'
    TEST_DIR = r'./Balanced_Dataset/test'
    
    IMG_HEIGHT, IMG_WIDTH = 48, 48
    BATCH_SIZE = 32
    EPOCHS = 30 # Increased because Early Stopping will catch it if it finishes early

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

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    model = create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=len(class_names))
    # Use a slower, steadier learning rate (0.0001 instead of default 0.001)
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 2. Early Stopping Callback
    # Monitors validation loss and stops if it doesn't improve for 3 epochs
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7, 
        restore_best_weights=True,
        verbose=1
    )

    print("--> Starting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # Saved in the new native Keras format to fix the legacy warning
    model.save('emotion_detection_model.keras')
    print("--> Model successfully trained and saved as 'emotion_detection_model.keras'")

if __name__ == '__main__':
    main()