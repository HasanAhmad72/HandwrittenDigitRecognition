import os
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# 1. LOAD AND PREPARE MNIST DATASET
# ============================================================
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0.0-1.0
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to include channel dimension: (batch_size, 28, 28, 1)
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

# ============================================================
# 2. BUILD CNN ARCHITECTURE WITH DATA AUGMENTATION
# ============================================================
# Data augmentation is active during training to make the model
# robust against camera angles, handwriting variations, and shifts.
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Augmentation layers (active during model.fit only)
    layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
    layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
    layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0),

    # Feature Extractor - Conv Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Feature Extractor - Conv Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Classification Head
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.35),
    layers.Dense(10, activation="softmax")
])

model.summary()

# ============================================================
# 3. COMPILE MODEL
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 4. TRAIN MODEL
# ============================================================
print("\nStarting CNN training with data augmentation...")
EPOCHS = 10
BATCH_SIZE = 128

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    verbose=1
)

# ============================================================
# 5. EVALUATE MODEL
# ============================================================
print("\nEvaluating model on MNIST test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ============================================================
# 6. SAVE MODEL
# ============================================================
os.makedirs("model", exist_ok=True)
MODEL_SAVE_PATH = "model/digit_model.keras"
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved successfully to '{MODEL_SAVE_PATH}'!")