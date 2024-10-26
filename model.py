import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Parameters
IMG_SIZE = (96, 96)  # Image resolution
BATCH_SIZE = 32
EPOCHS = 15
SAVE_TFLITE_PATH = "gesture_model.tflite"
DATASET_PATH = "gesture_photos"  # Path to the folder with gesture images

# Step 1: Define a function for Gaussian Blur
def apply_gaussian_blur(image):
    # Create a Gaussian kernel for 1 channel (grayscale)
    kernel = tf.constant([[1/16, 2/16, 1/16],
                          [2/16, 4/16, 2/16],
                          [1/16, 2/16, 1/16]], dtype=tf.float32)
    kernel = tf.expand_dims(kernel, axis=-1)  # Make it (3, 3, 1)
    kernel = tf.expand_dims(kernel, axis=-1)  # Make it (3, 3, 1, 1)
    return tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')

# Step 2: Load and Preprocess the Dataset
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_dataset = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    color_mode='grayscale',
    subset='training',
    seed=123
)

val_dataset = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    color_mode='grayscale',
    subset='validation',
    seed=123
)

# Step 3: Define the CNN Model with Regularization and Dropout
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(6, activation="softmax")  # 6 output neurons for 6 gestures
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Step 4: Train the Model with Early Stopping and Learning Rate Reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# Step 5: Evaluate the Model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

# Step 6: Convert the Model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables quantization
tflite_model = converter.convert()

# Save the TFLite model
with open(SAVE_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to {SAVE_TFLITE_PATH}")
