import tensorflow as tf
import numpy as np
import cv2
import time

# Parameters
TFLITE_MODEL_PATH = "gesture_model.tflite"
IMG_SIZE = (96, 96)  # Must match training resolution
GESTURE_CLASSES = ["closed", "gun", "ok", "open", "peace", "thumbs_up"]
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for displaying a prediction

# ESP32-CAM Stream URL (Try different URLs if needed)
ESP32_CAM_URL = "http://192.168.1.158:81/stream"  # Adjust port if necessary

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess a frame for the model
def preprocess_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
    frame = np.array(frame, dtype=np.float32) / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension for grayscale
    return np.expand_dims(frame, axis=0)  # Add batch dimension

# Run inference on a single frame
def predict(frame):
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the highest probability prediction
    predicted_class = np.argmax(output_data)
    confidence = output_data[0][predicted_class]
    
    # Debug: Print raw probabilities
    print(f"Raw output probabilities: {output_data[0]}")
    
    return GESTURE_CLASSES[predicted_class], confidence

# Open stream using VideoCapture
cap = cv2.VideoCapture(ESP32_CAM_URL)
time.sleep(1)  # Allow time for the camera to initialize

if not cap.isOpened():
    print("Failed to open stream from ESP32-CAM")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            time.sleep(1)  # Add delay before retrying
            continue  # Skip this iteration and retry

        # Make a prediction on the current frame
        gesture, confidence = predict(frame)

        # Display the predicted gesture and confidence if above threshold
        if confidence > CONFIDENCE_THRESHOLD:
            cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Uncertain", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame with prediction
        cv2.imshow("Gesture Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
