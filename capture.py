import requests
import cv2
import numpy as np
import os

# URL for capturing images from the ESP32-CAM (replace with your ESP32 IP address)
ESP32_CAM_URL = "http://192.168.1.158/capture"  # Replace with actual ESP32-CAM IP and endpoint

# Define gesture labels and corresponding keys
gesture_keys = {
    '1': 'open',        # Press '1' for 'open' gesture
    '2': 'closed',      # Press '2' for 'closed' gesture
    '3': 'ok',          # Press '3' for 'ok' gesture
    '4': 'gun',     # Press '4' for 'pointer' gesture
    '5': 'peace',       # Press '5' for 'peace' gesture
    '6': 'thumbs_up'    # Press '6' for 'thumbs up' gesture
}

# Create directories for each gesture if they don't exist
for gesture in gesture_keys.values():
    os.makedirs(f"gesture_photos/{gesture}", exist_ok=True)

# Function to capture an image from ESP32-CAM
def capture_image():
    try:
        response = requests.get(ESP32_CAM_URL)
        response.raise_for_status()  # Raise an error for bad responses
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error capturing image: {e}")
        return None

# Start capturing images based on key press
image_counts = {gesture: 0 for gesture in gesture_keys.values()}

print("Press the corresponding key to capture an image for each gesture:")
print("1 = open, 2 = closed, 3 = ok, 4 = gun, 5 = peace, 6 = thumbs up")
print("Press 'q' to quit.")

while True:
    # Capture an image from ESP32-CAM
    img = capture_image()
    
    if img is None:  # Check if the image was captured successfully
        continue

    cv2.imshow("ESP32-CAM Live Feed", img)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Check if key matches any gesture key
    if chr(key) in gesture_keys:
        gesture = gesture_keys[chr(key)]
        image_count = image_counts[gesture]

        # Save the image for the gesture
        filename = f"gesture_photos/{gesture}/image_{image_count}.jpg"
        cv2.imwrite(filename, img)
        image_counts[gesture] += 1
        print(f"Captured image for '{gesture}' - Total: {image_count + 1}")

    # Quit if 'q' is pressed
    elif key == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
