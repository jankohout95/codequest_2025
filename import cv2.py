import cv2
import os
import time
import models
import sys
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox



# Load your custom YOLOv7 model (best.pt) from the saved path
model_path = 'best.pt'  # Update with the path to your custom best.pt model
model = torch.load(model_path, weights_only=False)
model.eval()  # Set the model to evaluation mode


# Function to get all available camera devices and their names
def get_camera_devices():
    available_cameras = []
    for i in range(5):  # Check the first 5 possible camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get the device name (some webcams may not support this)
            device_name = cap.get(cv2.CAP_PROP_DEVICE_NAME)
            if device_name == 0:  # If it fails, fallback to generic "Camera X"
                device_name = f"Camera {i}"
            available_cameras.append((i, device_name))
        cap.release()
    return available_cameras

# Function to run YOLOv7 inference on the frame and draw detections
def process_frame_with_yolo(frame):
    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare the image for YOLO inference
    img = np.transpose(img, (2, 0, 1))  # Change dimensions to CHW
    img = np.expand_dims(img, 0)  # Add batch dimension
    img = torch.tensor(img).float()  # Convert to tensor and make it float
    img /= 255.0  # Normalize to [0, 1]

    # Run inference on the frame
    with torch.no_grad():
        pred = model(img)  # Get predictions from the model

    # Parse the results
    detections = pred[0]  # Get the first prediction (batch size of 1)
    detections = detections[detections[:, 4] > 0.5]  # Only keep detections with a confidence > 0.5
    
    # Loop through detections and draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])  # Extract bounding box coordinates
        label = int(det[5])  # Get the label index
        confidence = det[4].item()  # Get the confidence score

        # Convert label index to class name (optional, if you have a class list)
        class_name = f"Class {label}"

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame, detections

# Function to run the selected camera feed
def run_camera(selected_camera):
    vc = cv2.VideoCapture(selected_camera)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    cv2.namedWindow("preview", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while rval:
        start_time = time.time()
        filename = "temp_frame.jpg"

        # Process frame with YOLOv7 to get detectionodel_path, weigsodel_path, weig
        frame, detections = process_frame_with_yolo(frame)
        
        # Display frame with detections
        cv2.imshow("preview", frame)

        rval, frame = vc.read()
        os.remove(filename)  # Delete the saved frame

        key = cv2.waitKey(1)
        if key == 27:  # Exit on ESC
            break

        # Ensure a consistent capture rate
        elapsed_time = time.time() - start_time
        time.sleep(max(0.3 - elapsed_time, 0))

    vc.release()
    cv2.destroyAllWindows()

class CameraSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Selection")
        self.setGeometry(100, 100, 300, 200)

        # Layout setup
        self.layout = QVBoxLayout()

        # Get available cameras
        self.camera_devices = get_camera_devices()

        if not self.camera_devices:
            self.show_error_message("No cameras found.")
            sys.exit()

        # Create a button for each available camera
        for idx, camera in enumerate(self.camera_devices):
            button = QPushButton(camera[1], self)  # Use the device name for the button text
            button.clicked.connect(lambda checked, c=camera[0]: self.on_camera_select(c))
            self.layout.addWidget(button)

        self.setLayout(self.layout)

    def on_camera_select(self, selected_camera):
        try:
            run_camera(selected_camera)
        except Exception as e:
            self.show_error_message(f"Failed to start the camera: {str(e)}")

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)


# Create the application and run the window
app = QApplication(sys.argv)
window = CameraSelectionWindow()
window.show()
sys.exit(app.exec_())
