import socket
import numpy as np
import cv2
import torch
import sys
import os
import warnings
import threading
import time

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, './yolov7')

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Socket settings
HOST = '0.0.0.0'
PORT_RECEIVE = 5555  # From camera sender
PORT_SEND = 5556     # To GUI receiver
PORT_EVENT = 5557    # For detection event communication

# Global flag to indicate detection state (thread-safe using Lock)
detection_lock = threading.Lock()
object_detected = False  # True when an object is detected in the current frame

class YOLOv7_Detector:
    def __init__(self, weights='best.pt'):
        self.device = select_device('')
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect(self, frame):
        global object_detected
        # Flag to track if current frame has any detection
        frame_has_detection = False
        
        # Preprocess frame
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            pred = self.model(img)[0]
        # Adjust the thresholds as needed
        pred = non_max_suppression(pred, conf_thres=0.15, iou_thres=0.3)

        # Process detections
        for det in pred:
            if len(det):
                frame_has_detection = True  # Found at least one detection
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    color = self.colors[int(cls)]
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])), color, 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update the global detection flag (using a lock for thread safety)
        with detection_lock:
            object_detected = frame_has_detection
        
        return frame

def detection_event_monitor():
    """
    Monitor the global detection flag and, if an object is detected continuously
    (allowing for gaps of up to 0.1 sec) for 5 seconds, send an event message via a separate socket.
    """
    detection_start = None
    last_detection_time = None
    event_sent = False
    allowed_gap = 0.1  # Allow gaps up to 0.1 seconds

    # Create and bind the socket for sending detection events
    event_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    event_sock.bind((HOST, PORT_EVENT))
    event_sock.listen(1)
    print(f"Waiting for event receiver connection on port {PORT_EVENT}...")
    event_conn, _ = event_sock.accept()
    print("Event receiver connected.")

    try:
        while True:
            with detection_lock:
                current_detection = object_detected

            current_time = time.time()
            if current_detection:
                if detection_start is None:
                    detection_start = current_time
                last_detection_time = current_time
            else:
                # If we haven't seen detection recently, check the gap
                if last_detection_time is not None and (current_time - last_detection_time) <= allowed_gap:
                    # Within allowed gap, do nothing
                    pass
                else:
                    detection_start = None
                    last_detection_time = None
                    event_sent = False

            # Check if sustained detection time reached 5 seconds
            if detection_start is not None and (current_time - detection_start) >= 5 and not event_sent:
                message = "Object detected continuously for 5 seconds!"
                message_bytes = message.encode('utf-8')
                event_conn.sendall(len(message_bytes).to_bytes(4, 'big'))
                event_conn.sendall(message_bytes)
                print("Event sent:", message)
                event_sent = True

            time.sleep(0.1)
    except Exception as e:
        print(f"Event monitor error: {e}")
    finally:
        event_conn.close()
        event_sock.close()

def receive_and_process():
    detector = YOLOv7_Detector()

    # Start the detection event monitor thread
    monitor_thread = threading.Thread(target=detection_event_monitor, daemon=True)
    monitor_thread.start()

    # Setup socket for camera sender
    sock_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_receive.bind((HOST, PORT_RECEIVE))
    sock_receive.listen(1)
    print(f"Waiting for camera connection on port {PORT_RECEIVE}...")
    conn, _ = sock_receive.accept()
    print("Camera connected.")

    # Setup socket for GUI receiver
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_send.bind((HOST, PORT_SEND))
    sock_send.listen(1)
    print(f"Waiting for GUI connection on port {PORT_SEND}...")
    gui_conn, _ = sock_send.accept()
    print("GUI connected.")

    try:
        while True:
            size_bytes = conn.recv(4)
            if not size_bytes:
                break

            frame_size = int.from_bytes(size_bytes, 'big')
            frame_data = bytearray()

            while len(frame_data) < frame_size:
                chunk = conn.recv(min(4096, frame_size - len(frame_data)))
                if not chunk:
                    break
                frame_data.extend(chunk)

            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                processed_frame = detector.detect(frame)

                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                gui_conn.sendall(len(buffer).to_bytes(4, 'big'))
                gui_conn.sendall(buffer)
    except Exception as e:
        print(f"Processing error: {e}")
    finally:
        conn.close()
        gui_conn.close()
        sock_receive.close()
        sock_send.close()

if __name__ == "__main__":
    receive_and_process()
