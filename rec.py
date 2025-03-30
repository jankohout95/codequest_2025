import socket
import numpy as np
import cv2
import struct
import threading
import time
from playsound import playsound

HOST = '192.168.99.119'  # IP of the YOLO server machine
PORT_VIDEO = 5556        # Port for receiving video frames
PORT_EVENT = 5557        # Port for receiving detection events

ALERT_SOUND = 'audio.mp3'  # Path to the audio file

def receive_video():
    """ Receive and display video frames from the YOLO server. """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT_VIDEO))
    print("Connected to video feed.")

    try:
        while True:
            # Receive the length of the incoming frame (4 bytes)
            size_bytes = sock.recv(4)
            if not size_bytes:
                break

            frame_size = struct.unpack('!I', size_bytes)[0]
            frame_data = bytearray()

            while len(frame_data) < frame_size:
                chunk = sock.recv(min(4096, frame_size - len(frame_data)))
                if not chunk:
                    break
                frame_data.extend(chunk)

            # Decode the frame and show it
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("YOLOv7 Detection", frame)

            # Wait for 'q' key to exit
            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"Error receiving video: {e}")
    finally:
        sock.close()
        cv2.destroyAllWindows()

def receive_detection_event():
    """ Receive detection event notifications from the YOLO server and play alert sound. """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT_EVENT))
    print("Connected to detection event.")

    try:
        while True:
            # Receive the length of the event message (4 bytes)
            size_bytes = sock.recv(4)
            if not size_bytes:
                break

            event_size = struct.unpack('!I', size_bytes)[0]
            event_data = bytearray()

            while len(event_data) < event_size:
                chunk = sock.recv(min(4096, event_size - len(event_data)))
                if not chunk:
                    break
                event_data.extend(chunk)

            # Decode the event and print it
            event_message = event_data.decode('utf-8')
            print(f"Detection Event: {event_message}")
            
            # Play alert sound when the detection event occurs
            if "Object detected continuously for 3 seconds" in event_message:
                print("Playing alert sound...")
                playsound(ALERT_SOUND)
    except Exception as e:
        print(f"Error receiving detection event: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    # Create threads for both video and event reception
    video_thread = threading.Thread(target=receive_video, daemon=True)
    video_thread.start()

    event_thread = threading.Thread(target=receive_detection_event, daemon=True)
    event_thread.start()

    # Run the threads concurrently
    video_thread.join()