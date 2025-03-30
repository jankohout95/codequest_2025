import socket
import cv2
import numpy as np
import time  # Import time for the sleep function

def get_available_cameras(max_tests=5):
    """Check available cameras and return list of working indices"""
    available = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def select_camera(available_cams):
    """Show camera selection menu"""
    print("\nAvailable Cameras:")
    for i, cam_idx in enumerate(available_cams):
        print(f"{i+1}: Camera Index {cam_idx}")
    
    while True:
        try:
            selection = int(input("Select camera (1-{}): ".format(len(available_cams))))
            if 1 <= selection <= len(available_cams):
                return available_cams[selection-1]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")

def send_frame(sock, frame):
    """Send frame over socket with size header"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    size = len(buffer)
    sock.sendall(size.to_bytes(4, 'big'))
    sock.sendall(buffer)

def main():
    # Network configuration
    HOST = '192.168.99.119'  # Change to receiver IP
    PORT = 5555

    # Camera selection
    available_cams = get_available_cameras()
    if not available_cams:
        print("Error: No cameras found!")
        return
    
    cam_index = select_camera(available_cams)
    
    # Start capture and streaming
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {cam_index}")
        return
    
    # Set reasonable resolution (adjust if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Connect to receiver
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}, streaming from camera {cam_index}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Send frame
            send_frame(sock, frame)
            
            # Sleep for 1 second to achieve 1 FPS
            time.sleep(0.1)
                
    except ConnectionRefusedError:
        print(f"Could not connect to {HOST}:{PORT}")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        sock.close()
        print("Stream stopped")

if __name__ == "__main__":
    main()
