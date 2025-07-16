import cv2
import numpy as np
import pyrealsense2 as rs
from process.process_frame_serial import process_frame
import os
import time
import serial
import struct

# Constants
ERROR_THRESHOLD = 5  # Angular error threshold in degrees

# Set up the serial port and baud rate
try:
    ser = serial.Serial('COM10', 115200, timeout=1)
    print(f"Serial port {ser.port} opened successfully.")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}. Try running with sudo or check permissions.")
    exit(1)

def send_packet(gaz, yon, aci):
    # Check if the serial port is open
    if not ser.is_open:
        print("Serial port is not open! Check the connection.")
        return

    # Validate direction (must be 'r', 'l', or 'd')
    if yon not in ['r', 'l', 'd']:
        print("Invalid direction! Must be 'r', 'l', or 'd'.")
        return

    # Validate gaz range (1000-2000)
    if gaz < 1000 or gaz > 2000:
        print("Gaz value must be between 1000 and 2000!")
        return

    # Validate aci range (0-180)
    if aci < -180 or aci > 180:
        print("Aci value must be between 0 and 180!")
        return
    if aci < 90 and aci > 0:
        aci = 75
    if aci > -90 and aci < 0:
        aci = -75

    # Create packet with a newline character (\n) as terminator
    packet = f"S,{gaz},{yon},{aci}\n"
    ser.write(packet.encode())
    ser.flush()  # Ensure all data is written to the serial port
    print(packet)
    print(f"Sent -> Gaz: {gaz}, Yön: {yon}, Aci: {aci}")

def main():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        use_camera = True
        print("RealSense camera initialized successfully with depth scale:", depth_scale)
    except RuntimeError:
        print("RealSense camera not detected. Using fallback video.")
        use_camera = False
        cap = cv2.VideoCapture('/home/ansoatc/Videos/test_video.mp4')
        if not cap.isOpened():
            print("Fallback video not found. Please provide a valid video path.")
            return

    # Load model paths (relative path)
    seg_model_path = os.path.join('models', 'yolo11n_segmentation.pt')
    det_model_path = os.path.join('models', 'yolo12n_detection.pt')

    if not os.path.exists(seg_model_path) or not os.path.exists(det_model_path):
        print(f"Models not found at {seg_model_path} or {det_model_path}")
        return

    # Initialize variables
    last_time = time.time()
    frame_count = 0

    try:
        while True:
            if use_camera:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                rgb_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            else:
                ret, rgb_image = cap.read()
                if not ret:
                    break
                depth_image = np.zeros_like(rgb_image, dtype=np.float32)  # Placeholder
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Process frames with depth scale
            combined_frame, depth_frame_display, navigation_data = process_frame(
                rgb_image, depth_image, depth_scale, seg_model_path, det_model_path
            )

            # Navigation based on angular error with reference vector
            if navigation_data['angle_error'] is not None and navigation_data['reference_point'] is not None:
                gaz = 1300  # Default gaz value (adjustable)
                aci = 0    # Default aci value (center)
                if navigation_data['angle_error'] < -ERROR_THRESHOLD:
                    yon = 'l'  # Turn left
                    aci = int(navigation_data['angle_error'])   # Adjust aci angle
                elif navigation_data['angle_error'] > ERROR_THRESHOLD:
                    yon = 'r'  # Turn right
                    aci = int(navigation_data['angle_error'])  # Adjust aci angle
                else:
                    yon = 'd'  # Move forward
                send_packet(gaz, yon, aci)
            else:
                send_packet(1500, 'd', 90)  # Stop if no reference

            # Display frames
            cv2.imshow('Combined Frame', combined_frame)
            cv2.imshow('Depth Frame', depth_frame_display)

            # Calculate and display FPS
            frame_count += 1
            if time.time() - last_time >= 1.0:
                fps = frame_count / (time.time() - last_time)
                frame_count = 0
                last_time = time.time()

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                send_packet(1500, 'd', 90)  # Stop robot
                break

    finally:
        if use_camera:
            pipeline.stop()
        else:
            cap.release()
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
