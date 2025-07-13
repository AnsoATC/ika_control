import cv2
import numpy as np
import pyrealsense2 as rs
from process.process_frame_serial import process_frame
import os
import time
import serial

def main():
    # Initialize serial port (adjust port and baud rate as per your robot setup, e.g., '/dev/ttyUSB0' and 9600)
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)  # Wait for serial port to initialize

    # Try to initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        use_camera = True
        print("RealSense camera initialized successfully.")
    except RuntimeError:
        print("RealSense camera not detected. Using fallback video.")
        use_camera = False
        cap = cv2.VideoCapture('/home/ansoatc/Videos/test_video.mp4')  # Ajuste le chemin
        if not cap.isOpened():
            print("Fallback video not found. Please provide a valid video path.")
            return

    # Load model paths
    base_dir = os.path.expanduser('~/Teknofest/omu-ika/ika_control')
    seg_model_path = os.path.join(base_dir, 'models/yolo11n_segmentation.pt')
    det_model_path = os.path.join(base_dir, 'models/yolo12n_detection.pt')

    if not os.path.exists(seg_model_path):
        print(f"Segmentation model {seg_model_path} not found.")
        return
    if not os.path.exists(det_model_path):
        print(f"Detection model {det_model_path} not found.")
        return

    try:
        while True:
            if use_camera:
                # Wait for frames and align them
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert frames to numpy arrays
                rgb_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            else:
                ret, rgb_image = cap.read()
                if not ret:
                    break
                depth_image = np.zeros_like(rgb_image, dtype=np.float32)  # Placeholder depth
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Process frames
            original_frame, depth_frame_display, seg_frame, det_frame, combined_frame, navigation_data = process_frame(
                rgb_image, depth_image, seg_model_path, det_model_path, ser
            )

            # Display frames
            cv2.imshow('Original Frame', original_frame)
            cv2.imshow('Depth Frame', depth_frame_display)
            cv2.imshow('Navigable Road Frame', seg_frame)
            cv2.imshow('Detection Frame', det_frame)
            cv2.imshow('Combined Frame', combined_frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                ser.write(b'S')  # Stop robot before exiting
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
