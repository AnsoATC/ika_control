import cv2
import numpy as np
import pyrealsense2 as rs
from process.process_frame import process_frame
import os
import time

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
                break

    finally:
        if use_camera:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
