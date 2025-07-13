import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# Constants
ERROR_THRESHOLD = 5  # Angular error threshold in degrees

def get_road_vector(seg_image, seg_model):
    """Calculate road vector based on navigable_road mask."""
    h, w = seg_image.shape[:2]
    seg_results = seg_model.predict(seg_image, conf=0.8, iou=0.8)
    for result in seg_results:
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            for mask, box in zip(masks, boxes):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                green_mask = np.zeros_like(seg_image)
                green_mask[mask > 0] = [0, 255, 0]
                seg_image = cv2.addWeighted(seg_image, 0.7, green_mask, 0.3, 0)
                x1, y1, x2, y2 = map(int, box)
                ref_x, ref_y = (x1 + x2) // 2, (y1 + y2) // 2
                start_x, start_y = w // 2, h
                dx = ref_x - start_x
                dy = ref_y - start_y
                length = h // 2
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    dx, dy = (dx / norm) * length, (dy / norm) * length
                end_x, end_y = int(start_x + dx), int(start_y + dy)
                cv2.arrowedLine(seg_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
                cv2.circle(seg_image, (ref_x, ref_y), 6, (0, 255, 255), -1)
                return np.array([dx, dy]), (ref_x, ref_y), seg_image
    return None, None, seg_image

def get_barrier_vectors(det_image, depth_image, det_model):
    """Detect the two closest barriers (left and right) and calculate vectors."""
    h, w = det_image.shape[:2]
    det_results = det_model.predict(det_image, conf=0.8, iou=0.8)
    inference_time = det_results[0].speed['inference'] / 1000  # Convertir de ms à s
    barriers = []
    for result in det_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_names = result.names
        for box, cls, conf in zip(boxes, classes, confidences):
            if class_names[int(cls)].lower() == "traffic_barrier":
                x1, y1, x2, y2 = map(int, box)
                x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= y_c < depth_image.shape[0] and 0 <= x_c < depth_image.shape[1]:
                    depth_value = depth_image[y_c, x_c]
                    if 0.01 <= depth_value <= 10.0:
                        barriers.append((x_c, y_c, depth_value))

    det_image_with_bboxes = det_image.copy()
    if len(barriers) < 2:
        return None, None, det_image_with_bboxes, inference_time

    # Sort by x-coordinate to get left and right barriers
    barriers.sort(key=lambda x: x[0])
    left_barrier = barriers[0]
    right_barrier = barriers[-1]

    # Draw bounding boxes and labels
    for box, cls, conf in zip(boxes, classes, confidences):
        if class_names[int(cls)].lower() == "traffic_barrier":
            x1, y1, x2, y2 = map(int, box)
            color = (255, 0, 255)
            cv2.rectangle(det_image_with_bboxes, (x1, y1), (x2, y2), color, 2)
            x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(det_image_with_bboxes, (x_c, y_c), 6, color, -1)
            cv2.putText(det_image_with_bboxes, f"{conf:.2f}", (x_c, y_c - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Process vectors
    start_x, start_y = w // 2, h
    dx_l = left_barrier[0] - start_x
    dy_l = left_barrier[1] - start_y
    length = h // 2
    norm_l = np.sqrt(dx_l**2 + dy_l**2)
    if norm_l > 0:
        dx_l, dy_l = (dx_l / norm_l) * length, (dy_l / norm_l) * length
    end_x_l, end_y_l = int(start_x + dx_l), int(start_y + dy_l)
    cv2.arrowedLine(det_image_with_bboxes, (start_x, start_y), (end_x_l, end_y_l), (255, 0, 255), 1)
    cv2.putText(det_image_with_bboxes, f"Left: {left_barrier[2]:.1f}m", (left_barrier[0], left_barrier[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    dx_r = right_barrier[0] - start_x
    dy_r = right_barrier[1] - start_y
    norm_r = np.sqrt(dx_r**2 + dy_r**2)
    if norm_r > 0:
        dx_r, dy_r = (dx_r / norm_r) * length, (dy_r / norm_r) * length
    end_x_r, end_y_r = int(start_x + dx_r), int(start_y + dy_r)
    cv2.arrowedLine(det_image_with_bboxes, (start_x, start_y), (end_x_r, end_y_r), (255, 0, 255), 1)
    cv2.putText(det_image_with_bboxes, f"Right: {right_barrier[2]:.1f}m", (right_barrier[0], right_barrier[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return np.array([dx_l, dy_l]), np.array([dx_r, dy_r]), det_image_with_bboxes, inference_time

def get_robot_vector(image):
    """Calculate robot vector (from bottom center to frame center)."""
    h, w = image.shape[:2]
    start_x, start_y = w // 2, h
    end_x, end_y = w // 2, h // 2
    cv2.arrowedLine(image, (start_x, start_y), (end_x, end_y), (255, 255, 0), 1)
    return np.array([0, -(h // 2)]), (end_x, end_y), image

def calculate_angular_error(robot_vector, road_vector):
    """Calculate angular error between robot and road vectors."""
    if robot_vector is None or road_vector is None:
        return None
    robot_norm = np.sqrt(robot_vector[0]**2 + robot_vector[1]**2)
    road_norm = np.sqrt(road_vector[0]**2 + road_vector[1]**2)
    if robot_norm == 0 or road_norm == 0:
        return None
    robot_unit = robot_vector / robot_norm
    road_unit = road_vector / road_norm
    cross_product = robot_unit[0] * road_unit[1] - robot_unit[1] * road_unit[0]
    dot_product = np.dot(robot_unit, road_unit)
    angle_rad = np.arctan2(cross_product, dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def process_frame(rgb_image, depth_image, seg_model_path, det_model_path, ser):
    """Process frames for navigation and detection with serial control."""
    # Load models
    seg_model = YOLO(seg_model_path)
    det_model = YOLO(det_model_path)

    # Prepare frames
    original_frame = rgb_image.copy()
    depth_frame = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET).copy()
    seg_frame = rgb_image.copy()
    det_frame = rgb_image.copy()  # Use RGB image as base for detection frame
    combined_frame = rgb_image.copy()

    # Get road vector
    road_vector, road_ref_point, seg_frame = get_road_vector(seg_frame, seg_model)

    # Get barrier vectors
    barrier_left_vector, barrier_right_vector, det_frame, inference_time = get_barrier_vectors(det_frame, depth_image, det_model)

    # Get robot vector
    robot_vector, robot_ref_point, combined_frame = get_robot_vector(combined_frame)

    # Calculate angular error
    angular_error = calculate_angular_error(robot_vector, road_vector)

    # Send serial commands based on navigation data
    if road_ref_point is None:
        ser.write(b'S')  # Stop robot
    elif angular_error is not None:
        if angular_error < -ERROR_THRESHOLD:
            ser.write(b'L')  # Turn left
        elif angular_error > ERROR_THRESHOLD:
            ser.write(b'R')  # Turn right
        else:
            ser.write(b'F')  # Move forward

    # Display navigation data on Combined Frame
    h, w = combined_frame.shape[:2]
    if angular_error is not None:
        cv2.putText(combined_frame, f"Angle: {angular_error:.1f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if angular_error < -ERROR_THRESHOLD:
            cv2.putText(combined_frame, "Turn left", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        elif angular_error > ERROR_THRESHOLD:
            cv2.putText(combined_frame, "Turn right", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        else:
            cv2.putText(combined_frame, "Move forward", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    # Display message if road reference is None
    if road_ref_point is None:
        cv2.putText(combined_frame, "Reference none, stop robot", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    # Add vectors to Combined Frame
    if road_vector is not None:
        cv2.arrowedLine(combined_frame, (w // 2, h), (w // 2 + int(road_vector[0]), h - int(road_vector[1])),
                        (0, 0, 255), 1)
        cv2.circle(combined_frame, road_ref_point, 6, (0, 255, 255), -1)
    if barrier_left_vector is not None:
        cv2.arrowedLine(combined_frame, (w // 2, h), (w // 2 + int(barrier_left_vector[0]), h - int(barrier_left_vector[1])),
                        (255, 0, 255), 1)
    if barrier_right_vector is not None:
        cv2.arrowedLine(combined_frame, (w // 2, h), (w // 2 + int(barrier_right_vector[0]), h - int(barrier_right_vector[1])),
                        (255, 0, 255), 1)

    # Display FPS using YOLO inference time
    fps = 1.0 / inference_time if inference_time > 0 else 0.0
    cv2.putText(combined_frame, f"FPS: {fps:.1f}", (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    navigation_data = {
        'angle_error': angular_error,
        'road_ref_point': road_ref_point,
        'left_barrier_distance': barrier_left_vector[1] if barrier_left_vector is not None else None,
        'right_barrier_distance': barrier_right_vector[1] if barrier_right_vector is not None else None
    }

    return original_frame, depth_frame, seg_frame, det_frame, combined_frame, navigation_data
