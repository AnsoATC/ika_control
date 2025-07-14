import cv2
import numpy as np
from ultralytics import YOLO
import os

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
                cv2.arrowedLine(seg_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)  # Blue road vector
                cv2.circle(seg_image, (ref_x, ref_y), 6, (0, 255, 255), -1)  # Yellow reference point
                # Add robot orientation vector
                cv2.arrowedLine(seg_image, (w // 2, h), (w // 2, h // 2), (255, 0, 0), 2)  # Red robot vector
                return np.array([dx, dy]), (ref_x, ref_y), seg_image
    return None, None, seg_image

def get_barrier_vectors(det_image, depth_image, det_model, depth_scale):
    """Detect all objects and calculate vectors/distances for the two closest barriers."""
    h, w = det_image.shape[:2]
    det_results = det_model.predict(det_image, conf=0.5, iou=0.5)
    inference_time = det_results[0].speed['inference'] / 1000  # Convert to seconds
    barriers = []
    det_image_with_bboxes = det_image.copy()

    for result in det_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_names = result.names
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= y_c < depth_image.shape[0] and 0 <= x_c < depth_image.shape[1]:
                depth_value = depth_image[y_c, x_c] * depth_scale  # Convert to meters
                if 0.01 <= depth_value <= 10.0:
                    class_name = class_names[int(cls)].lower()
                    # Draw bounding box with color based on class
                    color = (0, 255, 0) if class_name == "traffic_barrier" else \
                            (0, 0, 255) if class_name == "traffic_cone" else (255, 0, 0)
                    cv2.rectangle(det_image_with_bboxes, (x1, y1), (x2, y2), color, 1)  # Light box
                    cv2.circle(det_image_with_bboxes, (x_c, y_c), 4, color, -1)  # Reference point
                    cv2.putText(det_image_with_bboxes, f"Dist: {depth_value:.2f}m", (x_c, y_c - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(det_image_with_bboxes, class_name, (x_c, y_c + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Add class name
                    if class_name == "traffic_barrier":
                        barriers.append((x_c, y_c, depth_value))

    # Process the two closest barriers
    if len(barriers) >= 2:
        barriers.sort(key=lambda x: x[0])  # Sort by x-coordinate
        left_barrier = barriers[0]
        right_barrier = barriers[-1]
        mid_x = (left_barrier[0] + right_barrier[0]) // 2
        mid_y = (left_barrier[1] + right_barrier[1]) // 2
        cv2.line(det_image_with_bboxes, (left_barrier[0], left_barrier[1]), (right_barrier[0], right_barrier[1]),
                 (255, 0, 255), 1)  # Segment between barriers
        start_x, start_y = w // 2, h
        dx = mid_x - start_x
        dy = mid_y - start_y
        length = h // 2
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            dx, dy = (dx / norm) * length, (dy / norm) * length
        end_x, end_y = int(start_x + dx), int(start_y + dy)
        cv2.arrowedLine(det_image_with_bboxes, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2)  # Magenta reference vector
        # Add robot orientation vector
        cv2.arrowedLine(det_image_with_bboxes, (w // 2, h), (w // 2, h // 2), (255, 0, 0), 2)  # Red robot vector
        return np.array([dx, dy]), (mid_x, mid_y), det_image_with_bboxes, inference_time
    return None, None, det_image_with_bboxes, inference_time

def get_robot_vector(image):
    """Calculate robot vector (from bottom center to frame center)."""
    h, w = image.shape[:2]
    start_x, start_y = w // 2, h
    end_x, end_y = w // 2, h // 2
    cv2.arrowedLine(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # Red robot orientation vector
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

def process_frame(rgb_image, depth_image, depth_scale, seg_model_path, det_model_path):
    """Process frames for navigation and detection."""
    # Load models
    seg_model = YOLO(seg_model_path)
    det_model = YOLO(det_model_path)

    # Prepare frames
    original_frame = rgb_image.copy()
    depth_frame_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET).copy()
    seg_frame = rgb_image.copy()
    det_frame = rgb_image.copy()
    combined_frame = rgb_image.copy()

    # Get road vector and mask
    road_vector, road_ref_point, seg_frame = get_road_vector(seg_frame, seg_model)

    # Get barrier vectors and detections
    barrier_vector, barrier_ref_point, det_frame, inference_time = get_barrier_vectors(det_frame, depth_image, det_model, depth_scale)

    # Get robot vector
    robot_vector, robot_ref_point, temp_frame = get_robot_vector(combined_frame.copy())  # Store robot vector separately

    # Calculate angular error
    angular_error = calculate_angular_error(robot_vector, road_vector)

    # Display navigation data on Combined Frame
    h, w = combined_frame.shape[:2]
    if angular_error is not None:
        cv2.putText(combined_frame, f"Angle: {angular_error:.1f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if angular_error < -ERROR_THRESHOLD:
            cv2.putText(combined_frame, "Turn Left", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif angular_error > ERROR_THRESHOLD:
            cv2.putText(combined_frame, "Turn Right", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(combined_frame, "Move Forward", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if road_ref_point is None:
        cv2.putText(combined_frame, "No Road Ref, Stop", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add road mask and vectors to Combined Frame
    if road_vector is not None:
        seg_results = seg_model.predict(rgb_image, conf=0.8, iou=0.8)
        for result in seg_results:
            if result.masks is not None:
                mask = result.masks.data.cpu().numpy()[0]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                green_mask = np.zeros_like(combined_frame)
                green_mask[mask > 0] = [0, 255, 0]
                combined_frame = cv2.addWeighted(combined_frame, 0.7, green_mask, 0.3, 0)
        cv2.arrowedLine(combined_frame, (w // 2, h), (w // 2 + int(road_vector[0]), h - int(road_vector[1])),
                        (0, 0, 255), 2)  # Blue road vector
        cv2.circle(combined_frame, road_ref_point, 6, (0, 255, 255), -1)  # Yellow reference point
    # Add robot orientation vector in red
    if robot_vector is not None:
        cv2.arrowedLine(combined_frame, (w // 2, h), (w // 2, h // 2), (255, 0, 0), 2)  # Red robot orientation vector

    # Add barrier detections to Combined Frame
    det_results = det_model.predict(rgb_image, conf=0.5, iou=0.5)
    for result in det_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        class_names = result.names
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls)].lower()
            color = (0, 255, 0) if class_name == "traffic_barrier" else \
                    (0, 0, 255) if class_name == "traffic_cone" else (255, 0, 0)
            cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, 1)

    # Display FPS
    fps = 1.0 / inference_time if inference_time > 0 else 0.0
    cv2.putText(combined_frame, f"FPS: {fps:.1f}", (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    navigation_data = {
        'angle_error': angular_error,
        'road_ref_point': road_ref_point,
        'barrier_ref_point': barrier_ref_point
    }

    return original_frame, depth_frame_display, seg_frame, det_frame, combined_frame, navigation_data
