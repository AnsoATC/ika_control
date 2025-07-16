import cv2
import numpy as np
from ultralytics import YOLO
import os

# Constants
ERROR_THRESHOLD = 5  # Angular error threshold in degrees
FRAME_CENTER_X = 320  # Centre horizontal for 640px width

def get_robot_vector(image):
    """Calculate robot vector (from bottom center to frame center)."""
    h, w = image.shape[:2]
    start_x, start_y = w // 2, h
    end_x, end_y = w // 2, h // 2
    cv2.arrowedLine(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)  # Red robot orientation vector
    return np.array([0, -(h // 2)]), (end_x, end_y), image

def get_cone_vector(det_image, depth_image, det_model, depth_scale):
    """Detect cones and display with distances."""
    h, w = det_image.shape[:2]
    det_results = det_model.predict(det_image, conf=0.5, iou=0.5)
    inference_time = det_results[0].speed['inference'] / 1000  # Convert to seconds

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
                    color = (0, 0, 255) if class_name == "traffic_cone" else \
                            (0, 255, 0) if class_name == "traffic_barrier" else \
                            (255, 0, 0) if class_name == "traffic_sign" else (255, 255, 255)
                    cv2.rectangle(det_image, (x1, y1), (x2, y2), color, 1)
                    cv2.circle(det_image, (x_c, y_c), 4, color, -1)
                    if class_name == "traffic_cone":
                        cv2.putText(det_image, f"Dist: {depth_value:.2f}m", (x_c, y_c - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(det_image, class_name, (x_c, y_c + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return det_image, inference_time  # Return the modified image and inference time

def get_barrier_vector(det_image, depth_image, det_model, depth_scale):
    """Detect barriers and display (no distance)."""
    h, w = det_image.shape[:2]
    det_results = det_model.predict(det_image, conf=0.5, iou=0.5)

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
                    color = (0, 0, 255) if class_name == "traffic_cone" else \
                            (0, 255, 0) if class_name == "traffic_barrier" else \
                            (255, 0, 0) if class_name == "traffic_sign" else (255, 255, 255)
                    cv2.rectangle(det_image, (x1, y1), (x2, y2), color, 1)
                    cv2.circle(det_image, (x_c, y_c), 4, color, -1)
                    cv2.putText(det_image, class_name, (x_c, y_c + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return None, None, det_image  # No reference vector from barriers
    
def get_road_mask(seg_image, seg_model):
    """Calculate road mask and return its center of the lower third of the bbox as reference."""
    h, w = seg_image.shape[:2]
    seg_results = seg_model.predict(seg_image, conf=0.6, iou=0.6)
    reference_vector = None
    reference_point = None

    for result in seg_results:
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            for mask, box in zip(masks, boxes):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                green_mask = np.zeros_like(seg_image)
                green_mask[mask > 0] = [0, 255, 0]
                seg_image = cv2.addWeighted(seg_image, 0.7, green_mask, 0.3, 0)

                # Extract bbox coordinates
                x1, y1, x2, y2 = map(int, box)
                # Calculate the center of the lower third of the bbox
                lower_third_y = y1 + 2 * (y2 - y1) // 3  # Center of the lower third
                mask_center_x = (x1 + x2) // 2
                mask_center_y = lower_third_y

                start_x, start_y = w // 2, h
                dx = mask_center_x - start_x
                dy = mask_center_y - start_y
                length = h // 2
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    dx, dy = (dx / norm) * length, (dy / norm) * length
                end_x, end_y = int(start_x + dx), int(start_y + dy)
                reference_vector = np.array([dx, dy])
                reference_point = (mask_center_x, mask_center_y)
                cv2.arrowedLine(seg_image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # Blue reference vector
                cv2.circle(seg_image, (mask_center_x, mask_center_y), 6, (0, 255, 255), -1)  # Yellow reference point

    return seg_image, reference_vector, reference_point

def calculate_angular_error(robot_vector, reference_vector):
    """Calculate angular error between robot and reference vectors."""
    if robot_vector is None or reference_vector is None:
        return None
    robot_norm = np.sqrt(robot_vector[0]**2 + robot_vector[1]**2)
    ref_norm = np.sqrt(reference_vector[0]**2 + reference_vector[1]**2)
    if robot_norm == 0 or ref_norm == 0:
        return None
    robot_unit = robot_vector / robot_norm
    ref_unit = reference_vector / ref_norm
    cross_product = robot_unit[0] * ref_unit[1] - robot_unit[1] * ref_unit[0]
    dot_product = np.dot(robot_unit, ref_unit)
    angle_rad = np.arctan2(cross_product, dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def process_frame(rgb_image, depth_image, depth_scale, seg_model_path, det_model_path):
    """Process frames for navigation based on road mask center."""
    # Load models
    seg_model = YOLO(seg_model_path)
    det_model = YOLO(det_model_path)

    # Prepare frames
    combined_frame = rgb_image.copy()
    depth_frame_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET).copy()

    # Get robot vector
    robot_vector, robot_ref_point, combined_frame = get_robot_vector(combined_frame)

    # Get road mask and reference
    combined_frame, reference_vector, reference_point = get_road_mask(combined_frame, seg_model)

    # Display detected objects (cones, barriers, signs) for illustration
    combined_frame, inference_time = get_cone_vector(combined_frame, depth_image, det_model, depth_scale)
    get_barrier_vector(combined_frame, depth_image, det_model, depth_scale)  # No return needed, just display

    # Calculate angular error
    angular_error = calculate_angular_error(robot_vector, reference_vector)

    # Display navigation data on Combined Frame
    h, w = combined_frame.shape[:2]
    if angular_error is not None:
        cv2.putText(combined_frame, f"Angle: {angular_error:.1f} deg", (10, int(h * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if angular_error < -ERROR_THRESHOLD:
            cv2.putText(combined_frame, "Turn Left", (50, int(h * 0.1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif angular_error > ERROR_THRESHOLD:
            cv2.putText(combined_frame, "Turn Right", (50, int(h * 0.1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(combined_frame, "Move Forward", (50, int(h * 0.1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if reference_point is None:
        cv2.putText(combined_frame, "No Ref, Stop", (50, int(h * 0.15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add reference vector and point to Combined Frame
    if reference_vector is not None:
        cv2.arrowedLine(combined_frame, (w // 2, h), (w // 2 + int(reference_vector[0]), h - int(reference_vector[1])),
                        (255, 0, 0), 2)  # Blue reference vector
        cv2.circle(combined_frame, reference_point, 6, (0, 255, 255), -1)  # Yellow reference point

    # Display FPS using YOLO's inference time
    fps = 1.0 / inference_time if inference_time > 0 else 0.0
    cv2.putText(combined_frame, f"FPS: {fps:.1f}", (w - 150, int(h * 0.05)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    navigation_data = {
        'angle_error': angular_error,
        'reference_point': reference_point
    }

    return combined_frame, depth_frame_display, navigation_data
