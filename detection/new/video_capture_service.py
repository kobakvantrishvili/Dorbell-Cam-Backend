# video_capture_service.py
import cv2
import queue
import threading
from ultralytics import YOLO
from .frame_processor import run_full_frame_pipeline
from utils.logger import logger


def start_detection(model_path):
    yolo_model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return
    else:
        logger.info("Webcam opened successfully.")

    desired_width = 1920
    desired_height = int(desired_width * (9 / 16))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Face Recognition with YOLOv8", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition with YOLOv8", 720, int(720 * (9 / 16)))

    frame_buffer = []
    max_buffer_len = 300
    fps = 30
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or cv2.waitKey(5) & 0xFF == ord("q"):
            break
        frame = cv2.resize(frame, (desired_width, desired_height))
        if cv2.getWindowProperty("Face Recognition with YOLOv8", cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_buffer.append(frame.copy())
        if len(frame_buffer) > max_buffer_len:
            frame_buffer.pop(0)

        frame_counter += 1
        results = yolo_model(frame)

        processed_frame = run_full_frame_pipeline(
            frame, results, frame_buffer, fps, frame_counter
        )
        cv2.imshow("Face Recognition with YOLOv8", processed_frame)

    cap.release()
    cv2.destroyAllWindows()
