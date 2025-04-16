import cv2
import queue
import threading
import time
from ultralytics import YOLO
from .frame_processor import run_full_frame_pipeline
from utils.logger import logger

def frame_grabber(cap, buffer_queue, width, height, fps):
    delay = 1 / fps
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (width, height))
        if buffer_queue.full():
            buffer_queue.get()
        buffer_queue.put(frame)
        time.sleep(delay)  # ensures consistent 30 fps grabbing

def start_detection(model_path):
    yolo_model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    logger.info("Webcam opened successfully.")

    desired_width = 1920
    desired_height = int(desired_width * (9 / 16))
    fps = 30

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    cv2.namedWindow("Face Recognition with YOLOv8", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition with YOLOv8", 720, int(720 * (9 / 16)))

    # Thread-safe circular buffer of 10 seconds
    frame_buffer = queue.Queue(maxsize=fps * 10)
    frame_counter = 0

    # Start background thread to grab raw frames at 30 FPS
    capture_thread = threading.Thread(
        target=frame_grabber,
        args=(cap, frame_buffer, desired_width, desired_height, fps),
        daemon=True
    )
    capture_thread.start()

    while True:
        if frame_buffer.empty():
            continue

        # Use the latest available frame for detection
        frame = frame_buffer.queue[-1].copy()
        frame_counter += 1

        results = yolo_model(frame)

        processed_frame = run_full_frame_pipeline(
            frame, results, list(frame_buffer.queue), fps, frame_counter
        )

        cv2.imshow("Face Recognition with YOLOv8", processed_frame)

        if cv2.waitKey(5) & 0xFF == ord("q") or \
           cv2.getWindowProperty("Face Recognition with YOLOv8", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
