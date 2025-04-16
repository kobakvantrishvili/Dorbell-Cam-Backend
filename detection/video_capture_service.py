import cv2
import queue
import threading
from ultralytics import YOLO
from .frame_processor import handle_frame
from utils.logger import logger


logger.info("Starting detection script...")

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

    initial_window_width = 720
    initial_window_height = int(initial_window_width * (9 / 16))
    cv2.namedWindow("Face Recognition with YOLOv8", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition with YOLOv8", initial_window_width, initial_window_height)

    frame_queue = queue.Queue(maxsize=30)
    output_queue = queue.Queue(maxsize=60)

    frame_buffer = []
    max_buffer_len = 300  # 10 seconds at 30fps
    fps = 30

    def process_frame():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                results = yolo_model(frame)
                processed = handle_frame(frame, results, frame_buffer, fps)
                if not output_queue.full():
                    output_queue.put(processed)

    threading.Thread(target=process_frame, daemon=True).start()

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

        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

        if not output_queue.empty():
            out_frame = output_queue.get()
            cv2.imshow("Face Recognition with YOLOv8", out_frame)

    cap.release()
    cv2.destroyAllWindows()
