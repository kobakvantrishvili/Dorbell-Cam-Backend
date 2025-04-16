import cv2
import face_recognition
import time
from concurrent.futures import ThreadPoolExecutor
from face_cache_loader import load_cached_faces
from utils.recording_utils import save_detection_clip
from utils.logger import logger

# Initialize thread pool
executor = ThreadPoolExecutor(max_workers=3)

# Load known faces
known_faces = load_cached_faces()

# Cooldowns to prevent spamming
last_unknown_clip_time = 0
last_car_clip_time = 0
UNKNOWN_COOLDOWN = 10
CAR_COOLDOWN = 10

def process_face_async(crop, frame_buffer, fps):
    global last_unknown_clip_time
    current_time = time.time()

    try:
        rgb_face = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_face)

        if not face_locations:
            logger.warning("[?] No face found in crop, skipping")
            return

        unknown_encodings = face_recognition.face_encodings(rgb_face, face_locations)
        if not unknown_encodings:
            logger.warning("[?] No encoding found even though face was detected")
            return

        unknown_encoding = unknown_encodings[0]
        matches = face_recognition.compare_faces(list(known_faces.values()), unknown_encoding)

        if True in matches:
            name = list(known_faces.keys())[matches.index(True)]
            logger.info(f"[+] Recognized: {name}")
        else:
            if current_time - last_unknown_clip_time > UNKNOWN_COOLDOWN:
                logger.info("[!] Unknown person detected")
                save_detection_clip(list(frame_buffer), fps, label="unknown_person")
                last_unknown_clip_time = current_time
            else:
                logger.info("[~] Unknown detected but on cooldown")
    except Exception as e:
        logger.error(f"[x] Error in process_face_async: {str(e)}")


def handle_frame(frame, results, frame_buffer, fps):
    global last_car_clip_time
    current_time = time.time()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            if class_id not in [0, 3]:  # 0: person, 3: car
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_crop = frame[y1:y2, x1:x2]
            if object_crop.size == 0:
                continue

            if class_id == 0:  # Person
                if executor._work_queue.qsize() < 3:
                    executor.submit(process_face_async, object_crop.copy(), list(frame_buffer), fps)
                else:
                    logger.debug("[~] Skipping face task - executor queue full")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person - {confidence*100:.2f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif class_id == 3:  # Car
                if current_time - last_car_clip_time > CAR_COOLDOWN:
                    if executor._work_queue.qsize() < 3:
                        executor.submit(save_detection_clip, list(frame_buffer), fps, "car")
                        last_car_clip_time = current_time
                        logger.info("[!] Car detected - clip saved")
                    else:
                        logger.debug("[~] Skipping car task - executor queue full")
                else:
                    logger.info("[~] Car detected but on cooldown")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Car - {confidence*100:.2f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame
