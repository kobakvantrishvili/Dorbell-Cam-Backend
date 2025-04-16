import cv2
import face_recognition
import time
from face_cache_loader import load_cached_faces
from utils.recording_utils import save_detection_clip
from utils.logger import logger


known_faces = load_cached_faces()
last_unknown_clip_time = 0  # Global cooldown tracker

COOLDOWN_SECONDS = 10


def handle_frame(frame, results, frame_buffer, fps):
    global last_unknown_clip_time

    current_time = time.time()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            if class_id not in [0, 3]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_crop = frame[y1:y2, x1:x2]
            if object_crop.size == 0:
                continue

            if class_id == 0:
                rgb_face = cv2.cvtColor(object_crop, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_face)

                if not face_locations:
                    logger.warning("[?] No face found, skipping")

                unknown_encodings = face_recognition.face_encodings(rgb_face, face_locations)
                name = "Person"
                is_unknown = False

                if unknown_encodings:
                    unknown_encoding = unknown_encodings[0]
                    matches = face_recognition.compare_faces(list(known_faces.values()), unknown_encoding)
                    if True in matches:
                        match_index = matches.index(True)
                        name = list(known_faces.keys())[match_index]
                    else:
                        is_unknown = True
                else:
                    logger.warning("[?] No encoding found even though face was detected")

                if is_unknown and (current_time - last_unknown_clip_time > COOLDOWN_SECONDS):
                    logger.info("[!] Unknown person detected")
                    save_detection_clip(list(frame_buffer), fps, label="unknown_person")
                    last_unknown_clip_time = current_time

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} - {confidence*100:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                logger.info("[!] Car detected")
                save_detection_clip(list(frame_buffer), fps, label="car")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Car - {confidence*100:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame
