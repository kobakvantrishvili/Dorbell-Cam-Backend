# frame_processor.py
import cv2
import time
from face_cache_loader import load_cached_faces
from utils.recording_utils import save_detection_clip
from utils.logger import logger
import face_recognition
import threading

known_faces = load_cached_faces()
face_track_cache = {}  # stores (bbox_coords) -> name
last_unknown_clip_time = 0
COOLDOWN_SECONDS = 10

def run_full_frame_pipeline(frame, results, frame_buffer, fps, frame_counter):
    global face_track_cache

    # Update known face tracking every N frames
    if frame_counter % 15 == 0:
        threading.Thread(
            target=lambda: update_face_track_cache(frame, results, frame_buffer, fps),
            daemon=True
        ).start()

    # Use the latest tracking info to draw labels
    return simple_drawer(frame, results, face_track_cache)

def update_face_track_cache(frame, results, frame_buffer, fps):
    global face_track_cache
    face_track_cache = deep_analyzer(frame, results, frame_buffer, fps)

def deep_analyzer(frame, results, frame_buffer, fps):
    global last_unknown_clip_time
    updated_tracks = {}
    current_time = time.time()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id != 0:  # Only analyze persons
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_crop = frame[y1:y2, x1:x2]
            if object_crop.size == 0:
                continue

            name = "Unknown"
            should_record = False

            rgb_face = cv2.cvtColor(object_crop, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_face)

            if not face_locations:
                should_record = True  # couldn't locate face at all
            else:
                unknown_encodings = face_recognition.face_encodings(rgb_face, face_locations)
                if not unknown_encodings:
                    should_record = True  # face located but no encoding
                else:
                    encoding = unknown_encodings[0]
                    matches = face_recognition.compare_faces(list(known_faces.values()), encoding)

                    if True in matches:
                        name = list(known_faces.keys())[matches.index(True)]
                        logger.info(f"[+] Recognized: {name}")
                    else:
                        name = "Unrecognized"
                        should_record = True

            if should_record and (current_time - last_unknown_clip_time > COOLDOWN_SECONDS):
                logger.info(f"[!] Detected {name}, saving clip...")
                threading.Thread(
                    target=save_detection_clip,
                    args=(list(frame_buffer), fps, "unknown_person"),
                    daemon=True
                ).start()
                last_unknown_clip_time = current_time

            updated_tracks[(x1, y1, x2, y2)] = name

    return updated_tracks



def simple_drawer(frame, results, track_info):
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in [0, 3]:
                continue

            confidence = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = "Car" if class_id == 3 else "Person"

            if class_id == 0:
                label = ""
                for (tx1, ty1, tx2, ty2), tracked_name in track_info.items():
                    if _iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2)) > 0.5:
                        label = tracked_name
                        name += f" ({tracked_name})"
                        break

                if label == "Unknown":
                    color = (0, 255, 255)  # Yellow
                elif label == "Unrecognized":
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Green

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}  - {confidence*100:.2f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)
