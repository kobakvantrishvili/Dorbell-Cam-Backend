import os
import cv2
import datetime
from .mongo_utils import fs_clips

def save_detection_clip(frames, fps, label):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.mp4"
    temp_path = f"temp/{filename}"

    os.makedirs("temp", exist_ok=True)
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for f in frames:
        out.write(f)
    out.release()

    utc_timestamp = datetime.datetime.now(datetime.timezone.utc)
    with open(temp_path, "rb") as f:
        fs_clips.put(f, filename=filename, metadata={"label": label, "timestamp": utc_timestamp})

    os.remove(temp_path)
    print(f"[+] Saved clip for {label}")
