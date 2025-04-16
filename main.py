from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from utils.mongo_utils import fs_faces, fs_clips
import sys
import subprocess
import gridfs
import io
import face_recognition
import pickle
import os


load_dotenv()
uri = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(uri, server_api=ServerApi('1'))

db = client["face_db"]
fs = gridfs.GridFS(db)

app = FastAPI()
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_image(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()

    fs_faces.put(contents, filename=name)

    image = face_recognition.load_image_file(io.BytesIO(contents))
    encoding = face_recognition.face_encodings(image)
    if not encoding:
        return JSONResponse(content={"error": "No face found"}, status_code=400)

    with open(f"{CACHE_DIR}/{name}.pkl", "wb") as f:
        pickle.dump(encoding[0], f)

    return {"message": "Image uploaded and cached successfully."}


@app.post("/run-detection/")
async def run_detection():
    try:
        subprocess.Popen([sys.executable, "run_detection.py"])
        return {"message": "Detection script started successfully."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/detections/")
def list_detections():
    detections = []
    for clip in fs_clips.find():
        metadata = clip.metadata or {}
        detections.append({
            "filename": clip.filename,
            "label": metadata.get("label", "unknown"),
            "timestamp": str(metadata.get("timestamp", "N/A"))
        })
    return detections



@app.get("/download/{filename}")
def download_video(filename: str):
    file = fs_clips.find_one({"filename": filename})
    if not file:
        raise HTTPException(status_code=404, detail="Video not found")

    return StreamingResponse(
        io.BytesIO(file.read()),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
