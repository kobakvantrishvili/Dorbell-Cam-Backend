import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import gridfs

load_dotenv()
uri = os.getenv("MONGO_CONNECTION_STRING")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
db = mongo_client["face_db"]

# Separate GridFS buckets
fs_faces = gridfs.GridFS(db, collection="face_images")
fs_clips = gridfs.GridFS(db, collection="detection_clips")
