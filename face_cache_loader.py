import os
import pickle

def load_cached_faces(cache_dir="cache"):
    known_faces = {}
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pkl"):
            name = filename[:-4]  # remove .pkl
            with open(os.path.join(cache_dir, filename), "rb") as f:
                known_faces[name] = pickle.load(f)
    return known_faces
