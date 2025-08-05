import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import faiss
import pickle
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from feature_extractor import FeatureExtractor
from torchvision import transforms

# --- Initialization ---

app = FastAPI(title="Visual Search Engine API")

# Add CORS Middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the high-resolution Faiss index and path mapping
print("Loading high-resolution Faiss index...")
index = faiss.read_index("index_high_res.faiss")
print("✅ Index loaded.")

print("Loading image paths mapping...")
with open("image_paths_high_res.pkl", "rb") as f:
    image_paths = pickle.load(f)
print("✅ Mapping loaded.")

# Initialize our feature extractor
print("Initializing feature extractor...")
extractor = FeatureExtractor()
print("✅ Feature extractor initialized.")

# Define the same transformations used during indexing for the query image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Visual Search Engine API!"}


@app.post("/search-high-res/")
async def search_high_res(file: UploadFile = File(...), k: int = 5):
    """
    Accepts a high-resolution image upload, finds the k most similar images,
    and returns their file paths.
    """
    contents = await file.read()
    query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    query_tensor = transform(query_image)

    query_embedding = extractor.get_embedding(query_tensor)
    query_embedding_numpy = query_embedding.unsqueeze(0).numpy()

    distances, indices = index.search(query_embedding_numpy, k)

    # Get the file paths for the results using the new mapping
    result_paths = [image_paths[i] for i in indices[0]]

    return JSONResponse(content={
        "message": "Search successful",
        "k": k,
        "result_paths": result_paths,
        "distances": distances[0].tolist()
    })


@app.get("/get-image-by-path/")
def get_image_by_path(path: str):
    """
    Returns an image file directly from its path.
    """
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Image not found"}, status_code=404)
    
    return FileResponse(path)

