import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import faiss
import pickle
import torch
from torchvision import datasets, transforms
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Import the CORS middleware

from feature_extractor import FeatureExtractor

# --- Initialization ---

# Initialize FastAPI app
app = FastAPI(title="Visual Search Engine API")

# --- Add CORS Middleware ---
# This is the new section that fixes the error.
# It allows requests from any origin, which is fine for development.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------


# Load the pre-built Faiss index
print("Loading Faiss index...")
index = faiss.read_index("index.faiss")
print("✅ Index loaded.")

# Load the image index mapping
print("Loading image indices mapping...")
with open("image_indices.pkl", "rb") as f:
    image_indices = pickle.load(f)
print("✅ Mapping loaded.")

# Initialize our feature extractor
print("Initializing feature extractor...")
extractor = FeatureExtractor()
print("✅ Feature extractor initialized.")

# Load the dataset (we need it to get the image paths for the results)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='fashion_mnist_data', train=True, download=False, transform=transform)


# --- API Endpoints ---

@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Visual Search Engine API!"}


@app.post("/search/")
async def search(file: UploadFile = File(...), k: int = 5):
    """
    Accepts an image upload, finds the k most similar images, 
    and returns their indices.
    """
    # 1. Read and process the uploaded image
    contents = await file.read()
    query_image = Image.open(io.BytesIO(contents)).convert("L") # Convert to grayscale
    query_tensor = transform(query_image)

    # 2. Get the embedding for the query image
    query_embedding = extractor.get_embedding(query_tensor)
    query_embedding_numpy = query_embedding.unsqueeze(0).numpy()

    # 3. Search the Faiss index
    # D: distances, I: indices of the nearest neighbors
    distances, indices = index.search(query_embedding_numpy, k)

    # 4. Get the original dataset indices for the results
    result_indices = [image_indices[i] for i in indices[0]]

    return JSONResponse(content={
        "message": "Search successful",
        "k": k,
        "result_indices": result_indices,
        "distances": distances[0].tolist()
    })


from fastapi.responses import StreamingResponse

@app.get("/get-image/{image_index}")
def get_image(image_index: int):
    """
    Returns the image file for a given index from the dataset.
    """
    # Get the image tensor from the dataset
    image_tensor, _ = train_dataset[image_index]

    # Convert the tensor to a PIL Image
    # The tensor is normalized, so we need to un-normalize it (multiply by 255)
    image_pil = transforms.ToPILImage()(image_tensor)

    # Save the PIL image to a byte stream
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0) # Go to the beginning of the stream

    return StreamingResponse(img_byte_arr, media_type="image/png")