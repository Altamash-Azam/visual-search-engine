import os
import torch
from torchvision import datasets, transforms
from feature_extractor import FeatureExtractor 
import faiss
import numpy as np
from tqdm import tqdm
import pickle

# --- Step 1: Load the Dataset ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='fashion_mnist_data', train=True, download=True, transform=transform)

print(f"Number of images in dataset: {len(train_dataset)}")

# --- Step 2: Initialize Feature Extractor ---
extractor = FeatureExtractor()
print("✅ Feature extractor initialized.")

# --- Test on a single image ---
sample_image, _ = train_dataset[0]
sample_embedding = extractor.get_embedding(sample_image)
print(f"Shape of sample embedding: {sample_embedding.shape}")



# --- Step 3: Build the Faiss Index ---

print("\n--- Starting to build Faiss index ---")

# The dimension of our embeddings is 512
d = 512
index = faiss.IndexFlatL2(d)

# We will also need a way to map from an index ID to an image
# Let's store the dataset indices
image_indices = []

# Loop through the dataset and add embeddings to the index
for i, (image, label) in enumerate(tqdm(train_dataset)):
    # Get the embedding
    embedding = extractor.get_embedding(image)
    
    # Faiss requires numpy arrays, so we convert the tensor
    # Also, the input to Faiss must be a 2D array (batch of vectors)
    embedding_numpy = embedding.unsqueeze(0).numpy()
    
    # Add the vector to the index
    index.add(embedding_numpy)
    
    # Store the original dataset index
    image_indices.append(i)

print(f"\n✅ Faiss index built successfully!")
print(f"Total vectors in index: {index.ntotal}")

# --- Step 4: Save the Index and Mapping ---

# Save the Faiss index to a file
faiss.write_index(index, "index.faiss")

# Save the image index mapping
with open("image_indices.pkl", "wb") as f:
    pickle.dump(image_indices, f)

print("\n--- Index and mapping saved to disk ---")
print("Offline indexing process complete.")
