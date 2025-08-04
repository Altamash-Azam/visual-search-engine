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



