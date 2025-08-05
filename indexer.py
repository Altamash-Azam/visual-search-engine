import os
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from feature_extractor import FeatureExtractor

# --- Step 1: Create a Custom Dataset for our Image Folder ---

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Get a list of all image file paths and sort them
        self.image_files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path # Return the image and its path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor and path if an image is corrupt
            return torch.zeros(3, 224, 224), img_path


# --- Step 2: Define Image Transformations for High-Res Images ---

# ResNet-18 was trained on 224x224 images. We will resize and normalize our images accordingly.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Step 3: Load the Dataset and Initialize ---

# Path to your high-resolution image folder
dataset_path = 'images' 
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The dataset folder '{dataset_path}' was not found. Please download and place it in the project directory.")

dataset = ImageFolderDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"Number of high-resolution images found: {len(dataset)}")

# Initialize our feature extractor
extractor = FeatureExtractor()
print("✅ Feature extractor initialized.")


# --- Step 4: Build and Save the Faiss Index ---

print("\n--- Starting to build Faiss index for high-res images ---")

d = 512  # Dimension of our embeddings
index = faiss.IndexFlatL2(d)

# We will store the actual image paths now instead of indices
image_paths = []

# Loop through the dataset using the dataloader
for image_tensor, path_list in tqdm(dataloader):
    path = path_list[0] # Dataloader returns a list of paths
    try:
        embedding = extractor.get_embedding(image_tensor.squeeze(0)) # Remove batch dim from dataloader
        embedding_numpy = embedding.unsqueeze(0).numpy()
        
        index.add(embedding_numpy)
        image_paths.append(path)
    except Exception as e:
        print(f"Skipping image {path} due to an error: {e}")


print(f"\n✅ Faiss index built successfully!")
print(f"Total vectors in index: {index.ntotal}")

# Save the updated index and the new path mapping
faiss.write_index(index, "index_high_res.faiss")

with open("image_paths_high_res.pkl", "wb") as f:
    pickle.dump(image_paths, f)

print("\n--- High-resolution index and path mapping saved to disk ---")
print("Offline indexing process complete.")
