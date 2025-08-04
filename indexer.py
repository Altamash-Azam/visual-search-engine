import torch
from torchvision import datasets, transforms
import os

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {datasets.__version__}")

# --- Step 1: Load the Dataset ---

# Define a transform to convert images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create a directory to store the data
data_dir = 'fashion_mnist_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download the training dataset
train_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

print("\n✅ Fashion-MNIST dataset downloaded successfully!")
print(f"Number of images in the dataset: {len(train_dataset)}")

# Let's inspect a single data point
image, label = train_dataset[0]
print("\n--- Sample Data Point ---")
print(f"Image shape: {image.shape}")
print(f"Label: {label}")
print("Image shape means [Color Channels, Height, Width]. [1, 28, 28] is a 28x28 grayscale image.")