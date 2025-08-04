import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor:
    def __init__(self):
        # Load the pre-trained ResNet-18 model
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

        # Modify the model to remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

        # Set the model to evaluation mode
        self.feature_extractor.eval()

    def get_embedding(self, image_tensor):
        """
        Takes a single image tensor and returns its feature embedding.
        """
        # The model expects a batch of images, so we add a batch dimension
        # and handle grayscale images by repeating the channel.
        if image_tensor.shape[0] == 1: # Check if it's a grayscale image
            image_tensor = image_tensor.repeat(3, 1, 1)
        
        image_tensor = image_tensor.unsqueeze(0) # Add batch dimension

        # Use torch.no_grad() for faster inference
        with torch.no_grad():
            embedding = self.feature_extractor(image_tensor)
        
        # Squeeze the output to get a 1D vector
        return embedding.squeeze()

# This allows us to test the extractor directly if we run this file
if __name__ == '__main__':
    from torchvision import datasets, transforms

    # Test the class
    extractor = FeatureExtractor()
    print("✅ FeatureExtractor class initialized successfully.")

    # Load a sample image
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root='fashion_mnist_data', train=True, download=True, transform=transform)
    sample_image, _ = train_dataset[0]

    # Get the embedding
    embedding = extractor.get_embedding(sample_image)
    print(f"Generated embedding shape: {embedding.shape}") # Should be torch.Size([512])