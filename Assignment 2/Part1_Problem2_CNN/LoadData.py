import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_dir, batch_size=32):
    """
    Loads the ReducedMNIST dataset from the given directory and creates PyTorch DataLoaders.
    
    Args:
        data_dir (str): The root directory containing 'train' and 'test' folders.
        batch_size (int): The number of samples per batch.
        
    Returns:
        train_loader, test_loader: PyTorch DataLoaders for training and testing.
    """
    # 1. Define transformations
    # - Grayscale: Ensure the images are read as 1 channel (grayscale) instead of 3 channels (RGB)
    # - Resize: Explicitly resize to 28x28 as instructed in the assignment
    # - ToTensor: Convert images to PyTorch tensors and scale pixel values to [0.0, 1.0]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Paths to train and test directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 2. Create Dataset objects using ImageFolder
    # ImageFolder automatically labels images based on the folder names (0, 1, ..., 9)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # 3. Create DataLoaders
    # DataLoaders handle batching, shuffling, and loading the data efficiently
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage to test the code:
if __name__ == "__main__":
    # We use raw string (r"") to handle Windows backslashes properly
    data_dir = r"d:\EECE4\NeuralNetworks\repo-clone\Materials\ReducedMNIST_generated"
    
    # Get the dataloaders
    train_loader, test_loader = get_dataloaders(data_dir, batch_size=32)
    
    # Check the size of the datasets
    print(f"Total training examples: {len(train_loader.dataset)}")
    print(f"Total testing examples: {len(test_loader.dataset)}")
    
    # Get one batch of data to check the shape
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}") # Expected: [32, 1, 28, 28]
    print(f"Batch labels shape: {labels.shape}") # Expected: [32]
