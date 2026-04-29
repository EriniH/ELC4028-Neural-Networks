import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    """
    Baseline CNN Model for ReducedMNIST (28x28 images).
    This architecture is heavily inspired by LeNet-5 but uses ReLU 
    as the activation function as explicitly requested in the assignment.
    """
    def __init__(self):
        super(BaseCNN, self).__init__()
        
        # 1. First Convolutional Layer
        # Input: 1 channel (grayscale), Output: 6 channels (filters), Kernel Size: 5x5
        # Padding=2 is used so the 28x28 image doesn't shrink during this convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        
        # 2. Second Convolutional Layer
        # Input: 6 channels (from conv1), Output: 16 channels, Kernel Size: 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 3. Fully Connected (Linear) Layers
        # After two 2x2 Max Pooling steps, the image dimension shrinks:
        # 28x28 -> pool1 -> 14x14 -> conv2 -> 10x10 -> pool2 -> 5x5
        # So the flattened size is 16 channels * 5 * 5 = 400
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        
        # Output layer (10 classes for digits 0-9)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Pass through Conv1 -> ReLU -> MaxPool (2x2)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Pass through Conv2 -> ReLU -> MaxPool (2x2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the tensor for the fully connected layers
        # x.size(0) is the batch size, -1 flattens all remaining dimensions
        x = x.view(x.size(0), -1)
        
        # Pass through Fully Connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Final output layer (no ReLU here, loss function Handles softmax)
        x = self.fc3(x)
        
        return x


class WiderCNN(nn.Module):
    """
    Variation: increase convolutional filters and hidden layer width.
    This tests model capacity as a hyperparameter change.
    """

    def __init__(self):
        super(WiderCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5)

        self.fc1 = nn.Linear(in_features=32 * 5 * 5, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeakyReLUCNN(nn.Module):
    """
    Variation: replace ReLU with LeakyReLU.
    This tests activation-function choice as a hyperparameter change.
    """

    def __init__(self, negative_slope=0.01):
        super(LeakyReLUCNN, self).__init__()

        self.negative_slope = negative_slope

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.negative_slope)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.negative_slope)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope)
        x = self.fc3(x)
        return x


class DropoutCNN(nn.Module):
    """
    Variation: add dropout before the last two fully connected layers.
    This tests regularization as a hyperparameter change.
    """

    def __init__(self, dropout_p=0.3):
        super(DropoutCNN, self).__init__()

        self.dropout = nn.Dropout(p=dropout_p)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x


def get_model_variants():
    """
    Returns experiment-ready model constructors and short descriptions.
    """
    return {
        "BaseCNN": {
            "builder": BaseCNN,
            "description": "Baseline LeNet-style CNN with ReLU",
        },
        "Variation1_Wider": {
            "builder": WiderCNN,
            "description": "Increased filters and FC widths",
        },
        "Variation2_LeakyReLU": {
            "builder": LeakyReLUCNN,
            "description": "ReLU replaced by LeakyReLU",
        },
        "Variation3_Dropout": {
            "builder": DropoutCNN,
            "description": "Added dropout regularization before FC2/FC3",
        },
    }

# Example usage to verify the model structure
if __name__ == "__main__":
    # Create an instance of the model
    model = BaseCNN()
    
    # Create a dummy input tensor matching our batch shape: [Batch_Size, Channels, Height, Width]
    dummy_input = torch.randn(32, 1, 28, 28)
    
    # Pass the dummy input through the model
    output = model(dummy_input)
    
    print(f"Model Structure:\n{model}")
    print(f"Output shape: {output.shape}") # Expected: [32, 10]
