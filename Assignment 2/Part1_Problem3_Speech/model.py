import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSpeechCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleSpeechCNN, self).__init__()
        # Input images are 1 channel (grayscale spectrograms) and size 64x64
        
        # Convolutional Layer 1
        # Extract features from the spectrogram image
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output will be 32x32

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output will be 16x16

        # Fully Connected Layer 1
        # Flatten the 32 feature maps of size 16x16
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)
        self.dropout = nn.Dropout(0.5)  # Helps prevent overfitting
        
        # Fully Connected Output Layer (10 digits)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Apply Conv1 + ReLU + MaxPool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Apply Conv2 + ReLU + MaxPool
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten into a vector sequence
        x = x.view(x.size(0), -1)

        # Apply FC1 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output target scores (unnormalized)
        x = self.fc2(x)
        return x
