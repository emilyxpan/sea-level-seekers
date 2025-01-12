import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from netCDF4 import Dataset as ncDataset
from datetime import datetime
import matplotlib.pyplot as plt


torch.manual_seed(42)


class CNNFeedforward(nn.Module):
    def __init__(self, input_shape=(1, 100, 160), num_classes=12):
        super(CNNFeedforward, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Flattened size calculation after pooling layers
        flattened_size = (input_shape[1] // 8) * (input_shape[2] // 8) * 128
        
        # Feedforward layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool((self.conv1(x)))
        x = self.pool((self.conv2(x)))
        x = self.pool((self.conv3(x)))

        # Flatten the tensor for feedforward layers
        x = x.view(x.size(0), -1)

        # Apply feedforward layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply sigmoid activation for multi-class classification
        return self.sigmoid(x)

# Attention-based CNN Architecture
class AttentionCNN(nn.Module):
    def __init__(self, input_shape=(1, 100, 160), num_classes=12):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        flattened_size = (input_shape[1] // 8) * (input_shape[2] // 8) * 128
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        attention_weights = self.attention(x)
        x = x * attention_weights
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)