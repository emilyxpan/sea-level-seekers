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


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM Cell.
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        self.conv_i = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size, padding=self.kernel_size//2)
        self.conv_f = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size, padding=self.kernel_size//2)
        self.conv_c = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size, padding=self.kernel_size//2)
        self.conv_o = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size, padding=self.kernel_size//2)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), 1)  # concatenate input and previous hidden state
        i = torch.sigmoid(self.conv_i(combined))  # input gate
        f = torch.sigmoid(self.conv_f(combined))  # forget gate
        c_tilde = torch.tanh(self.conv_c(combined))  # candidate memory cell
        o = torch.sigmoid(self.conv_o(combined))  # output gate
        
        c = f * c + i * c_tilde  # update cell state
        h = o * torch.tanh(c)  # update hidden state
        
        return h, c

class ConvLSTM(nn.Module):
    """
    ConvLSTM model for predicting flooding status for 12 cities.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_classes=12):
        super(ConvLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        # Define the convolutional layers stack
        self.conv_stack = nn.ModuleList()
        self.conv_stack.append(nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1))  # Initial convolution
        for _ in range(input_channels - 1):
            self.conv_stack.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))  # Subsequent convolutions
        
        # Create ConvLSTM cells
        self.lstm_cells = nn.ModuleList([ConvLSTMCell(hidden_channels, hidden_channels, kernel_size) for _ in range(input_channels)])

        # Fully connected layer for final prediction
        self.fc = nn.Linear(hidden_channels * 100 * 160, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, height, width = x.size()  # B, N, 100, 160

        # Initialize hidden and cell states
        h, c = torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device), torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device)
        
        for t in range(seq_len):
            xi = x[:, t, :, :]  # Extract the feature map at time step t
            xi = xi.unsqueeze(1)  # Add channel dimension
            
            # Apply convolutions sequentially
            for i in range(len(self.conv_stack)):
                xi = self.conv_stack[i](xi)  # Apply each convolution in the stack
            
            # Pass through the ConvLSTM cell
            h, c = self.lstm_cells[t](xi, h, c)
        
        # Flatten the output
        h = h.view(batch_size, -1)
        
        # Final binary classification for each class
        out = self.fc(h)
        
        return torch.sigmoid(out)  # Output the probabilities for each class (12 cities)

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