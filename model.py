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