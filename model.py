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

# Custom Dataset for Flooding Data
class FloodingDataset(Dataset):
    def __init__(self, nc_dir, label_dir, cities, start_year=1993, end_year=2013):
        self.nc_dir = nc_dir
        self.cities = cities
        self.data = []
        self.labels = []
        self._prepare_data(label_dir, start_year, end_year)

    def _prepare_data(self, label_dir, start_year, end_year):
        cities = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
        anomaly_data = []
        for file in cities:
            file_path = os.path.join(label_dir, file)
            df = pd.read_csv(file_path)
            anomaly_data.append(df['anomaly'].iloc[:7064].values)

        self.labels = np.array(anomaly_data).T

        ncfiles = [f for f in os.listdir(self.nc_dir) if f.endswith('.nc')]

        for file in ncfiles:
            file_path = os.path.join(self.nc_dir, file)
            self.data.append(file_path)
            if len(self.data) == 7064: break

        # for city in self.cities:
        #     # Load the city-specific CSV
        #     label_path = os.path.join(label_dir, f"{city.replace(' ', '_')}_{start_year}_{end_year}_training_data.csv")
        #     labels = pd.read_csv(label_path)
            
        #     # Iterate through the dates and match with .nc files
        #     for _, row in labels.iterrows():
        #         date = datetime.strptime(row['t'], "%Y-%m-%d")
        #         nc_file = os.path.join(
        #             self.nc_dir,
        #             f"dt_ena_{date.strftime('%Y%m%d')}_vDT2021.nc"
        #         )
        #         if os.path.exists(nc_file):
        #             if city == 'Atlantic City':
        #                 self.data.append(nc_file)
        #             # self.labels.append(row['anomaly'])  # 0 or 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load .nc file and anomaly label
        nc_file = self.data[idx]
        nc_data = ncDataset(nc_file, 'r')
        sea_level = np.array(np.squeeze(nc_data.variables['sla'][:]))  # Adjust for your variable name
        nc_data.close()
        sea_level = torch.tensor(sea_level, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sea_level, label

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

import matplotlib.pyplot as plt
import os

# Training Function with Accuracy and Plot Saving
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, output_dir="plots"):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    
    # To store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Create output directory for plots if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Match output shape for BCE loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate training accuracy
            preds = (outputs > 0.5).float()  # Threshold at 0.5 for multi-label classification
            correct_train += (preds == labels).sum().item()
            total_train += labels.numel()

        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate validation accuracy
                preds = (outputs > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.numel()

        val_accuracy = correct_val / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

    # Plotting training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save loss plot
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Save accuracy plot
    accuracy_plot_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy plot saved to {accuracy_plot_path}")

    plt.close()  # Close the figure to free up memory

# Main Function
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    nc_dir = "iharp_training_dataset/Copernicus_ENA_Satelite_Maps_Training_Data"
    label_dir = "iharp_training_dataset/Flooding_Data"
    cities = ["Atlantic City", "Baltimore", "Eastport", "Fort Pulaski", 
                "Lewes", "New London", "Newport", "Portland", "Sandy Hook",
                "Sewells Point", "The Battery", "Washington"]

    dataset = FloodingDataset(nc_dir, label_dir, cities)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(len(dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = AttentionCNN()
    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.1, device=device)
