import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from datasets import FloodingDataset, FloodingDatasetStack
from models import CNNFeedforward, AttentionCNN

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

    # Split dataset into train (first 80%) and validation (last 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # No random split, just slicing the dataset
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    print(f"Total dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    # model = AttentionCNN()
    model = CNNFeedforward()

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.0001, device=device)