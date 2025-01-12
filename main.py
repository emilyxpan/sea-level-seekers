import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import sys
from tqdm import tqdm

from datasets import FloodingDataset, FloodingDatasetStack
from models import CNNFeedforward, AttentionCNN, ConvLSTM

def calculate_metrics(outputs, labels):
    preds = (outputs > 0.5).float()
    tp = (preds * labels).sum()  # True Positives
    fp = (preds * (1 - labels)).sum()  # False Positives
    fn = ((1 - preds) * labels).sum()  # False Negatives
    tn = ((1 - preds) * (1 - labels)).sum()  # True Negatives

    # Calculate rates and scores
    tpr = tp / (tp + fn + 1e-8)  # True Positive Rate
    fpr = fp / (fp + tn + 1e-8)  # False Positive Rate
    precision = tp / (tp + fp + 1e-8)
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return fpr.item(), tpr.item(), f1.item()

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, output_dir="plots"):
    positive_weight = torch.tensor([30]).to(device)  # Example weight for imbalanced classes
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    # criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    # To store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_fprs, train_tprs, train_f1s = [], [], []
    val_fprs, val_tprs, val_f1s = [], [], []

    # Create output directory for plots if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_fprs, epoch_tprs, epoch_f1s = [], [], []

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate metrics
            preds = (outputs > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.numel()

            fpr, tpr, f1 = calculate_metrics(outputs, labels)
            epoch_fprs.append(fpr)
            epoch_tprs.append(tpr)
            epoch_f1s.append(f1)

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)
        train_fprs.append(sum(epoch_fprs) / len(epoch_fprs))
        train_tprs.append(sum(epoch_tprs) / len(epoch_tprs))
        train_f1s.append(sum(epoch_f1s) / len(epoch_f1s))

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        epoch_fprs, epoch_tprs, epoch_f1s = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate metrics
                preds = (outputs > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.numel()

                fpr, tpr, f1 = calculate_metrics(outputs, labels)
                epoch_fprs.append(fpr)
                epoch_tprs.append(tpr)
                epoch_f1s.append(f1)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)
        val_fprs.append(sum(epoch_fprs) / len(epoch_fprs))
        val_tprs.append(sum(epoch_tprs) / len(epoch_tprs))
        val_f1s.append(sum(epoch_f1s) / len(epoch_f1s))

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}, "
              f"Train FPR: {train_fprs[-1]:.4f}, Train TPR: {train_tprs[-1]:.4f}, Train F1: {train_f1s[-1]:.4f}, "
              f"Val FPR: {val_fprs[-1]:.4f}, Val TPR: {val_tprs[-1]:.4f}, Val F1: {val_f1s[-1]:.4f}")

    plot_graphs(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies, output_dir)  
    
def plot_graphs(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
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

import sys
import torch
from torch.utils.data import DataLoader

# Assuming your models are defined somewhere
# from models import AttentionCNN, CNNFeedforward
# from dataset import FloodingDataset
# from train import train_model

# Main Function
if __name__ == "__main__":
    # Check for command-line arguments
    model_flag = sys.argv[1] if len(sys.argv) > 1 else "CNNFeedforward"  # Default model is CNNFeedforward

    if model_flag == "load_model":
        model_path = sys.argv[2]
        model = ConvLSTM(input_channels=5, hidden_channels=64, kernel_size=3, num_classes=12)
        model.load_state_dict(torch.load(model_path))
        model.eval()  
        print("Model loaded successfully!")
    else:
            
        # Set device based on availability of GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
    
        # Directories for data
        nc_dir = "iharp_training_dataset/Copernicus_ENA_Satelite_Maps_Training_Data"
        label_dir = "iharp_training_dataset/Flooding_Data"
        cities = ["Atlantic City", "Baltimore", "Eastport", "Fort Pulaski", 
                  "Lewes", "New London", "Newport", "Portland", "Sandy Hook",
                  "Sewells Point", "The Battery", "Washington"]
    
        # Load dataset
        if model_flag != "ConvLSTM":
            dataset = FloodingDataset(nc_dir, label_dir, cities)
        else:
            dataset = FloodingDatasetStack(nc_dir, label_dir, cities, stack_size = 5)
    
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
        print(f"Total dataset size: {len(dataset)}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
        # Initialize the model based on the command-line flag
        if model_flag == "AttentionCNN":
            model = AttentionCNN()
        elif model_flag == "CNNFF":
            model = CNNFeedforward()
        elif model_flag == "ConvLSTM":
            model = ConvLSTM(input_channels=5, hidden_channels=64, kernel_size=3, num_classes=12)
    
        # Train the model
        train_model(model, train_loader, val_loader, num_epochs=4, learning_rate=0.0001, device=device)
    
        torch.save(model.state_dict(), "model_weights.pth")
