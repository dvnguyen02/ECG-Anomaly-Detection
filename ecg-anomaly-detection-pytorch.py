import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score

# Set random seed for reproducibility
torch.manual_seed(1024)
np.random.seed(1024)

# Assuming the data loading and preprocessing steps remain the same
# Load your data here (normal_df, anomaly_df)

# Convert data to PyTorch tensors
X_train, X_test = train_test_split(normal_df.values, test_size=0.15, random_state=45, shuffle=True)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
anomaly = torch.FloatTensor(anomaly_df.values)

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, padding=1),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, padding=1),
            nn.Conv1d(128, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
            nn.MaxPool1d(2, padding=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)  # Remove channel dimension

# Create the model
input_dim = X_train.shape[1]
latent_dim = 32
model = Autoencoder(input_dim, latent_dim)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Create DataLoader
train_dataset = TensorDataset(X_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Train the model
num_epochs = 100
train(model, train_loader, criterion, optimizer, num_epochs)

# Evaluation functions
def predict(model, X):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        loss = torch.mean(torch.abs(pred - X), dim=1)
    return pred, loss

# Calculate threshold
_, train_loss = predict(model, X_train)
threshold = torch.mean(train_loss) + torch.std(train_loss)

# Evaluation
_, test_loss = predict(model, X_test)
_, anomaly_loss = predict(model, anomaly)

# Plot histograms
plt.figure(figsize=(9, 5), dpi=100)
sns.histplot(train_loss.numpy(), bins=40, kde=True, label="Train Normal")
sns.histplot(test_loss.numpy(), bins=40, kde=True, label="Test Normal")
sns.histplot(anomaly_loss.numpy(), bins=40, kde=True, label="Anomaly")
plt.axvline(threshold.item(), color='k', linestyle='--', label='Threshold')
plt.legend()
plt.title("Distribution of Reconstruction Errors")
plt.show()

# Prepare labels and predictions
def prepare_labels(model, train, test, anomaly, threshold):
    ytrue = torch.cat([torch.ones(len(train) + len(test)), torch.zeros(len(anomaly))])
    _, train_loss = predict(model, train)
    _, test_loss = predict(model, test)
    _, anomaly_loss = predict(model, anomaly)
    ypred = torch.cat([(train_loss <= threshold).float(), 
                       (test_loss <= threshold).float(), 
                       (anomaly_loss <= threshold).float()])
    return ytrue.numpy(), ypred.numpy()

# Evaluate model
ytrue, ypred = prepare_labels(model, X_train, X_test, anomaly, threshold)
print(classification_report(ytrue, ypred, target_names=["Anomaly", "Normal"]))

# Plot confusion matrix and ROC curve
def plot_confusion_matrix_and_roc_curve(ytrue, ypred):
    # Confusion matrix
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(ytrue, ypred)
    auc = roc_auc_score(ytrue, ypred)
    plt.subplot(122)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix_and_roc_curve(ytrue, ypred)
