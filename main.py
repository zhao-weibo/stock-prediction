import os
import torch
import numpy as np
from scripts.data_preparation import load_and_preprocess_data
from scripts.dataset import StockDataset
from scripts.transformer_model import TransformerModel
from scripts.train import train_model
from scripts.predict import predict
from scripts.evaluate import evaluate_model, visualize_predictions
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt


# Paths
data_path = "data/full_history/A.csv"
log_file = "logs/training_log.txt"

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

# Create datasets and dataloaders
train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TransformerModel(input_dim=5, output_dim=5).to(device)

# Train the model
train_model(model, train_loader, test_loader, device, log_file=log_file)

# Save the model
torch.save(model.state_dict(), "models/transformer_stock_model.pth")

# Load and predict
model.load_state_dict(torch.load("models/transformer_stock_model.pth", weights_only=True))
predictions = predict(model, X_test, device)

# Print predictions
print("Predictions vs Actuals:")
for i in range(min(len(predictions), 5)): 
    print(f"Sample {i + 1}:")
    print(f"Predicted: {predictions[i]}")
    print(f"Actual: {y_test[i]}")

# Evaluate the model
evaluate_model(predictions, y_test)

# Visualize the predictions
visualize_predictions(predictions, y_test)