import os
import torch
from torch.utils.data import DataLoader
from scripts.data_preparation import load_and_preprocess_data
from scripts.dataset import StockDataset
from scripts.transformer_model import TransformerModel
from scripts.train import train_model
from scripts.test_model import test_model_on_stock

# Paths
train_data_folder = "data/few_stock/"
test_data_folder = "data/test_stock/"
log_file = "logs/training_log.txt"
model_save_path = "models/transformer_stock_model_on_selected_50_stock.pth"

# Step 1: Model setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TransformerModel(input_dim=5, output_dim=5).to(device)

# Step 2: Train the model with stocks in `few_stock/`
for file_name in os.listdir(train_data_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(train_data_folder, file_name)
        print(f"[INFO] Training on stock data: {file_name}")

        # Load and preprocess data for this stock
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to process file {file_name}: {e}")
            continue

        # Create datasets and dataloaders
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Train the model on this stock's data
        train_model(model, train_loader, test_loader, device, log_file=log_file)
        print(f"[INFO] Completed training on: {file_name}")

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"[INFO] Final model saved at: {model_save_path}")

# Step 3: Test the model on `test_stock/`
test_model_on_stock(test_data_folder, model_save_path, device, TransformerModel)
