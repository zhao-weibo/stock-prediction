import torch
from scripts.transformer_model import TransformerModel
from scripts.test_model import test_model_on_stock

# Paths
test_data_folder = "data/test_stock/" 
model_save_path = "models/transformer_stock_model_on_selected_50_stock.pth"

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Step 1: Call `test_model_on_stock` to evaluate the model
print("[INFO] Starting model evaluation on test data...")
try:
    test_model_on_stock(test_data_folder, model_save_path, device, TransformerModel)
    print("[INFO] Model evaluation completed successfully.")
except Exception as e:
    print(f"[ERROR] Model evaluation failed: {e}")