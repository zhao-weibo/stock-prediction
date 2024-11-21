import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from scripts.data_preparation import load_and_preprocess_data
from scripts.dataset import StockDataset
from scripts.transformer_model import TransformerModel
from scripts.train import train_model
from scripts.predict import predict
from scripts.evaluate import evaluate_model, visualize_predictions

# Paths
train_data_folder = "data/few_stock/"
test_data_folder = "data/test_stock/"
log_file = "logs/training_log.txt"

# Step 1: Load and preprocess all training data
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
for file_name in os.listdir(train_data_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(train_data_folder, file_name)
        print(f"[INFO] Processing file: {file_path}")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

# Combine all training data
X_train = np.concatenate(X_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# Step 2: Create training and testing datasets
train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 3: Model setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TransformerModel(input_dim=5, output_dim=5).to(device)

# Step 4: Train the model
train_model(model, train_loader, test_loader, device, log_file=log_file)

# Save the trained model
torch.save(model.state_dict(), "models/transformer_stock_model.pth")
print("[INFO] Model trained and saved successfully.")

# Step 5: Validate the model on `test_stock/`
for test_file in os.listdir(test_data_folder):
    if test_file.endswith(".csv"):
        test_file_path = os.path.join(test_data_folder, test_file)
        print(f"[INFO] Testing on stock data: {test_file}")

        # Load and preprocess test data
        try:
            X_test, _, y_test, _ = load_and_preprocess_data(test_file_path)
        except Exception as e:
            print(f"[ERROR] Failed to process test file {test_file}: {e}")
            continue

        # Make predictions
        predictions = predict(model, X_test, device)

        # Evaluate model performance
        evaluate_model(predictions, y_test)

        # Visualize predictions
        visualize_predictions(predictions, y_test)



# import os
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from scripts.data_preparation import load_and_preprocess_data
# from scripts.dataset import StockDataset
# from scripts.transformer_model import TransformerModel
# from scripts.train import train_model
# from scripts.predict import predict
# from scripts.evaluate import evaluate_model, visualize_predictions

# # Paths
# data_folder = "data/few_stock/"
# log_file = "logs/training_log.txt"

# # Step 1: Load and preprocess all CSV files
# X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
# for file_name in os.listdir(data_folder):
#     if file_name.endswith(".csv"):
#         file_path = os.path.join(data_folder, file_name)
#         if not os.path.exists(file_path):
#             print(f"File not found: {file_path}")
#         print(f"[INFO] Processing file: {file_path}")
#         # Load and preprocess each CSV file
#         X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
#         X_train_list.append(X_train)
#         X_test_list.append(X_test)
#         y_train_list.append(y_train)
#         y_test_list.append(y_test)

# # Step 2: Combine all data

# X_train = np.concatenate(X_train_list, axis=0)
# X_test = np.concatenate(X_test_list, axis=0)
# y_train = np.concatenate(y_train_list, axis=0)
# y_test = np.concatenate(y_test_list, axis=0)

# # Step 3: Create datasets and dataloaders
# train_dataset = StockDataset(X_train, y_train)
# test_dataset = StockDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# # Step 4: Model setup
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model = TransformerModel(input_dim=5, output_dim=5).to(device)

# # Step 5: Train the model
# train_model(model, train_loader, test_loader, device, log_file=log_file)

# # Step 6: Save the model
# torch.save(model.state_dict(), "models/transformer_stock_model.pth")


# # Step 7: Load and predict
# model.load_state_dict(torch.load("models/transformer_stock_model.pth", weights_only=True))
# predictions = predict(model, X_test, device)

# # Step 8: Print predictions
# print("Predictions vs Actuals:")
# for i in range(min(len(predictions), 5)): 
#     print(f"Sample {i + 1}:")
#     print(f"Predicted: {predictions[i]}")
#     print(f"Actual: {y_test[i]}")

# # Step 9: Evaluate the model
# evaluate_model(predictions, y_test)

# # Step 10: Visualize the predictions
# visualize_predictions(predictions, y_test)