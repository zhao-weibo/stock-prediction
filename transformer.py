import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# ========================
# 1. Data Preparation
# ========================
# Load data
data_folder = "data/full_history"
file_name = "A.csv"
file_path = os.path.join(data_folder, file_name)

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")

df = data

# Data preprocessing
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')  # Ensure the data is sorted by date

print(df)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['volume', 'open', 'high', 'low', 'close', 'adj close']])
scaled_df = pd.DataFrame(scaled_features, columns=['volume', 'open', 'high', 'low', 'close', 'adj close'])

# Construct sliding window
window_size = 30  # Use the past 30 days
target_size = 5   # Predict the next 5 days

X, y = [], []
for i in range(len(scaled_df) - window_size - target_size + 1):
    # Input: past 30 days' features
    past_window = scaled_df.iloc[i:i+window_size][['volume', 'open', 'high', 'low', 'close']].values
    # Output: next 5 days' close values
    future_window = scaled_df.iloc[i+window_size:i+window_size+target_size]['close'].values
    X.append(past_window)
    y.append(future_window)

# Convert to NumPy arrays
X = np.array(X)  # Shape: (number of samples, 30, 5)
y = np.array(y)  # Shape: (number of samples, 5)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================
# 2. Define PyTorch Dataset
# ========================
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Input (number of samples, 30, 5)
        self.y = torch.tensor(y, dtype=torch.float32)  # Output (number of samples, 5)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ========================
# 3. Define Transformer Model
# ========================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=2, num_layers=2, hidden_dim=64):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # Maximum sequence length is 1000
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=128,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        # src: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = src.size()
        
        # Add positional encoding
        src = self.input_layer(src) + self.positional_encoding[:, :seq_len, :]
        
        # Directly pass to the Transformer, no need to permute
        transformer_output = self.transformer(src, src)
        
        # Use the output from the last time step as the result
        output = self.output_layer(transformer_output[:, -1, :])  # (batch_size, output_dim)
        
        return output

# ========================
# 4. Model Training and Testing
# ========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TransformerModel(input_dim=5, output_dim=5).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader)}")

# Test the model
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader)}")