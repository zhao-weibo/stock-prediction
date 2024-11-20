import torch
import logging
from torch.utils.data import DataLoader
from scripts.transformer_model import TransformerModel

def train_model(model, train_loader, test_loader, device, epochs=10, lr=0.001, log_file="logs/training_log.txt"):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
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

        avg_train_loss = train_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss}")

    # Testing loop
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        logging.info(f"Test Loss: {avg_test_loss}")
        print(f"Test Loss: {avg_test_loss}")