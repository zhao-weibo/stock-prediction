# import torch

# def predict(model, X_test, device, sample_size=5):
#     model.eval()
#     sample_input = torch.tensor(X_test[:sample_size], dtype=torch.float32).to(device)
#     with torch.no_grad():
#         predictions = model(sample_input).cpu().numpy()
#     return predictions

import torch
import numpy as np

def predict(model, X_test, device):
    model.eval()
    predictions = []  # Store predictions for all samples

    with torch.no_grad():
        for i in range(len(X_test)):
            # Prepare input tensor for a single sample
            input_tensor = torch.tensor(X_test[i]).unsqueeze(0).float().to(device)  # Shape: (1, 50, 5)
            
            # Generate prediction
            output = model(input_tensor).cpu().numpy()  # Shape: (1, 5)
            predictions.append(output.squeeze())  # Append (5,) to predictions list

    # Convert list of predictions to numpy array
    return np.array(predictions)  # Shape: (N, 5)