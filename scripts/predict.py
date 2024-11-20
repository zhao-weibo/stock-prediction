import torch

def predict(model, X_test, device, sample_size=5):
    model.eval()
    sample_input = torch.tensor(X_test[:sample_size], dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(sample_input).cpu().numpy()
    return predictions