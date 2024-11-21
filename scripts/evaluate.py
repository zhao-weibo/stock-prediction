import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def evaluate_model(predictions, y_test):
    predictions = predictions[:len(y_test)]
    y_test = y_test[:len(predictions)]

    # print('length of predictions:',len(predictions))
    # print('predictions:',predictions)
    # print('length of y_test:',len(y_test))
    # print('y_test:',y_test)


    mse = mean_squared_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    smape = symmetric_mean_absolute_percentage_error(y_test, predictions)

    print("\nEvaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")

import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(predictions, y_test, x_test, num_samples=4):
    # Check if x_test is 3D
    if x_test.ndim == 3:
        # If there are multiple features, only select closing price
        x_test = x_test[:, :, 4]
    
    # Ensure inputs are 2D after transformation
    if predictions.ndim != 2 or y_test.ndim != 2 or x_test.ndim != 2:
        raise ValueError("All inputs (predictions, y_test, x_test) must be 2D arrays.")
    
    # Limit the number of samples to plot
    num_samples = min(num_samples, predictions.shape[0], y_test.shape[0], x_test.shape[0])
    
    # Create a figure with subplots
    cols = 2  # Number of columns for subplots
    rows = (num_samples + cols - 1) // cols  # Calculate required rows
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = axes.flatten()  # Flatten the axes for easier indexing
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Combine x_test and y_test for complete x-axis
        full_context = list(range(x_test.shape[1] + y_test.shape[1]))
        
        # Plot x_test as historical context
        ax.plot(range(x_test.shape[1]), x_test[i], label="Historical (x_test)", color='blue', marker='o')
        
        # Plot y_test as actual future values
        ax.plot(range(x_test.shape[1], x_test.shape[1] + y_test.shape[1]), y_test[i], label="Actual (y_test)", color='green', marker='o')
        
        # Plot predictions as predicted future values
        ax.plot(range(x_test.shape[1], x_test.shape[1] + predictions.shape[1]), predictions[i], label="Predicted", color='red', linestyle='--', marker='x')
        
        # Add labels and titles
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Normalized Values")
        ax.legend()
    
    # Hide unused subplots if num_samples < total subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()