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

def visualize_predictions(predictions, y_test):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test[0])), y_test[0], label="Actual", marker='o')
    plt.plot(range(len(predictions[0])), predictions[0], label="Predicted", marker='x')
    plt.title("Actual vs Predicted Close Prices (First Test Sample)")
    plt.xlabel("Future Days")
    plt.ylabel("Normalized Close Price")
    plt.legend()
    plt.show()