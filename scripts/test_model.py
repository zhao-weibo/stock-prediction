import os
import torch
from scripts.data_preparation import load_and_preprocess_data
from scripts.predict import predict
from scripts.evaluate import evaluate_model, visualize_predictions

def test_model_on_stock(test_data_folder, model_path, device, TransformerModel):
    # Step 1: Load the trained model
    model = TransformerModel(input_dim=5, output_dim=5).to(device)
    # model.load_state_dict(torch.load(model_path))
    model_weights = torch.load(model_path, weights_only=True)
    model.load_state_dict(model_weights)
    print("[INFO] Model loaded successfully.")

    # Step 2: Test the model on each file in the folder
    for test_file in os.listdir(test_data_folder):
        if test_file.endswith(".csv"):
            test_file_path = os.path.join(test_data_folder, test_file)
            print(f"\n[INFO] Testing on stock data: {test_file}")

            try:
                # Load and preprocess test data (with scaler for normalization)
                X_test, _, y_test, _ = load_and_preprocess_data(test_file_path)
                # print('X_test:',X_test)
                # print('length of X_test:',len(X_test))
                # print('y_test:',y_test)
                # print('length of y_test:',len(y_test))
                # Make predictions
                predictions = predict(model, X_test, device)
                # print('predictions:',predictions)

                # Denormalize predictions and true values
                # predictions_denormalized = scaler.inverse_transform(predictions)
                # y_test_denormalized = scaler.inverse_transform(y_test)

                # print(predictions_denormalized)
                # print('/n')
                # print(y_test_denormalized)
                print('X_test:', X_test)
                print('y_test:',y_test)
                print('predictions:',predictions)
                # Evaluate model performance
                print(f"Results for {test_file}:")
                evaluate_model(predictions, y_test)

                # Visualize predictions
                visualize_predictions(predictions, y_test, X_test)

            except Exception as e:
                print(f"[ERROR] Failed to process {test_file}: {e}")



# import os
# import torch
# from scripts.data_preparation import load_and_preprocess_data
# from scripts.predict import predict
# from scripts.evaluate import evaluate_model, visualize_predictions

# def test_model_on_stock(test_data_folder, model_path, device, TransformerModel):
#     # Step 1: Load the trained model
#     model = TransformerModel(input_dim=5, output_dim=5).to(device)
#     model.load_state_dict(torch.load(model_path))
#     print("[INFO] Model loaded successfully.")

#     # Step 2: Test the model on each file in the folder
#     for test_file in os.listdir(test_data_folder):
#         if test_file.endswith(".csv"):
#             test_file_path = os.path.join(test_data_folder, test_file)
#             print(f"\n[INFO] Testing on stock data: {test_file}")

#             try:
#                 # Load and preprocess test data
#                 X_test, _, y_test, _ = load_and_preprocess_data(test_file_path)

#                 # Make predictions
#                 predictions = predict(model, X_test, device)

#                 # print('predictions:', len(predictions))
#                 # print('y_test:',len(y_test))

#                 # Evaluate model performance
#                 print(f"Results for {test_file}:")
#                 evaluate_model(predictions, y_test)

#                 # Visualize predictions
#                 visualize_predictions(predictions, y_test)

#             except Exception as e:
#                 print(f"[ERROR] Failed to process {test_file}: {e}")