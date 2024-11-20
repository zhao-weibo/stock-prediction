import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, window_size=30, target_size=5):
    # Load data
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[['volume', 'open', 'high', 'low', 'close']])
    scaled_df = pd.DataFrame(scaled_features, columns=['volume', 'open', 'high', 'low', 'close'])

    # Create sliding windows
    X, y = [], []
    for i in range(len(scaled_df) - window_size - target_size + 1):
        past_window = scaled_df.iloc[i:i+window_size][['volume', 'open', 'high', 'low', 'close']].values
        future_window = scaled_df.iloc[i+window_size:i+window_size+target_size]['close'].values
        X.append(past_window)
        y.append(future_window)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split into train/test
    return train_test_split(X, y, test_size=0.2, random_state=42)