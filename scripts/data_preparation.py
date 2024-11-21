import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view

def load_and_preprocess_data(file_path, window_size=50, target_size=5):
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert 'date' column to datetime and filter data after 2000
    df['date'] = pd.to_datetime(df['date'])
    # df = df[df['date'] >= pd.Timestamp('2000-01-01')]
    # if df.empty:
    #     raise ValueError(f"No data available after 2000 in {file_path}.")

    # Sort by date to ensure chronological order
    df = df.sort_values('date', ascending=True).reset_index(drop=True)

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print(f"[WARNING] Missing values detected in {file_path}. Filling missing values...")
        df = df.bfill().ffill()

    # Ensure there are no remaining missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError(f"Unresolved missing values in {file_path} after filling.")

    # Scale features
    # feature_scaler = MinMaxScaler()
    # target_scaler = MinMaxScaler()

    # feature_columns = ['volume', 'open', 'high', 'low']
    # df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])
    # df['close'] = target_scaler.fit_transform(df[['close']])
    scaler = MinMaxScaler()
    feature_columns = ['volume', 'open', 'high', 'low', 'close']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    # print(df)
    # Create sliding windows
    # X = sliding_window_view(df[['volume', 'open', 'high', 'low', 'close']].values, (window_size, len(feature_columns)))
    # X = X[:-target_size].squeeze()
    # y = sliding_window_view(df['close'].values, target_size)[:-window_size]
    # print(X)
    # print('yyyyyyyyyyyyyyyyyyyy')
    # print(y)

    X, y = [], []
    for i in range(len(df) - window_size - target_size + 1):
        past_window = df.iloc[i:i+window_size][['volume', 'open', 'high', 'low', 'close']].values
        future_window = df.iloc[i+window_size:i+window_size+target_size]['close'].values
        X.append(past_window)
        y.append(future_window)

    # Convert to numpy arrays
    
    X = np.array(X)
    y = np.array(y)
    # print(X)
    # print('yyyyyyyyyyyy')
    # print(y)
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test




# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split


# def load_and_preprocess_data(file_path, window_size=30, target_size=5, return_scaler=False):
#     # Load data
#     df = pd.read_csv(file_path)
    
#     # Convert 'date' column to datetime and filter data after 2000
#     df['date'] = pd.to_datetime(df['date'])
#     df = df[df['date'] >= pd.Timestamp('2000-01-01')]
#     if df.empty:
#         raise ValueError(f"No data available after 2000 in {file_path}.")

#     # Sort by date to ensure chronological order
#     df = df.sort_values('date', ascending=True).reset_index(drop=True)

#     # Handle missing values
#     if df.isnull().sum().sum() > 0:
#         print(f"[WARNING] Missing values detected in {file_path}. Filling missing values...")
#         df = df.bfill().ffill()

#     # Ensure there are no remaining missing values
#     if df.isnull().sum().sum() > 0:
#         raise ValueError(f"Unresolved missing values in {file_path} after filling.")

#     # Scale features
#     # scaler = MinMaxScaler()
#     # scaled_features = scaler.fit_transform(df[['volume', 'open', 'high', 'low', 'close']])
#     # scaled_df = pd.DataFrame(scaled_features, columns=['volume', 'open', 'high', 'low', 'close'])

#     feature_scaler = MinMaxScaler()
#     target_scaler = MinMaxScaler()

#     feature_columns = ['volume', 'open', 'high', 'low']
#     df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])
#     df['close'] = target_scaler.fit_transform(df[['close']])

#     # Create sliding windows
#     X, y = [], []
#     for i in range(len(df) - window_size - target_size + 1):
#         past_window = df.iloc[i:i+window_size][['volume', 'open', 'high', 'low', 'close']].values
#         future_window = df.iloc[i+window_size:i+window_size+target_size]['close'].values
#         X.append(past_window)
#         y.append(future_window)

#     # Convert to numpy arrays
#     X = np.array(X)
#     y = np.array(y)

#     # Split into train/test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     if return_scaler:
#         return X_train, X_test, y_train, y_test, target_scaler
#     else:
#         return X_train, X_test, y_train, y_test


