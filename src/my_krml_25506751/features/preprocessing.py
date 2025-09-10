import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataframe based on the specified strategy.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strategy : str, default 'mean'
        Strategy to handle missing values: 'mean', 'median', 'mode', or 'drop'.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

def scale_features(X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Scale the features using the specified method.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features dataframe (numeric columns assumed).
    method : str, default 'standard'
        Scaling method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
    
    Returns
    -------
    pd.DataFrame
        Scaled features dataframe.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
    scaled = scaler.fit_transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)

def encode_categorical(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    One-hot encode categorical columns in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        List of columns to encode. If None, encodes all object-type columns.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with categorical columns one-hot encoded.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return pd.get_dummies(df, columns=columns)

def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop specified columns from the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list
        List of columns to drop.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with columns dropped.
    """
    return df.drop(columns=columns, errors='ignore')

def detect_outliers(df: pd.DataFrame, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers in numeric columns using specified method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (numeric columns assumed).
    method : str, default 'zscore'
        Method to detect outliers: 'zscore' or 'iqr'.
    threshold : float, default 3.0
        Threshold for z-score or multiplier for IQR.
    
    Returns
    -------
    pd.DataFrame
        Boolean dataframe indicating outliers.
    """
    if method == 'zscore':
        z = np.abs((df - df.mean()) / df.std())
        return z > threshold
    elif method == 'iqr':
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))
    else:
        raise ValueError(f"Unsupported method: {method}")

def remove_outliers(df: pd.DataFrame, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from numeric columns using specified method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (numeric columns assumed).
    method : str, default 'zscore'
        Method to detect outliers: 'zscore' or 'iqr'.
    threshold : float, default 3.0
        Threshold for z-score or multiplier for IQR.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with outliers removed.
    """
    outliers = detect_outliers(df.select_dtypes(include=np.number), method, threshold)
    return df[~outliers.any(axis=1)]