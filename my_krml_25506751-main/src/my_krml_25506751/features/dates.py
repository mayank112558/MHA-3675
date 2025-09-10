def convert_to_date(df, cols: list):
    """Convert specified columns from a Pandas dataframe into datetime

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : list
        List of columns to be converted

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with converted columns
    """
    import pandas as pd

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df
    
def extract_date_features(df: pd.DataFrame, date_cols: list, features: list = ['year', 'month', 'day', 'weekday']) -> pd.DataFrame:
    """
    Extract specified features from datetime columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime columns.
    date_cols : list
        List of datetime columns to extract features from.
    features : list, default ['year', 'month', 'day', 'weekday']
        List of features to extract: 'year', 'month', 'day', 'weekday', 'hour', 'minute', etc.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with new date features added.
    """
    for col in date_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            dt = df[col].dt
            for feat in features:
                if feat == 'year':
                    df[f'{col}_year'] = dt.year
                elif feat == 'month':
                    df[f'{col}_month'] = dt.month
                elif feat == 'day':
                    df[f'{col}_day'] = dt.day
                elif feat == 'weekday':
                    df[f'{col}_weekday'] = dt.weekday
                elif feat == 'hour':
                    df[f'{col}_hour'] = dt.hour
                elif feat == 'minute':
                    df[f'{col}_minute'] = dt.minute
                # Add more as needed
    return df