import numpy as np
from sklearn.dummy import DummyRegressor

def create_null_model(strategy: str = 'mean'):
    """
    Create a null (baseline) regressor model.
    
    Parameters
    ----------
    strategy : str, default 'mean'
        Strategy for the dummy regressor: 'mean', 'median', 'constant', etc.
    
    Returns
    -------
    DummyRegressor
        Initialized null model.
    """
    return DummyRegressor(strategy=strategy)

def train_null_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Train the null model (fits the baseline).
    
    Parameters
    ----------
    model : DummyRegressor
        The null model to train.
    X_train : np.ndarray
        Training features (not really used for null model).
    y_train : np.ndarray
        Training target.
    
    Returns
    -------
    DummyRegressor
        Trained null model.
    """
    model.fit(X_train, y_train)
    return model