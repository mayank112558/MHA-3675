
import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> object:
    """
    Train a linear regression model.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training target.
    
    Returns
    -------
    object
        Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_with_model(model: object, X: np.ndarray) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Parameters
    ----------
    model : object
        Trained model.
    X : np.ndarray
        Features to predict on.
    
    Returns
    -------
    np.ndarray
        Predictions.
    """
    return model.predict(X)