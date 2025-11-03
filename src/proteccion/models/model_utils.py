"""
Model training and evaluation utilities.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate regression model performance.

    Parameters
    ----------
    y_true : pandas.Series or numpy.ndarray
        True values.
    y_pred : pandas.Series or numpy.ndarray
        Predicted values.

    Returns
    -------
    metrics : dict of str to float
        Dictionary of metric names and values: mse, rmse, r2.
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }


def log_mlflow_experiment(
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    experiment_name: str,
    run_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> None:
    """
    Log model training results to MLflow.

    Parameters
    ----------
    model : object
        Trained model.
    params : dict
        Model parameters.
    metrics : dict
        Evaluation metrics.
    experiment_name : str
        Name of the MLflow experiment.
    run_name : str, optional
        Optional name for this run.
    model_name : str, optional
        Optional name for the model.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        if model_name:
            mlflow.sklearn.log_model(model, model_name) # type: ignore
        else:
            mlflow.sklearn.log_model(model, "model") # type: ignore


def save_model(
    model: Any,
    filepath: Union[str, Path],
    engine: str = 'joblib'
) -> None:
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model.
    filepath : str or pathlib.Path
        Path to save the model.
    engine : str, optional
        Engine to save the model in ('joblib' or 'pickle'). Default is 'joblib'.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if engine == 'joblib':
        joblib.dump(model, filepath)
    elif engine == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def load_model(
    filepath: Union[str, Path],
    engine: str = 'joblib'
) -> Any:
    """
    Load a trained model from disk.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the saved model.
    engine : str, optional
        Engine the model was saved in ('joblib' or 'pickle'). Default is 'joblib'.

    Returns
    -------
    model : object
        The loaded model.
    """
    filepath = Path(filepath)

    if engine == 'joblib':
        return joblib.load(filepath)
    if engine == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
