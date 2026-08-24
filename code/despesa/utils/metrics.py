"""Métricas de erro usadas na avaliação (RMSE, MAE, MAPE)."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro percentual absoluto médio. Pula os meses com y_true == 0 (divisão por
    zero) e devolve nan se a série inteira for zero."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Junta RMSE, MAE e MAPE num dicionário."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE" : mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
