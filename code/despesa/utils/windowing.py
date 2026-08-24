"""Janelas deslizantes de lags para os modelos de regressão (SVR, MLP, etc.)."""

import numpy as np


def criar_janela(serie: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Transforma a série 1-D em matriz de lags (X) e vetor alvo (y). Com
    n_lags=3 e serie=[1,2,3,4,5], dá X=[[1,2,3],[2,3,4]] e y=[4,5]."""
    X, y = [], []
    for i in range(n_lags, len(serie)):
        X.append(serie[i - n_lags : i])
        y.append(serie[i])
    return np.array(X), np.array(y)
