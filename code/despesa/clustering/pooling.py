"""
Monta o dataset pooled a partir das séries de um cluster e faz a previsão
recursiva com o modelo treinado nesse pool.

Cuidados contra vazamento: cada série é normalizada pelo (mean, std) do próprio
treino; o StandardScaler é ajustado só no pool de treino; e a previsão é
recursiva, realimentando a própria previsão — nenhum valor real do conjunto de teste é
usado, igual aos modelos individuais.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


# Normalizacao por serie

def calcular_norma(treino_vals: np.ndarray) -> tuple[float, float]:
    """Retorna (mean, std) do treino. std minimo de 1e-9 para evitar /0."""
    v    = np.asarray(treino_vals, dtype=float)
    mean = float(np.mean(v))
    std  = float(np.std(v, ddof=1)) if len(v) > 1 else 1.0
    return mean, max(std, 1e-9)


def normalizar(vals: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(vals, dtype=float) - mean) / std


def desnormalizar(vals: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(vals, dtype=float) * std + mean


# Matriz de lags

def construir_lag_matrix(
    vals_norm: np.ndarray,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de lags (X) e vetor alvo (y) a partir da série normalizada. X fica
    vazio se a série tiver menos de n_lags + 1 pontos."""
    v = np.asarray(vals_norm, dtype=float)
    X, y = [], []
    for i in range(n_lags, len(v)):
        X.append(v[i - n_lags : i])
        y.append(v[i])
    if not X:
        return np.empty((0, n_lags), dtype=float), np.empty(0, dtype=float)
    return np.array(X, dtype=float), np.array(y, dtype=float)


# Dataset pooled

def construir_dataset_pooled(
    codigos_cluster: list[str],
    treinos: dict,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray, dict, StandardScaler]:
    """Junta o treino de todas as séries do cluster num pool: cada série é
    normalizada pelo próprio (mean, std), vira matriz de lags e é empilhada; o
    StandardScaler é ajustado no resultado. Devolve X_pool (já escalonado),
    y_pool (normalizado), o dict {codigo: (mean, std)} e o scaler."""
    norm_params: dict = {}
    X_parts: list     = []
    y_parts: list     = []

    for cod in codigos_cluster:
        if cod not in treinos:
            continue
        vals = np.asarray(treinos[cod], dtype=float)
        mean, std = calcular_norma(vals)
        norm_params[cod] = (mean, std)

        vals_norm = normalizar(vals, mean, std)
        X_s, y_s  = construir_lag_matrix(vals_norm, n_lags)

        if len(X_s) > 0:
            X_parts.append(X_s)
            y_parts.append(y_s)

    # Scaler fallback caso pool vazio
    if not X_parts:
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, n_lags)))
        return (
            np.empty((0, n_lags), dtype=float),
            np.empty(0, dtype=float),
            norm_params,
            scaler,
        )

    X_pool = np.vstack(X_parts)
    y_pool = np.concatenate(y_parts)

    scaler = StandardScaler()
    X_pool = scaler.fit_transform(X_pool)

    return X_pool, y_pool, norm_params, scaler


# Previsao recursiva multi-step

def prever_rolling_pooled(
    modelo,
    scaler: StandardScaler,
    treino_vals: np.ndarray,
    teste_vals: np.ndarray,
    n_lags: int,
    norm_mean: float,
    norm_std: float,
) -> np.ndarray:
    """Previsão recursiva com o modelo pooled.

    A cada passo: pega os últimos n_lags do histórico, normaliza pela estatística
    do treino da série, passa pelo scaler do pool, prevê, desnormaliza e
    realimenta o histórico com a própria previsão. Os valores reais do conjunto de teste não
    entram — `teste_vals` só define o número de passos. Devolve um array com
    len(teste_vals) previsões."""
    history   = list(np.asarray(treino_vals, dtype=float))
    predicoes = []

    for _ in range(len(np.asarray(teste_vals, dtype=float))):
        lags_raw  = np.array(history[-n_lags:], dtype=float)
        lags_norm = normalizar(lags_raw, norm_mean, norm_std)
        lags_sc   = scaler.transform(lags_norm.reshape(1, -1))

        y_norm = float(modelo.predict(lags_sc)[0])
        y_pred = desnormalizar(np.array([y_norm]), norm_mean, norm_std)[0]

        predicoes.append(float(y_pred))
        history.append(float(y_pred))   # valor PREVISTO alimenta o proximo passo

    return np.array(predicoes, dtype=float)
