"""
Monta o dataset pooled a partir das séries de um cluster e faz a previsão
recursiva com o modelo treinado nesse pool.

Cuidados contra vazamento: só séries não-ruído-branco entram no pool (o chamador
garante isso); cada série é normalizada pelo (mean, std) do próprio treino; o
StandardScaler é ajustado só no pool de treino; e a previsão é recursiva,
realimentando a própria previsão (nenhum valor real do conjunto de teste é usado).

O dataset sempre traz os lags (z-score do treino) e, quando os índices temporais
são passados, também month_sin, month_cos, quarter e year_offset, todos
derivados só do índice (sem vazamento).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Features de calendário

_ANO_BASE_DEFAULT = 2019


def _cal_feats(periodo: pd.Period, ano_base: int = _ANO_BASE_DEFAULT) -> list:
    """
    Retorna [month_sin, month_cos, quarter, year_offset] para um pd.Period mensal.
    Codificação cíclica do mês garante continuidade entre dezembro e janeiro.
    """
    m = periodo.month
    return [
        float(np.sin(2.0 * np.pi * m / 12.0)),
        float(np.cos(2.0 * np.pi * m / 12.0)),
        float((m - 1) // 3 + 1),          # quarter 1–4
        float(periodo.year - ano_base),    # tendência secular
    ]


# Normalização por série

def calcular_norma(treino_vals: np.ndarray) -> tuple[float, float]:
    """Retorna (mean, std) do treino. std mínimo de 1e-9 para evitar /0."""
    v    = np.asarray(treino_vals, dtype=float)
    mean = float(np.mean(v))
    std  = float(np.std(v, ddof=1)) if len(v) > 1 else 1.0
    return mean, max(std, 1e-9)


def normalizar(vals: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(vals, dtype=float) - mean) / std


def desnormalizar(vals: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(vals, dtype=float) * std + mean


# Matriz de lags (+ calendário opcional)

def construir_lag_matrix(
    vals_norm: np.ndarray,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Matriz só de lags (X) e vetor alvo (y). Mantida por compatibilidade."""
    v = np.asarray(vals_norm, dtype=float)
    X, y = [], []
    for i in range(n_lags, len(v)):
        X.append(v[i - n_lags : i])
        y.append(v[i])
    if not X:
        return np.empty((0, n_lags), dtype=float), np.empty(0, dtype=float)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _construir_lags_cal(
    vals_norm : np.ndarray,
    n_lags    : int,
    idx       : pd.PeriodIndex | None,
    ano_base  : int = _ANO_BASE_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Lags e, se `idx` vier, também as features de calendário. O X tem n_lags
    colunas (sem calendário) ou n_lags + 4 (com)."""
    v      = np.asarray(vals_norm, dtype=float)
    use_cal = idx is not None
    n_feat  = n_lags + (4 if use_cal else 0)
    X, y   = [], []

    for i in range(n_lags, len(v)):
        row = list(v[i - n_lags : i])
        if use_cal:
            row += _cal_feats(idx[i], ano_base)
        X.append(row)
        y.append(float(v[i]))

    if not X:
        return np.empty((0, n_feat), dtype=float), np.empty(0, dtype=float)
    return np.array(X, dtype=float), np.array(y, dtype=float)


# Dataset pooled

def construir_dataset_pooled(
    codigos_cluster : list[str],
    treinos         : dict,
    n_lags          : int,
    treino_indices  : dict | None = None,   # {cod: pd.PeriodIndex}
    ano_base        : int = _ANO_BASE_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, dict, StandardScaler]:
    """Junta o treino de todas as séries do cluster num pool. Passando os índices
    em `treino_indices`, inclui as features de calendário. Devolve X_pool (já
    escalonado), y_pool (normalizado), o dict {codigo: (mean, std)} e o scaler."""
    use_cal     = treino_indices is not None
    n_feat      = n_lags + (4 if use_cal else 0)
    norm_params : dict = {}
    X_parts     : list = []
    y_parts     : list = []

    for cod in codigos_cluster:
        if cod not in treinos:
            continue
        vals = np.asarray(treinos[cod], dtype=float)
        mean, std = calcular_norma(vals)
        norm_params[cod] = (mean, std)

        vals_norm = normalizar(vals, mean, std)
        idx = treino_indices.get(cod) if use_cal else None
        X_s, y_s = _construir_lags_cal(vals_norm, n_lags, idx, ano_base)

        if len(X_s) > 0:
            X_parts.append(X_s)
            y_parts.append(y_s)

    # Fallback: pool vazio
    if not X_parts:
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, n_feat)))
        return (
            np.empty((0, n_feat), dtype=float),
            np.empty(0, dtype=float),
            norm_params,
            scaler,
        )

    X_pool = np.vstack(X_parts)
    y_pool = np.concatenate(y_parts)

    scaler = StandardScaler()
    X_pool = scaler.fit_transform(X_pool)

    return X_pool, y_pool, norm_params, scaler


# Previsão recursiva multi-step

def prever_rolling_pooled(
    modelo      : object,
    scaler      : StandardScaler,
    treino_vals : np.ndarray,
    teste_vals  : np.ndarray,
    n_lags      : int,
    norm_mean   : float,
    norm_std    : float,
    teste_idx   : pd.PeriodIndex | None = None,   # para features de calendário
    ano_base    : int = _ANO_BASE_DEFAULT,
) -> np.ndarray:
    """Previsão recursiva com o modelo pooled.

    A cada passo: pega os últimos n_lags do histórico, normaliza pela estatística
    do treino da série, junta o calendário (se `teste_idx` vier), passa pelo
    scaler do pool, prevê, desnormaliza e realimenta o histórico com a própria
    previsão. Os valores reais do conjunto de teste não entram — `teste_vals` só dá o número
    de passos. Devolve um array com len(teste_vals) previsões."""
    history   = list(np.asarray(treino_vals, dtype=float))
    predicoes = []
    use_cal   = teste_idx is not None

    for t, real_val in enumerate(np.asarray(teste_vals, dtype=float)):
        lags_raw  = np.array(history[-n_lags:], dtype=float)
        lags_norm = normalizar(lags_raw, norm_mean, norm_std)

        if use_cal:
            feats = np.concatenate([lags_norm, _cal_feats(teste_idx[t], ano_base)])
        else:
            feats = lags_norm

        feats_sc = scaler.transform(feats.reshape(1, -1))
        y_norm   = float(modelo.predict(feats_sc)[0])
        y_pred   = desnormalizar(np.array([y_norm]), norm_mean, norm_std)[0]

        predicoes.append(float(y_pred))
        history.append(float(y_pred))   # multi-step: alimenta com a PREVISÃO

    return np.array(predicoes, dtype=float)
