"""
Modelo global LightGBM: um único modelo treinado sobre todas as séries (menos as
de ruído branco) ao mesmo tempo, aproveitando padrões que se repetem entre as
fontes. Costuma ajudar quando há muitas séries curtas que dividem sazonalidade
fiscal e drivers macroeconômicos (Montero-Manso et al., 2020).

Para não vazar informação: cada série é normalizada pelo (mean, std) do próprio
treino; as features de calendário saem só do índice temporal; o series_id entra
como categórica nativa do LightGBM; e o número de árvores é definido por early
stopping no treino, sem tocar no conjunto de teste. A previsão é recursiva (h=12): cada
passo realimenta a própria previsão.

Cada linha do dataset traz lag_1..lag_N (z-score, lag_1 = mais recente),
month_sin/month_cos (mês em codificação cíclica), quarter, year_offset
(year - ANO_BASE, como proxy de tendência) e o series_id.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clustering.pooling import calcular_norma, normalizar, desnormalizar
from utils.metrics import calcular_metricas

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    logger.warning("lightgbm nao instalado. Execute: pip install lightgbm")

# Importa parametros do config (usa defaults se config nao disponivel)
try:
    from config import (
        N_LAGS_GLOBAL, LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE,
        LGBM_NUM_LEAVES, LGBM_MIN_CHILD_SAMPLES, LGBM_SUBSAMPLE,
        LGBM_COLSAMPLE_BYTREE, LGBM_REG_ALPHA, LGBM_REG_LAMBDA,
        ANO_BASE_LGBM, GRID_LGBM_PARAMS,
    )
except ImportError:
    N_LAGS_GLOBAL           = 12
    LGBM_N_ESTIMATORS       = 1000
    LGBM_LEARNING_RATE      = 0.05
    LGBM_NUM_LEAVES         = 31
    LGBM_MIN_CHILD_SAMPLES  = 20
    LGBM_SUBSAMPLE          = 0.8
    LGBM_COLSAMPLE_BYTREE   = 0.8
    LGBM_REG_ALPHA          = 0.1
    LGBM_REG_LAMBDA         = 0.1
    ANO_BASE_LGBM           = 2019
    GRID_LGBM_PARAMS        = {
        "num_leaves"       : [15, 31, 63],
        "min_child_samples": [5, 20, 50],
    }

CAT_FEATURES = ["series_id"]


# Features de calendário

def _calendar_features(periodo) -> dict:
    """
    Extrai month_sin, month_cos, quarter, year_offset de um pd.Period mensal.
    Codificação cíclica do mês garante continuidade entre dezembro e janeiro.
    """
    month = periodo.month
    year  = periodo.year
    return {
        "month_sin"  : np.sin(2.0 * np.pi * month / 12.0),
        "month_cos"  : np.cos(2.0 * np.pi * month / 12.0),
        "quarter"    : int((month - 1) // 3 + 1),
        "year_offset": int(year - ANO_BASE_LGBM),
    }


def _col_names(n_lags: int) -> List[str]:
    return (
        [f"lag_{k}" for k in range(1, n_lags + 1)]
        + ["month_sin", "month_cos", "quarter", "year_offset", "series_id"]
    )


# Construção de features

def construir_features_serie(
    vals_norm : np.ndarray,
    periodos  : object,
    n_lags    : int,
    series_id : int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monta (X, y) normalizado de uma série, com observações de t=n_lags em
    diante. Se a série tiver menos de n_lags+1 pontos, devolve arrays vazios."""
    v = np.asarray(vals_norm, dtype=float)
    n = len(v)
    n_feat = n_lags + 5

    rows_X, rows_y = [], []
    for t in range(n_lags, n):
        # lag_1 = t-1 (mais recente), lag_N = t-N (mais antigo)
        lags = v[t - n_lags : t][::-1].tolist()
        cal  = _calendar_features(periodos[t])
        row  = lags + [
            cal["month_sin"], cal["month_cos"],
            cal["quarter"],   cal["year_offset"],
            series_id,
        ]
        rows_X.append(row)
        rows_y.append(float(v[t]))

    if not rows_X:
        return np.empty((0, n_feat), dtype=float), np.empty(0, dtype=float)

    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float)


def construir_dataset_global(
    series_data : Dict,
    n_lags      : int = N_LAGS_GLOBAL,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Tuple[float, float]], LabelEncoder]:
    """Empilha o treino de todas as séries num só dataset. Cada série é
    normalizada pelo próprio (mean, std), vira matriz de features (lags +
    calendário + id) e é concatenada.

    Recebe {codigo: (treino_vals, treino_index)} e devolve o DataFrame de
    features, o vetor de alvos normalizados, o dict {codigo: (mean, std)} e o
    LabelEncoder dos códigos."""
    codigos   = sorted(str(c) for c in series_data.keys())
    label_enc = LabelEncoder()
    label_enc.fit(codigos)

    norm_params : Dict[str, Tuple[float, float]] = {}
    X_parts, y_parts = [], []

    for cod, (treino_vals, treino_idx) in series_data.items():
        vals      = np.asarray(treino_vals, dtype=float)
        mean, std = calcular_norma(vals)
        norm_params[str(cod)] = (mean, std)

        vals_norm = normalizar(vals, mean, std)
        sid       = int(label_enc.transform([str(cod)])[0])

        X_s, y_s  = construir_features_serie(vals_norm, treino_idx, n_lags, sid)
        if len(X_s) > 0:
            X_parts.append(X_s)
            y_parts.append(y_s)

    cols = _col_names(n_lags)

    if not X_parts:
        return pd.DataFrame(columns=cols), np.empty(0), norm_params, label_enc

    X_global             = np.vstack(X_parts)
    y_global             = np.concatenate(y_parts)
    X_df                 = pd.DataFrame(X_global, columns=cols)
    X_df["series_id"]    = X_df["series_id"].astype(int)

    logger.info(
        "Dataset global: %d observacoes | %d series | %d features",
        len(X_df), len(series_data), len(cols),
    )
    return X_df, y_global, norm_params, label_enc


# Treinamento

def _build_lgbm_params(random_state: int = 42) -> dict:
    return {
        "objective"        : "regression",
        "metric"           : "rmse",
        "learning_rate"    : LGBM_LEARNING_RATE,
        "num_leaves"       : LGBM_NUM_LEAVES,
        "min_child_samples": LGBM_MIN_CHILD_SAMPLES,
        "subsample"        : LGBM_SUBSAMPLE,
        "subsample_freq"   : 1,
        "colsample_bytree" : LGBM_COLSAMPLE_BYTREE,
        "reg_alpha"        : LGBM_REG_ALPHA,
        "reg_lambda"       : LGBM_REG_LAMBDA,
        "random_state"     : random_state,
        "verbosity"        : -1,
        "force_col_wise"   : True,
    }


def _otimizar_params_lgbm(
    X_df     : pd.DataFrame,
    y        : np.ndarray,
    params_base : dict,
    n_splits : int = 3,
) -> tuple:
    """Testa as combinações de num_leaves x min_child_samples (GRID_LGBM_PARAMS)
    na última dobra do TimeSeriesSplit e devolve a de menor RMSE."""
    from itertools import product as iterproduct

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_df))
    if not splits:
        return LGBM_NUM_LEAVES, LGBM_MIN_CHILD_SAMPLES

    tr_idx, val_idx = splits[-1]
    X_tr,  y_tr     = X_df.iloc[tr_idx],  y[tr_idx]
    X_val, y_val    = X_df.iloc[val_idx], y[val_idx]

    best_rmse = float("inf")
    best_nl   = LGBM_NUM_LEAVES
    best_mcs  = LGBM_MIN_CHILD_SAMPLES

    for nl, mcs in iterproduct(
        GRID_LGBM_PARAMS["num_leaves"],
        GRID_LGBM_PARAMS["min_child_samples"],
    ):
        p = dict(params_base)
        p["num_leaves"]        = nl
        p["min_child_samples"] = mcs

        dtrain = lgb.Dataset(
            X_tr, label=y_tr,
            categorical_feature=CAT_FEATURES, free_raw_data=False
        )
        dval   = lgb.Dataset(
            X_val, label=y_val, reference=dtrain,
            categorical_feature=CAT_FEATURES, free_raw_data=False
        )
        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        m = lgb.train(
            p, dtrain,
            valid_sets=[dval],
            callbacks=callbacks,
            num_boost_round=300,
        )
        preds = m.predict(X_val)
        rmse  = float(np.sqrt(np.mean((preds - y_val) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_nl   = nl
            best_mcs  = mcs

    logger.info(
        "LGBM params busca: num_leaves=%d  min_child_samples=%d  CV-RMSE=%.4f",
        best_nl, best_mcs, best_rmse,
    )
    return best_nl, best_mcs


def _otimizar_n_estimators(
    X_df     : pd.DataFrame,
    y        : np.ndarray,
    params   : dict,
    n_splits : int = 3,
) -> int:
    """Define o número de árvores por early stopping, usando a última dobra do
    TimeSeriesSplit (a mais próxima do futuro) como validação. Piso de 50."""
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_df))

    if not splits:
        return LGBM_N_ESTIMATORS // 2

    tr_idx, val_idx = splits[-1]
    X_tr,  y_tr     = X_df.iloc[tr_idx],  y[tr_idx]
    X_val, y_val    = X_df.iloc[val_idx], y[val_idx]

    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature=CAT_FEATURES, free_raw_data=False
    )
    dval   = lgb.Dataset(
        X_val, label=y_val, reference=dtrain,
        categorical_feature=CAT_FEATURES, free_raw_data=False
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model_cv = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        callbacks=callbacks,
        num_boost_round=LGBM_N_ESTIMATORS,
    )

    best_n = max(50, model_cv.best_iteration)
    logger.info("Early stopping: n_estimators otimo = %d", best_n)
    return best_n


def treinar_lgbm_global(
    X_df         : pd.DataFrame,
    y            : np.ndarray,
    n_splits     : int = 3,
    random_state : int = 42,
) -> Tuple[object, int]:
    """Treina o LightGBM global: primeiro busca os hiperparâmetros e o número de
    árvores por TimeSeriesSplit/early stopping, depois reajusta com todo o treino.
    Devolve (modelo, n_estimators)."""
    if not _HAS_LGBM:
        raise ImportError("lightgbm nao instalado. Execute: pip install lightgbm")

    if len(X_df) == 0:
        raise ValueError("Dataset global vazio — nenhuma serie contribuiu com dados.")

    params = _build_lgbm_params(random_state)

    # Etapa 1a: busca de num_leaves e min_child_samples via TimeSeriesSplit
    best_nl, best_mcs = _otimizar_params_lgbm(X_df, y, params, n_splits=n_splits)
    params["num_leaves"]        = best_nl
    params["min_child_samples"] = best_mcs

    # Etapa 1b: n_estimators via early stopping
    n_est = _otimizar_n_estimators(X_df, y, params, n_splits=n_splits)

    # Etapa 2: treino final em todos os dados
    dtrain = lgb.Dataset(
        X_df, label=y,
        categorical_feature=CAT_FEATURES, free_raw_data=False
    )
    modelo = lgb.train(
        params,
        dtrain,
        num_boost_round=n_est,
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    logger.info(
        "Modelo global treinado: %d arvores | %d obs | %d features",
        n_est, len(X_df), X_df.shape[1],
    )
    return modelo, n_est


# Importância de features

def importancia_features(modelo, n_lags: int = N_LAGS_GLOBAL) -> pd.DataFrame:
    """
    Retorna DataFrame com importância das features do modelo treinado.
    Tipo 'gain' (redução acumulada de impureza) é mais informativo que 'split'.
    """
    cols = _col_names(n_lags)
    imp  = modelo.feature_importance(importance_type="gain")
    df   = pd.DataFrame({"feature": cols, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# Previsão recursiva multi-step

def prever_rolling_lgbm(
    modelo      : object,
    treino_vals : np.ndarray,
    teste_vals  : np.ndarray,
    treino_idx  : object,
    teste_idx   : object,
    n_lags      : int,
    norm_mean   : float,
    norm_std    : float,
    series_id   : int,
) -> np.ndarray:
    """Previsão recursiva multi-step com o LightGBM global.

    A cada passo: pega os últimos n_lags do histórico, normaliza pelo (mean, std)
    do treino da série, monta o vetor [lags, calendário, series_id], prevê,
    desnormaliza e realimenta o histórico com a própria previsão. Os valores reais
    do conjunto de teste não entram — `teste_vals` serve só para saber quantos passos prever.
    Devolve um array com len(teste_vals) previsões."""
    cols    = _col_names(n_lags)
    history = list(np.asarray(treino_vals, dtype=float))
    preds   = []

    for i, real_val in enumerate(np.asarray(teste_vals, dtype=float)):
        # Lags: últimos n_lags do histórico, normalizados
        raw_lags  = np.array(history[-n_lags:], dtype=float)
        norm_lags = normalizar(raw_lags, norm_mean, norm_std)

        # lag_1 = mais recente → reverter (history[-n_lags:] está em ordem cronológica)
        lag_vec = norm_lags[::-1].tolist()

        # Padding esquerdo com 0 se histórico < n_lags (séries muito curtas)
        while len(lag_vec) < n_lags:
            lag_vec.append(0.0)

        # Features de calendário do período atual
        cal = _calendar_features(teste_idx[i])

        row = lag_vec + [
            cal["month_sin"], cal["month_cos"],
            cal["quarter"],   cal["year_offset"],
            series_id,
        ]

        X_df_pred = pd.DataFrame(
            [row], columns=cols, dtype=float
        )
        X_df_pred["series_id"] = X_df_pred["series_id"].astype(int)

        y_norm = float(modelo.predict(X_df_pred)[0])
        y_pred = float(desnormalizar(np.array([y_norm]), norm_mean, norm_std)[0])

        preds.append(y_pred)
        history.append(float(y_pred))   # multi-step: alimenta com a PREVISÃO

    return np.array(preds, dtype=float)


# Interface de avaliação

def avaliar_lgbm(
    modelo      : object,
    treino_vals : np.ndarray,
    teste_vals  : np.ndarray,
    treino_idx  : object,
    teste_idx   : object,
    n_lags      : int,
    norm_mean   : float,
    norm_std    : float,
    series_id   : int,
) -> dict:
    """Avalia o LightGBM global numa série (previsão recursiva) e devolve RMSE,
    MAE, MAPE e o array de previsões."""
    p = prever_rolling_lgbm(
        modelo, treino_vals, teste_vals,
        treino_idx, teste_idx, n_lags,
        norm_mean, norm_std, series_id,
    )
    m = calcular_metricas(
        np.asarray(teste_vals, dtype=float),
        p,
    )
    m["preds"] = p
    return m
