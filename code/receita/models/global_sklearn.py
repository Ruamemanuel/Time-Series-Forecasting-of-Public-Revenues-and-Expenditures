"""
Modelos globais do sklearn (Ridge, Random Forest e MLP), treinados sobre todas as
séries não-ruído-branco com o mesmo dataset e protocolo do LGBM global.

  Ridge -- linear com L2. O series_id entra one-hot, dando um intercepto por
           série enquanto os coeficientes de lags/calendário são compartilhados
           (na prática, um painel com efeitos fixos).
  RF    -- floresta de árvores; series_id como número, que as árvores separam
           sozinhas, capturando não-linearidades e interações.
  MLP   -- rede (64, 32) com Adam e early stopping; series_id como número.

Todos reaproveitam o construir_dataset_global e a previsão recursiva do
lgbm_global (o prever_rolling chama modelo.predict, que serve a qualquer
estimador sklearn). Os cuidados contra vazamento são os mesmos do global.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRID_RF, GRID_MLP, GRID_RIDGE_ALPHAS, RANDOM_STATE, N_SPLITS_CV

logger = logging.getLogger(__name__)

# grid de alpha do Ridge, vindo do config
_RIDGE_ALPHAS = GRID_RIDGE_ALPHAS


# Pré-processador para Ridge (one-hot em series_id)

def _preprocessor_ridge(feature_cols: list) -> ColumnTransformer:
    """One-hot no series_id (vira efeito fixo por série) e StandardScaler nos
    lags e features de calendário."""
    lag_cal_cols = [c for c in feature_cols if c != "series_id"]
    return ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
             ["series_id"]),
            ("scaler", StandardScaler(), lag_cal_cols),
        ],
        remainder="drop",
    )


# Treinamento

def treinar_global_ridge(
    X_df         : pd.DataFrame,
    y            : np.ndarray,
    random_state : int = 42,
) -> Tuple[Pipeline, dict]:
    """Treina o Ridge global; o alpha é escolhido na última dobra do
    TimeSeriesSplit. Devolve (pipeline, info) com alpha, n_series e n_obs."""
    if len(X_df) == 0:
        raise ValueError("Dataset global vazio.")

    prep = _preprocessor_ridge(list(X_df.columns))

    # TimeSeriesSplit para RidgeCV — usa última dobra como validação
    tscv    = TimeSeriesSplit(n_splits=3)
    cv_list = list(tscv.split(X_df))
    if cv_list:
        tr_idx, val_idx = cv_list[-1]
        X_tr  = X_df.iloc[tr_idx]
        y_tr  = y[tr_idx]
        X_val = X_df.iloc[val_idx]
        y_val = y[val_idx]

        # Testa cada alpha manualmente (RidgeCV não suporta Pipeline diretamente)
        best_alpha = _RIDGE_ALPHAS[0]
        best_rmse  = float("inf")
        for alpha in _RIDGE_ALPHAS:
            p = Pipeline([
                ("prep",  _preprocessor_ridge(list(X_df.columns))),
                ("ridge", Ridge(alpha=alpha)),
            ])
            p.fit(X_tr, y_tr)
            rmse = float(np.sqrt(np.mean((p.predict(X_val) - y_val) ** 2)))
            if rmse < best_rmse:
                best_rmse  = rmse
                best_alpha = alpha
        logger.info("Ridge global: alpha ótimo = %.4g  (RMSE CV = %.4f)",
                    best_alpha, best_rmse)
    else:
        best_alpha = 1.0

    # Treino final em todos os dados
    pipeline = Pipeline([
        ("prep",  _preprocessor_ridge(list(X_df.columns))),
        ("ridge", Ridge(alpha=best_alpha)),
    ])
    pipeline.fit(X_df, y)

    n_series = int(X_df["series_id"].nunique())
    logger.info("Ridge global treinado: alpha=%g  n_series=%d  n_obs=%d",
                best_alpha, n_series, len(X_df))
    return pipeline, {"alpha": best_alpha, "n_series": n_series, "n_obs": len(X_df)}


def treinar_global_rf(
    X_df         : pd.DataFrame,
    y            : np.ndarray,
    random_state : int = 42,
) -> Tuple[RandomForestRegressor, dict]:
    """Treina a Random Forest global com RandomizedSearchCV (GRID_RF, n_iter=15)
    via TimeSeriesSplit; series_id entra como número. Devolve (modelo, info)."""
    if len(X_df) == 0:
        raise ValueError("Dataset global vazio.")

    # Converte series_id para float para compatibilidade total com sklearn
    X_arr = X_df.values.astype(float)

    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=GRID_RF,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_arr, y)
    modelo      = search.best_estimator_
    best_params = search.best_params_

    logger.info("Random Forest global treinado: best=%s  n_obs=%d",
                best_params, len(X_df))
    return modelo, {"n_estimators": best_params["n_estimators"],
                    "best_params": best_params, "n_obs": len(X_df)}


def treinar_global_mlp(
    X_df         : pd.DataFrame,
    y            : np.ndarray,
    random_state : int = 42,
) -> Tuple[Pipeline, dict]:
    """Treina o MLP global com GridSearchCV (GRID_MLP) via TimeSeriesSplit, com
    early stopping interno. Devolve (pipeline, info)."""
    if len(X_df) == 0:
        raise ValueError("Dataset global vazio.")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    param_grid = {
        "mlp__hidden_layer_sizes": GRID_MLP["hidden_layer_sizes"],
        "mlp__alpha"             : GRID_MLP["alpha"],
        "mlp__learning_rate_init": GRID_MLP["learning_rate_init"],
    }
    pipeline_base = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            activation="relu", solver="adam", max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=random_state, verbose=False,
        )),
    ])
    gs = GridSearchCV(pipeline_base, param_grid, cv=tscv,
                      scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_df.values, y)
    pipeline = gs.best_estimator_
    best_p   = gs.best_params_

    n_iter = pipeline.named_steps["mlp"].n_iter_
    hls    = pipeline.named_steps["mlp"].hidden_layer_sizes
    logger.info("MLP global treinado: best=%s  n_iter=%d  n_obs=%d",
                best_p, n_iter, len(X_df))
    return pipeline, {"hidden_layer_sizes": hls, "n_iter": n_iter,
                      "best_params": best_p, "n_obs": len(X_df)}


# Previsão rolling adaptada para sklearn

def prever_rolling_global_sklearn(
    modelo      : object,
    treino_vals : np.ndarray,
    teste_vals  : np.ndarray,
    teste_idx   : object,          # pd.PeriodIndex do conjunto de teste
    n_lags      : int,
    norm_mean   : float,
    norm_std    : float,
    series_id   : int,
    col_names   : list,            # _col_names(n_lags) da lgbm_global
    usa_array   : bool = False,    # True → passa np.array (RF); False → DataFrame (Ridge/MLP via Pipeline)
) -> np.ndarray:
    """Previsão recursiva para os modelos sklearn globais, com o mesmo protocolo
    do prever_rolling_lgbm: lags normalizados pelo treino da série, calendário a
    partir do índice e histórico realimentado com a própria previsão. Os valores
    reais do conjunto de teste não entram — `teste_vals` só dá o número de passos. Use
    `usa_array=True` para estimadores que recebem np.array (caso da RF). Devolve
    um array com len(teste_vals) previsões."""
    from clustering.pooling import normalizar, desnormalizar

    def _cal(periodo) -> list:
        m = periodo.month
        return [
            float(np.sin(2.0 * np.pi * m / 12.0)),
            float(np.cos(2.0 * np.pi * m / 12.0)),
            float((m - 1) // 3 + 1),
            float(periodo.year - 2019),
        ]

    history = list(np.asarray(treino_vals, dtype=float))
    preds   = []

    for i, real_val in enumerate(np.asarray(teste_vals, dtype=float)):
        raw_lags  = np.array(history[-n_lags:], dtype=float)
        norm_lags = normalizar(raw_lags, norm_mean, norm_std)
        lag_vec   = norm_lags[::-1].tolist()   # lag_1 = mais recente

        # Padding esquerdo para séries muito curtas
        while len(lag_vec) < n_lags:
            lag_vec.append(0.0)

        cal  = _cal(teste_idx[i])
        row  = lag_vec + cal + [float(series_id)]

        if usa_array:
            X_in = np.array([row], dtype=float)
        else:
            X_in = pd.DataFrame([row], columns=col_names)
            X_in["series_id"] = X_in["series_id"].astype(int)

        y_norm = float(modelo.predict(X_in)[0])
        y_pred = float(desnormalizar(np.array([y_norm]), norm_mean, norm_std)[0])

        preds.append(y_pred)
        history.append(float(y_pred))   # multi-step: alimenta com a PREVISÃO

    return np.array(preds, dtype=float)
