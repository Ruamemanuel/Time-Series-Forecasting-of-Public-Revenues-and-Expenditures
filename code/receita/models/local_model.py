"""
Modelos locais (pooled por cluster): Ridge, SVR e MLP treinados no pool de séries
de cada cluster. Os hiperparâmetros saem de uma busca por TimeSeriesSplit
(n_splits adaptativo) sobre o conjunto de treino pooled; o conjunto de teste nunca entra
aqui.

Resumo das escolhas de grid, com a literatura que as embasa:

SVR — só kernel RBF, que supera linear/polinomial em séries financeiras mensais
(Kim 2003; Tay & Cao 2001). C em {0.01..1000}, grade exponencial base 10 (Hsu,
Chang & Lin 2016). epsilon em {0.001, 0.01, 0.1, 0.5}, unindo as faixas de Tay &
Cao (2001) e Kim (2003). gamma fixo em "scale": como o pool já passa pelo
StandardScaler, equivale a 1/n_features (Hsu et al. 2016).

MLP — arquiteturas {(10,),(25,),(50,),(100,),(50,25)}, de uma camada rasa a uma de
duas camadas (Zhang 2003; Göçken et al. 2016; Makridakis et al. 2020). Ativação
tanh ou relu (Zhang 2003; Glorot et al. 2011). L2 alpha em {1e-4, 1e-3, 1e-2}
(Göçken et al. 2016). Adam com lr=0.001 fixo (Kingma & Ba 2015), max_iter=500 e
early stopping quando o pool tem ao menos 20 amostras.

As referências completas estão em models/REFERENCIAS.md.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from clustering.pooling import construir_dataset_pooled, prever_rolling_pooled
from utils.metrics import calcular_metricas

from config import GRID_SVR, GRID_MLP, GRID_RIDGE_ALPHAS

logger = logging.getLogger(__name__)


# Os grids vêm todos do config, para a comparação entre paradigmas ser justa.

# Teto de linhas do pool no SVR: acima disso, subamostra com semente fixa antes
# do grid search. O libsvm é O(n^2) em memória, então ~5000 linhas já dão uma
# matriz de kernel de ~200 MB — inviável num loop sobre 100+ séries.
_SVR_MAX_ROWS: int = 5_000


# Helpers internos

def _metricas(real: np.ndarray, pred: np.ndarray, params: str) -> dict:
    m = calcular_metricas(np.asarray(real, dtype=float),
                          np.asarray(pred, dtype=float))
    m["params"] = params
    m["preds"]  = pred
    return m


def _nan_result(params: str) -> dict:
    return {
        "RMSE": float("nan"), "MAE": float("nan"),
        "MAPE": float("nan"), "params": params,
        "preds": np.array([]),
    }


def _n_splits(n_pool: int) -> int:
    """TimeSeriesSplit adaptativo: 2 splits para pools pequenos, 3 para grandes."""
    return max(2, min(3, n_pool // 10))


def _subsample_pool(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Amostragem aleatoria simples do pool quando n > max_rows.

    Mantemos a ordem temporal dentro de cada serie (o subsample e feito no nivel
    do pool agregado, nao por serie), o que e suficiente porque o TimeSeriesSplit
    ja garante a avaliacao temporal correta.
    """
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    idx.sort()   # preserva ordem cronologica relativa
    return X[idx], y[idx]


# Ridge Pooled

def treinar_avaliar_pooled_ridge(
    codigos_cluster : list[str],
    treinos         : dict,
    codigo_alvo     : str,
    teste_vals      : np.ndarray,
    n_lags          : int,
    alpha           : float = None,
    treino_indices  : dict | None = None,   # {cod: pd.PeriodIndex}
    teste_idx       = None,                 # pd.PeriodIndex do conjunto de teste
) -> dict:
    """Treina o Ridge no pool do cluster e avalia no conjunto de teste da série-alvo com
    previsão recursiva. Se `alpha` for None, busca na última dobra do
    TimeSeriesSplit. Passando os índices, inclui features de calendário. Devolve
    RMSE, MAE, MAPE, params e previsões."""
    X_pool, y_pool, norm_params, scaler = construir_dataset_pooled(
        codigos_cluster, treinos, n_lags,
        treino_indices=treino_indices,
    )

    if len(X_pool) == 0 or codigo_alvo not in norm_params:
        return _nan_result(f"alpha={alpha}")

    # Busca de alpha via TimeSeriesSplit se não fornecido
    if alpha is None:
        tscv   = TimeSeriesSplit(n_splits=_n_splits(len(X_pool)))
        splits = list(tscv.split(X_pool))
        if splits:
            tr_idx, val_idx = splits[-1]
            best_alpha = GRID_RIDGE_ALPHAS[0]
            best_rmse  = float("inf")
            for a in GRID_RIDGE_ALPHAS:
                m = Ridge(alpha=a).fit(X_pool[tr_idx], y_pool[tr_idx])
                # RMSE no espaço normalizado (suficiente para comparar alphas)
                r = float(np.sqrt(np.mean(
                    (m.predict(X_pool[val_idx]) - y_pool[val_idx]) ** 2
                )))
                if r < best_rmse:
                    best_rmse  = r
                    best_alpha = a
            alpha = best_alpha
        else:
            alpha = 1.0

    modelo = Ridge(alpha=alpha)
    modelo.fit(X_pool, y_pool)

    mean, std   = norm_params[codigo_alvo]
    treino_alvo = treinos[codigo_alvo]

    preds = prever_rolling_pooled(
        modelo, scaler, treino_alvo, teste_vals, n_lags, mean, std,
        teste_idx=teste_idx,
    )
    return _metricas(teste_vals, preds, f"alpha={alpha}")


# SVR Pooled

def treinar_avaliar_pooled_svr(
    codigos_cluster : list[str],
    treinos         : dict,
    codigo_alvo     : str,
    teste_vals      : np.ndarray,
    n_lags          : int,
    random_state    : int = 42,
    treino_indices  : dict | None = None,
    teste_idx       = None,
) -> dict:
    """Treina o SVR com grid search (GRID_SVR, ver cabeçalho) no pool do cluster
    e avalia no conjunto de teste da série-alvo com previsão recursiva. Pools acima de
    _SVR_MAX_ROWS são subamostrados antes da busca."""
    X_pool, y_pool, norm_params, scaler = construir_dataset_pooled(
        codigos_cluster, treinos, n_lags,
        treino_indices=treino_indices,
    )

    if len(X_pool) < 6 or codigo_alvo not in norm_params:
        return _nan_result("{}")

    # Subamostrar pool grande antes do grid search
    X_fit, y_fit = _subsample_pool(X_pool, y_pool, _SVR_MAX_ROWS, random_state)

    cv = TimeSeriesSplit(n_splits=_n_splits(len(X_fit)))
    gs = GridSearchCV(
        SVR(),
        GRID_SVR,
        cv      = cv,
        scoring = "neg_mean_squared_error",
        n_jobs  = -1,
        refit   = True,
    )
    gs.fit(X_fit, y_fit)

    mean, std   = norm_params[codigo_alvo]
    treino_alvo = treinos[codigo_alvo]

    preds = prever_rolling_pooled(
        gs.best_estimator_, scaler, treino_alvo, teste_vals, n_lags, mean, std,
        teste_idx=teste_idx,
    )
    params_str = (
        f"C={gs.best_params_['C']}  "
        f"eps={gs.best_params_['epsilon']}  "
        f"kernel={gs.best_params_['kernel']}"
    )
    return _metricas(teste_vals, preds, params_str)


# MLP Pooled

def treinar_avaliar_pooled_mlp(
    codigos_cluster : list[str],
    treinos         : dict,
    codigo_alvo     : str,
    teste_vals      : np.ndarray,
    n_lags          : int,
    random_state    : int = 42,
    treino_indices  : dict | None = None,
    teste_idx       = None,
) -> dict:
    """Treina o MLP com grid search (GRID_MLP, ver cabeçalho) no pool do cluster
    e avalia no conjunto de teste da série-alvo com previsão recursiva. O early stopping só
    liga quando o pool tem ao menos 20 amostras."""
    X_pool, y_pool, norm_params, scaler = construir_dataset_pooled(
        codigos_cluster, treinos, n_lags,
        treino_indices=treino_indices,
    )

    if len(X_pool) < 6 or codigo_alvo not in norm_params:
        return _nan_result("{}")

    n_pool = len(X_pool)
    use_es = n_pool >= 20
    cv     = TimeSeriesSplit(n_splits=_n_splits(n_pool))

    base = MLPRegressor(
        solver           = "adam",
        max_iter         = 500,
        early_stopping   = use_es,
        n_iter_no_change = 10,
        random_state     = random_state,
    )
    gs = GridSearchCV(
        base,
        GRID_MLP,
        cv      = cv,
        scoring = "neg_mean_squared_error",
        n_jobs  = -1,
        refit   = True,
    )
    gs.fit(X_pool, y_pool)

    mean, std   = norm_params[codigo_alvo]
    treino_alvo = treinos[codigo_alvo]

    preds = prever_rolling_pooled(
        gs.best_estimator_, scaler, treino_alvo, teste_vals, n_lags, mean, std,
        teste_idx=teste_idx,
    )
    p = gs.best_params_
    params_str = (
        f"layers={p['hidden_layer_sizes']}  "
        f"alpha={p['alpha']}  "
        f"lr={p['learning_rate_init']}"
    )
    return _metricas(teste_vals, preds, params_str)
