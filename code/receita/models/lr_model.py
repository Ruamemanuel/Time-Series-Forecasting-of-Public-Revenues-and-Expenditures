"""
Regressão linear regularizada (Ridge) sobre janelas de lags. Normaliza X e y
com StandardScaler ajustado só no treino e, se o alpha não vier pronto, escolhe
o melhor por TimeSeriesSplit. A avaliação no conjunto de teste e a previsão final são
recursivas: cada passo realimenta o valor previsto como lag seguinte.
"""

import logging
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from utils.metrics import calcular_metricas
from utils.windowing import criar_janela

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_STATE, GRID_RIDGE_ALPHAS

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _buscar_alpha_ridge(
    X_tr_sc: np.ndarray,
    y_tr: np.ndarray,
    y_tr_sc: np.ndarray,
    scaler_y: StandardScaler,
    n_splits: int = 3,
) -> float:
    """Escolhe o alpha do Ridge testando GRID_RIDGE_ALPHAS na última dobra do
    TimeSeriesSplit e devolve o de menor RMSE."""
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_tr_sc))
    if not splits:
        return 1.0

    tr_idx, val_idx = splits[-1]
    if len(tr_idx) < 2:
        return 1.0

    best_alpha = GRID_RIDGE_ALPHAS[0]
    best_rmse  = float("inf")
    for alpha in GRID_RIDGE_ALPHAS:
        m = Ridge(alpha=alpha).fit(X_tr_sc[tr_idx], y_tr_sc[tr_idx])
        pred_sc = m.predict(X_tr_sc[val_idx])
        pred    = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
        r       = float(np.sqrt(np.mean((y_tr[val_idx] - pred) ** 2)))
        if r < best_rmse:
            best_rmse  = r
            best_alpha = alpha

    logger.debug("LR alpha busca: best_alpha=%.4g  CV-RMSE=%.2f", best_alpha, best_rmse)
    return best_alpha


def treinar_avaliar_lr(
    treino_vals: np.ndarray,
    teste_vals: np.ndarray,
    n_lags: int,
    alpha: float = None,
) -> dict:
    """Treina o Ridge no treino e mede o erro no conjunto de teste com previsão recursiva.
    Se `alpha` for None, busca por CV. Devolve params, previsões e métricas."""
    X_tr, y_tr = criar_janela(treino_vals, n_lags)

    # scalers só com o treino, pra não vazar informação do conjunto de teste
    scaler_X = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_sc  = scaler_X.transform(X_tr)
    y_tr_sc  = scaler_y.transform(y_tr.reshape(-1, 1)).ravel()

    if alpha is None:
        n_splits = min(3, max(2, len(X_tr_sc) - 1))
        alpha    = _buscar_alpha_ridge(X_tr_sc, y_tr, y_tr_sc, scaler_y, n_splits)

    modelo = Ridge(alpha=alpha).fit(X_tr_sc, y_tr_sc)

    # previsão recursiva: cada passo realimenta o que foi previsto
    seq   = list(treino_vals)
    preds = []
    for step in range(len(teste_vals)):
        janela    = np.array(seq[-n_lags:]).reshape(1, -1)
        janela_sc = scaler_X.transform(janela)
        pred_sc   = modelo.predict(janela_sc)
        pred      = scaler_y.inverse_transform(
            pred_sc.reshape(-1, 1)
        ).ravel()[0]
        preds.append(pred)
        seq.append(pred)

    metricas = calcular_metricas(teste_vals, np.array(preds))
    logger.debug(
        "LR: RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%",
        metricas["RMSE"], metricas["MAE"], metricas["MAPE"],
    )

    return {
        "params"     : f"Ridge(alpha={alpha})",
        "alpha"      : alpha,
        "scaler_X"   : scaler_X,
        "scaler_y"   : scaler_y,
        "modelo"     : modelo,
        "n_lags"     : n_lags,
        "preds_teste": preds,
        **metricas,
    }


def prever_lr(
    todos_vals: np.ndarray,
    n_lags: int,
    alpha: float,
    horizonte: int,
) -> list[float]:
    """Reajusta com a série inteira e projeta `horizonte` passos à frente."""
    X_all, y_all = criar_janela(todos_vals, n_lags)
    scaler_X = StandardScaler().fit(X_all)
    scaler_y = StandardScaler().fit(y_all.reshape(-1, 1))
    modelo   = Ridge(alpha=alpha).fit(
        scaler_X.transform(X_all),
        scaler_y.transform(y_all.reshape(-1, 1)).ravel(),
    )
    seq = list(todos_vals)
    fc  = []
    for _ in range(horizonte):
        jan = scaler_X.transform(np.array(seq[-n_lags:]).reshape(1, -1))
        p   = scaler_y.inverse_transform(
            modelo.predict(jan).reshape(-1, 1)
        ).ravel()[0]
        fc.append(p)
        seq.append(p)
    return fc
